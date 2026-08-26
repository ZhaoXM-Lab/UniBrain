#!/usr/bin/env python3
"""
=============================================================================
File      : 16_L4_cluster.py
Directory : 05_annotation
Purpose   : Level 4 clustering of each type_2 subpopulation with a resolution sweep and consensus partition.
=============================================================================
"""

# ============================================================================
# Configuration: set PROJ_DIR to the project root before running
# ============================================================================
import os

proj_dir = os.path.join(
    os.environ.get("PROJ_DIR", "path/to/your/project"), "data", "human"
)
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import anndata
import numpy as np
import pandas as pd
import scipy.sparse as sparse
import scanpy as sc
from joblib import Parallel, delayed
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD

print("Scanpy version: %s" % sc.__version__)
print("Anndata version: %s\n" % anndata.__version__)


def _leiden_process(adata, res):
    """Run Leiden clustering at one resolution and return the labels."""
    key = "leiden_l4_%.1f" % res
    sc.tl.leiden(adata, resolution=res, key_added=key)
    print("Finish leiden, resolution: %.1f" % res)
    return adata.obs[key].astype(str)


def run_leiden(adata, resolution):
    """Run the Leiden resolution sweep in parallel."""
    with Pool(15) as pool:
        process_func = partial(_leiden_process, adata)
        results = pool.map(process_func, resolution)

    for res, result in zip(resolution, results):
        adata.obs["leiden_l4_%.1f" % res] = result
    return adata


def leiden_pipeline(cell_type, resolutions=np.linspace(0.1, 1.5, 15),
                    data_dir=None):
    """Run the Leiden sweep for one cell type and save the clusterings.

    Args:
        cell_type: Type name used in file names.
        resolutions: Resolutions to sweep.
        data_dir: Directory containing the harmonized input object.

    Returns:
        AnnData object with all sweep results in ``obs``.
    """
    if data_dir is None:
        data_dir = os.path.join(proj_dir, "output/Annotation_hvg5000/L4")
    data_path = Path(data_dir)

    adata = sc.read_h5ad(data_path / "hs_anno_3_%s_inter.h5ad" % cell_type)
    print(adata)

    # Run the sweep on a lightweight copy (graph only)
    adata_light = adata.copy()
    del adata_light.X
    del adata_light.raw
    adata_light = run_leiden(adata_light, resolutions)
    adata.obs = adata_light.obs.copy()
    del adata_light

    # Save clustering results
    cluster_cols = ["leiden_l4_%.1f" % r for r in resolutions]
    adata.obs[cluster_cols].to_csv(
        data_path / "clustering_results_4_%s.csv" % cell_type
    )
    adata.write_h5ad(filename=data_path / "hs_anno_4_%s_cluster.h5ad" % cell_type)

    return adata


def hybrid_bipartite_consensus(adata, resolution=0.5, n_runs=100, optimalk=7,
                               maxDropPct=0.5, verbose=True, n_jobs=10):
    """Hybrid bipartite-graph consensus clustering.

    Runs Leiden clustering ``n_runs`` times at a fixed resolution, builds the
    cell-to-cluster incidence matrix, and applies spectral clustering followed
    by K-means on the joint cell/cluster embedding to obtain a consensus
    partition.

    Args:
        adata: AnnData object to cluster.
        resolution: Leiden resolution used for each run.
        n_runs: Number of Leiden runs.
        optimalk: Target number of consensus clusters (K-means k).
        maxDropPct: Maximum tolerated fraction of clusterings to discard.
        verbose: Whether to print progress information.
        n_jobs: Number of parallel jobs.

    Returns:
        AnnData object with consensus labels stored in ``obs['cluster_4']``.
    """
    # 1. Generate the clustering solutions
    if verbose:
        print(
            "Generating %d clusterings at resolution %s..." % (n_runs, resolution)
        )

    adata_light = adata.copy()
    del adata_light.X, adata_light.raw

    def _single_run(i):
        sc.tl.leiden(
            adata_light, resolution=resolution, key_added="tmp_cluster",
            random_state=i,
        )
        return adata_light.obs["tmp_cluster"].values

    results = Parallel(n_jobs=n_jobs)(
        delayed(_single_run)(i) for i in range(n_runs)
    )
    del adata_light

    print("Finish consensus clustering.")

    cluster_df = pd.DataFrame(
        np.array(results).T,
        index=adata.obs_names,
        columns=["Solution_%d" % i for i in range(n_runs)],
    )

    # 2. Drop invalid solutions (single-cluster runs)
    if verbose:
        print("Filtering invalid clusterings...")

    valid_cols = [
        col for col in cluster_df.columns if len(cluster_df[col].unique()) > 1
    ]
    drop_pct = 1 - len(valid_cols) / n_runs
    if verbose:
        print(
            "Dropped %.2f%% of clusterings (%d remain)"
            % (drop_pct * 100, len(valid_cols))
        )

    if drop_pct > maxDropPct:
        raise ValueError(
            "Would drop %.2f%% of clusterings, exceeds maxDropPct of %.2f%%"
            % (drop_pct * 100, maxDropPct * 100)
        )

    cluster_df = cluster_df[valid_cols]

    # 3. Build the sparse bipartite cell-to-cluster incidence matrix A
    if verbose:
        print("Building bipartite matrix A...")

    rows, cols, data = [], [], []
    col_idx = 0
    for col in cluster_df.columns:
        clusters = cluster_df[col].values
        unique_clusters = np.unique(clusters)
        cluster_map = {
            cluster: idx for idx, cluster in enumerate(unique_clusters)
        }

        for i, cluster in enumerate(clusters):
            rows.append(i)
            cols.append(col_idx + cluster_map[cluster])
            data.append(1)

        col_idx += len(unique_clusters)

    n_cells = cluster_df.shape[0]
    n_clusters = col_idx
    A = sparse.coo_matrix((data, (rows, cols)), shape=(n_cells, n_clusters))
    A = A.astype(np.float32)

    # 4. Spectral embedding via truncated SVD
    if verbose:
        print("Performing spectral clustering with k=%d..." % optimalk)

    svd = TruncatedSVD(n_components=optimalk, algorithm="arpack")
    U = svd.fit_transform(A)
    V = svd.components_.T

    # Normalize rows of U and V
    U_norm = U / np.linalg.norm(U, axis=1, keepdims=True)
    V_norm = V / np.linalg.norm(V, axis=1, keepdims=True)

    # 5. Joint embedding of cells and clusters
    Z = np.vstack((U_norm, V_norm))

    # 6. K-means clustering of the joint embedding
    if verbose:
        print("Running K-means clustering...")

    kmeans = KMeans(n_clusters=optimalk, n_init=50, random_state=0)
    cluster_labels = kmeans.fit_predict(Z)

    # 7. Extract the cell labels
    cell_clusters = cluster_labels[:n_cells]
    adata.obs["cluster_4"] = pd.Categorical(cell_clusters.astype(str))

    if verbose:
        print("Consensus clustering completed")

    return adata


def process(adata, type_i, optimalk, resolution=0.5):
    """Run consensus clustering for one cell type and save the result."""
    adata = hybrid_bipartite_consensus(
        adata, resolution=resolution, n_runs=100, optimalk=optimalk,
        maxDropPct=0.5, verbose=True, n_jobs=10,
    )
    sc.tl.paga(adata, groups="cluster_4")
    sc.tl.umap(adata, init_pos="paga", maxiter=1000)

    adata.write_h5ad(
        filename=os.path.join(
            proj_dir,
            "output/Annotation_hvg5000/L4/hs_anno_4_" + type_i + "_cluster.h5ad",
        )
    )


data_dir = os.path.join(proj_dir, "output/Annotation_hvg5000/L4")
resolutions = np.linspace(0.1, 1.5, 15)

# --- type_2 population: 5-HT ---
# Note: the sanitized name (15_L4_subcluster_integration) is "type_5-HT"
type_i = "type_5-HT"
ad_hs_i = leiden_pipeline(type_i, resolutions=resolutions, data_dir=data_dir)
resolution = 1.2
optimalk = len(ad_hs_i.obs["leiden_l4_1.2"].value_counts())
process(ad_hs_i, type_i, optimalk, resolution=resolution)

# --- type_2 population: ACh ---
type_i = "ACh"
ad_hs_i = leiden_pipeline(type_i, resolutions=resolutions, data_dir=data_dir)
resolution = 0.5
optimalk = len(ad_hs_i.obs["leiden_l4_0.5"].value_counts())
process(ad_hs_i, type_i, optimalk, resolution=resolution)

# --- type_2 population: Ast ---
type_i = "Ast"
ad_hs_i = leiden_pipeline(type_i, resolutions=resolutions, data_dir=data_dir)
resolution = 0.2
optimalk = len(ad_hs_i.obs["leiden_l4_0.2"].value_counts())
process(ad_hs_i, type_i, optimalk, resolution=resolution)

# --- type_2 population: CALB2+ GABA ---
type_i = "CALB2__GABA"
ad_hs_i = leiden_pipeline(type_i, resolutions=resolutions, data_dir=data_dir)
resolution = 0.5
optimalk = len(ad_hs_i.obs["leiden_l4_0.5"].value_counts())
process(ad_hs_i, type_i, optimalk, resolution=resolution)

# --- type_2 population: COPC ---
type_i = "COPC"
ad_hs_i = leiden_pipeline(type_i, resolutions=resolutions, data_dir=data_dir)
resolution = 0.8
optimalk = len(ad_hs_i.obs["leiden_l4_0.8"].value_counts())
process(ad_hs_i, type_i, optimalk, resolution=resolution)

# --- type_2 population: CP ---
type_i = "CP"
ad_hs_i = leiden_pipeline(type_i, resolutions=resolutions, data_dir=data_dir)
resolution = 0.5
optimalk = len(ad_hs_i.obs["leiden_l4_0.5"].value_counts())
process(ad_hs_i, type_i, optimalk, resolution=resolution)

# --- type_2 population: EC ---
type_i = "EC"
ad_hs_i = leiden_pipeline(type_i, resolutions=resolutions, data_dir=data_dir)
resolution = 0.8
optimalk = len(ad_hs_i.obs["leiden_l4_0.8"].value_counts())
process(ad_hs_i, type_i, optimalk, resolution=resolution)

# --- type_2 population: Epc ---
type_i = "Epc"
ad_hs_i = leiden_pipeline(type_i, resolutions=resolutions, data_dir=data_dir)
resolution = 0.6
optimalk = len(ad_hs_i.obs["leiden_l4_0.6"].value_counts())
process(ad_hs_i, type_i, optimalk, resolution=resolution)

# --- type_2 population: Fib ---
type_i = "Fib"
ad_hs_i = leiden_pipeline(type_i, resolutions=resolutions, data_dir=data_dir)
resolution = 0.2
optimalk = len(ad_hs_i.obs["leiden_l4_0.2"].value_counts())
process(ad_hs_i, type_i, optimalk, resolution=resolution)

# --- type_2 population: GB ---
type_i = "GB"
ad_hs_i = leiden_pipeline(type_i, resolutions=resolutions, data_dir=data_dir)
resolution = 1.0
optimalk = len(ad_hs_i.obs["leiden_l4_1.0"].value_counts())
process(ad_hs_i, type_i, optimalk, resolution=resolution)

# --- type_2 population: Granule ---
type_i = "Granule"
ad_hs_i = leiden_pipeline(type_i, resolutions=resolutions, data_dir=data_dir)
resolution = 0.9
optimalk = len(ad_hs_i.obs["leiden_l4_0.9"].value_counts())
process(ad_hs_i, type_i, optimalk, resolution=resolution)

# --- type_2 population: HIP Glu ---
type_i = "HIP_Glu"
ad_hs_i = leiden_pipeline(type_i, resolutions=resolutions, data_dir=data_dir)
resolution = 0.7
optimalk = len(ad_hs_i.obs["leiden_l4_0.7"].value_counts())
process(ad_hs_i, type_i, optimalk, resolution=resolution)

# --- type_2 population: HT Glu ---
type_i = "HT_Glu"
ad_hs_i = leiden_pipeline(type_i, resolutions=resolutions, data_dir=data_dir)
resolution = 0.7
optimalk = len(ad_hs_i.obs["leiden_l4_0.7"].value_counts())
process(ad_hs_i, type_i, optimalk, resolution=resolution)

# --- type_2 population: IPC ---
type_i = "IPC"
ad_hs_i = leiden_pipeline(type_i, resolutions=resolutions, data_dir=data_dir)
resolution = 0.7
optimalk = len(ad_hs_i.obs["leiden_l4_0.7"].value_counts())
process(ad_hs_i, type_i, optimalk, resolution=resolution)

# --- type_2 population: L2/3-CUX2 Glu ---
type_i = "L2_3-CUX2_Glu"
ad_hs_i = leiden_pipeline(type_i, resolutions=resolutions, data_dir=data_dir)
resolution = 0.6
optimalk = len(ad_hs_i.obs["leiden_l4_0.6"].value_counts())
process(ad_hs_i, type_i, optimalk, resolution=resolution)

# --- type_2 population: L2/3-SATB2 Glu ---
type_i = "L2_3-SATB2_Glu"
ad_hs_i = leiden_pipeline(type_i, resolutions=resolutions, data_dir=data_dir)
resolution = 1.2
optimalk = len(ad_hs_i.obs["leiden_l4_1.2"].value_counts())
process(ad_hs_i, type_i, optimalk, resolution=resolution)

# --- type_2 population: L4-RORB Glu ---
type_i = "L4-RORB_Glu"
ad_hs_i = leiden_pipeline(type_i, resolutions=resolutions, data_dir=data_dir)
resolution = 0.2
optimalk = len(ad_hs_i.obs["leiden_l4_0.2"].value_counts())
process(ad_hs_i, type_i, optimalk, resolution=resolution)

# --- type_2 population: L5-BCL11B Glu ---
type_i = "L5-BCL11B_Glu"
ad_hs_i = leiden_pipeline(type_i, resolutions=resolutions, data_dir=data_dir)
resolution = 1.3
optimalk = len(ad_hs_i.obs["leiden_l4_1.3"].value_counts())
process(ad_hs_i, type_i, optimalk, resolution=resolution)

# --- type_2 population: L5/6-THEMIS Glu ---
type_i = "L5_6-THEMIS_Glu"
ad_hs_i = leiden_pipeline(type_i, resolutions=resolutions, data_dir=data_dir)
resolution = 0.4
optimalk = len(ad_hs_i.obs["leiden_l4_0.4"].value_counts())
process(ad_hs_i, type_i, optimalk, resolution=resolution)

# --- type_2 population: L5/6-TLE4 Glu ---
type_i = "L5_6-TLE4_Glu"
ad_hs_i = leiden_pipeline(type_i, resolutions=resolutions, data_dir=data_dir)
resolution = 0.8
optimalk = len(ad_hs_i.obs["leiden_l4_0.8"].value_counts())
process(ad_hs_i, type_i, optimalk, resolution=resolution)

# --- type_2 population: LHX6+LAMP5+ GABA ---
type_i = "LHX6_LAMP5__GABA"
ad_hs_i = leiden_pipeline(type_i, resolutions=resolutions, data_dir=data_dir)
resolution = 0.8
optimalk = len(ad_hs_i.obs["leiden_l4_0.8"].value_counts())
process(ad_hs_i, type_i, optimalk, resolution=resolution)

# --- type_2 population: MB GABA ---
type_i = "MB_GABA"
ad_hs_i = leiden_pipeline(type_i, resolutions=resolutions, data_dir=data_dir)
resolution = 0.5
optimalk = len(ad_hs_i.obs["leiden_l4_0.5"].value_counts())
process(ad_hs_i, type_i, optimalk, resolution=resolution)

# --- type_2 population: MSN ---
type_i = "MSN"
ad_hs_i = leiden_pipeline(type_i, resolutions=resolutions, data_dir=data_dir)
resolution = 0.6
optimalk = len(ad_hs_i.obs["leiden_l4_0.6"].value_counts())
process(ad_hs_i, type_i, optimalk, resolution=resolution)

# --- type_2 population: Mac ---
type_i = "Mac"
ad_hs_i = leiden_pipeline(type_i, resolutions=resolutions, data_dir=data_dir)
resolution = 0.6
optimalk = len(ad_hs_i.obs["leiden_l4_0.6"].value_counts())
process(ad_hs_i, type_i, optimalk, resolution=resolution)

# --- type_2 population: Mic ---
type_i = "Mic"
ad_hs_i = leiden_pipeline(type_i, resolutions=resolutions, data_dir=data_dir)
resolution = 0.6
optimalk = len(ad_hs_i.obs["leiden_l4_0.6"].value_counts())
process(ad_hs_i, type_i, optimalk, resolution=resolution)

# --- type_2 population: NB ---
type_i = "NB"
ad_hs_i = leiden_pipeline(type_i, resolutions=resolutions, data_dir=data_dir)
resolution = 0.6
optimalk = len(ad_hs_i.obs["leiden_l4_0.6"].value_counts())
process(ad_hs_i, type_i, optimalk, resolution=resolution)

# --- type_2 population: OPC ---
type_i = "OPC"
ad_hs_i = leiden_pipeline(type_i, resolutions=resolutions, data_dir=data_dir)
resolution = 0.8
optimalk = len(ad_hs_i.obs["leiden_l4_0.8"].value_counts())
process(ad_hs_i, type_i, optimalk, resolution=resolution)

# --- type_2 population: Oli ---
type_i = "Oli"
ad_hs_i = leiden_pipeline(type_i, resolutions=resolutions, data_dir=data_dir)
resolution = 0.9
optimalk = len(ad_hs_i.obs["leiden_l4_0.9"].value_counts())
process(ad_hs_i, type_i, optimalk, resolution=resolution)

# --- type_2 population: PC ---
type_i = "PC"
ad_hs_i = leiden_pipeline(type_i, resolutions=resolutions, data_dir=data_dir)
resolution = 0.7
optimalk = len(ad_hs_i.obs["leiden_l4_0.7"].value_counts())
process(ad_hs_i, type_i, optimalk, resolution=resolution)

# --- type_2 population: PVALB+ GABA ---
type_i = "PVALB__GABA"
ad_hs_i = leiden_pipeline(type_i, resolutions=resolutions, data_dir=data_dir)
resolution = 0.2
optimalk = len(ad_hs_i.obs["leiden_l4_0.2"].value_counts())
process(ad_hs_i, type_i, optimalk, resolution=resolution)

# --- type_2 population: Per (retry with the consensus approach) ---
type_i = "Per"
ad_hs_i = leiden_pipeline(type_i, resolutions=resolutions, data_dir=data_dir)
resolution = 0.4
optimalk = len(ad_hs_i.obs["leiden_l4_0.4"].value_counts())
process(ad_hs_i, type_i, optimalk, resolution=resolution)

# --- type_2 population: RELN+ GABA ---
type_i = "RELN__GABA"
ad_hs_i = leiden_pipeline(type_i, resolutions=resolutions, data_dir=data_dir)
resolution = 0.6
optimalk = len(ad_hs_i.obs["leiden_l4_0.6"].value_counts())
process(ad_hs_i, type_i, optimalk, resolution=resolution)

# --- type_2 population: RG ---
type_i = "RG"
ad_hs_i = leiden_pipeline(type_i, resolutions=resolutions, data_dir=data_dir)
resolution = 0.4
optimalk = len(ad_hs_i.obs["leiden_l4_0.4"].value_counts())
process(ad_hs_i, type_i, optimalk, resolution=resolution)

# --- type_2 population: SST+ GABA ---
type_i = "SST__GABA"
ad_hs_i = leiden_pipeline(type_i, resolutions=resolutions, data_dir=data_dir)
resolution = 0.6
optimalk = len(ad_hs_i.obs["leiden_l4_0.6"].value_counts())
process(ad_hs_i, type_i, optimalk, resolution=resolution)

# Note: Th GABA (187 cells) was skipped, too small for consensus clustering.

# --- type_2 population: Th Glu ---
type_i = "Th_Glu"
ad_hs_i = leiden_pipeline(type_i, resolutions=resolutions, data_dir=data_dir)
resolution = 0.3
optimalk = len(ad_hs_i.obs["leiden_l4_0.3"].value_counts())
process(ad_hs_i, type_i, optimalk, resolution=resolution)

# --- type_2 population: VIP+ GABA ---
type_i = "VIP__GABA"
ad_hs_i = leiden_pipeline(type_i, resolutions=resolutions, data_dir=data_dir)
resolution = 0.6
optimalk = len(ad_hs_i.obs["leiden_l4_0.6"].value_counts())
process(ad_hs_i, type_i, optimalk, resolution=resolution)

# --- type_2 population: VSMC ---
type_i = "VSMC"
ad_hs_i = leiden_pipeline(type_i, resolutions=resolutions, data_dir=data_dir)
resolution = 0.2
optimalk = len(ad_hs_i.obs["leiden_l4_0.2"].value_counts())
process(ad_hs_i, type_i, optimalk, resolution=resolution)
