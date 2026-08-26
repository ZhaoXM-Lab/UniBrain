#!/usr/bin/env python3
"""
=============================================================================
File      : 18_L5_cluster.py
Directory : 05_annotation
Purpose   : Level 5 clustering of each type_3 subpopulation with a resolution sweep and consensus partition.
=============================================================================
"""

# ============================================================================
# Configuration: set PROJ_DIR to the project root before running
# ============================================================================
import os

proj_dir = os.path.join(
    os.environ.get("PROJ_DIR", "path/to/your/project"), "data", "human"
)
import re
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import anndata
import numpy as np
import pandas as pd
import scipy.sparse as sparse
import scanpy as sc
from joblib import Parallel, delayed
from scipy import signal
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD

print("Scanpy version: %s" % sc.__version__)
print("Anndata version: %s\n" % anndata.__version__)


def calculate_compactness(adata, key_name):
    """Compute the mean within-cluster distance (compactness) for a clustering.

    Args:
        adata: AnnData object.
        key_name: ``obs`` column with the cluster labels.

    Returns:
        Mean over clusters of the mean cell-to-centroid distance in the
        harmony-corrected PCA embedding.
    """
    cluster_list = np.unique(adata.obs[key_name])
    compactness = []

    for cluster in cluster_list:
        emb = adata.obsm["X_pca_harmony"][adata.obs[key_name] == cluster]
        emb_centered = emb - emb.mean(axis=0)
        distances = np.linalg.norm(emb_centered, axis=1)
        compactness.append(distances.mean())

    return np.mean(compactness)


def _leiden_process(adata, res):
    """Run Leiden clustering at one resolution and return the labels."""
    key = "leiden_l5_%.1f" % res
    sc.tl.leiden(adata, resolution=res, key_added=key)
    print("Finish leiden, resolution: %.1f" % res)
    return adata.obs[key].astype(str)


def run_leiden(adata, resolution):
    """Run the Leiden resolution sweep in parallel."""
    with Pool(15) as pool:
        process_func = partial(_leiden_process, adata)
        results = pool.map(process_func, resolution)

    for res, result in zip(resolution, results):
        adata.obs["leiden_l5_%.1f" % res] = result
    return adata


def find_optimal_resolution(compactness, resolutions):
    """Select the optimal resolution from the compactness curve.

    Uses the inflection point (second-derivative sign change) of the
    smoothed compactness curve, falling back to the global minimum.

    Args:
        compactness: Compactness values for each resolution.
        resolutions: The corresponding resolutions.

    Returns:
        Index of the optimal resolution.
    """
    y = np.array(compactness)
    x = np.array(resolutions)

    # Fallback: direct minimum
    min_idx = np.argmin(y)

    # Preferred: inflection point of the smoothed curve
    try:
        # Smooth the curve to reduce noise (odd window size)
        window_size = min(5, len(y) // 3)
        if window_size % 2 == 0:
            window_size += 1

        if len(y) > window_size:
            y_smooth = signal.savgol_filter(
                y, window_length=window_size, polyorder=2
            )
            dy = np.gradient(y_smooth)
            d2y = np.gradient(dy)

            # Find all inflection points (second-derivative zero crossings)
            inflection_points = []
            for i in range(1, len(d2y)):
                if (d2y[i - 1] <= 0 and d2y[i] >= 0) or (
                    d2y[i - 1] >= 0 and d2y[i] <= 0
                ):
                    inflection_points.append(i)

            # Pick the inflection point with the lowest compactness
            if inflection_points:
                min_inflection_idx = min(
                    inflection_points, key=lambda i: y[i]
                )
                return min_inflection_idx
    except Exception as e:
        print("Inflection detection failed, using the minimum method: %s" % e)

    return min_idx


def leiden_pipeline(adata, cell_type, resolutions=np.linspace(0.1, 1.5, 15)):
    """Run the Leiden sweep for one subpopulation and pick the resolution.

    Args:
        adata: AnnData object for the subpopulation.
        cell_type: Type name (used in log messages).
        resolutions: Resolutions to sweep.

    Returns:
        Tuple of (adata with sweep results in ``obs``, optimal resolution).
    """
    adata_light = adata.copy()
    del adata_light.X
    del adata_light.raw
    adata_light = run_leiden(adata_light, resolutions)
    adata.obs = adata_light.obs.copy()
    del adata_light

    # Compute compactness and select the optimal resolution
    compactness = [
        calculate_compactness(adata, "leiden_l5_%.1f" % r) for r in resolutions
    ]
    optimal_idx = find_optimal_resolution(compactness, resolutions)
    optimal_resolution = resolutions[optimal_idx]

    return adata, optimal_resolution


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
        AnnData object with consensus labels stored in ``obs['cluster_5']``.
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
    adata.obs["cluster_5"] = pd.Categorical(cell_clusters.astype(str))

    if verbose:
        print("Consensus clustering completed")

    return adata


def process(adata, cell_type_i):
    """Cluster every type_3 subpopulation within one cell type.

    Subpopulations with fewer than 50 cells keep their type_3 label as a
    single cluster. Larger subpopulations are clustered with the Leiden sweep
    plus consensus clustering.

    Args:
        adata: AnnData object containing multiple type_3 subpopulations.
        cell_type_i: Cell type name (used in log messages).

    Returns:
        Tuple of (concatenated AnnData object, clustering summary DataFrame).
    """
    adata_list = []
    clustering_records = []

    for type_i in adata.obs["type_3_abbr"].unique():
        sub_adata = adata[adata.obs["type_3_abbr"] == type_i].copy()
        if sub_adata.n_obs < 50:
            # Too few cells: keep the original type_3 label as one cluster
            sub_adata.obs["cluster_5"] = sub_adata.obs["type_3_abbr"]
            adata_list.append(sub_adata)
            record = {
                "type_3_abbr": type_i,
                "optimal_resolution": np.nan,
                "number_of_clusters": 1,
            }
            clustering_records.append(record)
            print(
                "Skipping %s, only %d cells." % (type_i, sub_adata.n_obs)
            )
            continue

        sub_adata, resolution = leiden_pipeline(
            sub_adata, type_i, resolutions=np.linspace(0.1, 1.5, 15)
        )

        # Round the resolution to match the sweep column names
        optimal_resolution = round(resolution, 1)
        optimalk = len(
            sub_adata.obs["leiden_l5_%.1f" % optimal_resolution].unique()
        )

        sub_adata = hybrid_bipartite_consensus(
            sub_adata, resolution=optimal_resolution, n_runs=100,
            optimalk=optimalk, maxDropPct=0.5, verbose=True, n_jobs=10,
        )
        adata_list.append(sub_adata)
        final_num_clusters = len(sub_adata.obs["cluster_5"].cat.categories)
        record = {
            "type_3_abbr": type_i,
            "optimal_resolution": optimal_resolution,
            "number_of_clusters": final_num_clusters,
        }
        clustering_records.append(record)

    adata = anndata.concat(adata_list, axis=0)
    clustering_df = pd.DataFrame(clustering_records)

    return adata, clustering_df


def sanitize_type_name(cell_type_i):
    """Convert a cell type name into a filesystem-safe name."""
    type_name_i = re.sub(r'[\\/:*?"<>|+ ]', "_", cell_type_i)
    if type_name_i and type_name_i[0] in "0123456789!@#$%^&*()-=+":
        type_name_i = "type_" + type_name_i
    return type_name_i


# ----------------------------------------------------------------------------
# Main: L5 clustering for every cell type
# ----------------------------------------------------------------------------
type_list = [
    "5-HT", "ACh", "Ast", "CALB2+ GABA", "COPC", "CP", "EC", "Epc",
    "Fib", "GB", "Granule", "HIP Glu", "HT Glu", "IPC",
    "L2/3-CUX2 Glu", "L2/3-SATB2 Glu", "L4-RORB Glu", "L5-BCL11B Glu",
    "L5/6-THEMIS Glu", "L5/6-TLE4 Glu", "LHX6+LAMP5+ GABA", "MB GABA",
    "MSN", "Mac", "Mic", "NB", "OPC", "Oli", "PC", "PVALB+ GABA",
    "Per", "RELN+ GABA", "RG", "SST+ GABA",  # "Remove" and "Th GABA" excluded
    "Th Glu", "VIP+ GABA", "VSMC",
]

l4_dir = os.path.join(proj_dir, "output/Annotation_hvg5000/L4")
l5_dir = os.path.join(proj_dir, "output/Annotation_hvg5000/L5")

for cell_type_i in type_list:
    print("Processing type: %s" % cell_type_i)

    type_name_i = sanitize_type_name(cell_type_i)

    save_dir = Path(os.path.join(l5_dir, type_name_i))
    output_file = save_dir / ("hs_anno_5_%s_cluster.h5ad" % type_name_i)
    if output_file.exists():
        print(
            "Output file %s already exists. Skipping %s."
            % (output_file, cell_type_i)
        )
        continue

    os.makedirs(save_dir, exist_ok=True)

    adata = sc.read_h5ad(
        os.path.join(l4_dir, "hs_anno_4_" + type_name_i + ".h5ad")
    )
    adata, clustering_df = process(adata, cell_type_i)

    adata.write_h5ad(filename=output_file)
    clustering_df.to_csv(
        path_or_buf=save_dir / ("%s_cluster_info.csv" % type_name_i),
        index=False,
    )
