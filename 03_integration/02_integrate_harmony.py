#!/usr/bin/env python3
"""
=============================================================================
File      : 02_integrate_harmony.py
Directory : 03_integration
Purpose   : Run Harmony integration on the PCA embedding, evaluate batch mixing (iLISI, batch silhouette), and save the integrated object with UMAP coordinates.
=============================================================================
"""

import os
import random
from multiprocessing import Pool

import anndata
import numpy as np
import pandas as pd
import scanpy as sc
import scib
import torch
from sklearn.metrics import silhouette_samples, silhouette_score
from tqdm import tqdm

# ============================================================================
# Configuration: Set the project root directory before running
# ============================================================================
proj_dir = os.environ.get("PROJ_DIR", "/path/to/your/project")  # <-- MODIFY THIS or export PROJ_DIR

os.environ["OPENBLAS_NUM_THREADS"] = "32"
os.environ["MKL_NUM_THREADS"] = "32"
os.environ["OMP_NUM_THREADS"] = "32"
os.environ["OPENBLAS_DYNAMIC"] = "FALSE"
os.environ["MKL_DYNAMIC"] = "FALSE"

random.seed(123)
torch.set_num_threads(16)
sc.settings.n_jobs = 8

data_dir = os.path.join(proj_dir, "data", "human")
it_dir = os.path.join(data_dir, "output", "Integration")
umap_dir = os.path.join(data_dir, "output", "Umap")

BATCH_KEY = "sample_new"
EMBED_KEY = "X_pca_harmony"

print(f"Scanpy version: {sc.__version__}")
print(f"Anndata version: {anndata.__version__}\n")


def save_umap_coordinates(adata, obsm_key, out_path):
    """Save UMAP coordinates stored under ``obsm_key`` to a CSV file.

    Args:
        adata: AnnData object containing the UMAP embedding.
        obsm_key: Key in ``adata.obsm`` holding the UMAP embedding.
        out_path: Output CSV path.
    """
    coordinates = pd.DataFrame(
        data=adata.obsm[obsm_key],
        index=adata.obs_names,
        columns=["UMAP_1", "UMAP_2"],
    )
    coordinates.to_csv(out_path)
    print(f"Saved: {out_path}")


def calculate_batch_silhouette(adata, embedding_key, n_sample=20000, random_state=123):
    """Compute the batch silhouette score on a subsample of cells.

    Args:
        adata: AnnData object with the integration embedding in ``obsm``
            and the batch labels in ``obs[BATCH_KEY]``.
        embedding_key: Key in ``adata.obsm`` holding the embedding.
        n_sample: Number of cells to subsample for speed.
        random_state: Random seed for the subsampling.

    Returns:
        float: Batch silhouette score (lower = better batch mixing).
    """
    if BATCH_KEY not in adata.obs:
        raise ValueError(f"Batch label '{BATCH_KEY}' not found in adata.obs")
    if embedding_key not in adata.obsm:
        raise ValueError(f"Embedding '{embedding_key}' not found in adata.obsm")

    n_sample = min(n_sample, adata.n_obs)
    rng = np.random.RandomState(random_state)
    sample_idx = rng.choice(adata.n_obs, size=n_sample, replace=False)

    X_emb = adata.obsm[embedding_key][sample_idx]
    batches = adata.obs[BATCH_KEY].values[sample_idx]
    return silhouette_score(X_emb, batches, metric="euclidean")


def compute_asw(adata, umap_key):
    """Compute the average silhouette width (ASW) of the batch labels.

    Higher ASW means the embedding separates batches; the batch-mixing
    score is its negation, so ``-ASW`` is reported by ``compute_avg_batch``.

    Args:
        adata: AnnData object with a UMAP embedding and batch labels.
        umap_key: Key in ``adata.obsm`` holding the UMAP embedding.

    Returns:
        float: Negative mean batch silhouette (lower = better mixing).
    """
    batch_labels = adata.obs[BATCH_KEY]
    silhouette_vals = silhouette_samples(adata.obsm[umap_key], batch_labels)
    return -np.mean(silhouette_vals)


def compute_single_graph_conn(args):
    """Return the fraction of neighbors sharing the cell's batch label."""
    batch_labels, cell_idx, neighbors = args
    return np.mean(batch_labels[neighbors] == batch_labels[cell_idx])


def compute_graph_conn(adata, connect_key, n_workers=16):
    """Compute the average graph connectivity (batch mixing on the KNN graph).

    Args:
        adata: AnnData object with a connectivity matrix in ``obsp``.
        connect_key: Key in ``adata.obsp`` holding the connectivity matrix.
        n_workers: Number of parallel processes.

    Returns:
        float: Average fraction of same-batch neighbors (higher = better).
    """
    batch_labels = adata.obs[BATCH_KEY].astype("category").cat.codes.to_numpy()
    connectivity = adata.obsp[connect_key]

    tasks = [
        (batch_labels, i, connectivity[i].indices)
        for i in range(connectivity.shape[0])
    ]
    with Pool(processes=n_workers) as pool:
        results = list(
            tqdm(pool.imap(compute_single_graph_conn, tasks), total=len(tasks))
        )
    return float(np.mean(results))


def compute_avg_batch(adata, umap_key, connect_key):
    """Compute the composite AvgBatch metric (ASW + graph connectivity).

    Args:
        adata: AnnData object with a UMAP embedding and connectivity.
        umap_key: Key in ``adata.obsm`` holding the UMAP embedding.
        connect_key: Key in ``adata.obsp`` holding the connectivity matrix.

    Returns:
        float: AvgBatch score (lower = better batch mixing).
    """
    print(f"Computing graph connectivity for '{connect_key}' ...")
    graph_conn = compute_graph_conn(adata, connect_key)
    print(f"Computing ASW for '{umap_key}' ...")
    asw_batch = compute_asw(adata, umap_key)
    return (asw_batch + graph_conn) / 2


# ---------------------------------------------------------------------------
# 1. Load the integration-ready object and run Harmony
# ---------------------------------------------------------------------------
ad_hs = sc.read_h5ad(os.path.join(it_dir, "hs_it_temp4.h5ad"))
ad_hs.raw = None

sc.external.pp.harmony_integrate(
    ad_hs, key=BATCH_KEY, basis="X_pca", max_iter_harmony=30
)
np.savetxt(
    os.path.join(it_dir, "hs_it_harmony_hvg5000.csv"),
    ad_hs.obsm[EMBED_KEY],
    delimiter=",",
)

# ---------------------------------------------------------------------------
# 2. Batch-mixing evaluation
# ---------------------------------------------------------------------------
ilisi_harmony = scib.me.ilisi_graph(
    ad_hs, batch_key=BATCH_KEY, type_="embed", use_rep=EMBED_KEY, n_cores=2
)
print(f"iLISI (harmony): {ilisi_harmony}")

batch_sil = calculate_batch_silhouette(ad_hs, EMBED_KEY, n_sample=200000)
print(f"Batch silhouette (harmony): {batch_sil:.4f}")

# ---------------------------------------------------------------------------
# 3. UMAP coordinates at several neighborhood sizes
# ---------------------------------------------------------------------------
for k in [100, 50, 30]:
    sc.pp.neighbors(ad_hs, n_neighbors=k, n_pcs=50, use_rep=EMBED_KEY)
    sc.tl.umap(ad_hs)
    ad_hs.obsm[f"X_umap_harmony_k_{k}"] = ad_hs.obsm["X_umap"]
    if k == 30:
        ad_hs.obsp["connectivities_harmony_k30"] = ad_hs.obsp["connectivities"]
    save_umap_coordinates(
        ad_hs,
        f"X_umap_harmony_k_{k}",
        os.path.join(umap_dir, f"umap_harmony_k_{k}_hvg5000.csv"),
    )

avg_batch_harmony = compute_avg_batch(
    ad_hs, "X_umap_harmony_k_30", "connectivities_harmony_k30"
)
print(f"Harmony AvgBatch: {avg_batch_harmony}")

# ---------------------------------------------------------------------------
# 4. Save the integrated object
# ---------------------------------------------------------------------------
qc_columns = [
    "pct_counts_in_top_50_genes",
    "pct_counts_in_top_100_genes",
    "pct_counts_in_top_200_genes",
    "pct_counts_in_top_500_genes",
]
ad_hs.obs = ad_hs.obs.drop(columns=[c for c in qc_columns if c in ad_hs.obs.columns])

out_file = os.path.join(it_dir, "hs_it_harmony_hvg5000.h5ad")
ad_hs.write(out_file)
print("Saved:", out_file)
