#!/usr/bin/env python3
"""
=============================================================================
File      : 03_integrate_scanorama.py
Directory : 03_integration
Purpose   : Run Scanorama integration on the PCA embedding, evaluate batch mixing (iLISI, batch silhouette), and save the integrated object with UMAP coordinates.
=============================================================================
"""

import os
import random

import anndata
import numpy as np
import pandas as pd
import scanpy as sc
import scib
import torch
from sklearn.metrics import silhouette_score

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
EMBED_KEY = "X_scanorama"

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


# ---------------------------------------------------------------------------
# 1. Load the integration-ready object and run Scanorama
# ---------------------------------------------------------------------------
ad_hs = sc.read_h5ad(os.path.join(it_dir, "hs_it_temp4.h5ad"))
ad_hs.raw = None

# Scanorama requires cells of the same batch to be contiguous
idx = ad_hs.obs.sort_values(BATCH_KEY).index
ad_hs = ad_hs[idx,]

sc.external.pp.scanorama_integrate(
    ad_hs,
    key=BATCH_KEY,
    basis="X_pca",
    adjusted_basis=EMBED_KEY,
    batch_size=1000,
    verbose=1,
)
np.savetxt(
    os.path.join(it_dir, "hs_it_scanorama_hvg5000.csv"),
    ad_hs.obsm[EMBED_KEY],
    delimiter=",",
)

# ---------------------------------------------------------------------------
# 2. Batch-mixing evaluation
# ---------------------------------------------------------------------------
ilisi_scanorama = scib.me.ilisi_graph(
    ad_hs, batch_key=BATCH_KEY, type_="embed", use_rep=EMBED_KEY, n_cores=2
)
print(f"iLISI (scanorama): {ilisi_scanorama}")

batch_sil = calculate_batch_silhouette(ad_hs, EMBED_KEY, n_sample=200000)
print(f"Batch silhouette (scanorama): {batch_sil:.4f}")

# ---------------------------------------------------------------------------
# 3. UMAP coordinates at several neighborhood sizes
# ---------------------------------------------------------------------------
for k in [100, 15, 30]:
    sc.pp.neighbors(ad_hs, n_neighbors=k, n_pcs=50, use_rep=EMBED_KEY)
    sc.tl.umap(ad_hs)
    ad_hs.obsm[f"X_umap_scanorama_k_{k}"] = ad_hs.obsm["X_umap"]
    if k == 30:
        ad_hs.obsp["connectivities_scanorama_k30"] = ad_hs.obsp["connectivities"]
    save_umap_coordinates(
        ad_hs,
        f"X_umap_scanorama_k_{k}",
        os.path.join(umap_dir, f"umap_scanorama_k_{k}.csv"),
    )

# ---------------------------------------------------------------------------
# 4. Save the integrated object
# ---------------------------------------------------------------------------
out_file = os.path.join(it_dir, "hs_it_scanorama_hvg5000.h5ad")
ad_hs.write(out_file)
print("Saved:", out_file)
