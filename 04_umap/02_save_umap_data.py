#!/usr/bin/env python3
"""
=============================================================================
File      : 02_save_umap_data.py
Directory : 04_umap
Purpose   : Assemble the final UMAP-ready AnnData object with aligned embeddings from all integration methods.
=============================================================================
"""

import os
import random

import anndata
import numpy as np
import scanpy as sc
import torch

# ============================================================================
# Configuration: Set the project root directory before running
# ============================================================================
proj_dir = os.environ.get("PROJ_DIR", "/path/to/your/project")  # <-- MODIFY THIS or export PROJ_DIR

random.seed(123)
torch.set_num_threads(16)

data_dir = os.path.join(proj_dir, "data", "human")
it_dir = os.path.join(data_dir, "output", "Integration")
umap_dir = os.path.join(data_dir, "output", "Umap")

print(f"Scanpy version: {sc.__version__}")
print(f"Anndata version: {anndata.__version__}\n")


def align_embedding(base_cells, embedding_adata, embed_key):
    """Align an embedding to the base cell order, padding missing cells with NaN.

    Args:
        base_cells: Index of the base AnnData (target cell order).
        embedding_adata: AnnData whose cells are a subset of ``base_cells``.
        embed_key: Key in ``embedding_adata.obsm`` holding the embedding.

    Returns:
        numpy.ndarray: Embedding matrix aligned to ``base_cells``.
    """
    embedding = embedding_adata.obsm[embed_key]
    full = np.full((len(base_cells), embedding.shape[1]), np.nan)
    cell_index_map = {cell: idx for idx, cell in enumerate(embedding_adata.obs_names)}
    for idx, cell in enumerate(base_cells):
        if cell in cell_index_map:
            full[idx] = embedding[cell_index_map[cell]]
    return full


# ---------------------------------------------------------------------------
# 1. Load the base object and the integration embeddings
# ---------------------------------------------------------------------------
ad_hs = sc.read_h5ad(os.path.join(it_dir, "hs_it_temp4.h5ad"))

ad_hs_harmony = sc.read_h5ad(os.path.join(it_dir, "hs_it_harmony_hvg5000.h5ad"))
ad_hs_scanorama = sc.read_h5ad(os.path.join(it_dir, "hs_it_scanorama_hvg5000.h5ad"))
ad_hs_scvi = sc.read_h5ad(os.path.join(it_dir, "hs_it_scVI_hvg5000.h5ad"))

ad_hs.obsm["X_pca_harmony"] = align_embedding(
    ad_hs.obs_names, ad_hs_harmony, "X_pca_harmony"
)
ad_hs.obsm["X_pca_scanorama"] = align_embedding(
    ad_hs.obs_names, ad_hs_scanorama, "X_scanorama"
)
ad_hs.obsm["X_scvi"] = align_embedding(ad_hs.obs_names, ad_hs_scvi, "X_scvi")

# ---------------------------------------------------------------------------
# 2. Drop redundant QC columns and save
# ---------------------------------------------------------------------------
qc_columns = [
    "pct_counts_in_top_50_genes",
    "pct_counts_in_top_100_genes",
    "pct_counts_in_top_200_genes",
    "pct_counts_in_top_500_genes",
]
ad_hs.obs = ad_hs.obs.drop(columns=[c for c in qc_columns if c in ad_hs.obs.columns])

out_file = os.path.join(umap_dir, "hs_umap_hvg5000.h5ad")
ad_hs.write_h5ad(out_file)
print("Saved:", out_file)
