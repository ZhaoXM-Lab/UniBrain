#!/usr/bin/env python3
"""
=============================================================================
File      : 01_run_umap.py
Directory : 04_umap
Purpose   : Run UMAP for all integration methods and save the coordinates as CSV files.
=============================================================================
"""

import os
import random

import anndata
import numpy as np
import pandas as pd
import scanpy as sc
import torch

# ============================================================================
# Configuration: Set the project root directory before running
# ============================================================================
proj_dir = os.environ.get("PROJ_DIR", "/path/to/your/project")  # <-- MODIFY THIS or export PROJ_DIR

random.seed(123)
torch.set_num_threads(16)
sc.settings.n_jobs = 8

data_dir = os.path.join(proj_dir, "data", "human")
it_dir = os.path.join(data_dir, "output", "Integration")
umap_dir = os.path.join(data_dir, "output", "Umap")

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


def compute_umap(adata, embedding_key, neighbor_k, out_prefix, hvg_suffix=""):
    """Compute UMAP embeddings at several neighborhood sizes and save them.

    Args:
        adata: AnnData object with the integration embedding in ``obsm``.
        embedding_key: Key in ``adata.obsm`` holding the embedding.
        neighbor_k: List of neighborhood sizes to evaluate.
        out_prefix: Output file prefix (method name).
        hvg_suffix: Optional HVG suffix appended to the CSV names.
    """
    for k in neighbor_k:
        sc.pp.neighbors(adata, n_neighbors=k, n_pcs=50, use_rep=embedding_key)
        sc.tl.umap(adata)
        adata.obsm[f"X_umap_{out_prefix}_k_{k}"] = adata.obsm["X_umap"]
        if k == 30:
            adata.obsp[f"connectivities_{out_prefix}_k30"] = adata.obsp["connectivities"]
        save_umap_coordinates(
            adata,
            f"X_umap_{out_prefix}_k_{k}",
            os.path.join(umap_dir, f"umap_{out_prefix}_k_{k}{hvg_suffix}.csv"),
        )


# ---------------------------------------------------------------------------
# Harmony
# ---------------------------------------------------------------------------
ad_hs = sc.read_h5ad(os.path.join(it_dir, "hs_it_harmony_hvg5000.h5ad"))
compute_umap(ad_hs, "X_pca_harmony", [100, 50, 30], "harmony")

# ---------------------------------------------------------------------------
# Scanorama
# ---------------------------------------------------------------------------
ad_hs = sc.read_h5ad(os.path.join(it_dir, "hs_it_scanorama_hvg5000.h5ad"))
compute_umap(ad_hs, "X_scanorama", [100, 15, 30], "scanorama")

# ---------------------------------------------------------------------------
# scVI (latent dimension 50)
# ---------------------------------------------------------------------------
ad_hs = sc.read_h5ad(os.path.join(it_dir, "hs_it_scVI_hvg5000.h5ad"))
compute_umap(ad_hs, "X_scvi", [100, 50, 30], "scvi_l50")
