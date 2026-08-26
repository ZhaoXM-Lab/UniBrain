#!/usr/bin/env python3
"""
=============================================================================
File      : 05_integrate_scvi.py
Directory : 03_integration
Purpose   : Train a scVI VAE for batch correction, extract the latent embedding and imputed expression, evaluate batch mixing, and save the integrated objects.
=============================================================================
"""

import os
import random
from typing import Tuple

import anndata
import numpy as np
import pandas as pd
import scanpy as sc
import scib
import scipy.sparse as sp
import torch
from sklearn.metrics import silhouette_score
from tqdm import tqdm

# scVI imports
import scvi
from scvi.dataset import AnnDatasetFromAnnData
from scvi.inference import UnsupervisedTrainer
from scvi.models.vae import VAE

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
sc.settings.n_jobs = 16

data_dir = os.path.join(proj_dir, "data", "human")
it_dir = os.path.join(data_dir, "output", "Integration")
merge_dir = os.path.join(data_dir, "output", "MergeData")
umap_dir = os.path.join(data_dir, "output", "Umap")

BATCH_KEY = "sample_new"
BATCH_ENCODED = "sample_new_encoded"
HVG = 5000
N_EPOCHS = 20
N_LATENT = 50
USE_CUDA = True

print(f"Scanpy version: {sc.__version__}")
print(f"Anndata version: {anndata.__version__}\n")


def compute_scvi_latent(
    adata: sc.AnnData,
    n_latent: int = 5,
    n_epochs: int = 100,
    lr: float = 1e-3,
    batch: str = BATCH_ENCODED,
    use_cuda: bool = False,
) -> Tuple[scvi.inference.Posterior, np.ndarray]:
    """Train a scVI VAE and return the posterior and the latent space.

    Args:
        adata: AnnData object with raw (non-normalized) counts.
        n_latent: Dimension of the latent space.
        n_epochs: Number of training epochs.
        lr: Learning rate.
        batch: Column in ``adata.obs`` with the encoded batch labels.
        use_cuda: Whether to train on GPU.

    Returns:
        tuple: (scvi.Posterior, latent_space)
    """
    scvi_dataset = AnnDatasetFromAnnData(adata, batch_label=batch)
    n_batch = scvi_dataset.n_batches

    vae = VAE(
        scvi_dataset.nb_genes,
        n_batch=n_batch,
        n_latent=n_latent,
    )
    trainer = UnsupervisedTrainer(
        vae,
        scvi_dataset,
        train_size=1,
        use_cuda=use_cuda,
    )
    trainer.train(n_epochs=n_epochs, lr=lr)

    posterior = trainer.create_posterior(
        trainer.model, scvi_dataset, indices=np.arange(len(scvi_dataset))
    ).sequential()
    latent, _, _ = posterior.get_latent()
    return posterior, latent


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


# ---------------------------------------------------------------------------
# 1. Build the memory-efficient HVG AnnData
# ---------------------------------------------------------------------------
highly_variable_genes = pd.read_csv(
    os.path.join(it_dir, "hvg", f"highly_variable_genes_{HVG}.txt"),
    header=None,
)[0].tolist()
ad_hs_orig = sc.read_h5ad(os.path.join(merge_dir, "hs_qc.h5ad"), backed="r")
hvg_idx = ad_hs_orig.var_names.isin(highly_variable_genes)
selected_genes = ad_hs_orig.var_names[hvg_idx].tolist()

n_cells = ad_hs_orig.n_obs
n_genes = len(selected_genes)
X_new = sp.lil_matrix((n_cells, n_genes), dtype=ad_hs_orig.X.dtype)
chunk_size = 100000
for start in tqdm(range(0, n_cells, chunk_size), desc="Building HVG matrix"):
    end = min(start + chunk_size, n_cells)
    chunk = ad_hs_orig.X[start:end, hvg_idx]
    if sp.issparse(chunk):
        X_new[start:end] = chunk.tolil()
    else:
        X_new[start:end] = sp.lil_matrix(chunk)
X_new = X_new.tocsr()

ad_hs = sc.AnnData(
    X=X_new,
    obs=ad_hs_orig.obs.copy(),
    var=ad_hs_orig.var.loc[selected_genes].copy(),
    dtype=ad_hs_orig.X.dtype,
)
ad_hs_orig.file.close()

# Batch key and its encoded version used by scVI
ad_hs.obs[BATCH_KEY] = ad_hs.obs["sample"].astype(str)
for dataset in ["hs09", "hs11", "hs12", "hs14"]:
    idx = ad_hs.obs["dataset"] == dataset
    ad_hs.obs.loc[idx, BATCH_KEY] = dataset
ad_hs.obs[BATCH_KEY] = ad_hs.obs[BATCH_KEY].astype("category")
ad_hs.obs[BATCH_ENCODED] = pd.Categorical(ad_hs.obs[BATCH_KEY]).codes.astype(float)
ad_hs.obs[BATCH_ENCODED] = ad_hs.obs[BATCH_ENCODED].astype("category")

# ---------------------------------------------------------------------------
# 2. Train scVI and extract the latent space
# ---------------------------------------------------------------------------
scvi_posterior, scvi_latent = compute_scvi_latent(
    ad_hs,
    n_epochs=N_EPOCHS,
    n_latent=N_LATENT,
    use_cuda=USE_CUDA,
    batch=BATCH_ENCODED,
)
np.savetxt(
    os.path.join(it_dir, f"hs_it_scVI_hvg{HVG}.csv"), scvi_latent, delimiter=","
)

scvi_dataset = AnnDatasetFromAnnData(ad_hs, batch_label=BATCH_ENCODED)
hs_it_scVI_cell = scvi_dataset.cell_names
np.savetxt(
    os.path.join(it_dir, f"hs_it_scVI_cell_hvg{HVG}.csv"),
    hs_it_scVI_cell,
    delimiter=",",
    fmt="%s",
)

ad_hs = ad_hs[hs_it_scVI_cell,]
ad_hs.obsm["X_scvi"] = scvi_latent

# ---------------------------------------------------------------------------
# 3. Imputed expression (average of 10 scale draws from the posterior)
# ---------------------------------------------------------------------------
scale = scvi_posterior.get_sample_scale()
for _ in range(9):
    scale += scvi_posterior.get_sample_scale()
scale /= 10

ad_hs_scVI = sc.AnnData(X=np.squeeze(scale))
ad_hs_scVI.var_names = ad_hs.var.index
ad_hs_scVI.obs = ad_hs.obs.copy()
ad_hs_scVI.var = ad_hs.var.copy()
ad_hs_scVI.obsm = ad_hs.obsm.copy()

geex_file = os.path.join(it_dir, f"hs_scVI_geex_l{N_LATENT}_hvg{HVG}.h5ad")
ad_hs_scVI.write_h5ad(geex_file)
print("Saved:", geex_file)

# ---------------------------------------------------------------------------
# 4. Batch-mixing evaluation
# ---------------------------------------------------------------------------
ilisi_scvi = scib.me.ilisi_graph(
    ad_hs, batch_key=BATCH_KEY, type_="embed", use_rep="X_scvi", n_cores=2
)
print(f"iLISI (scVI): {ilisi_scvi}")

batch_sil = calculate_batch_silhouette(ad_hs, "X_scvi", n_sample=200000)
print(f"Batch silhouette (scVI): {batch_sil:.4f}")

# ---------------------------------------------------------------------------
# 5. UMAP coordinates at several neighborhood sizes
# ---------------------------------------------------------------------------
for k in [100, 50, 30]:
    sc.pp.neighbors(ad_hs, n_neighbors=k, n_pcs=50, use_rep="X_scvi")
    sc.tl.umap(ad_hs)
    ad_hs.obsm[f"X_umap_scvi_k_{k}"] = ad_hs.obsm["X_umap"]
    if k == 30:
        ad_hs.obsp["connectivities_scvi_k30"] = ad_hs.obsp["connectivities"]
    save_umap_coordinates(
        ad_hs,
        f"X_umap_scvi_k_{k}",
        os.path.join(umap_dir, f"umap_scvi_k_{k}_hvg{HVG}.csv"),
    )

# ---------------------------------------------------------------------------
# 6. Save the integrated object
# ---------------------------------------------------------------------------
out_file = os.path.join(it_dir, f"hs_it_scVI_hvg{HVG}.h5ad")
ad_hs.write(out_file)
print("Saved:", out_file)
