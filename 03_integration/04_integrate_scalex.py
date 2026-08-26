#!/usr/bin/env python3
"""
=============================================================================
File      : 04_integrate_scalex.py
Directory : 03_integration
Purpose   : Run SCALEX integration on a memory-efficient HVG-only AnnData, evaluate batch mixing (iLISI, batch silhouette), and save the integrated object with UMAP embedding.
=============================================================================
"""

import gc
import os

import anndata
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
from sklearn.metrics import silhouette_score
from tqdm import tqdm

from scalex.function import SCALEX

# ============================================================================
# Configuration: Set the project root directory before running
# ============================================================================
proj_dir = os.environ.get("PROJ_DIR", "/path/to/your/project")  # <-- MODIFY THIS or export PROJ_DIR

os.environ["OPENBLAS_NUM_THREADS"] = "32"
os.environ["MKL_NUM_THREADS"] = "32"
os.environ["OMP_NUM_THREADS"] = "32"
os.environ["OPENBLAS_DYNAMIC"] = "FALSE"
os.environ["MKL_DYNAMIC"] = "FALSE"

sc.settings.verbosity = 3
sc.settings.n_jobs = 8
sc.logging.print_header()

data_dir = os.path.join(proj_dir, "data", "human")
it_dir = os.path.join(data_dir, "output", "Integration")
merge_dir = os.path.join(data_dir, "output", "MergeData")
umap_dir = os.path.join(data_dir, "output", "Umap")

BATCH_KEY = "sample_new"
N_TOP_FEATURES = 5000
N_EPOCHS = 100
BATCH_SIZE = 32


def optimized_gene_selection(ad_hs_orig, highly_variable_genes_path):
    """Select the HVG indices and names from a saved gene list.

    Args:
        ad_hs_orig: Backed AnnData object with the full gene space.
        highly_variable_genes_path: Path to a text file with one HVG name
            per line.

    Returns:
        tuple: (selected_gene_indices, selected_genes, hvg_mask)
    """
    highly_variable_genes = set(
        pd.read_csv(highly_variable_genes_path, header=None, dtype=str)[0].values
    )

    hvg_mask = np.array(
        [gene in highly_variable_genes for gene in ad_hs_orig.var_names], dtype=bool
    )
    selected_gene_indices = np.where(hvg_mask)[0]
    selected_genes = ad_hs_orig.var_names[hvg_mask].tolist()

    print(f"Original genes: {ad_hs_orig.n_vars}")
    print(f"Selected genes: {len(selected_genes)}")
    return selected_gene_indices, selected_genes, hvg_mask


def build_optimized_sparse_matrix(ad_hs_orig, selected_gene_indices, chunk_size=50000):
    """Build a sparse gene-subset matrix in chunks (COO format).

    Avoids loading the full dense matrix into memory; only the selected
    gene columns are extracted, chunk by chunk.

    Args:
        ad_hs_orig: Backed AnnData object with the full gene space.
        selected_gene_indices: Column indices of the genes to keep.
        chunk_size: Number of cells processed per chunk.

    Returns:
        scipy.sparse.csr_matrix: The gene-subset expression matrix.
    """
    n_cells = ad_hs_orig.n_obs
    n_genes = len(selected_gene_indices)

    data_list = []
    row_list = []
    col_list = []

    gene_idx_map = {
        orig_idx: new_idx for new_idx, orig_idx in enumerate(selected_gene_indices)
    }

    for start in tqdm(range(0, n_cells, chunk_size), desc="Building sparse matrix"):
        end = min(start + chunk_size, n_cells)
        chunk = ad_hs_orig.X[start:end]

        if sp.issparse(chunk):
            chunk_csr = chunk.tocsr()
            for i in range(chunk_csr.shape[0]):
                row_data = chunk_csr[i].toarray().flatten()
                nonzero_indices = np.where(row_data != 0)[0]
                for orig_col_idx in nonzero_indices:
                    if orig_col_idx in gene_idx_map:
                        data_list.append(row_data[orig_col_idx])
                        row_list.append(start + i)
                        col_list.append(gene_idx_map[orig_col_idx])
        else:
            chunk_subset = chunk[:, selected_gene_indices]
            coo_chunk = sp.coo_matrix(chunk_subset)
            data_list.extend(coo_chunk.data)
            row_list.extend(coo_chunk.row + start)
            col_list.extend(coo_chunk.col)

        del chunk
        gc.collect()

    if data_list:
        X_new = sp.coo_matrix(
            (data_list, (row_list, col_list)),
            shape=(n_cells, n_genes),
            dtype=ad_hs_orig.X.dtype,
        ).tocsr()
    else:
        X_new = sp.csr_matrix((n_cells, n_genes), dtype=ad_hs_orig.X.dtype)

    return X_new


def memory_efficient_annada_creation(ad_hs_orig, selected_genes, X_new):
    """Create an AnnData object from the subset matrix and minimal metadata.

    Args:
        ad_hs_orig: Original (backed) AnnData object.
        selected_genes: Gene names kept in the subset matrix.
        X_new: Subset expression matrix.

    Returns:
        anndata.AnnData: The memory-efficient AnnData object.
    """
    obs_data = ad_hs_orig.obs.copy()

    var_columns_to_keep = ["gene_ids"]
    available_columns = [
        col for col in var_columns_to_keep if col in ad_hs_orig.var.columns
    ]
    if available_columns:
        var_data = ad_hs_orig.var.loc[selected_genes, available_columns].copy()
    else:
        var_data = pd.DataFrame(index=selected_genes)

    return sc.AnnData(X=X_new, obs=obs_data, var=var_data, dtype=ad_hs_orig.X.dtype)


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


def calculate_optimal_iterations(n_cells, batch_size, target_epochs=10):
    """Convert a target number of epochs into training iterations."""
    iterations_per_epoch = n_cells // batch_size
    return iterations_per_epoch * target_epochs


def main():
    # -----------------------------------------------------------------------
    # 1. Build the memory-efficient HVG AnnData
    # -----------------------------------------------------------------------
    hvg_path = os.path.join(it_dir, "hvg", f"highly_variable_genes_{N_TOP_FEATURES}.txt")
    ad_hs_orig = sc.read_h5ad(os.path.join(merge_dir, "hs_qc.h5ad"), backed="r")

    selected_gene_indices, selected_genes, _ = optimized_gene_selection(
        ad_hs_orig, hvg_path
    )
    X_new = build_optimized_sparse_matrix(ad_hs_orig, selected_gene_indices)
    ad_hs_orig.file.close()

    ad_hs = memory_efficient_annada_creation(ad_hs_orig, selected_genes, X_new)

    # Batch key: pool the four public datasets, keep sample-level labels otherwise
    ad_hs.obs[BATCH_KEY] = ad_hs.obs["sample"].astype(str)
    dataset_mapping = {"hs09": "hs09", "hs11": "hs11", "hs12": "hs12", "hs14": "hs14"}
    ad_hs.obs[BATCH_KEY] = (
        ad_hs.obs["dataset"].map(dataset_mapping).fillna(ad_hs.obs[BATCH_KEY])
    )
    ad_hs.obs[BATCH_KEY] = ad_hs.obs[BATCH_KEY].astype("category")

    print(f"Final data dimensions: {ad_hs.n_obs} x {ad_hs.n_vars}")

    # -----------------------------------------------------------------------
    # 2. Run SCALEX
    # -----------------------------------------------------------------------
    n_cells = ad_hs.n_obs
    target_iterations = calculate_optimal_iterations(n_cells, BATCH_SIZE, N_EPOCHS)

    ad_hs_integrated = SCALEX(
        ad_hs,
        batch_key=BATCH_KEY,
        min_features=0,
        min_cells=0,
        n_top_features=N_TOP_FEATURES,
        profile="RNA",
        outdir=os.path.join(it_dir, "SCALEX_results", f"hvg{N_TOP_FEATURES}"),
        impute=False,
        batch_size=BATCH_SIZE,
        max_iteration=target_iterations,
        ignore_umap=True,
        gpu=1,
        verbose=True,
    )

    np.savetxt(
        os.path.join(
            it_dir,
            "SCALEX_results",
            f"hs_it_SCALEX_hvg{N_TOP_FEATURES}_epoch{N_EPOCHS}.csv",
        ),
        ad_hs_integrated.obsm["latent"],
        delimiter=",",
    )

    # -----------------------------------------------------------------------
    # 3. Batch-mixing evaluation
    # -----------------------------------------------------------------------
    batch_sil = calculate_batch_silhouette(
        ad_hs_integrated, "latent", n_sample=200000
    )
    print(f"Batch silhouette (SCALEX): {batch_sil:.4f}")

    # -----------------------------------------------------------------------
    # 4. UMAP embedding from the SCALEX latent space
    # -----------------------------------------------------------------------
    sc.pp.neighbors(ad_hs_integrated, n_neighbors=100, n_pcs=10, use_rep="latent")
    sc.tl.umap(ad_hs_integrated)
    ad_hs_integrated.obsm["X_umap_SCALEX_k_100"] = ad_hs_integrated.obsm["X_umap"]
    save_umap_coordinates(
        ad_hs_integrated,
        "X_umap_SCALEX_k_100",
        os.path.join(umap_dir, f"umap_SCALEX_k_100_hvg{N_TOP_FEATURES}.csv"),
    )

    # -----------------------------------------------------------------------
    # 5. Save the integrated object
    # -----------------------------------------------------------------------
    out_file = os.path.join(it_dir, f"hs_it_SCALEX_hvg{N_TOP_FEATURES}.h5ad")
    ad_hs_integrated.write(out_file)
    print("Saved:", out_file)


if __name__ == "__main__":
    main()
