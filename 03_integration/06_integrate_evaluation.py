#!/usr/bin/env python3
"""
=============================================================================
File      : 06_integrate_evaluation.py
Directory : 03_integration
Purpose   : Reproduce integration preprocessing and evaluate Harmony batch mixing with the iLISI metric on a downsampled 2M-cell subset.
=============================================================================
"""

import os

import anndata
import numpy as np
import pandas as pd
import scanpy as sc
import scib

# ============================================================================
# Configuration: Set the project root directory before running
# ============================================================================
proj_dir = os.environ.get("PROJ_DIR", "/path/to/your/project")  # <-- MODIFY THIS or export PROJ_DIR

os.environ["OPENBLAS_NUM_THREADS"] = "32"
os.environ["MKL_NUM_THREADS"] = "32"
os.environ["OMP_NUM_THREADS"] = "32"
os.environ["OPENBLAS_DYNAMIC"] = "FALSE"
os.environ["MKL_DYNAMIC"] = "FALSE"

np.random.seed(123)
sc.settings.n_jobs = 8

data_dir = os.path.join(proj_dir, "data", "human")
it_dir = os.path.join(data_dir, "output", "Integration")
merge_dir = os.path.join(data_dir, "output", "MergeData")

BATCH_KEY = "sample_new"
N_TOP_GENES = 5000
SAMPLE_CELL_NUM = 2000000

print(f"Scanpy version: {sc.__version__}")
print(f"Anndata version: {anndata.__version__}\n")

# ---------------------------------------------------------------------------
# 1. Preprocessing (same steps as 01_integrate_preprocess)
# ---------------------------------------------------------------------------
ad_hs = sc.read_h5ad(os.path.join(merge_dir, "hs_qc.h5ad"))

qc_columns = [
    "log1p_n_genes_by_counts",
    "log1p_total_counts",
    "pct_counts_in_top_50_genes",
    "pct_counts_in_top_100_genes",
    "pct_counts_in_top_200_genes",
    "pct_counts_in_top_500_genes",
    "total_counts_mt",
    "log1p_total_counts_mt",
]
ad_hs.obs = ad_hs.obs.drop(columns=qc_columns)

# Attach the cell-cycle scores computed in 01_integrate_preprocess
cc_score = pd.read_csv(os.path.join(it_dir, "cc_score.csv"), index_col=0)
ad_hs.obs["S_score"] = cc_score["S_score"]
ad_hs.obs["G2M_score"] = cc_score["G2M_score"]
ad_hs.obs["phase"] = cc_score["phase"].astype(str)

# Remove mitochondrial, hemoglobin, ribosomal, and MALAT1 genes
remove_gene = (
    ad_hs.var_names.str.startswith("MT-")
    | ad_hs.var_names.str.contains("^HB[^(P)]")
    | ad_hs.var_names.str.contains("^RP[SL]")
    | ad_hs.var_names.str.contains("MALAT1")
)
ad_hs = ad_hs[:, ~remove_gene]

sc.pp.normalize_total(ad_hs, target_sum=1e4)
sc.pp.log1p(ad_hs)
ad_hs.write_h5ad(os.path.join(it_dir, "hs_norm.h5ad"))

# ---------------------------------------------------------------------------
# 2. HVG selection, regress-out, scaling, and PCA
# ---------------------------------------------------------------------------
sc.pp.highly_variable_genes(ad_hs, n_top_genes=N_TOP_GENES, flavor="seurat")
print(f"Num. HVGs: {sum(ad_hs.var.highly_variable)}")

ad_hs = ad_hs[:, ad_hs.var.highly_variable]
sc.pp.regress_out(ad_hs, keys=["total_counts", "S_score", "G2M_score"], n_jobs=8)
sc.pp.scale(ad_hs)
sc.tl.pca(ad_hs, svd_solver="arpack", n_comps=50)

ad_hs.obs[BATCH_KEY] = ad_hs.obs["sample"].astype(str)
for dataset in ["hs09", "hs11", "hs12", "hs14"]:
    idx = ad_hs.obs["dataset"] == dataset
    ad_hs.obs.loc[idx, BATCH_KEY] = dataset
ad_hs.obs[BATCH_KEY] = ad_hs.obs[BATCH_KEY].astype("category")

ad_hs.write_h5ad(os.path.join(it_dir, "hvg", f"hs_it_hvg{N_TOP_GENES}.h5ad"))
pd.Series(ad_hs.var_names.tolist()).to_csv(
    os.path.join(it_dir, "hvg", f"highly_variable_genes_{N_TOP_GENES}.txt"),
    index=False,
    header=False,
)

# ---------------------------------------------------------------------------
# 3. Downsample and run Harmony on the subset
# ---------------------------------------------------------------------------
sampled_indices = np.random.choice(
    ad_hs.obs_names, size=SAMPLE_CELL_NUM, replace=False
)
sampled_ad_hs = ad_hs[sampled_indices]

sc.external.pp.harmony_integrate(
    sampled_ad_hs, key=BATCH_KEY, basis="X_pca", max_iter_harmony=30
)

sc.pp.neighbors(sampled_ad_hs, n_neighbors=40, n_pcs=50, use_rep="X_pca_harmony")
sc.tl.umap(sampled_ad_hs)
sampled_ad_hs.obsm["X_umap_harmony_k_40"] = sampled_ad_hs.obsm["X_umap"]

# ---------------------------------------------------------------------------
# 4. Batch-mixing evaluation on the subset
# ---------------------------------------------------------------------------
ilisi_harmony = scib.me.ilisi_graph(
    sampled_ad_hs,
    batch_key=BATCH_KEY,
    type_="embed",
    use_rep="X_pca_harmony",
    n_cores=2,
)
print(f"iLISI (harmony, 2M subset): {ilisi_harmony}")

sampled_ad_hs.write_h5ad(
    os.path.join(it_dir, "hvg", f"hs_it_hvg{N_TOP_GENES}_sample2m.h5ad")
)
print("Saved:", os.path.join(it_dir, "hvg", f"hs_it_hvg{N_TOP_GENES}_sample2m.h5ad"))
