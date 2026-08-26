#!/usr/bin/env python3
"""
=============================================================================
File      : 01_integrate_preprocess.py
Directory : 03_integration
Purpose   : Build the integration-ready AnnData object: cell-cycle scoring, artifact-gene removal, HVG selection, regress-out, scaling, PCA, and batch-key definition.
=============================================================================
"""

import os

import anndata
import pandas as pd
import scanpy as sc

# ============================================================================
# Configuration: Set the project root directory before running
# ============================================================================
proj_dir = os.environ.get("PROJ_DIR", "/path/to/your/project")  # <-- MODIFY THIS or export PROJ_DIR

os.environ["OPENBLAS_NUM_THREADS"] = "32"
os.environ["MKL_NUM_THREADS"] = "32"
os.environ["OMP_NUM_THREADS"] = "32"
os.environ["OPENBLAS_DYNAMIC"] = "FALSE"
os.environ["MKL_DYNAMIC"] = "FALSE"

sc.settings.n_jobs = 8

data_dir = os.path.join(proj_dir, "data", "human")
out_dir = os.path.join(data_dir, "output", "Integration")
merge_dir = os.path.join(data_dir, "output", "MergeData")

print(f"Scanpy version: {sc.__version__}")
print(f"Anndata version: {anndata.__version__}\n")

# ---------------------------------------------------------------------------
# 1. Load QC-filtered data and drop redundant QC columns
# ---------------------------------------------------------------------------
ad_hs = sc.read_h5ad(os.path.join(merge_dir, "hs_qc.h5ad"))
ad_hs_backup = ad_hs.copy()

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

# ---------------------------------------------------------------------------
# 2. Cell-cycle scoring (S/G2M phase) on a subset of HVGs
# ---------------------------------------------------------------------------
ad_hs.raw = ad_hs.copy()
sc.pp.normalize_total(ad_hs, target_sum=1e4)
sc.pp.log1p(ad_hs)
sc.pp.highly_variable_genes(ad_hs, n_top_genes=2000, flavor="seurat")

with open(os.path.join(data_dir, "cell_cycle_genes.txt")) as f:
    cell_cycle_genes = [line.strip() for line in f]
s_genes = cell_cycle_genes[:43]
g2m_genes = cell_cycle_genes[43:]
ad_hs.var["cell_cycle"] = ad_hs.var_names.isin(cell_cycle_genes)

# Restrict to HVGs + cell-cycle genes for a stable score computation
selected_genes = ad_hs.var.highly_variable | ad_hs.var.cell_cycle
ad_hs = ad_hs[:, selected_genes]
sc.pp.regress_out(ad_hs, keys=["total_counts"], n_jobs=8)
sc.pp.scale(ad_hs)
sc.tl.score_genes_cell_cycle(ad_hs, s_genes=s_genes, g2m_genes=g2m_genes)

# Keep only the scores and restore the full expression matrix
cc_score = ad_hs.obs.loc[:, ["S_score", "G2M_score", "phase"]]
cc_score.to_csv(os.path.join(out_dir, "cc_score.csv"))

ad_hs = ad_hs_backup.copy()
ad_hs.obs["S_score"] = cc_score["S_score"]
ad_hs.obs["G2M_score"] = cc_score["G2M_score"]
ad_hs.obs["phase"] = cc_score["phase"].astype(str)

# ---------------------------------------------------------------------------
# 3. Remove mitochondrial, hemoglobin, ribosomal, and MALAT1 genes
# ---------------------------------------------------------------------------
remove_gene = (
    ad_hs.var_names.str.startswith("MT-")
    | ad_hs.var_names.str.contains("^HB[^(P)]")
    | ad_hs.var_names.str.contains("^RP[SL]")
    | ad_hs.var_names.str.contains("MALAT1")
)
ad_hs = ad_hs[:, ~remove_gene]

sc.pp.normalize_total(ad_hs, target_sum=1e4)
sc.pp.log1p(ad_hs)

# ---------------------------------------------------------------------------
# 4. HVG selection restricted to protein-coding genes
# ---------------------------------------------------------------------------
sc.pp.highly_variable_genes(ad_hs, n_top_genes=5000, flavor="seurat")

hgtf = pd.read_csv(
    os.path.join(proj_dir, "data", "ref", "gencode.v42.annotation.csv"),
    sep=",",
    header=0,
    low_memory=False,
)
merged = ad_hs.var.merge(
    hgtf[["gene_id", "gene_type", "gene_name"]],
    left_on="ensemble_id",
    right_on="gene_id",
    how="left",
).drop_duplicates()
merged.set_index(ad_hs.var.index, inplace=True)
ad_hs.var["gene_type"] = merged["gene_type"]

protein_coding_hvg = ad_hs.var[
    ad_hs.var["highly_variable"] & (ad_hs.var["gene_type"] == "protein_coding")
]
ad_hs = ad_hs[:, protein_coding_hvg.index]
print(f"Protein-coding HVGs: {ad_hs.n_vars}")

# ---------------------------------------------------------------------------
# 5. Regress out confounders, scale, and run PCA
# ---------------------------------------------------------------------------
sc.pp.regress_out(ad_hs, keys=["total_counts", "S_score", "G2M_score"], n_jobs=8)
sc.pp.scale(ad_hs)
sc.tl.pca(ad_hs, svd_solver="arpack", n_comps=50)

# ---------------------------------------------------------------------------
# 6. Define the integration batch key (sample_new)
# ---------------------------------------------------------------------------
ad_hs.obs["sample_new"] = ad_hs.obs["sample"].astype(str)
for dataset in ["hs09", "hs11", "hs12", "hs14"]:
    idx = ad_hs.obs["dataset"] == dataset
    ad_hs.obs.loc[idx, "sample_new"] = dataset
ad_hs.obs["sample_new"] = ad_hs.obs["sample_new"].astype("category")

# ---------------------------------------------------------------------------
# 7. Save the integration-ready outputs
# ---------------------------------------------------------------------------
pd.Series(ad_hs.var_names.tolist()).to_csv(
    os.path.join(out_dir, "highly_variable_genes.txt"), index=False, header=False
)
ad_hs.var.loc[:, ["ensemble_id"]].to_csv(
    os.path.join(out_dir, "hs_hv_gene.csv"), index=True
)

ad_hs_clear = ad_hs.copy()
ad_hs_clear.raw = None
ad_hs_clear.write_h5ad(os.path.join(out_dir, "hs_it_temp4.h5ad"))
print("Saved:", os.path.join(out_dir, "hs_it_temp4.h5ad"))
