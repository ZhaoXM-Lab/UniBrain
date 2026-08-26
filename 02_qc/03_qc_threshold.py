#!/usr/bin/env python3
"""
=============================================================================
File      : 03_qc_threshold.py
Directory : 02_qc
Purpose   : Apply per-sample MAD-based QC thresholds, mitochondrial-fraction filtering, and doublet removal to produce the QC-passed AnnData object.
=============================================================================
"""

import os

import anndata
import numpy as np
import pandas as pd
import scanpy as sc
from joblib import Parallel, delayed
from scipy.stats import median_abs_deviation

# ============================================================================
# Configuration: Set the project root directory before running
# ============================================================================
proj_dir = "/path/to/your/project"  # <-- MODIFY THIS

data_dir = os.path.join(proj_dir, "data", "human")
out_dir = os.path.join(data_dir, "output", "MergeData")

# Load the merged human data
ad_hs = sc.read_h5ad(
    filename=os.path.join(out_dir, "hs_merged_all.h5ad")
)

# Remove irrelevant data
ad_hs = ad_hs[ad_hs.obs.dataset != "hs12", :]
ad_hs = ad_hs[ad_hs.obs.region_1 != "Retina", :]
ad_hs = ad_hs[ad_hs.obs.region_1 != "Organoid", :]

ad_hs.var['mt'] = ad_hs.var_names.str.startswith('MT-')
sc.pp.calculate_qc_metrics(ad_hs, qc_vars=['mt'], inplace=True)


def is_outlier(adata, metric: str, nmads: int):
    """Return cells outside median +/- nmads * MAD for the given metric."""
    M = adata.obs[metric]
    median = np.median(M)
    mad = median_abs_deviation(M)
    outlier = (M < median - nmads * mad) | (M > median + nmads * mad)
    return outlier


def process_sample(adata_sample, sample):
    """Detect outlier cells for one sample and report the outlier flags."""
    outliers_info = []

    if adata_sample.n_obs < 50:
        # Too few cells for MAD-based detection: flag all cells as outliers
        adata_sample.obs["outlier"] = pd.Series(True, index=adata_sample.obs.index)
        adata_sample.obs["mt_outlier"] = pd.Series(True, index=adata_sample.obs.index)

        outliers_info.append(pd.DataFrame({
            "cell": adata_sample.obs_names,
            "sample": sample,
            "outlier": adata_sample.obs["outlier"],
            "mt_outlier": adata_sample.obs["mt_outlier"],
            "doublets_prediction": adata_sample.obs["doublets_prediction"]
        }))
    else:
        sc.pp.calculate_qc_metrics(
            adata_sample, qc_vars=["mt"], inplace=True, percent_top=[20], log1p=True
        )

        # Calculate outlier cells
        adata_sample.obs["outlier"] = (
            is_outlier(adata_sample, "log1p_total_counts", 8)
            | is_outlier(adata_sample, "log1p_n_genes_by_counts", 8)
            | is_outlier(adata_sample, "pct_counts_in_top_20_genes", 8)
            | (adata_sample.obs['n_genes_by_counts'] < 100)
        )

        # Calculate mitochondrial outliers
        adata_sample.obs["mt_outlier"] = (
            is_outlier(adata_sample, "pct_counts_mt", 8)
            | (adata_sample.obs["pct_counts_mt"] > 15)
        )

        # Save outliers, mt_outliers, and doublets_prediction
        if not outliers_info:
            outliers_info.append(pd.DataFrame({
                "cell": adata_sample.obs_names,
                "sample": sample,
                "outlier": adata_sample.obs["outlier"],
                "mt_outlier": adata_sample.obs["mt_outlier"],
                "doublets_prediction": adata_sample.obs["doublets_prediction"]
            }))

    return adata_sample, pd.concat(outliers_info, ignore_index=True)


def qc_adata(adata, n_jobs=4):
    """Apply per-sample QC thresholds and return the filtered datasets."""
    # Mitochondrial, ribosomal and hemoglobin gene markers
    adata.var["mt"] = adata.var_names.str.startswith("MT-")
    adata.var["ribo"] = adata.var_names.str.startswith(("RPS", "RPL"))
    adata.var["hb"] = adata.var_names.str.contains(("^HB[^(P)]"))

    # Remove genes expressed in fewer than 300 cells
    sc.pp.filter_genes(adata, min_cells=300)
    adata.var = adata.var.drop(columns=["n_cells"])

    # List to store outlier flags and doublets predictions
    outliers_info = []
    # List to store sample names and cell counts
    sample_counts = []
    ad_hs_list = []

    # Process each dataset
    for proj in adata.obs['dataset'].unique():
        adata_proj = adata[adata.obs['dataset'] == proj].copy()

        # Process each sample in parallel
        results = Parallel(n_jobs=n_jobs)(
            delayed(process_sample)(adata_proj[adata_proj.obs['sample'] == sample], sample)
            for sample in adata_proj.obs['sample'].unique()
        )

        # Collect results
        for adata_sample, outlier_info in results:
            outliers_info.append(outlier_info)
            adata_proj.obs.loc[adata_sample.obs.index, "outlier"] = adata_sample.obs["outlier"]
            adata_proj.obs.loc[adata_sample.obs.index, "mt_outlier"] = adata_sample.obs["mt_outlier"]

        adata_proj.obs["outlier"] = adata_proj.obs["outlier"].astype(bool)
        adata_proj.obs["mt_outlier"] = adata_proj.obs["mt_outlier"].astype(bool)
        adata_proj.obs["doublets_prediction"] = adata_proj.obs["doublets_prediction"].astype(bool)

        # Filter out outlier cells
        adata_proj = adata_proj[
            (~adata_proj.obs.outlier) & (~adata_proj.obs.mt_outlier) & (~adata_proj.obs.doublets_prediction)
        ].copy()

        ad_hs_list.append(adata_proj)
        print(f"Finished processing {proj}!")

        # Store sample counts
        for sample in adata_proj.obs['sample'].unique():
            sample_counts.append([sample, adata_proj[adata_proj.obs['sample'] == sample].n_obs])

    # Combine results
    outliers_info_df = pd.concat(outliers_info, ignore_index=True)
    sample_counts_df = pd.DataFrame(sample_counts, columns=["sample", "cell_count"])

    return ad_hs_list, outliers_info_df, sample_counts_df


ad_hs_merge, outliers_info_df, sample_counts_df = qc_adata(ad_hs, n_jobs=8)
ad_hs = sc.concat(ad_hs_merge, join='outer', index_unique=None)

outliers_info_df.to_csv(
    os.path.join(out_dir, "hs_qc_info.csv"), index=True
)
sample_counts_df.to_csv(
    os.path.join(out_dir, "hs_sample_counts.csv"), index=True
)

hs_meta = ad_hs.obs.loc[:, ["species", "dataset", "individual", "sample", "age", "stage", "sex",
                            "region_original", "region_1", "region_1_abbr", "region_2", "region_2_abbr",
                            "platform", "diagnosis"]]
hs_meta.to_csv(
    os.path.join(out_dir, "human_meta_qc.csv"), index=True
)

gtf = pd.read_csv(
    os.path.join(proj_dir, "data", "ref", "human_reference_unique.txt"),
    sep="\t", header=None
)
gtf.columns = ["id", "name"]
gtf.index = gtf.name
ad_hs.var["ensemble_id"] = gtf.loc[ad_hs.var_names, "id"].values

hs_gene = ad_hs.var.loc[:, ["ensemble_id"]]
hs_gene.to_csv(
    os.path.join(out_dir, "hs_gene.csv"), index=True
)

# Save the QC-passed data
ad_hs.write_h5ad(
    filename=os.path.join(out_dir, "hs_qc.h5ad")
)
print("Saved QC-passed data to", os.path.join(out_dir, "hs_qc.h5ad"))
