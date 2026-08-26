#!/usr/bin/env python3
"""
=============================================================================
File      : 03_hippo_npc.py
Directory : 06_cellular phylogeny & regulatory factors
Purpose   : Hippocampal NPC analysis (Hippo pathway).
=============================================================================
"""
import os

import scanpy as sc
import pandas as pd

# ============================================================================
# Configuration: Set the project root directory before running
# ============================================================================
# proj_dir = "path/to/your/project"  # <-- MODIFY THIS
proj_dir = os.environ.get("PROJ_DIR", "path/to/your/project")
os.chdir(path=proj_dir)

# Load the hippocampal NPC dataset (replace with the actual path if needed)
adata = sc.read_h5ad("data/human/output/Analysis/lineage/hippo/hs_hippo.h5ad")
adata.obs["stage_abbr"] = adata.obs["stage"].str.split("-").str[0]

# Define cell-type marker gene sets
marker_genes = {
    "RG": ["NES", "PAX6", "ASCL1", "SOX2"],
    "Neuronal_IPC": ["EOMES", "TFAP2C", "MKI67", "IGFBPL1"],
    "Neuroblast": ["ST8SIA2", "DCX", "SOX4", "SOX11"],
    "Neuron": ["SLC17A7", "GAD1", "GAD2"],
}
all_markers = []
for group in marker_genes.values():
    for g in group:
        if g not in all_markers:
            all_markers.append(g)

# Filter 1: keep only IPC, NB, and RG cells
cell_mask = adata.obs["type_2_abbr"].isin(["IPC", "NB", "RG"])
adata_sub = adata[cell_mask, :].copy()

# Filter 2: keep only marker genes (reduce memory usage)
gene_mask = adata_sub.var_names.isin(all_markers)
adata_sub = adata_sub[:, gene_mask].copy()

valid_markers = [g for g in all_markers if g in adata_sub.var_names]

# Restrict to the developmental stages used in the downstream analysis
adata_sub = adata_sub[adata_sub.obs["stage_abbr"].isin(["08", "09", "10", "11", "12", "13", "14", "15"]), :].copy()

# Cell-type composition across stages
stage_order = [
    "03", "04", "05", "06", "08", "10", "11", "13", "14", "15"
]

valid_stages = [s for s in stage_order if s in adata.obs["stage_abbr"].unique()]
adata.obs["stage_abbr"] = adata.obs["stage_abbr"].astype("category")
adata.obs["stage_abbr"] = adata.obs["stage_abbr"].cat.set_categories(valid_stages, ordered=True)

# Step 1: count all cell types per stage
count_df_all = pd.crosstab(adata.obs["stage_abbr"], adata.obs["type_2_abbr"])
# Reorder by stage
count_df_all = count_df_all.reindex(valid_stages, fill_value=0)

# Step 2: rebuild the table (NB/IPC/RG as separate columns, rest merged as Others)
stats_df = pd.DataFrame(index=valid_stages)
stats_df["IPC"] = count_df_all["IPC"] if "IPC" in count_df_all.columns else 0
stats_df["NB"] = count_df_all["NB"] if "NB" in count_df_all.columns else 0
stats_df["RG"] = count_df_all["RG"] if "RG" in count_df_all.columns else 0
other_cols = [col for col in count_df_all.columns if col not in ["IPC", "NB", "RG"]]
stats_df["Others"] = count_df_all[other_cols].sum(axis=1)

# Step 3: totals and ratios (Total must include RG, otherwise RG_ratio can
# exceed 100%)
stats_df["Total"] = stats_df[["IPC", "NB", "RG", "Others"]].sum(axis=1)
stats_df["IPC_ratio"] = (stats_df["IPC"] / stats_df["Total"] * 100).round(2)
stats_df["NB_ratio"] = (stats_df["NB"] / stats_df["Total"] * 100).round(2)
stats_df["RG_ratio"] = (stats_df["RG"] / stats_df["Total"] * 100).round(2)
stats_df["Others_ratio"] = (stats_df["Others"] / stats_df["Total"] * 100).round(2)

# Print the summary table
print("Cell counts and ratios of NB/IPC/Others per stage:")
print(stats_df)
