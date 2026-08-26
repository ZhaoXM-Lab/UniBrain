#!/usr/bin/env python3
"""
=============================================================================
File      : 06_marker_gene_trajectory.py
Directory : 10_uniBrain agent
Purpose   : Marker gene trajectory validation for model-generated microglia.
=============================================================================
"""

import os
import warnings

import numpy as np
import pandas as pd
import anndata
from scipy.sparse import issparse

warnings.filterwarnings("ignore")

## Configuration ##
# Set the base directory for input/output data below.
BASE_DIR = os.environ.get("PROJ_DIR", "path/to/your/project")  # <-- MODIFY THIS

# Input h5ad: VQVAE-predicted / LLM-generated microglia expression data
DATA_PATH = os.path.join(BASE_DIR, "output", "microglia_generation.h5ad")

# Directory for exported expression matrices
OUTPUT_DIR = os.path.join(BASE_DIR, "output", "validation_data")

# ============================================================
# 1. Data loading and stage filtering
# ============================================================
print(">>> Reading h5ad file ...")
adata = anndata.read_h5ad(DATA_PATH)
print(f"    Shape: {adata.shape} (cells x genes)")
print(f"    obs columns: {list(adata.obs.columns)}")
print(f"    Unique stages: {sorted(adata.obs['stage'].unique())}")

# Drop the early first-trimester stage (insufficient cells)
n_before = adata.n_obs
adata = adata[adata.obs["stage"] != "01-Early first trimester"].copy()
n_after = adata.n_obs
print(f">>> Filtered '01-Early first trimester': "
      f"{n_before} -> {n_after} cells (removed {n_before - n_after})")

# Ordered developmental stages (15), phases (8) and periods (3)
STAGE_ORDER = [
    "02-Late first trimester",
    "03-Early second trimester",
    "04-Mid second trimester",
    "05-Late second trimester",
    "06-Early third trimester",
    "07-Late third trimester",
    "08-Neonatal and early infancy",
    "09-Late infancy",
    "10-Early childhood",
    "11-Middle and late childhood",
    "12-Adolescence",
    "13-Young adulthood",
    "14-Middle adulthood",
    "15-Late adulthood",
]

PHASE_ORDER = [
    "First trimester",
    "Second trimester",
    "Third trimester",
    "Infancy",
    "Childhood",
    "Adolescence",
    "Adulthood",
    "Aged",
]

PERIOD_ORDER = [
    "Prenatal",
    "Postnatal developmental",
    "Adult and aged",
]

# ============================================================
# 2. Marker gene modules
# ============================================================
GENE_MODULES = {
    "A: Homeostatic (surveillance)": {
        "core": [
            "P2RY12", "TMEM119", "HEXB", "CX3CR1", "TGFBR1",
            "OLFML3", "F13A1", "SLC2A5", "SPIC", "SIGLEC8", "SALL1", "FCRLS",
        ],
    },
    "B: Developing (embryonic/infant)": {
        "core": [
            "SPP1", "GPNMB", "IGF1", "CD68", "CST7",
            "LPL", "APOE", "LGALS3", "MMP9", "CTSB",
        ],
    },
    "C: Complement (synaptic pruning)": {
        "core": [
            "C1QA", "C1QB", "C1QC", "C3", "ITGAM",
            "ITGB2", "CFH", "CFB", "C3AR1",
        ],
    },
    "D: Proliferation (cell cycle)": {
        "core": ["MKI67", "TOP2A", "CDK1", "PCNA", "CENPF"],
    },
    "E: Pro-inflammatory (priming/aging)": {
        "core": [
            "IL1B", "TNF", "NLRP3", "CYBB", "CCL2", "CD74", "IL6", "CXCL10", "IL18",
        ],
    },
    "F: DAM (disease-assoc.)": {
        "core": ["TREM2", "APOE", "LPL", "CTSB", "CTSD", "CD9"],
    },
    "G: IFN-alpha (human aging)": {
        "core": ["IFIT3", "ISG15", "IRF7", "IFIT1", "MX1", "BST2"],
    },
}

# Collect all marker genes
ALL_MARKER_GENES = []
for mod_name, mod_dict in GENE_MODULES.items():
    for g in mod_dict["core"]:
        if g not in ALL_MARKER_GENES:
            ALL_MARKER_GENES.append(g)

# Check which genes are present in the data
AVAILABLE_GENES = [g for g in ALL_MARKER_GENES if g in adata.var_names]
MISSING_GENES = [g for g in ALL_MARKER_GENES if g not in adata.var_names]

print(f"\n>>> Gene availability:")
print(f"    Total marker genes: {len(ALL_MARKER_GENES)}")
print(f"    Present in data   : {len(AVAILABLE_GENES)}")
if MISSING_GENES:
    print(f"    Missing           : {', '.join(MISSING_GENES)}")

# Keep only modules with at least one available gene
GENE_MODULES_FILTERED = {}
for mod_name, mod_dict in GENE_MODULES.items():
    available = [g for g in mod_dict["core"] if g in adata.var_names]
    if available:
        GENE_MODULES_FILTERED[mod_name] = {"core": available}

# ============================================================
# 3. Mean expression matrices (stage / phase / period)
# ============================================================
def compute_mean_expression(adata, groupby, group_order):
    """
    Compute the mean expression of every gene within each group.

    Parameters
    ----------
    adata : anndata.AnnData
        Expression data (cells x genes).
    groupby : str
        Column of adata.obs used to define groups.
    group_order : list of str
        Ordered group labels; becomes the columns of the result.

    Returns
    -------
    pandas.DataFrame
        Mean expression matrix, genes as index, groups as columns.
    """
    adata.obs[groupby] = pd.Categorical(
        adata.obs[groupby], categories=group_order, ordered=True
    )
    adata.obs[groupby] = adata.obs[groupby].cat.remove_unused_categories()

    means = []
    for grp in group_order:
        sub = adata[adata.obs[groupby] == grp, :]
        expr = sub.X
        if issparse(expr):
            expr = expr.toarray()
        means.append(np.asarray(expr.mean(axis=0)).flatten())

    mean_mat = np.vstack(means).T  # genes x groups
    return pd.DataFrame(mean_mat, index=adata.var_names, columns=group_order)


print("\n>>> Computing mean expression matrices ...")

# Ensure stage / phase / period are stored as strings
for col in ["stage", "phase", "period"]:
    if col in adata.obs.columns:
        adata.obs[col] = adata.obs[col].astype(str)

expr_stage = compute_mean_expression(adata, "stage", STAGE_ORDER)
expr_phase = compute_mean_expression(adata, "phase", PHASE_ORDER)
expr_period = compute_mean_expression(adata, "period", PERIOD_ORDER)

print("    Stage matrix :", expr_stage.shape)
print("    Phase matrix :", expr_phase.shape)
print("    Period matrix:", expr_period.shape)

# Export the computed matrices for downstream analysis and plotting
os.makedirs(OUTPUT_DIR, exist_ok=True)
expr_stage.to_csv(os.path.join(OUTPUT_DIR, "expr_stage_15.csv"))
expr_phase.to_csv(os.path.join(OUTPUT_DIR, "expr_phase_08.csv"))
expr_period.to_csv(os.path.join(OUTPUT_DIR, "expr_period_03.csv"))
print(f"Expression matrices saved to: {OUTPUT_DIR}")
