#!/usr/bin/env python3
"""
=============================================================================
File      : 06_Fig6.py
Directory : 11_figures
Purpose   : Fig6 — UniBrain agent validation. Panel B: atlas coverage heatmap
            (brain region view). Panel C: cell-type prediction confusion matrix
            (hs29). Panel D: marker gene dotplot of generated cells (hs20).
            Panel F: microglia lifespan module trajectory.
Input     : output/Analysis/model/BrianAtlasAgent/output/
Output    : Visualization/Fig6/
=============================================================================
"""

import os
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from matplotlib.patches import Patch
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import pdist

plt.rcParams["font.family"] = "Arial"
plt.rcParams.update({"font.size": 8})
mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42

# ============================================================================
# Configuration
# ============================================================================
proj_dir = os.environ.get("PROJ_DIR", "path/to/your/project")  # <-- MODIFY THIS
model_dir = os.path.join(proj_dir, "data", "human", "output",
                         "Analysis", "model", "BrianAtlasAgent")
out_dir = os.path.join(proj_dir, "data", "human", "output", "Visualization", "Fig6")
os.makedirs(out_dir, exist_ok=True)

# ============================================================================
# Shared palettes
# ============================================================================
col_stage = {
    "01-Early first trimester": "#1A318B", "02-Late first trimester": "#2E4CB5",
    "03-Early second trimester": "#496CCE", "04-Mid second trimester": "#658AD7",
    "05-Late second trimester": "#82AAE7", "06-Early third trimester": "#A2C0EC",
    "07-Late third trimester": "#B9D2F3", "08-Neonatal and early infancy": "#CFE0F9",
    "09-Mid infancy": "#E6F0FE", "10-Early childhood": "#F0E8F1",
    "11-Middle and late childhood": "#F9DBE5", "12-Adolescence": "#F8BED6",
    "13-Young adulthood": "#F7A6BF", "14-Middle adulthood": "#EA89A0",
    "15-Late adulthood": "#9A133D",
}

col_region = {
    "Brain": "#5c8e9b", "Diencephalon": "#85b76e", "Mesencephalon": "#decf98",
    "Metencephalon ": "#874e94", "Myelencephalon ": "#e97c17",
    "Prosencephalon": "#da84b6", "Rhombencephalon": "#4f9446",
    "Spinal cord": "#2a71aa", "Telencephalon": "#ae94d7",
}

# ============================================================================
# Panel B: Atlas coverage heatmap (brain region view)
# ============================================================================
cov_file = os.path.join(model_dir, "output", "validation_data",
                        "atlas_coverage_matrix_stage_celltype.csv")
atlas_h5ad = os.path.join(proj_dir, "data", "human", "output",
                          "Annotation_hvg5000", "final", "hs_sample380k_norm.h5ad")

if os.path.exists(cov_file) and os.path.exists(atlas_h5ad):
    # Load metadata for region annotation
    adata = sc.read_h5ad(atlas_h5ad, backed="r")
    meta = adata.obs[["stage", "region_L2", "type_2"]].dropna()
    regions_uniq = sorted(meta["region_L2"].unique())

    # Load coverage matrix
    cov_df = pd.read_csv(cov_file, index_col=[0, 1])
    # Cluster regions by total coverage
    region_counts = cov_df.sum(axis=0)
    regions_clustered = region_counts.sort_values(ascending=False).index.tolist()

    # Build heatmap data per stage
    stage_order = list(col_stage.keys())
    heatmap_data = []
    row_labels = []
    for stage in stage_order:
        if stage in cov_df.index.get_level_values(0):
            stage_data = cov_df.loc[stage]
            stage_data = stage_data.reindex(columns=regions_clustered, fill_value=0)
            heatmap_data.append(stage_data.values)
            row_labels.append(stage)
    heatmap_mat = np.vstack(heatmap_data)

    # Hierarchical clustering of stages
    if heatmap_mat.shape[0] > 1:
        Z = linkage(pdist(heatmap_mat > 0, metric="hamming"), method="average")
        order = leaves_list(Z)
        heatmap_mat = heatmap_mat[order]
        row_labels = [row_labels[i] for i in order]

    fig, ax = plt.subplots(figsize=(14, 8))
    im = ax.imshow(heatmap_mat, aspect="auto", cmap="Greys",
                   vmin=0, vmax=1, interpolation="nearest")

    ax.set_xticks(range(len(regions_clustered)))
    ax.set_xticklabels(regions_clustered, rotation=90, fontsize=7)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=7)

    # Stage colour bar
    for i, s in enumerate(row_labels):
        if s in col_stage:
            ax.add_patch(plt.Rectangle((-0.8, i - 0.5), 0.6, 1,
                        color=col_stage[s], clip_on=False, transform=ax.transData))

    # Region colour bar
    for j, r in enumerate(regions_clustered):
        if r in col_region:
            ax.add_patch(plt.Rectangle((j - 0.5, -0.8), 1, 0.6,
                        color=col_region[r], clip_on=False, transform=ax.transData))

    ax.set_xlabel("Brain Region")
    ax.set_ylabel("Developmental Stage")
    ax.set_title("Atlas Coverage (Observed = dark, Imputed = light)")

    # Legend
    legend_els = [Patch(facecolor="#0B1F3A", label="Observed"),
                  Patch(facecolor="#D9D9D9", label="Imputed")]
    fig.legend(handles=legend_els, loc="center right", frameon=False, fontsize=7)
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "brain_atlas_coverage_heatmap_br.pdf"),
                dpi=600, bbox_inches="tight")
    plt.close()
    print("Panel B saved")

# ============================================================================
# Panel C: Cell-type prediction confusion matrix (hs29)
# ============================================================================
pred_file = os.path.join(model_dir, "output", "prediction_proportion.csv")
if os.path.exists(pred_file):
    cm_prop = pd.read_csv(pred_file, index_col=0)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm_prop, annot=True, fmt=".2f",
                cmap=sns.light_palette("darkred", as_cmap=True),
                linewidths=0.5, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "hg29_ct_prediction_heatmap.pdf"),
                dpi=300, bbox_inches="tight")
    plt.close()
    print("Panel C saved")

# ============================================================================
# Panel D: Marker gene dotplot of generated cells (hs20)
# ============================================================================
gen_h5ad = os.path.join(model_dir, "output", "v6.2_model_hs33_generation.h5ad")
if os.path.exists(gen_h5ad):
    generated_adata = sc.read_h5ad(gen_h5ad)

    # Define marker genes per cell type
    marker_genes_dict = {
        "Radial glia": ["HES1", "SOX2", "VIM"],
        "Glioblast": ["OLIG2", "ASCL1", "DLL3"],
        "Neuroblast": ["DCX", "STMN2", "TUBB3"],
        "Glutamatergic neuron": ["SLC17A7", "NEUROD6", "TBR1"],
        "GABAergic neuron": ["GAD1", "GAD2", "DLX2"],
        "Oligodendrocyte": ["MBP", "MOG", "PLP1"],
        "Astrocyte": ["GFAP", "AQP4", "SLC1A3"],
        "Microglia": ["CX3CR1", "TMEM119", "P2RY12"],
    }

    # Filter to genes present in data
    valid = {ct: [g for g in genes if g in generated_adata.var_names]
             for ct, genes in marker_genes_dict.items()}
    valid = {k: v for k, v in valid.items() if v}

    if valid:
        sc.settings.set_figure_params(dpi=300, facecolor="white")
        sc.pl.dotplot(generated_adata, valid, groupby="type_1",
                      standard_scale="var", cmap="Reds", dot_max=0.6,
                      figsize=(10, 4),
                      save=os.path.join(out_dir, "marker_genes_generation_hs20.pdf"))
        print("Panel D saved")
else:
    print(f"Generation h5ad not found: {gen_h5ad}")

# ============================================================================
# Panel F: Microglia lifespan module trajectory
# ============================================================================
mg_h5ad = os.path.join(model_dir, "output", "microglia_lifespan_generation.h5ad")
if os.path.exists(mg_h5ad):
    adata_mg = sc.read_h5ad(mg_h5ad)

    # Define functional modules and their marker genes
    modules = {
        "Proliferative": ["MKI67", "TOP2A", "PCNA", "CCNB1", "CENPF"],
        "Enhanced redox": ["SOD2", "GPX1", "PRDX1", "TXN", "CAT"],
        "Inflammatory": ["IL1B", "TNF", "CCL2", "NFKB1", "TLR2"],
        "Lipid associated": ["APOE", "TREM2", "LPL", "ABCA1", "CH25H"],
        "Surveilling": ["P2RY12", "CX3CR1", "TMEM119", "SALL1", "SIGLEC1"],
        "Reacting": ["IFI44L", "ISG15", "MX1", "IFIT3", "OAS1"],
    }

    PHASE_ORDER = ["First trimester", "Second trimester", "Third trimester",
                   "Infancy", "Childhood", "Adolescence", "Adulthood", "Aged"]

    module_colors = {"Proliferative": "#D175A9", "Enhanced redox": "#A09CBF",
                     "Inflammatory": "#C47401", "Lipid associated": "#4C949F",
                     "Surveilling": "#2A71AA", "Reacting": "grey"}

    # Compute mean expression per phase per module
    n_rows = len(modules)
    n_cols = 5
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 2.5 * n_rows))

    for row, (module, genes) in enumerate(modules.items()):
        genes_present = [g for g in genes if g in adata_mg.var_names]
        if not genes_present:
            continue
        expr = adata_mg[:, genes_present].X
        expr = expr.toarray() if hasattr(expr, "toarray") else expr
        phase_means = []
        for ph in PHASE_ORDER:
            mask = adata_mg.obs["phase"] == ph
            phase_means.append(expr[mask].mean(axis=0) if mask.sum() > 0 else np.zeros(len(genes_present)))

        for col in range(min(n_cols, len(genes_present))):
            ax = axes[row, col] if n_rows > 1 else axes[col]
            gene = genes_present[col]
            gidx = list(genes_present).index(gene)
            vals = [pm[gidx] for pm in phase_means]
            x = range(len(PHASE_ORDER))
            ax.plot(x, vals, "o-", color=module_colors[module], markersize=3, linewidth=1)
            ax.set_title(gene, fontsize=8, pad=2)
            ax.set_xticks(x)
            ax.set_xticklabels(PHASE_ORDER, rotation=90, fontsize=5)
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_linewidth(0.8)
        # Module label on left
        if n_rows > 1:
            axes[row, 0].text(-0.4, 0.5, module, transform=axes[row, 0].transAxes,
                              rotation=90, va="center", ha="center",
                              fontsize=9, fontweight="bold",
                              color=module_colors[module])

    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "microglia_lifespan_modules.pdf"),
                dpi=300, bbox_inches="tight")
    plt.close()
    print("Panel F saved")
else:
    print(f"Microglia lifespan h5ad not found: {mg_h5ad}")

print("Fig6 complete.")
