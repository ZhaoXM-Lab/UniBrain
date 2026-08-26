#!/usr/bin/env python3
"""
=============================================================================
File      : 02_Fig2.py
Directory : 11_figures
Purpose   : Fig2 — Lineage tree and key regulator barplot.
            Panel A: Developmental lineage tree with time-point columns and
                     cell-type rows, edges coloured by transition probability.
            Panel B: Stacked barplot of candidate key regulators (TF / RBP /
                     TF and RBP) per cell type, with cell-type/lineage colour bars.
Input     : output/Analysis/lineage/all_cell/human_edge_pca_k5.txt
            output/Analysis/lineage/all_cell/TF_RBP/lineage_regulator_info.csv
Output    : output/Visualization/Fig2/type_lineage_tree.pdf
            output/Visualization/Fig2/type_lineage_regulator_counts.pdf
            output/Visualization/Fig2/type_lineage_regulator_colorbar.pdf
=============================================================================
"""

import os
import re
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
from matplotlib.path import Path

# ============================================================================
# Configuration
# ============================================================================
proj_dir = os.environ.get("PROJ_DIR", "path/to/your/project")  # <-- MODIFY THIS
data_dir = os.path.join(proj_dir, "data", "human", "output")
out_dir  = os.path.join(data_dir, "Visualization", "Fig2")
os.makedirs(out_dir, exist_ok=True)

# ============================================================================
# Shared colour palettes
# ============================================================================
LINEAGE_COLORS = {
    "Glutamatergic Neuron": "#DC9720",
    "GABAergic Neuron":     "#EA7075",
    "Neural Precursor Cell": "#7AA8D9",
    "Glia":                 "#BBD642",
    "Non-neural":           "#8E8E90",
    "default":              "black",
}

TYPE_COLOR = {
    "Astrocyte": "#2E8B57", "Ependymal": "#FF69B4",
    "Oligodendrocyte": "#DC143C", "Glioblast": "#90EE90",
    "Oligodendrocyte_precursor": "#32CD32",
    "Oligodendrocyte_precursor_committed": "#ADFF2F",
    "Neuroblast": "#9370DB", "Neuronal_IPC": "#8A2BE2",
    "Radial_glia": "#4682B4",
    "L2/3-CUX2 Glu": "#87CEEB", "L2/3-SATB2 Glu": "#87CEFA",
    "L4-RORB Glu": "#6495ED", "L5-BCL11B Glu": "#4169E1",
    "L5/6-TLE4 Glu": "#1E90FF", "L5/6-THEMIS Glu": "#0000CD",
    "Thalamic_Glu": "#7B68EE", "Hippocampal_Glu": "#6A5ACD",
    "Granule_neuron": "#B0C4DE", "Hypothalamic_Glu": "#5F9EA0",
    "PVALB+ GABA": "#FF6347", "SST+ GABA": "#FF4500",
    "LHX6+LAMP5+ GABA": "#DC143C", "CALB2+ GABA": "#FF1493",
    "VIP+ GABA": "#C71585", "RELN+ GABA": "#8B008B",
    "Medium_spiny_neuron": "#9400D3", "Purkinje_neuron": "#BA55D3",
    "Thalamic_GABA": "#9932CC", "Midbrain-derived_GABA": "#800080",
    "Serotonergic": "#FFA500", "Cholinergic": "#FFFF00",
    "Microglia": "#8B4513", "Macrophage": "#A0522D",
    "Fibroblast": "#D2B48C", "Endothelial_cell": "#1E90FF",
    "Pericyte": "#6495ED", "Choroid_plexus_epithelial": "#FF6347",
    "Vascular_smooth_muscle_cell": "#FF8C00",
    "Glutamatergic neuron": "#f2a297", "GABAergic neuron": "#f8ccbe",
}

REGULATOR_COLOR = {"TF": "#D2342B", "RBP": "#4A7BB3", "TF and RBP": "#8F519D"}

# ============================================================================
# Panel A: Lineage tree
# ============================================================================
def create_type_lineage_tree(final_tree, figsize=(7.87, 4.72)):
    """Create a developmental lineage tree with time points as columns and
    cell types as rows. Edges are coloured by transition probability.

    Parameters
    ----------
    final_tree : pd.DataFrame
        Columns: pre, nex, prob.
    figsize : tuple
        Figure size in inches.
    """
    fig, ax = plt.subplots(figsize=figsize)

    all_nodes = set(final_tree["pre"]).union(set(final_tree["nex"]))

    # Cell type → row index
    cell_type_order = [
        "Microglia", "Macrophage", "Fibroblast", "Endothelial_cell", "Pericyte",
        "Choroid_plexus_epithelial", "Vascular_smooth_muscle_cell", "Astrocyte",
        "Ependymal", "Oligodendrocyte", "Root",
        "Oligodendrocyte_precursor_committed", "Oligodendrocyte_precursor",
        "Glioblast", "Radial_glia", "Neuronal_IPC", "Neuroblast",
        "GABAergic neuron", "PVALB+ GABA", "SST+ GABA",
        "LHX6+LAMP5+ GABA", "CALB2+ GABA", "VIP+ GABA", "RELN+ GABA",
        "Medium_spiny_neuron", "Purkinje_neuron", "Thalamic_GABA",
        "Midbrain-derived_GABA", "Glutamatergic neuron", "L2/3-CUX2 Glu",
        "L2/3-SATB2 Glu", "L4-RORB Glu", "L5-BCL11B Glu", "L5/6-TLE4 Glu",
        "L5/6-THEMIS Glu", "Thalamic_Glu", "Hippocampal_Glu",
        "Granule_neuron", "Hypothalamic_Glu",
    ]
    cell_type_to_row = {ct: i for i, ct in enumerate(cell_type_order)}

    # Parse time points and assign x-coordinates
    time_points = sorted(
        {node.split(":")[0] for node in all_nodes if ":" in node},
        key=lambda s: float(s[1:]),
    )
    int_tps = [tp for tp in time_points if float(tp[1:]).is_integer()]
    base_spacing = 1.7
    time_to_x = {tp: i * base_spacing for i, tp in enumerate(int_tps)}
    for tp in time_points:
        if tp not in time_to_x:
            tv = float(tp[1:])
            prev_int = f"S{int(tv):02d}"
            next_int = f"S{int(tv) + 1:02d}"
            if prev_int in time_to_x and next_int in time_to_x:
                time_to_x[tp] = (time_to_x[prev_int] + time_to_x[next_int]) / 2
            else:
                time_to_x[tp] = max(time_to_x.values()) + base_spacing if time_to_x else 0

    # Node positions: (time_x, row_y)
    node_positions = {}
    for tp in time_points:
        time_nodes = [n for n in all_nodes if n.startswith(tp + ":")]
        cell_groups = {}
        for node in time_nodes:
            ct_full = node.split(":", 1)[1]
            matched = ct_full if ct_full in cell_type_order else next(
                (dt for dt in cell_type_order if re.search(r"\b" + re.escape(dt) + r"\b", ct_full)), None
            )
            if matched is None:
                cell_type_to_row[ct_full] = len(cell_type_order)
                cell_type_order.append(ct_full)
                matched = ct_full
            cell_groups.setdefault(matched, []).append(node)
        for ct, nodes_in_group in cell_groups.items():
            row = cell_type_to_row[ct]
            ys = [row] if len(nodes_in_group) == 1 else np.linspace(row - 0.3, row + 0.3, len(nodes_in_group))
            for i, node in enumerate(nodes_in_group):
                node_positions[node] = (time_to_x[tp], ys[i])

    # Draw edges (Bezier curves, colour by probability)
    for _, row in final_tree.iterrows():
        parent, child = row["pre"], row["nex"]
        if parent not in node_positions or child not in node_positions:
            continue
        x1, y1 = node_positions[parent]
        x2, y2 = node_positions[child]
        cy = (y1 + y2) / 2
        verts = [(x1, y1), (x1 + 0.3, cy), (x2 - 0.3, cy), (x2, y2)]
        path = Path(verts, [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4])
        prob = row.get("prob", 0.5)
        gv = 0.87 - prob * 0.87
        ax.add_patch(FancyArrowPatch(path=path, facecolor=(gv, gv, gv),
                     edgecolor=(gv, gv, gv), linewidth=4, alpha=0.8,
                     arrowstyle="->", mutation_scale=10))

    # Cell type → lineage category colour
    ct_colors = {}
    for ct in cell_type_order:
        if ct in ["L2/3-CUX2 Glu", "L2/3-SATB2 Glu", "L4-RORB Glu", "L5-BCL11B Glu",
                  "L5/6-TLE4 Glu", "L5/6-THEMIS Glu", "Thalamic_Glu",
                  "Hippocampal_Glu", "Granule_neuron", "Hypothalamic_Glu"]:
            ct_colors[ct] = LINEAGE_COLORS["Glutamatergic Neuron"]
        elif ct in ["PVALB+ GABA", "SST+ GABA", "LHX6+LAMP5+ GABA", "CALB2+ GABA",
                    "VIP+ GABA", "RELN+ GABA", "Medium_spiny_neuron",
                    "Purkinje_neuron", "Thalamic_GABA", "Midbrain-derived_GABA"]:
            ct_colors[ct] = LINEAGE_COLORS["GABAergic Neuron"]
        elif ct in ["Astrocyte", "Ependymal", "Oligodendrocyte"]:
            ct_colors[ct] = LINEAGE_COLORS["Glia"]
        elif ct in ["Microglia", "Macrophage", "Fibroblast", "Endothelial_cell",
                    "Pericyte", "Choroid_plexus_epithelial", "Vascular_smooth_muscle_cell"]:
            ct_colors[ct] = LINEAGE_COLORS["Non-neural"]
        elif ct in ["Glioblast", "Neuroblast", "Neuronal_IPC", "Radial_glia",
                    "Oligodendrocyte_precursor_committed", "Oligodendrocyte_precursor"]:
            ct_colors[ct] = LINEAGE_COLORS["Neural Precursor Cell"]
        else:
            ct_colors[ct] = LINEAGE_COLORS["default"]

    # Draw nodes
    for node, (x, y) in node_positions.items():
        ct = node.split(":", 1)[1] if ":" in node else node
        color = ct_colors.get(ct, "gray")
        ax.plot(x, y, "o", markersize=14, color="white",
                markeredgecolor=color, markeredgewidth=4.5, zorder=5)

    # Cell type labels (right side)
    max_x = max(p[0] for p in node_positions.values())
    for ct, row in cell_type_to_row.items():
        ax.text(max_x + 0.5, row, ct, ha="left", va="center", fontsize=13,
                weight="bold", color="black")

    # Time point labels (bottom)
    for tp, x in time_to_x.items():
        ax.text(x, -1, tp, ha="center", va="top", fontsize=12,
                fontweight="bold", color="black")

    all_x = [p[0] for p in node_positions.values()]
    all_y = [p[1] for p in node_positions.values()]
    ax.set_xlim(min(all_x) - 0.5, max(all_x) + 3)
    ax.set_ylim(min(all_y) - 1, max(all_y) + 1)
    ax.set_title("Cell Lineage Tree", pad=30, fontsize=18, weight="bold")
    ax.axis("off")

    # Category legend
    leg = [plt.Line2D([0], [0], marker="o", color="w", markersize=8,
                      markerfacecolor="white", markeredgewidth=3,
                      markeredgecolor=c, label=lab)
           for lab, c in LINEAGE_COLORS.items() if lab != "default"]
    ax.legend(handles=leg, loc="upper left", ncol=1, fontsize=12,
              title="Cell Lineage", title_fontsize=12, frameon=False)

    # Probability colour bar
    cmap = mpl.colors.LinearSegmentedColormap.from_list(
        "prob_gray",
        [(0.87, 0.87, 0.87), (0.0, 0.0, 0.0)], N=256,
    )
    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=mpl.colors.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cbar_ax = fig.add_axes([0.05, 0.65, 0.015, 0.1])
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation="vertical")
    cbar.set_ticks([0.0, 0.25, 0.5, 0.75, 1.0])
    cbar.ax.tick_params(labelsize=10, length=0)
    cbar.outline.set_visible(False)
    title_x = cbar_ax.get_xlim()[0] + (cbar_ax.get_xlim()[1] - cbar_ax.get_xlim()[0]) / 2
    cbar_ax.text(title_x, cbar_ax.get_ylim()[1] + 0.03, "Edge Probability",
                 ha="center", va="bottom", fontsize=12, weight="bold")

    plt.tight_layout()
    return fig


# ============================================================================
# Panel B: Regulator barplot
# ============================================================================
def process_regulator_data(file_path):
    """Load regulator info CSV and prepare stacked-barplot data.

    Returns: stack_df, cell_type_to_color, cell_type_to_trunk, trunk_to_color
    """
    df = pd.read_csv(file_path)
    regulator_order = ["TF", "RBP", "TF and RBP"]

    counts = df.groupby(["cell_type", "regulator_type"])["gene"].nunique().reset_index()
    counts.columns = ["cell_type", "regulator_type", "count"]

    stack_df = counts.pivot(index="cell_type", columns="regulator_type", values="count").fillna(0)
    for rt in regulator_order:
        if rt not in stack_df.columns:
            stack_df[rt] = 0
    stack_df = stack_df[regulator_order]
    stack_df["total"] = stack_df.sum(axis=1)
    stack_df = stack_df.sort_values("total", ascending=False).drop("total", axis=1)

    cell_type_to_color = {c: TYPE_COLOR.get(c, "#000000") for c in stack_df.index}

    trunk_to_color = {
        "Neural Precursor Cell": "#7AA8D9",
        "Glutamatergic Neuron": "#DC9720",
        "GABAergic Neuron": "#EA7075",
        "Glia": "#BBD642",
        "Non-neural": "#8E8E90",
    }

    cell_type_to_trunk = {}
    for ct in stack_df.index:
        if ct in ["L2/3-CUX2 Glu", "L2/3-SATB2 Glu", "L4-RORB Glu", "L5-BCL11B Glu",
                  "L5/6-TLE4 Glu", "L5/6-THEMIS Glu", "Thalamic_Glu",
                  "Hippocampal_Glu", "Granule_neuron", "Hypothalamic_Glu",
                  "Glutamatergic neuron"]:
            cell_type_to_trunk[ct] = "Glutamatergic Neuron"
        elif ct in ["PVALB+ GABA", "SST+ GABA", "LHX6+LAMP5+ GABA", "CALB2+ GABA",
                    "VIP+ GABA", "RELN+ GABA", "Medium_spiny_neuron",
                    "Purkinje_neuron", "Thalamic_GABA", "Midbrain-derived_GABA",
                    "GABAergic neuron"]:
            cell_type_to_trunk[ct] = "GABAergic Neuron"
        elif ct in ["Astrocyte", "Ependymal", "Oligodendrocyte"]:
            cell_type_to_trunk[ct] = "Glia"
        elif ct in ["Microglia", "Macrophage", "Fibroblast", "Endothelial_cell",
                    "Pericyte", "Choroid_plexus_epithelial", "Vascular_smooth_muscle_cell"]:
            cell_type_to_trunk[ct] = "Non-neural"
        elif ct in ["Glioblast", "Neuroblast", "Neuronal_IPC", "Radial_glia",
                    "Oligodendrocyte_precursor_committed", "Oligodendrocyte_precursor"]:
            cell_type_to_trunk[ct] = "Neural Precursor Cell"
        else:
            cell_type_to_trunk[ct] = "Unknown"
            trunk_to_color["Unknown"] = "#000000"

    return stack_df, cell_type_to_color, cell_type_to_trunk, trunk_to_color


def plot_regulator_bar(stack_df, cell_type_to_trunk, trunk_to_color, figsize=None):
    """Stacked barplot of TF / RBP / TF+RBP counts per cell type."""
    if figsize is None:
        figsize = (max(12, len(stack_df) * 0.4), 6)
    fig, ax = plt.subplots(figsize=figsize)

    regulator_order = stack_df.columns.tolist()
    colors = [REGULATOR_COLOR[rt] for rt in regulator_order]
    total_counts = stack_df.sum(axis=1).values

    bottom = np.zeros(len(stack_df))
    for i, rt in enumerate(regulator_order):
        ax.bar(range(len(stack_df)), stack_df[rt].values, bottom=bottom,
               color=colors[i], alpha=0.8, linewidth=0.5, label=rt, zorder=1)
        bottom += stack_df[rt].values

    for i, total in enumerate(total_counts):
        ax.text(i, total + 0.1, str(int(total)), ha="center", va="bottom",
                fontsize=11, zorder=4)

    ax.set_xlabel("Cell type", fontsize=14, fontweight="bold")
    ax.set_ylabel("Number of candidate key regulators", fontsize=14, fontweight="bold")
    ax.set_xticks(range(len(stack_df)))
    ax.set_xticklabels(stack_df.index, rotation=90, ha="right", fontsize=12)
    ax.set_ylim(0, max(total_counts) * 1.15)
    ax.set_xlim(-0.6, len(stack_df) - 0.4)

    reg_leg = [mpatches.Patch(color=REGULATOR_COLOR[rt], label=rt) for rt in regulator_order]
    unique_trunks = list(set(cell_type_to_trunk.values()))
    trunk_leg = [mpatches.Patch(color=trunk_to_color[t], label=t)
                 for t in unique_trunks if t != "Unknown"]
    ax.legend(handles=reg_leg + trunk_leg, bbox_to_anchor=(0.98, 0.98),
              loc="upper right", fontsize=11, frameon=False)
    plt.tight_layout()
    return fig


def plot_color_bars(stack_df, cell_type_to_color, cell_type_to_trunk,
                    trunk_to_color, figsize=None):
    """Two-row colour bar (cell type + lineage) aligned with the barplot."""
    if figsize is None:
        figsize = (max(12, len(stack_df) * 0.4), 1.5)
    fig, ax = plt.subplots(figsize=figsize)

    cell_types = stack_df.index.tolist()
    bar_width = 0.8
    x_positions = np.arange(len(cell_types))

    y1, h = 0.6, 0.1
    for i, cell in enumerate(cell_types):
        ax.add_patch(mpatches.Rectangle((x_positions[i] - bar_width / 2, y1),
                     bar_width, h, color=cell_type_to_color[cell],
                     linewidth=0.5, edgecolor="black"))

    y2 = y1 - h - 0.05
    for i, cell in enumerate(cell_types):
        trunk = cell_type_to_trunk[cell]
        ax.add_patch(mpatches.Rectangle((x_positions[i] - bar_width / 2, y2),
                     bar_width, h, color=trunk_to_color[trunk],
                     linewidth=0.5, edgecolor="black"))

    ax.text(-0.5, y1 + h / 2, "Cell Type", ha="right", va="center", fontsize=10)
    ax.text(-0.5, y2 + h / 2, "Cell Lineage", ha="right", va="center", fontsize=10)
    ax.set_xlim(-0.6, len(cell_types) - 0.4)
    ax.set_ylim(0, 1)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(cell_types, rotation=90, ha="right", fontsize=12)
    ax.set_yticks([])
    ax.set_xlabel("Cell type", fontsize=14, fontweight="bold")
    for sp in ["top", "right", "left"]:
        ax.spines[sp].set_visible(False)
    plt.tight_layout()
    return fig


# ============================================================================
# Main
# ============================================================================
if __name__ == "__main__":
    # --- Panel A: Lineage tree ---
    tree_file = os.path.join(data_dir, "Analysis", "lineage", "all_cell",
                             "human_edge_pca_k5.txt")
    final_tree = pd.read_csv(tree_file, sep="\t", header=None,
                             names=["pre", "nex", "prob"])
    fig_a = create_type_lineage_tree(final_tree)
    fig_a.savefig(os.path.join(out_dir, "type_lineage_tree.pdf"),
                  dpi=600, bbox_inches="tight")
    fig_a.savefig(os.path.join(out_dir, "type_lineage_tree.png"),
                  dpi=300, bbox_inches="tight")
    plt.close(fig_a)
    print(f"Panel A saved to: {out_dir}")

    # --- Panel B: Regulator barplot ---
    reg_file = os.path.join(data_dir, "Analysis", "lineage", "all_cell",
                            "TF_RBP", "lineage_regulator_info.csv")
    stack_df, ct_color, ct_trunk, trunk_color = process_regulator_data(reg_file)

    fig_b1 = plot_regulator_bar(stack_df, ct_trunk, trunk_color)
    fig_b1.savefig(os.path.join(out_dir, "type_lineage_regulator_counts.pdf"),
                   dpi=600, bbox_inches="tight")
    fig_b1.savefig(os.path.join(out_dir, "type_lineage_regulator_counts.png"),
                   dpi=300, bbox_inches="tight")
    plt.close(fig_b1)

    fig_b2 = plot_color_bars(stack_df, ct_color, ct_trunk, trunk_color)
    fig_b2.savefig(os.path.join(out_dir, "type_lineage_regulator_colorbar.pdf"),
                   dpi=600, bbox_inches="tight")
    fig_b2.savefig(os.path.join(out_dir, "type_lineage_regulator_colorbar.png"),
                   dpi=300, bbox_inches="tight")
    plt.close(fig_b2)
    print(f"Panel B saved to: {out_dir}")
