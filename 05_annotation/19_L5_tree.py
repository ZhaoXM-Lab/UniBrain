#!/usr/bin/env python3
"""
=============================================================================
File      : 19_L5_tree.py
Directory : 05_annotation
Purpose   : Level 5 hierarchical merging of consensus clusters into type_4 cell types.
=============================================================================
"""

# ============================================================================
# Configuration: set PROJ_DIR to the project root before running
# ============================================================================
import os

proj_dir = os.path.join(
    os.environ.get("PROJ_DIR", "path/to/your/project"), "data", "human"
)
import pickle
import re
from pathlib import Path

import anndata
import numpy as np
import pandas as pd
import scanpy as sc
from scipy.sparse import issparse
from sklearn.metrics import pairwise_distances
from treelib import Tree

print("Scanpy version: %s" % sc.__version__)
print("Anndata version: %s\n" % anndata.__version__)


def build_mrtree(adata, cluster_key="cluster_5_merge"):
    """Build the initial hierarchical tree of clusters.

    The root holds all cells; each leaf holds one cluster with its cells,
    centroid (harmony-corrected PCA), cell count and dominant-sample ratio.

    Args:
        adata: AnnData object.
        cluster_key: ``obs`` column with the clusters to merge.

    Returns:
        treelib.Tree object.
    """
    clusters = adata.obs[cluster_key].unique()
    tree = Tree()

    # Root node
    root_id = "root"
    tree.create_node(identifier=root_id, data={
        "cells": adata.obs_names.tolist(),
        "centroid": np.mean(adata.obsm["X_pca_harmony"], axis=0),
        "cell_count": adata.n_obs,
        "sample_ratio": adata.obs["sample_new"].value_counts(normalize=True).max(),
    })

    # Leaf nodes (one per cluster)
    for cluster in clusters:
        cluster_cells = adata.obs[cluster_key] == cluster
        cells = adata.obs_names[cluster_cells].tolist()
        centroid = np.mean(adata[cluster_cells].obsm["X_pca_harmony"], axis=0)

        tree.create_node(
            identifier=cluster,
            parent=root_id,
            data={
                "cells": cells,
                "centroid": centroid,
                "cell_count": len(cells),
                "sample_ratio": (
                    adata[cells].obs["sample_new"].value_counts(normalize=True).max()
                ),
            },
        )
    return tree


def calculate_markers(adata, cluster_key="cluster_5_merge",
                      min_pts=0.5, min_log2fc=0.5,
                      pval_cutoff=0.05, min_cells_per_cluster=10):
    """Compute per-cluster marker genes (Wilcoxon, Bonferroni corrected).

    Clusters that are too small, have no expressed genes, or come from a
    single sample are skipped (markers set to an empty DataFrame).

    Args:
        adata: AnnData object.
        cluster_key: ``obs`` column used as the grouping factor.
        min_pts: Minimum expression fraction per cluster.
        min_log2fc: Minimum absolute log2 fold change.
        pval_cutoff: Adjusted p-value cutoff.
        min_cells_per_cluster: Minimum cells required for DE analysis.

    Returns:
        Dict mapping each cluster to a filtered marker DataFrame.
    """
    markers_dict = {}
    clusters = adata.obs[cluster_key].unique()

    valid_clusters_for_de = []  # clusters that can be analyzed for DE
    for cluster in clusters:
        cluster_cells = adata.obs[cluster_key] == cluster
        cluster_adata = adata[cluster_cells]

        # Check 1: too few cells for DE analysis
        if cluster_adata.n_obs < min_cells_per_cluster:
            print(
                "Warning: cluster %s has only %d cells (< %d), "
                "skipped for DE analysis."
                % (cluster, cluster_adata.n_obs, min_cells_per_cluster)
            )
            markers_dict[cluster] = pd.DataFrame()
            continue

        # Check 2: all genes unexpressed in this cluster
        if issparse(cluster_adata.X):
            gene_sums = cluster_adata.X.sum(axis=0).A1  # sparse sum
        else:
            gene_sums = cluster_adata.X.sum(axis=0)  # dense sum

        zero_genes_in_cluster = gene_sums == 0
        if zero_genes_in_cluster.all():
            print(
                "Warning: all genes are unexpressed in cluster %s, "
                "skipped for DE analysis." % cluster
            )
            markers_dict[cluster] = pd.DataFrame()
            continue

        # Check 3: cells must come from more than one sample
        unique_samples_in_cluster = adata.obs.loc[
            cluster_cells, "sample"
        ].nunique()
        if unique_samples_in_cluster > 1:
            valid_clusters_for_de.append(cluster)
        else:
            print(
                "Warning: cluster %s comes from %d sample(s), "
                "skipped for DE analysis." % (cluster, unique_samples_in_cluster)
            )
            markers_dict[cluster] = pd.DataFrame()

    if not valid_clusters_for_de:
        print("No cluster can be analyzed for differential expression.")
        return markers_dict

    sc.tl.rank_genes_groups(
        adata,
        groupby=cluster_key,
        groups=valid_clusters_for_de,
        method="wilcoxon",
        corr_method="bonferroni",
        pts=True,  # enable expression-fraction computation
    )

    result = adata.uns["rank_genes_groups"]

    for cluster in valid_clusters_for_de:
        df = pd.DataFrame({
            "gene": result["names"][cluster],
            "log2fc": result["logfoldchanges"][cluster],
            "pvals": result["pvals"][cluster],
            "qvals": result["pvals_adj"][cluster],
            "expr_frac": result["pts"][cluster],
        })
        df["cluster"] = cluster

        # Apply the filtering criteria
        df_filtered = df[
            (df["qvals"] < pval_cutoff)
            & (df["expr_frac"] >= min_pts)
            & (df["log2fc"].abs() >= min_log2fc)
        ].sort_values("log2fc", ascending=False)

        # Sort by absolute log2 fold change
        df_filtered["abs_log2fc"] = df_filtered["log2fc"].abs()
        markers_dict[cluster] = df_filtered.sort_values(
            "abs_log2fc", ascending=False
        )

    return markers_dict


def merge_nodes(tree, adata, cluster_key="cluster_5_merge", min_marker_counts=2,
                min_cell_count=50, max_sample_ratio=1.0):
    """Iteratively merge the weakest clusters into their closest sibling.

    A leaf node is a merge candidate if it has too few marker genes, too few
    cells, or is dominated by a single sample. Candidates are processed in
    ascending cell-count order; each candidate is merged into its nearest
    sibling by centroid distance.

    Args:
        tree: treelib.Tree with the current cluster hierarchy.
        adata: AnnData object (updated in place).
        cluster_key: ``obs`` column with the clusters to merge.
        min_marker_counts: Minimum marker genes to keep a cluster.
        min_cell_count: Minimum cells to keep a cluster.
        max_sample_ratio: Maximum dominant-sample ratio to keep a cluster.

    Returns:
        Tuple of (adata, tree).
    """
    iteration = 0
    merged_id_counter = max(
        [
            int(c[1:])
            for c in adata.obs[cluster_key].cat.categories
            if c.startswith("M")
        ],
        default=0,
    )

    while True:
        iteration += 1
        print("\n=== Merge iteration %d ===" % iteration)

        # Collect the leaf nodes (depth 1)
        nodes = [
            node for node in tree.all_nodes() if tree.depth(node) == 1
        ]
        if not nodes:
            print("No remaining nodes to merge")
            break

        if len(nodes) <= 1:
            print("Only %d node(s) remain, stop merging" % len(nodes))
            break

        all_markers = calculate_markers(adata, cluster_key=cluster_key)
        print(
            "Marker gene counts per cluster: %s"
            % {
                k: len(v) if not v.empty else 0
                for k, v in all_markers.items()
            }
        )

        # Find merge candidates
        candidates = []
        for node in nodes:
            cluster_id = node.identifier
            marker_count = len(all_markers.get(cluster_id, pd.DataFrame()))

            status = []
            if marker_count < min_marker_counts:
                status.append("too few markers")
            if node.data["cell_count"] < min_cell_count:
                status.append("too few cells")
            if node.data["sample_ratio"] > max_sample_ratio:
                status.append("sample ratio too high")

            if status:
                print(
                    "Candidate node %s: %s (cells=%d)"
                    % (cluster_id, ", ".join(status), node.data["cell_count"])
                )
                candidates.append(node)

        if not candidates:
            print("No candidate nodes to merge")
            break

        # Process candidates in ascending cell-count order
        candidates.sort(key=lambda x: x.data["cell_count"])
        target_node = candidates[0]

        def find_closest(node):
            """Find the sibling with the closest centroid."""
            siblings = [
                n for n in nodes
                if n != node
                and tree.parent(n.identifier) == tree.parent(node.identifier)
                and (n.identifier in tree)
            ]
            if not siblings:
                return None

            centroids = np.array([n.data["centroid"] for n in siblings])
            dist = pairwise_distances([node.data["centroid"]], centroids).flatten()
            closest = siblings[np.argmin(dist)]
            print(
                "Closest sibling of %s is %s (distance=%.2f)"
                % (node.identifier, closest.identifier, dist.min())
            )
            return closest

        closest_node = find_closest(target_node)
        if not closest_node or closest_node.identifier not in tree:
            print("No valid closest sibling found, skipping this round")
            continue

        # Perform the merge
        try:
            merged_id_counter += 1
            new_parent_id = "M%d" % merged_id_counter
            print(
                "Merging %s and %s" % (target_node.identifier, closest_node.identifier)
            )

            if new_parent_id not in adata.obs[cluster_key].cat.categories:
                adata.obs[cluster_key] = (
                    adata.obs[cluster_key].cat.add_categories([new_parent_id])
                )

            merged_cells = (
                target_node.data["cells"] + closest_node.data["cells"]
            )
            adata.obs.loc[merged_cells, cluster_key] = new_parent_id
            adata.obs[cluster_key] = (
                adata.obs[cluster_key]
                .cat.remove_categories(
                    [target_node.identifier, closest_node.identifier]
                )
            )
            merged_sample_ratio = (
                adata.obs.loc[merged_cells, "sample_new"]
                .value_counts(normalize=True)
                .max()
            )

            # Create the new parent node under the original parent
            parent = tree.parent(target_node.identifier)
            if parent and parent.identifier not in tree:
                parent = None

            tree.create_node(
                identifier=new_parent_id,
                parent=parent.identifier if parent else "root",
                data={
                    "cells": merged_cells,
                    "centroid": (
                        target_node.data["centroid"] * target_node.data["cell_count"]
                        + closest_node.data["centroid"] * closest_node.data["cell_count"]
                    ) / (target_node.data["cell_count"] + closest_node.data["cell_count"]),
                    "cell_count": len(merged_cells),
                    "sample_ratio": merged_sample_ratio,
                },
            )

            # Hang the original nodes under the new parent
            tree.move_node(target_node.identifier, new_parent_id)
            tree.move_node(closest_node.identifier, new_parent_id)

            # Update the cluster labels
            adata.obs.loc[merged_cells, cluster_key] = new_parent_id
            print("Current tree depth: %d" % tree.depth())
            print("Merge successful, new cluster distribution:\n%s" % tree.show())

        except Exception as e:
            print("Merge failed: %s" % str(e))
            import traceback
            traceback.print_exc()
            continue

        # Proceed directly to the next iteration
        continue

    return adata, tree


def apply_merging(adata):
    """Run the full merging workflow for one type_3 subpopulation.

    Args:
        adata: AnnData object with ``obs['cluster_5']``.

    Returns:
        Tuple of (adata, tree).
    """
    # Step 1: build the initial hierarchical tree
    print("=== Initializing the hierarchy tree ===")
    tree = build_mrtree(adata)

    adata.obs["cluster_5_merge"] = (
        adata.obs["cluster_5_merge"]
        .astype("category")
        .cat.add_categories([])  # make the category extendable
    )

    # Step 2: iteratively merge nodes
    print("\n=== Starting the merging workflow ===")
    adata, tree = merge_nodes(tree, adata)

    # Step 3: derive the final type_4 labels
    base_type = adata.obs["type_3"].iloc[0]
    base_abbr = adata.obs["type_3_abbr"].iloc[0]

    # Order merged clusters by cell count (largest first)
    cluster_counts = adata.obs["cluster_5_merge"].value_counts().sort_values(
        ascending=False
    )
    type3_abbr_mapping = {}
    type3_mapping = {}
    for i, (cluster_id, _) in enumerate(cluster_counts.items(), start=1):
        type3_abbr = "%s-%d" % (base_abbr, i)
        type3 = "%s-%d" % (base_type, i)
        type3_abbr_mapping[cluster_id] = type3_abbr
        type3_mapping[cluster_id] = type3

    adata.obs["type_4_abbr"] = adata.obs["cluster_5_merge"].map(type3_abbr_mapping)
    adata.obs["type_4"] = adata.obs["cluster_5_merge"].map(type3_mapping)

    # Rename the depth-1 nodes with their type_4_abbr names
    root_id = tree.root
    for node_id in list(tree.nodes):
        if tree.depth(node_id) == 1:
            new_name = type3_abbr_mapping.get(node_id, node_id)
            if new_name != node_id:
                node_data = tree.get_node(node_id).data
                tree.create_node(
                    tag=new_name, identifier=new_name, parent=root_id, data=node_data
                )
                for child in tree.children(node_id):
                    tree.move_node(child.identifier, new_name)
                tree.remove_node(node_id)

    return adata, tree


def sanitize_type_name(type_name_i):
    """Convert a cell type name into a filesystem-safe name."""
    type_i = re.sub(r'[\\/:*?"<>|+ ]', "_", type_name_i)
    if type_i and type_i[0] in "0123456789!@#$%^&*()-=+":
        type_i = "type_" + type_i
    return type_i


# ----------------------------------------------------------------------------
# Main: merge L5 clusters for every cell type
# ----------------------------------------------------------------------------
type_list = [
    "5-HT", "ACh", "Ast", "CALB2+ GABA", "COPC", "CP", "EC", "Epc",
    "Fib", "GB", "Granule", "HIP Glu", "HT Glu", "IPC",
    "L2/3-CUX2 Glu", "L2/3-SATB2 Glu", "L4-RORB Glu", "L5-BCL11B Glu",
    "L5/6-THEMIS Glu", "L5/6-TLE4 Glu", "LHX6+LAMP5+ GABA", "MB GABA",
    "MSN", "Mac", "Mic", "NB", "OPC", "Oli", "PC", "PVALB+ GABA",
    "Per", "RELN+ GABA", "RG", "SST+ GABA",  # "Remove" and "Th GABA" excluded
    "Th Glu", "VIP+ GABA", "VSMC",
]

l5_dir = os.path.join(proj_dir, "output/Annotation_hvg5000/L5")

for type_name_i in type_list:
    print("Processing type: %s" % type_name_i)

    type_i = sanitize_type_name(type_name_i)

    save_dir = Path(os.path.join(l5_dir, type_i))
    os.makedirs(save_dir, exist_ok=True)
    output_file = save_dir / ("hs_anno_5_%s.h5ad" % type_i)

    ad_hs_i = sc.read_h5ad(save_dir / ("hs_anno_5_%s_cluster.h5ad" % type_i))
    ad_hs_i.obs["cluster_5_merge"] = ad_hs_i.obs["cluster_5"].copy()

    # Merge each type_3 subpopulation separately
    adata_list = []
    tree_list = []
    for type_3_abbr_i in ad_hs_i.obs["type_3_abbr"].unique():
        sub_adata = ad_hs_i[
            ad_hs_i.obs["type_3_abbr"] == type_3_abbr_i
        ].copy()
        sub_adata, tree = apply_merging(sub_adata)
        adata_list.append(sub_adata)
        tree_list.append(tree)

    adata_merge = anndata.concat(adata_list, axis=0)

    adata_merge.obs[["type_4", "type_4_abbr"]].to_csv(
        path_or_buf=save_dir / ("%s_type_4.csv" % type_i), index=True
    )

    adata_merge.write_h5ad(filename=output_file)

    with open(save_dir / "%s_tree.pkl" % type_i, "wb") as f:
        pickle.dump(tree_list, f)

# ----------------------------------------------------------------------------
# Combine the per-type type_4 labels into a single metadata table
# ----------------------------------------------------------------------------
combined_meta = None
type_list = [
    "5-HT", "Ast", "CALB2+ GABA", "COPC", "CP", "EC", "Epc",
    "Fib", "GB", "Granule", "HIP Glu", "HT Glu", "IPC",
    "L2/3-CUX2 Glu", "L2/3-SATB2 Glu", "L4-RORB Glu", "L5-BCL11B Glu",
    "L5/6-THEMIS Glu", "L5/6-TLE4 Glu", "LHX6+LAMP5+ GABA", "MB GABA",
    "MSN", "Mac", "Mic", "NB", "OPC", "Oli", "PC", "PVALB+ GABA",
    "Per", "RELN+ GABA", "RG", "SST+ GABA",  # "Remove", "Th GABA", "ACh" excluded
    "Th Glu", "VIP+ GABA", "VSMC",
]

for type_name_i in type_list:
    print("Processing type: %s" % type_name_i)

    type_i = sanitize_type_name(type_name_i)

    save_dir = Path(os.path.join(l5_dir, type_i))
    file_path = save_dir / ("%s_type_4.csv" % type_i)

    if os.path.exists(file_path):
        print("Loading metadata for %s..." % type_i)
        meta_i = pd.read_csv(file_path, index_col=0)

        if combined_meta is None:
            combined_meta = meta_i
        else:
            combined_meta = pd.concat([combined_meta, meta_i], axis=0)
            type_4_abbr_counts = len(meta_i["type_4_abbr"].unique())
            print(
                "%s cell population: %d types, %d cells"
                % (type_name_i, type_4_abbr_counts, meta_i.shape[0])
            )
    else:
        print("Warning: file does not exist - %s" % file_path)

combined_meta.to_csv(
    os.path.join(l5_dir, "type_4_metadata.csv"), index=True
)
