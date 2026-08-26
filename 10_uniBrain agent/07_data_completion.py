#!/usr/bin/env python3
"""
=============================================================================
File      : 07_data_completion.py
Directory : 10_uniBrain agent
Purpose   : Data completion (atlas coverage) validation against the reference human brain atlas.
=============================================================================
"""

import os

import numpy as np
import pandas as pd
import scanpy as sc
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import pdist

## Configuration ##
# Set the base directory for the project below.
BASE_DIR = os.environ.get("PROJ_DIR", "path/to/your/project")  # <-- MODIFY THIS

# Reference annotated atlas used as ground truth
ATLAS_PATH = os.path.join(
    BASE_DIR, "data", "human", "output",
    "Annotation_hvg5000", "final", "hs_sample380k_norm.h5ad",
)

# Directory for exported outputs
OUTPUT_DIR = os.path.join(BASE_DIR, "output", "validation_data")

STAGE_ORDER = [
    "01-Early first trimester",
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


def load_metadata(atlas_path):
    """
    Load (stage, region_L2, type_2) cell metadata from the reference atlas.

    Parameters
    ----------
    atlas_path : str
        Path to the reference h5ad atlas.

    Returns
    -------
    pandas.DataFrame
        Metadata rows with no missing values.
    """
    adata = sc.read_h5ad(atlas_path, backed="r")
    meta = adata.obs[["stage", "region_L2", "type_2"]].copy()
    return meta.dropna()


def build_view_a(meta, stage_order, regions, celltypes):
    """
    Build View A: (stage, region) × cell type binary presence matrix.

    Within each stage, regions are ordered by hierarchical clustering of
    their cell type profiles (hamming distance, average linkage).

    Parameters
    ----------
    meta : pandas.DataFrame
        Columns: stage, region_L2, type_2.
    stage_order : list of str
    regions : list of str
    celltypes : list of str

    Returns
    -------
    matrix : numpy.ndarray, shape (n_stages * n_regions, n_celltypes)
    row_labels : list of (stage, region) tuples
    """
    rows = []
    row_labels = []

    for stage in stage_order:
        stage_rows = []
        stage_regions = []

        for region in regions:
            sub = meta[
                (meta["stage"] == stage) & (meta["region_L2"] == region)
            ]
            observed_types = set(sub["type_2"])
            row = [1 if ct in observed_types else 0 for ct in celltypes]
            stage_rows.append(row)
            stage_regions.append(region)

        stage_matrix = np.array(stage_rows)

        if len(stage_matrix) > 1:
            dist = pdist(stage_matrix, metric="hamming")
            Z = linkage(dist, method="average")
            order = leaves_list(Z)
        else:
            order = np.array([0])

        rows.extend(stage_matrix[order])
        row_labels.extend([(stage, stage_regions[i]) for i in order])

    return np.array(rows), row_labels


def build_view_b(meta, stage_order, regions, celltypes):
    """
    Build View B: (stage, cell type) × region binary presence matrix.

    Regions are sorted by total observed (stage, cell type) combinations,
    descending.

    Parameters
    ----------
    meta : pandas.DataFrame
        Columns: stage, region_L2, type_2.
    stage_order : list of str
    regions : list of str
    celltypes : list of str

    Returns
    -------
    matrix : numpy.ndarray, shape (n_stages * n_celltypes, n_regions)
    row_labels : list of (stage, celltype) tuples
    regions_clustered : list of region labels in descending coverage order
    """
    rows = []
    row_labels = []

    for stage in stage_order:
        for ct in celltypes:
            row = []
            for region in regions:
                sub = meta[
                    (meta["stage"] == stage)
                    & (meta["region_L2"] == region)
                    & (meta["type_2"] == ct)
                ]
                row.append(1 if len(sub) > 0 else 0)
            rows.append(row)
            row_labels.append((stage, ct))

    matrix = np.array(rows)

    # Sort regions by observed counts, descending
    region_counts = matrix.sum(axis=0)
    region_order = np.argsort(-region_counts)
    matrix = matrix[:, region_order]
    regions_clustered = [regions[i] for i in region_order]

    return matrix, row_labels, regions_clustered


def main():
    # 1. Load metadata
    meta = load_metadata(ATLAS_PATH)

    # 2. Category ordering
    regions = sorted(meta["region_L2"].unique())
    celltypes = sorted(meta["type_2"].unique())

    # 3. Build View A: (stage, region) × cell type
    matrix_a, row_labels_a = build_view_a(meta, STAGE_ORDER, regions, celltypes)
    print("View A — (stage, region) × cell type matrix shape:", matrix_a.shape)

    # 4. Build View B: (stage, cell type) × region
    matrix_b, row_labels_b, regions_clustered = build_view_b(
        meta, STAGE_ORDER, regions, celltypes
    )
    print("View B — (stage, cell type) × region matrix shape:", matrix_b.shape)
    print("Regions by coverage (top 5):", regions_clustered[:5])

    # 5. Coverage statistics (identical for both views)
    n_observed = int(matrix_a.sum())
    n_total = matrix_a.size
    n_missing = n_total - n_observed

    print(f"\nCoverage statistics (same for both views):")
    print(f"  Observed combinations  : {n_observed}")
    print(f"  Model-completed space  : {n_missing}")
    print(f"  Total combinations     : {n_total}")
    print(f"  Coverage ratio         : {n_observed / n_total:.4f}")

    # 6. Export both matrices
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # View A
    df_a = pd.DataFrame(
        matrix_a,
        index=pd.MultiIndex.from_tuples(
            row_labels_a, names=["stage", "region_L2"]
        ),
        columns=celltypes,
    )
    out_a = os.path.join(OUTPUT_DIR, "atlas_coverage_matrix_stage_region.csv")
    df_a.to_csv(out_a)
    print(f"\nView A saved to: {out_a}")

    # View B
    df_b = pd.DataFrame(
        matrix_b,
        index=pd.MultiIndex.from_tuples(
            row_labels_b, names=["stage", "type_2"]
        ),
        columns=regions_clustered,
    )
    out_b = os.path.join(OUTPUT_DIR, "atlas_coverage_matrix_stage_celltype.csv")
    df_b.to_csv(out_b)
    print(f"View B saved to: {out_b}")


if __name__ == "__main__":
    main()
