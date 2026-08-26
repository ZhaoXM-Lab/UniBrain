#!/usr/bin/env python3
"""
=============================================================================
File      : 15_L4_subcluster_integration.py
Directory : 05_annotation
Purpose   : Level 4 subcluster integration merging L2/L3 metadata and harmonizing each type_2 subpopulation.
=============================================================================
"""

# ============================================================================
# Configuration: set PROJ_DIR to the project root before running
# ============================================================================
import os

proj_dir = os.path.join(
    os.environ.get("PROJ_DIR", "path/to/your/project"), "data", "human"
)
import re

# Limit the number of threads used by BLAS/parallel libraries
os.environ["OPENBLAS_NUM_THREADS"] = "8"
os.environ["MKL_NUM_THREADS"] = "8"
os.environ["OMP_NUM_THREADS"] = "8"
os.environ["NUMBA_NUM_THREADS"] = "8"
os.environ["VECLIB_MAXIMUM_THREADS"] = "8"
os.environ["NUMEXPR_NUM_THREADS"] = "8"

import anndata
import numpy as np
import pandas as pd
import scanpy as sc


def sanitize_type_name(type_i):
    """Convert a cell type name into a filesystem-safe name."""
    cleaned = re.sub(r'[\\/:*?"<>|+ ]', "_", type_i)
    if cleaned and cleaned[0] in "0123456789!@#$%^&*()-=+":
        cleaned = "type_" + cleaned
    return cleaned


if __name__ == "__main__":
    print("Scanpy version: %s" % sc.__version__)
    print("Anndata version: %s\n" % anndata.__version__)

    # Create the L4 output directory
    os.makedirs(os.path.join(proj_dir, "output/Annotation_hvg5000/L4"), exist_ok=True)

    # Load the L1 object
    ad_hs = sc.read_h5ad(
        os.path.join(proj_dir, "output/Annotation_hvg5000/L1/hs_anno_1.h5ad")
    )

    # Load and merge L2 metadata (one file per L1 type group)
    file_path = [
        "output/Annotation_hvg5000/L2/Neuron_cell_metadata.csv",
        "output/Annotation_hvg5000/L2/glia_cell_metadata.csv",
        "output/Annotation_hvg5000/L2/Others_cell_metadata.csv",
        "output/Annotation_hvg5000/L2/progenitors_cell_metadata.csv",
    ]
    meta_list = []
    for file in file_path:
        meta_list.append(pd.read_csv(os.path.join(proj_dir, file)))
    meta_df = pd.concat(meta_list, axis=0, ignore_index=False)
    meta_df = meta_df.set_index("cell_name")

    # Load and merge L3 metadata (neurons only)
    neuron_file = [
        "output/Annotation_hvg5000/L3/GABA_cell_metadata.csv",
        "output/Annotation_hvg5000/L3/Glu_cell_metadata.csv",
    ]
    neuron_list = []
    for file in neuron_file:
        neuron_list.append(pd.read_csv(os.path.join(proj_dir, file)))
    neuron_df = pd.concat(neuron_list, axis=0, ignore_index=False)
    neuron_df = neuron_df.set_index("cell_name")

    for col in neuron_df.columns:
        if col in meta_df.columns:
            # Only update overlapping rows (shared index)
            mask = meta_df.index.isin(neuron_df.index)
            meta_df.loc[mask, col] = neuron_df.loc[meta_df.index[mask], col]
        else:
            # Add new columns (pandas aligns on the index, NaN for
            # non-neuron cells)
            meta_df[col] = neuron_df[col]

    # Harmonize naming conventions
    meta_df["type_group"][meta_df["type_group"] == "Progenitor"] = "Progenitors"
    meta_df["type_2_abbr"][meta_df["type_2_abbr"] == "TH Glu"] = "Th Glu"

    # Replace the L1 annotations with the merged L2/L3 annotations
    ad_hs.obs = ad_hs.obs.drop(
        columns=["type_group", "type_1", "type_2", "type_2_abbr"]
    )
    ad_hs.obs = ad_hs.obs.drop(
        columns=[
            "leiden_coarse_0.1", "leiden_coarse_0.2", "leiden_coarse_0.3",
            "leiden_coarse_0.4", "leiden_coarse_0.5", "leiden_coarse_0.6",
            "leiden_coarse_0.7", "leiden_coarse_0.8", "leiden_coarse_0.9",
            "leiden_coarse_1.0",
        ]
    )
    ad_hs.obs = ad_hs.obs.join(meta_df, how="left")
    ad_hs.obs["type_group"] = ad_hs.obs["type_group"].astype("category")
    ad_hs.obs["type_1"] = ad_hs.obs["type_1"].astype("category")
    ad_hs.obs["type_2"] = ad_hs.obs["type_2"].astype("category")
    ad_hs.obs["type_2_abbr"] = ad_hs.obs["type_2_abbr"].astype("category")

    ad_hs.write_h5ad(
        filename=os.path.join(
            proj_dir, "output/Annotation_hvg5000/L4/hs_anno_3_all_tmp.h5ad"
        )
    )
    ad_hs.obs.to_csv(
        os.path.join(proj_dir, "output/Annotation_hvg5000/L4/hs_anno_3_all_metadata.csv")
    )
    print(ad_hs)

    # Per type_2 population: recompute HVG/PCA/Harmony for downstream clustering
    type_list = np.unique(ad_hs.obs["type_2_abbr"])
    for type_i in type_list:
        cleaned = sanitize_type_name(type_i)
        print("Processing type: %s" % type_i)

        ad_hs_i = ad_hs[ad_hs.obs["type_2_abbr"] == type_i]
        ad_hs_i = ad_hs_i.raw.to_adata()
        ad_hs_i.raw = ad_hs_i.copy()
        ad_hs_i.write_h5ad(
            filename=os.path.join(
                proj_dir,
                "output/Annotation_hvg5000/L4/hs_anno_3_" + cleaned + "_tmp.h5ad",
            )
        )

        # HVG selection, regression, scaling and PCA
        k = 2000
        sc.pp.highly_variable_genes(ad_hs_i, n_top_genes=k, flavor="seurat")
        print("Finish HVG selection for %s with %d genes\n" % (type_i, k))

        highly_variable = ad_hs_i.var.highly_variable
        ad_hs_i = ad_hs_i[:, highly_variable]
        sc.pp.regress_out(
            ad_hs_i, keys=["total_counts", "S_score", "G2M_score"], n_jobs=8
        )
        sc.pp.scale(ad_hs_i)
        sc.tl.pca(ad_hs_i, svd_solver="arpack", n_comps=50)

        # Harmony integration to correct for sample batch effects
        sc.external.pp.harmony_integrate(
            ad_hs_i, key="sample_new", basis="X_pca", max_iter_harmony=30
        )

        # Build k-NN graph on the harmony-corrected PCA embedding
        ad_hs_i.obsm["X_pca_copy"] = ad_hs_i.obsm["X_pca"]
        ad_hs_i.obsm["X_pca"] = ad_hs_i.obsm["X_pca_harmony"]
        sc.pp.neighbors(ad_hs_i, n_neighbors=30, n_pcs=50)
        print("\nFinish KNN\n")

        ad_hs_i.write_h5ad(
            filename=os.path.join(
                proj_dir,
                "output/Annotation_hvg5000/L4/hs_anno_3_" + cleaned + "_inter.h5ad",
            )
        )
        print(
            "Save as <output/Annotation_hvg5000/L4/hs_anno_3_%s_inter.h5ad>"
            % cleaned
        )
        del ad_hs_i
