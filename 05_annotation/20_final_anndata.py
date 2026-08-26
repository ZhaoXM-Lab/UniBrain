#!/usr/bin/env python3
"""
=============================================================================
File      : 20_final_anndata.py
Directory : 05_annotation
Purpose   : Assemble the final annotated human brain atlas AnnData objects and metadata.
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
import scipy.io
import scipy.sparse


def sanitize_type_name(type_name_i):
    """Convert a cell type name into a filesystem-safe name."""
    type_i = re.sub(r'[\\/:*?"<>|+ ]', "_", type_name_i)
    if type_i and type_i[0] in "0123456789!@#$%^&*()-=+":
        type_i = "type_" + type_i
    return type_i


def main():
    print("Scanpy version: %s" % sc.__version__)
    print("Anndata version: %s\n" % anndata.__version__)

    final_dir = os.path.join(proj_dir, "output/Annotation_hvg5000/final")
    l4_dir = os.path.join(proj_dir, "output/Annotation_hvg5000/L4")
    l5_dir = os.path.join(proj_dir, "output/Annotation_hvg5000/L5")
    os.makedirs(final_dir, exist_ok=True)

    # ------------------------------------------------------------------------
    # 1. Attach the L4 annotations to the full object
    # ------------------------------------------------------------------------
    ad_hs = sc.read_h5ad(
        os.path.join(l4_dir, "hs_anno_3_all_tmp.h5ad")
    )
    ad_hs.obs.to_csv(
        os.path.join(l4_dir, "hs_anno_3_all_metadata.csv")
    )

    L4_meta = pd.read_csv(
        os.path.join(l4_dir, "type_3_metadata.csv"), index_col=0
    )
    ad_hs.obs = ad_hs.obs.join(L4_meta, how="left")
    ad_hs.obs["type_3"] = np.where(
        ad_hs.obs["type_3"].isna(),
        ad_hs.obs["type_2"],
        ad_hs.obs["type_3"],
    )

    ad_hs.write_h5ad(filename=os.path.join(final_dir, "hs_anno_all.h5ad"))

    del ad_hs.raw
    ad_hs.write_h5ad(filename=os.path.join(final_dir, "hs_anno_all_light.h5ad"))

    # Version without the "Remove" cells
    ad_hs_clear = ad_hs[ad_hs.obs["type_2"] != "Remove"].copy()
    ad_hs_clear.write_h5ad(filename=os.path.join(final_dir, "hs_anno_all_clear.h5ad"))

    # ------------------------------------------------------------------------
    # 2. Per-cell metadata with hierarchical region mapping
    # ------------------------------------------------------------------------
    cell_type = pd.read_csv(
        os.path.join(proj_dir, "output/Visualization/cell_metadata_all.csv"),
        index_col=0, header=0,
    )
    cell_info = pd.read_csv(
        os.path.join(l4_dir, "hs_anno_3_all_metadata.csv"), index_col=0
    )
    brain_mapping = pd.read_csv(
        os.path.join(proj_dir, "qc/brain_region_mapping.csv"), header=0
    )

    meta = cell_type.join(
        cell_info[
            [
                "species", "dataset", "individual", "sample", "age",
                "stage", "sex", "region_original", "region_2",
            ]
        ],
        how="left",
    )

    meta = meta.rename(columns={"region_2": "region_temp"})
    mapping_info_unique = brain_mapping.drop_duplicates(
        subset="Raw name", keep="first"
    )
    mapping_dict = mapping_info_unique.set_index("Raw name").to_dict("index")

    def map_regions(row):
        """Map the raw region name to the hierarchical region labels."""
        raw_name = row["region_temp"]
        if raw_name in mapping_dict:
            mapped = mapping_dict[raw_name]
            return pd.Series({
                "region_new": mapped["New name"],
                "region_L3": mapped["L3"],
                "region_L2": mapped["L2"],
                "region_L1": mapped["L1"],
            })
        return pd.Series({
            "region_new": None,
            "region_L3": None,
            "region_L2": None,
            "region_L1": None,
        })

    mapped_region = meta.apply(map_regions, axis=1)
    meta = pd.concat([meta, mapped_region], axis=1)

    meta.to_csv(os.path.join(final_dir, "hs_all_metadata.csv"))

    # ------------------------------------------------------------------------
    # 3. Developmental stage mapping from age
    # ------------------------------------------------------------------------
    meta = pd.read_csv(
        os.path.join(final_dir, "hs_all_metadata.csv"), index_col=0
    )
    meta.loc[meta["region_L2"] == "Frontal Lobe", "region_L2"] = "Frontal lobe"

    stage_info = {
        "PCW4": "01-Early first trimester",
        "PCW5": "02-Late first trimester",
        "PCW5.85": "02-Late first trimester",
        "PCW6": "02-Late first trimester",
        "PCW7": "02-Late first trimester",
        "PCW8": "02-Late first trimester",
        "PCW9": "02-Late first trimester",
        "PCW10": "03-Early second trimester",
        "PCW11": "03-Early second trimester",
        "PCW11.5": "03-Early second trimester",
        "PCW12": "03-Early second trimester",
        "PCW13": "03-Early second trimester",
        "PCW13.3": "03-Early second trimester",
        "PCW14": "03-Early second trimester",
        "PCW14.2": "03-Early second trimester",
        "PCW14.5": "03-Early second trimester",
        "PCW14.7": "03-Early second trimester",
        "PCW15": "04-Mid second trimester",
        "PCW16": "04-Mid second trimester",
        "PCW16.3": "04-Mid second trimester",
        "PCW16.5": "04-Mid second trimester",
        "PCW17": "04-Mid second trimester",
        "PCW17.3": "04-Mid second trimester",
        "PCW17.5": "04-Mid second trimester",
        "PCW17.7": "04-Mid second trimester",
        "PCW18": "04-Mid second trimester",
        "PCW18.5": "04-Mid second trimester",
        "PCW19": "04-Mid second trimester",
        "PCW20": "05-Late second trimester",
        "PCW21": "05-Late second trimester",
        "PCW21.5": "05-Late second trimester",
        "PCW22": "05-Late second trimester",
        "PCW23": "05-Late second trimester",
        "PCW24": "05-Late second trimester",
        "PCW25": "06-Early third trimester",
        "PCW26": "06-Early third trimester",
        "PCW27": "06-Early third trimester",
        "PCW28": "06-Early third trimester",
        "PCW32": "07-Late third trimester",
        "PCW33": "07-Late third trimester",
        "PCW34": "07-Late third trimester",
        "PCW37": "07-Late third trimester",
        "PCW38": "07-Late third trimester",
        "D118": "08-Neonatal and early infancy",
        "D179": "08-Neonatal and early infancy",
        "D2": "08-Neonatal and early infancy",
        "D34": "08-Neonatal and early infancy",
        "D86": "08-Neonatal and early infancy",
        "M5": "08-Neonatal and early infancy",
        "Y0": "08-Neonatal and early infancy",
        "Y0.1": "08-Neonatal and early infancy",
        "Y0.2": "08-Neonatal and early infancy",
        "D301": "09-Late infancy",
        "Y0.6": "09-Late infancy",
        "M9": "09-Late infancy",
        "Y1": "10-Early childhood",
        "Y1.2": "10-Early childhood",
        "Y1.4": "10-Early childhood",
        "Y1.5": "10-Early childhood",
        "Y1.7": "10-Early childhood",
        "Y1.8": "10-Early childhood",
        "Y2": "10-Early childhood",
        "Y2.1": "10-Early childhood",
        "Y2.5": "10-Early childhood",
        "Y2.6": "10-Early childhood",
        "Y2.7": "10-Early childhood",
        "Y2.8": "10-Early childhood",
        "Y3": "10-Early childhood",
        "Y3.2": "10-Early childhood",
        "Y4": "10-Early childhood",
        "Y4.3": "10-Early childhood",
        "Y5.7": "10-Early childhood",
        "Y6": "11-Middle and late childhood",
        "Y6.7": "11-Middle and late childhood",
        "Y8": "11-Middle and late childhood",
        "Y10": "11-Middle and late childhood",
        "Y12": "12-Adolescence",
        "Y13": "12-Adolescence",
        "Y14": "12-Adolescence",
        "Y14.7": "12-Adolescence",
        "Y15": "12-Adolescence",
        "Y15.3": "12-Adolescence",
        "Y16": "12-Adolescence",
        "Y17": "12-Adolescence",
        "Y18": "12-Adolescence",
        "Y18.5": "12-Adolescence",
        "Y19": "12-Adolescence",
        "Y20": "13-Young adulthood",
        "Y21": "13-Young adulthood",
        "Y22": "13-Young adulthood",
        "Y25": "13-Young adulthood",
        "Y27": "13-Young adulthood",
        "Y28": "13-Young adulthood",
        "Y29": "13-Young adulthood",
        "Y32": "13-Young adulthood",
        "Y35": "13-Young adulthood",
        "Y36": "13-Young adulthood",
        "Y39": "13-Young adulthood",
        "Y40": "14-Middle adulthood",
        "Y42": "14-Middle adulthood",
        "Y43": "14-Middle adulthood",
        "Y44": "14-Middle adulthood",
        "Y45": "14-Middle adulthood",
        "Y47": "14-Middle adulthood",
        "Y48": "14-Middle adulthood",
        "Y49": "14-Middle adulthood",
        "Y50": "14-Middle adulthood",
        "Y50.2": "14-Middle adulthood",
        "Y51": "14-Middle adulthood",
        "Y52": "14-Middle adulthood",
        "Y53": "14-Middle adulthood",
        "Y54": "14-Middle adulthood",
        "Y55": "14-Middle adulthood",
        "Y56": "14-Middle adulthood",
        "Y59": "14-Middle adulthood",
        "Y60": "15-Late adulthood",
        "Y62": "15-Late adulthood",
        "Y63": "15-Late adulthood",
        "Y64": "15-Late adulthood",
        "Y65": "15-Late adulthood",
        "Y66": "15-Late adulthood",
        "Y69": "15-Late adulthood",
        "Y70": "15-Late adulthood",
        "Y71": "15-Late adulthood",
        "Y72": "15-Late adulthood",
        "Y73": "15-Late adulthood",
        "Y74": "15-Late adulthood",
        "Y75": "15-Late adulthood",
        "Y77": "15-Late adulthood",
        "Y78": "15-Late adulthood",
        "Y79": "15-Late adulthood",
        "Y80": "15-Late adulthood",
        "Y81": "15-Late adulthood",
        "Y82": "15-Late adulthood",
        "Y83": "15-Late adulthood",
        "Y84": "15-Late adulthood",
        "Y85": "15-Late adulthood",
        "Y86": "15-Late adulthood",
        "Y87": "15-Late adulthood",
        "Y88": "15-Late adulthood",
        "Y89": "15-Late adulthood",
        "Y90": "15-Late adulthood",
        "Y92": "15-Late adulthood",
        "Y93": "15-Late adulthood",
        "Y95": "15-Late adulthood",
    }

    meta["stage"] = meta["age"].map(stage_info)
    meta.to_csv(os.path.join(final_dir, "hs_all_metadata_update.csv"))

    # ------------------------------------------------------------------------
    # 4. Export per-type matrices and embeddings for downstream R analysis
    # ------------------------------------------------------------------------
    cell_type = [
        "5-HT", "ACh", "Ast", "CALB2+ GABA", "COPC", "CP", "EC", "Epc",
        "Fib", "GB", "Granule", "HIP Glu", "HT Glu", "IPC",
        "L2/3-CUX2 Glu", "L2/3-SATB2 Glu", "L4-RORB Glu", "L5-BCL11B Glu",
        "L5/6-THEMIS Glu", "L5/6-TLE4 Glu", "LHX6+LAMP5+ GABA", "MB GABA",
        "MSN", "Mac", "Mic", "NB", "OPC", "Oli", "PC", "PVALB+ GABA",
        "Per", "RELN+ GABA", "RG", "SST+ GABA",  # "Remove" and "Th GABA" excluded
        "Th Glu", "VIP+ GABA", "VSMC",
    ]

    for type_name_i in cell_type:
        print("Processing type: %s" % type_name_i)

        type_i = sanitize_type_name(type_name_i)

        ad_hs = sc.read_h5ad(
            os.path.join(l4_dir, "hs_anno_4_%s.h5ad" % type_i)
        )
        dir_path = os.path.join(l4_dir, "R", type_i)
        os.makedirs(dir_path, exist_ok=True)
        scipy.io.mmwrite(os.path.join(dir_path, "matrix.mtx"), ad_hs.X.T)
        ad_hs.obs.to_csv(os.path.join(dir_path, "cells.csv"))
        ad_hs.var.to_csv(os.path.join(dir_path, "hvg.csv"))
        raw = ad_hs.raw.to_adata()
        scipy.io.mmwrite(os.path.join(dir_path, "raw_matrix.mtx"), raw.X.T)
        raw.var.to_csv(os.path.join(dir_path, "raw_genes.csv"))
        pd.DataFrame(
            ad_hs.obsm["X_pca"], index=ad_hs.obs_names
        ).to_csv(os.path.join(dir_path, "pca.csv"))
        pd.DataFrame(
            ad_hs.obsm["X_pca_harmony"], index=ad_hs.obs_names
        ).to_csv(os.path.join(dir_path, "pca_harmony.csv"))
        pd.DataFrame(
            ad_hs.obsm["X_umap"], index=ad_hs.obs_names
        ).to_csv(os.path.join(dir_path, "umap.csv"))

        del raw
        del ad_hs

    # Export raw count matrices from the QC-passed object
    ad_hs_orig = sc.read_h5ad(
        os.path.join(proj_dir, "output/MergeData/hs_qc.h5ad")
    )

    # The QC object carries no annotations, so build the per-type cell
    # index from the L2/L3 annotation metadata
    anno_meta_files = [
        "output/Annotation_hvg5000/L2/Neuron_cell_metadata.csv",
        "output/Annotation_hvg5000/L2/glia_cell_metadata.csv",
        "output/Annotation_hvg5000/L2/Others_cell_metadata.csv",
        "output/Annotation_hvg5000/L2/progenitors_cell_metadata.csv",
        "output/Annotation_hvg5000/L3/GABA_cell_metadata.csv",
        "output/Annotation_hvg5000/L3/Glu_cell_metadata.csv",
    ]
    anno_meta = pd.concat(
        [pd.read_csv(os.path.join(proj_dir, f)) for f in anno_meta_files],
        axis=0,
    )
    anno_meta = anno_meta.set_index("cell_name")
    # Harmonize naming conventions (same as 15_L4_subcluster_integration)
    anno_meta.loc[anno_meta["type_2_abbr"] == "TH Glu", "type_2_abbr"] = "Th Glu"
    cells_by_type = {
        abbr: anno_meta.index[anno_meta["type_2_abbr"] == abbr].tolist()
        for abbr in anno_meta["type_2_abbr"].unique()
    }

    for type_name_i in cell_type:
        print("Processing type: %s" % type_name_i)

        type_i = sanitize_type_name(type_name_i)

        cells_i = cells_by_type.get(type_name_i, [])
        if not cells_i:
            print("Warning: no cells found for type %s" % type_name_i)
            continue
        ad_hs = ad_hs_orig[cells_i].copy()
        dir_path = os.path.join(l4_dir, "R", type_i)
        os.makedirs(dir_path, exist_ok=True)
        scipy.io.mmwrite(
            os.path.join(dir_path, "counts_matrix.mtx"), ad_hs.X.T
        )
        ad_hs.obs.to_csv(os.path.join(dir_path, "counts_cells.csv"))
        ad_hs.var.to_csv(os.path.join(dir_path, "counts_genes.csv"))

        del ad_hs

    # ------------------------------------------------------------------------
    # 5. Final metadata: attach L5 labels and build type_3 abbreviations
    # ------------------------------------------------------------------------
    meta = pd.read_csv(
        os.path.join(final_dir, "hs_all_metadata_update.csv"), index_col=0
    )
    L5_meta = pd.read_csv(
        os.path.join(l5_dir, "type_4_metadata.csv"), index_col=0
    )

    meta = meta.join(L5_meta, how="left")

    meta["type_4"] = np.where(
        meta["type_4"].isna(),
        meta["type_2"],
        meta["type_4"],
    )
    meta["type_4_abbr"] = np.where(
        meta["type_4_abbr"].isna(),
        meta["type_2_abbr"],
        meta["type_4_abbr"],
    )

    # Map type_2 values to their abbreviations
    abbr_map = (
        meta.drop_duplicates(subset=["type_2"])
        .set_index("type_2")["type_2_abbr"]
        .to_dict()
    )

    # Sort type_2 values by length (descending) so longer names match first
    type_2_vals_sorted = sorted(abbr_map.keys(), key=lambda x: len(x), reverse=True)

    def map_type_3_to_abbr(type_3_val):
        """Map a type_3 value to its abbreviation, keeping the numeric suffix.

        Identifies the base type prefix of the type_3 value, replaces it with
        its abbreviation, and preserves any trailing suffix (e.g. -2).
        """
        if type_3_val == "Remove":
            return "Remove"

        for type_2_val in type_2_vals_sorted:
            if type_3_val.startswith(type_2_val + "-"):
                suffix = type_3_val[len(type_2_val):]
                abbr = abbr_map[type_2_val]
                return abbr + suffix

        print(
            "Warning: no matching abbreviation for '%s', returning the "
            "original value" % type_3_val
        )
        return type_3_val

    meta["type_3_abbr"] = meta["type_3"].apply(map_type_3_to_abbr)

    # Manual corrections
    meta.loc[meta["type_3_abbr"] == "Thalamic_GABA", "type_3_abbr"] = "Th_GABA"
    meta.loc[meta["type_2"] == "Cholinergic", "type_3"] = "Cholinergic"
    meta.loc[meta["type_2"] == "Cholinergic", "type_3_abbr"] = "ACh"
    meta.loc[meta["type_2"] == "Cholinergic", "type_4"] = "Cholinergic"
    meta.loc[meta["type_2"] == "Cholinergic", "type_4_abbr"] = "ACh"

    meta.to_csv(os.path.join(final_dir, "hs_all_metadata_final.csv"))

    # ------------------------------------------------------------------------
    # 6. Final normalized AnnData objects
    # ------------------------------------------------------------------------
    ad_hs = sc.read_h5ad(os.path.join(proj_dir, "output/MergeData/hs_qc.h5ad"))
    meta = pd.read_csv(
        os.path.join(final_dir, "hs_all_metadata_final.csv"), index_col=0
    )
    meta_aligned = meta.loc[ad_hs.obs_names]
    ad_hs.obs = meta_aligned
    ad_hs.write_h5ad(filename=os.path.join(final_dir, "hs_anno_all_norm.h5ad"))

    # Downsampled version (380k cells)
    ad_hs_downsampled = pd.read_csv(
        os.path.join(final_dir, "hs_sample380k_cell_ids.txt"),
        header=None, names=["cell_id"],
    )
    ad_hs_sample = ad_hs[
        ad_hs.obs_names.isin(ad_hs_downsampled["cell_id"])
    ].copy()
    ad_hs_sample.write_h5ad(
        filename=os.path.join(final_dir, "hs_sample380k_norm.h5ad")
    )


if __name__ == "__main__":
    main()
