#!/usr/bin/env python3
"""
=============================================================================
File      : 02_L1_annotation.py
Directory : 05_annotation
Purpose   : Level 1 cell type annotation assigning broad cell type groups to coarse Leiden clusters.
=============================================================================
"""

# ============================================================================
# Configuration: set PROJ_DIR to the project root before running
# ============================================================================
import os

proj_dir = os.path.join(
    os.environ.get("PROJ_DIR", "path/to/your/project"), "data", "human"
)

import anndata
import numpy as np
import pandas as pd
import scanpy as sc

print("Scanpy version: %s" % sc.__version__)
print("Anndata version: %s\n" % anndata.__version__)

# Load dataset and attach the normalized raw data
ad_hs = sc.read_h5ad(
    os.path.join(proj_dir, "output/Annotation_hvg5000/L1/hs_anno_1_hvg5000_tmp.h5ad")
)
ad_hs_norm = sc.read_h5ad(os.path.join(proj_dir, "output/Integration/hs_norm.h5ad"))
ad_hs.raw = ad_hs_norm
ad_hs_scale = sc.read_h5ad(os.path.join(proj_dir, "output/Umap/hs_umap_hvg5000.h5ad"))
ad_hs.X = ad_hs_scale.X
print(ad_hs)

# Load the original UMAP coordinates
umap_coordinate = pd.read_csv(
    os.path.join(proj_dir, "output/Umap/umap_harmony_k_50.csv"), index_col=0
)
ad_hs.obsm["X_umap_raw"] = umap_coordinate.loc[ad_hs.obs_names].values

# Cluster compactness across resolutions (guides the resolution choice)
cp_index = []
resolution = np.linspace(0.1, 1.0, 10)
for res_i in resolution:
    key_name = "leiden_coarse_" + ("%.1f" % res_i)
    cp_list = []
    cluster_list = np.unique(ad_hs.obs[key_name].values)
    for cluster_i in cluster_list:
        emb_i = ad_hs.obsm["X_pca_harmony"][
            ad_hs.obs[key_name].values == cluster_i, :
        ]
        emb_i = emb_i - emb_i.mean(axis=0)
        distance_i = np.sqrt(np.sum(emb_i ** 2, axis=1)).mean()
        cp_list.append(distance_i)
    cp_index.append(sum(cp_list) / len(cp_list))
    print("%s compactness: %.3f\n" % (key_name, cp_index[-1]))

# Final resolution chosen manually (checked with R clustree, clustree_1.R)
ad_hs.obs["cluster_1"] = ad_hs.obs["leiden_coarse_0.9"]
sc.tl.paga(ad_hs, groups="cluster_1")
sc.tl.umap(ad_hs, init_pos="paga", maxiter=1000)

umap = pd.DataFrame(
    data=ad_hs.obsm["X_umap"],
    index=ad_hs.obs_names,
    columns=["UMAP_1", "UMAP_2"],
)
umap.to_csv(os.path.join(proj_dir, "output/Annotation_hvg5000/L1/umap_all_paga.csv"))

# Marker genes used to interpret each cluster (canonical literature markers)
marker_genes_dict = {
    # Glia
    "Astrocyte": ["AQP4", "GFAP", "SLC1A2", "FGFR3"],
    "Bergmann glia": ["ADAMTSL1", "LINC01727", "ALDH1A1"],
    "Oligodendrocyte": ["MBP", "MOG", "PLP1", "MAG"],
    "Oligodendrocyte_precursor": ["PDGFRA", "OLIG2", "ST18", "SOX6"],
    "Oligodendrocyte_precursor_committed": ["PTPRZ1", "VCAN", "PCDH15", "BCAS1"],
    "Ependymal": ["CFAP54", "DNAH9", "CFAP299"],

    # Neurons
    "Neuron": ["SLC17A7", "GAD1", "GAD2", "SYT1", "RBFOX3", "INA"],
    "GABAergic": ["GAD1", "GAD2", "SLC32A1"],
    # Classic MGE-derived subtypes (PMID: 30737435, 32350463)
    "MGE-dev": ["NKX2-1", "LHX6", "NXPH1"],
    "PVALB+ GABA": ["PVALB", "SST", "CALB1", "KCNC1"],  # Fast-spiking
    "SST+ GABA": ["SST", "NPY", "LHX6", "LAMP5"],  # Martinotti/Non-Martinotti
    "LHX6+LAMP5+ GABA": ["LHX6", "LAMP5"],
    # CGE-derived subtypes
    "CGE-dev": ["NFIX", "PROX1", "NR2F2"],
    "VIP+ GABA": ["VIP", "CCK", "CALB2", "CRH"],  # Vasoactive intestinal peptide
    "CALB2+ GABA": ["CALB2", "RELN", "SCUBE3", "HTR3A"],
    "Reelin+ GABA": ["RELN", "CALB2", "CXCL12"],
    # LGE-derived subtypes
    "LGE-dev": ["ZFHX3", "FOXP1", "EBF1"],
    "Medium_spiny": ["DRD1", "ISL1", "DRD2", "PENK"],  # Striatal medium spiny
    "OB Interneuron": ["CHD7", "ID2", "PAX6"],  # Olfactory bulb interneuron

    # Specialized subtypes
    "Purkinje": ["CALB1", "GRM1", "ITPR1", "CA8"],  # Purkinje cells
    "Thalamic GABA": ["LHX5", "TAL1", "GATA3"],
    "Midbrain-derived GABA": ["GPRIN2", "NOS1", "ANKRD34B"],  # Substantia nigra

    # Glutamatergic neurons (PMID: 32915944, 34615802)
    "Glutamatergic": ["SLC17A6", "SLC17A7", "NEUROD2"],
    "L2/3-CUX2": ["CUX2", "SATB2"],
    "L4-RORB": ["RORB", "NTNG2"],
    "L5/6-THEMIS": ["THEMIS", "FEZF2"],
    "L5/6-TLE4": ["TLE4", "BCL11B"],
    "Thalamic Glu": ["NTNG1", "PCP4", "FSTL4"],
    "Hypothalamic Glu": ["SIM1", "POMC", "NPY2R", "SLC17A6"],
    "Amygdala Glu": ["ENC1", "DSP", "TAFA2"],
    "Hippocampal_CA1-3 Glu": ["COL24A1", "COL5A2", "CCBE1"],
    "Hippocampal_CA4 Glu": ["SHISA6", "NRP1", "TRHDE"],
    "Hippocampal_dentate_gyrus Glu": ["SEMA5A", "ADRA1A", "ALDH1A2"],
    # Cerebellum (PMID: 32296178)
    "Granule": ["ZIC4", "GABRA6", "CNPY1"],

    # Other neurons (PMID: 30455457)
    "Dopaminergic": ["TH", "SLC6A3", "DRD2", "FOXA2"],  # Substantia nigra/VTA
    "Serotonergic": ["TPH2", "SLC6A4", "HTR1A", "GCH1"],  # Raphe nuclei
    "Cholinergic": ["CHAT", "SLC5A7", "ACHE"],  # Basal forebrain

    # Neuropeptidergic neurons (PMID: 28394325)
    "CRH+ Neuron": ["CRH", "NR4A2", "RGS4", "SST"],  # Stress-responsive
    "POMC+ Neuron": ["POMC", "LEPR", "MC4R", "NPY1R"],  # Energy balance
    "Cajal_Retzius": ["RELN", "CALB2", "TBR1", "NECAB1"],

    # Neural progenitors and stem cells
    "Progenitors": ["PAX6", "SOX2", "HES1", "HES5"],
    "Neuronal_IPC": ["EOMES", "TAC3", "NEUROG2", "NHLH1"],  # Intermediate progenitor
    "Radial_glia": ["SOX2", "VIM", "PAX6", "HES1"],  # Radial glia
    "Glioblast": ["TNC", "BCAN"],
    "Neuroblast": ["NHLH1", "DCX"],

    # Vascular
    "Endothelial_cell": ["FLT1", "PECAM1", "CLDN5", "ESM1"],
    "Pericyte": ["PDGFRB", "HIGD1B", "CSPG4", "DES"],
    "VLMC": ["ACTA2", "TAGLN", "MYH11", "CNN1"],
    "Fibroblast": ["COL1A1", "THY1", "FAP", "DCN"],
    "VSMC": ["ACTA2", "MYH11", "CNN1", "DES"],  # Vascular smooth muscle cell

    # Immune cells
    "Microglia": ["CX3CR1", "PTPRC", "TMEM119", "CSF1R"],
    "Macrophage": ["LYVE1", "MRC1", "CD163", "SIGLEC1"],
    "Mast_cell": ["TPSAB1", "KIT", "HDC"],
    "Infiltrat_Immune": ["CD3D", "CD3E", "CD79A", "CD79B"],

    # Special cell types
    "Choroid_plexus_epithelial": ["PRLR", "TTR", "NWD1", "CLIC6"],
    "Blood": ["HEMGN", "AHSP", "PTPRC", "CD34"],
}

# Map L1 clusters to broad cell type groups
cluster_anno = {
    "Glia": ["3", "5", "11", "22", "46"],
    "Neuron": [
        "0", "1", "2", "6", "8", "9", "10",
        "14", "15", "16", "17", "19",
        "20", "21", "23", "25", "26", "27", "28", "29",
        "31", "32", "33", "34", "35", "36", "37", "38", "39", "40", "44", "45",
    ],
    "Progenitors": ["4", "7", "12", "30", "41", "42", "43"],
    "Others": ["13", "18", "24"],
}

cluster_anno_rev = {}
for type_i in cluster_anno.keys():
    for cluster_i in cluster_anno[type_i]:
        cluster_anno_rev[cluster_i] = type_i

ad_hs.obs["type_group"] = ad_hs.obs["cluster_1"].map(cluster_anno_rev)
ad_hs.obs["type_group"] = ad_hs.obs["type_group"].astype("category")

# Save annotated object
ad_hs.write_h5ad(
    filename=os.path.join(proj_dir, "output/Annotation_hvg5000/L1/hs_anno_1.h5ad")
)
print("Save as <output/Annotation_hvg5000/L1/hs_anno_1.h5ad>")

# Split dataset by type group for the next annotation level
for type_i in cluster_anno.keys():
    ad_hs_type = ad_hs[ad_hs.obs["type_group"] == type_i, :]
    ad_hs_type.write_h5ad(
        filename=os.path.join(
            proj_dir, "output/Annotation_hvg5000/L1/hs_anno_1_%s.h5ad" % type_i
        )
    )
    print("Save as <output/Annotation_hvg5000/L1/hs_anno_1_%s.h5ad>" % type_i)
