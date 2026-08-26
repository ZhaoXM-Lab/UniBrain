#!/usr/bin/env python3
"""
=============================================================================
File      : 10_L2_neuron_step2.py
Directory : 05_annotation
Purpose   : Level 2 annotation of the Neuron subset, saving the annotated object and L3 subsets.
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

# Load dataset
ad_hs = sc.read_h5ad(
    os.path.join(
        proj_dir,
        "output/Annotation_hvg5000/L2/hs_anno_2_Neuron_hvg5000_cluster.h5ad",
    )
)
print(ad_hs)

# Cluster compactness across resolutions (guides the resolution choice)
cp_index = []
resolution = np.linspace(0.1, 1.5, 15)
for res_i in resolution:
    key_name = "leiden_fine_" + ("%.1f" % res_i)
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

# Final resolution chosen manually (checked with R clustree, clustree_2.R)
ad_hs.obs["cluster_2"] = ad_hs.obs["leiden_fine_0.9"]
ad_hs.obsm["X_umap_all"] = ad_hs.obsm["X_umap"]
sc.tl.paga(ad_hs, groups="cluster_2")
sc.tl.umap(ad_hs, init_pos="paga", maxiter=1000)

# Marker genes used to interpret each cluster (canonical literature markers)
marker_genes_dict = {
    # Glia
    "Astrocyte": ["AQP4", "GFAP", "SLC1A2", "FGFR3"],
    "Pre-astrocyte": ["GFAP", "S100B", "SOX9", "NFIA"],
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

# Differential expression between clusters (Wilcoxon, Bonferroni corrected)
sc.tl.rank_genes_groups(
    ad_hs,
    groupby="cluster_2",
    method="wilcoxon",
    pts=True,
    corr_method="bonferroni",
)

result = ad_hs.uns["rank_genes_groups"]
groups = result["names"].dtype.names
dfs_list = []
for cluster in groups:
    df = pd.DataFrame({
        "gene": result["names"][cluster],
        "log2fc": result["logfoldchanges"][cluster],
        "pvals": result["pvals"][cluster],
        "qvals": result["pvals_adj"][cluster],
        "expr_frac": result["pts"][cluster],
    })
    df["cluster"] = cluster
    dfs_list.append(df)

merged_df = pd.concat(dfs_list, axis=0, ignore_index=True)
merged_df.sort_values(
    by=["cluster", "log2fc"], ascending=[True, False], inplace=True
)
merged_df.to_csv(
    os.path.join(proj_dir, "output/Annotation_hvg5000/L2/clusters_2_deg_neuron.csv"),
    index=False,
)

# Map L2 clusters to cell type labels
cluster_anno = {
    "GABAergic_neuron": [
        "5", "7", "9",
        "11", "13", "16", "17", "18",
        "23", "25", "29",
        "37",
        "40", "43", "46",
    ],
    "Glutamatergic_neuron": [
        "2", "3", "4", "6",
        "10", "12", "15", "19",
        "20", "21", "24", "26", "27", "28",
        "31", "32", "33", "34", "38", "39",
        "42", "44", "45", "48",
        "0", "8", "30",  # Immature neuron
        "47",  # Amygdala Glu
    ],
    "Serotonergic": ["35"],
    "Radial_glia": ["14"],
    "Remove": ["1", "22", "36", "41", "49", "50"],
}

cluster_anno_rev = {}
for type_i in cluster_anno.keys():
    for cluster_i in cluster_anno[type_i]:
        cluster_anno_rev[cluster_i] = type_i

ad_hs.obs["type_2"] = ad_hs.obs["cluster_2"].map(cluster_anno_rev)
ad_hs.obs["type_1"] = ad_hs.obs["type_2"]
ad_hs.obs["type_1"] = ad_hs.obs["type_1"].astype("category")
ad_hs.obs["type_2"] = ad_hs.obs["type_2"].astype("category")

# Cell type abbreviations
abbreviation = {
    "Glutamatergic_neuron": "Glu",
    "GABAergic_neuron": "GABA",
    "Serotonergic": "5-HT",
    "Radial_glia": "RG",
    "Remove": "Remove",
}
ad_hs.obs["type_2_abbr"] = ad_hs.obs["type_2"].map(abbreviation)
ad_hs.obs["type_2_abbr"] = ad_hs.obs["type_2_abbr"].astype("category")

# Reassign broad type groups / level-1 labels for re-classified clusters
ad_hs.obs["type_group"] = ad_hs.obs["type_group"].astype(str)
ad_hs.obs["type_group"][ad_hs.obs["type_2"] == "Radial_glia"] = "Progenitors"
ad_hs.obs["type_group"] = ad_hs.obs["type_group"].astype("category")

ad_hs.obs["type_1"] = ad_hs.obs["type_1"].astype(str)
ad_hs.obs["type_1"][ad_hs.obs["type_2"] == "Serotonergic"] = "Other neuron"
ad_hs.obs["type_1"] = ad_hs.obs["type_1"].astype("category")

# Save result
ad_hs.write_h5ad(
    filename=os.path.join(proj_dir, "output/Annotation_hvg5000/L2/hs_anno_2_Neuron.h5ad")
)
print("Save as <output/Annotation_hvg5000/L2/hs_anno_2_Neuron.h5ad>")

cell_anno_info = ad_hs.obs[["type_group", "type_1", "type_2", "type_2_abbr"]]
cell_anno_info.insert(0, "cell_name", cell_anno_info.index)
cell_anno_info.to_csv(
    os.path.join(proj_dir, "output/Annotation_hvg5000/L2/Neuron_cell_metadata.csv"),
    index=False,
)

# Write the L3 (GABA / Glu) subsets for the next annotation level
l3_dir = os.path.join(proj_dir, "output/Annotation_hvg5000/L3")
os.makedirs(l3_dir, exist_ok=True)
for type_i in ["GABA", "Glu"]:
    ad_hs_i = ad_hs[ad_hs.obs["type_2_abbr"] == type_i]
    ad_hs_i.write_h5ad(
        filename=os.path.join(
            proj_dir, "output/Annotation_hvg5000/L3/hs_anno_3_" + type_i + "_tmp.h5ad"
        )
    )
    print(
        "Save as <output/Annotation_hvg5000/L3/hs_anno_3_%s_tmp.h5ad>" % type_i
    )
    del ad_hs_i
