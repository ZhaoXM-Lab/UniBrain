#!/usr/bin/env python3
"""
=============================================================================
File      : 07_L2_others_step2.py
Directory : 05_annotation
Purpose   : Level 2 annotation assigning cell type labels to the Others subset clusters.
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
    os.path.join(proj_dir, "output/Annotation_hvg5000/L2/hs_anno_2_Others_cluster.h5ad")
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
ad_hs.obs["cluster_2"] = ad_hs.obs["leiden_fine_1.2"]
ad_hs.obsm["X_umap_all"] = ad_hs.obsm["X_umap"]
sc.tl.paga(ad_hs, groups="cluster_2")
sc.tl.umap(ad_hs, init_pos="paga", maxiter=1000)

# Marker genes used to interpret each cluster (canonical literature markers)
marker_genes_dict = {
    # Glia
    "Astrocyte": ["AQP4", "GFAP", "SLC1A2", "FGFR3"],
    "pre-Astrocyte": ["GFAP", "S100B", "SOX9", "NFIA"],
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
    os.path.join(proj_dir, "output/Annotation_hvg5000/L2/clusters_2_deg_others.csv"),
    index=False,
)

# Map L2 clusters to cell type labels
cluster_anno = {
    "Endothelial_cell": ["4", "11", "19"],
    "Pericyte": ["8"],
    "Vascular_smooth_muscle_cell": ["16"],
    "Fibroblast": ["6", "7", "20", "24", "26", "29", "31"],
    "Microglia": ["0", "1", "2", "5", "9", "14", "17", "23", "25", "27", "28", "33", "34", "35"],
    "Macrophage": ["12"],
    "Oligodendrocyte": ["18"],
    "Oligodendrocyte_precursor_committed": ["3", "10", "30"],
    "GABAergic_neuron": ["15", "21"],
    "Glutamatergic_neuron": ["13", "32"],
    "Astrocyte": ["22"],
    "Remove": ["36"],
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
    "Endothelial_cell": "EC",
    "Pericyte": "Per",
    "Vascular_smooth_muscle_cell": "VSMC",
    "Fibroblast": "Fib",
    "Microglia": "Mic",
    "Macrophage": "Mac",
    "Oligodendrocyte": "Oli",
    "Oligodendrocyte_precursor_committed": "COPC",
    "Glutamatergic_neuron": "Glu",
    "GABAergic_neuron": "GABA",
    "Astrocyte": "Ast",
    "Remove": "Remove",
}
ad_hs.obs["type_2_abbr"] = ad_hs.obs["type_2"].map(abbreviation)
ad_hs.obs["type_2_abbr"] = ad_hs.obs["type_2_abbr"].astype("category")

# Reassign broad type groups / level-1 labels for re-classified clusters
ad_hs.obs["type_group"] = ad_hs.obs["type_group"].astype(str)
ad_hs.obs["type_group"][ad_hs.obs["type_2"] == "Astrocyte"] = "Glia"
ad_hs.obs["type_group"][ad_hs.obs["type_2"] == "Oligodendrocyte"] = "Glia"
ad_hs.obs["type_group"][
    ad_hs.obs["type_2"] == "Oligodendrocyte_precursor_committed"
] = "Progenitors"
ad_hs.obs["type_group"][ad_hs.obs["type_2"] == "Glutamatergic_neuron"] = "Neuron"
ad_hs.obs["type_group"][ad_hs.obs["type_2"] == "GABAergic_neuron"] = "Neuron"
ad_hs.obs["type_group"] = ad_hs.obs["type_group"].astype("category")

ad_hs.obs["type_1"] = ad_hs.obs["type_1"].astype(str)
ad_hs.obs["type_1"][ad_hs.obs["type_2"] == "Astrocyte"] = "AstroEpen"
ad_hs.obs["type_1"][
    ad_hs.obs["type_2"] == "Oligodendrocyte_precursor_committed"
] = "Glial_precursor"
ad_hs.obs["type_1"][ad_hs.obs["type_2"] == "Microglia"] = "Immune"
ad_hs.obs["type_1"][ad_hs.obs["type_2"] == "Macrophage"] = "Immune"
ad_hs.obs["type_1"] = ad_hs.obs["type_1"].astype("category")

# Refine Glu subclusters within this subset
ad_hs_glu = ad_hs[ad_hs.obs["type_2"] == "Glutamatergic_neuron"]
ad_hs_gaba = ad_hs[ad_hs.obs["type_2"] == "GABAergic_neuron"]

sc.tl.leiden(ad_hs_glu, key_added="leiden_refine", resolution=0.5)
ad_hs_glu.obs["cluster_3"] = ad_hs_glu.obs["leiden_refine"]
sc.tl.paga(ad_hs_glu, groups="cluster_3")
sc.tl.umap(ad_hs_glu, init_pos="paga", maxiter=1000)

cluster_anno = {
    "L2/3-CUX2 Glu": ["0", "5", "8"],
    "L4-RORB Glu": ["3"],
    "L5/6-TLE4 Glu": ["2"],
    "Hippocampal_Glu": ["1", "6", "9"],
    "Granule_neuron": ["4", "7"],
}

cluster_anno_rev = {}
for type_i in cluster_anno.keys():
    for cluster_i in cluster_anno[type_i]:
        cluster_anno_rev[cluster_i] = type_i

ad_hs_glu.obs["type_2"] = ad_hs_glu.obs["cluster_3"].map(cluster_anno_rev)
ad_hs_glu.obs["type_2"] = ad_hs_glu.obs["type_2"].astype("category")

abbreviation = {
    "L2/3-CUX2 Glu": "L2/3-CUX2 Glu",
    "L4-RORB Glu": "L4-RORB Glu",
    "L5/6-TLE4 Glu": "L5/6-TLE4 Glu",
    "Hippocampal_Glu": "HIP Glu",
    "Granule_neuron": "Granule",
}
ad_hs_glu.obs["type_2_abbr"] = ad_hs_glu.obs["type_2"].map(abbreviation)
ad_hs_glu.obs["type_2_abbr"] = ad_hs_glu.obs["type_2_abbr"].astype("category")

# Refine GABA subclusters within this subset
sc.tl.leiden(ad_hs_gaba, key_added="leiden_refine", resolution=0.5)
ad_hs_gaba.obs["cluster_3"] = ad_hs_gaba.obs["leiden_refine"]
sc.tl.paga(ad_hs_gaba, groups="cluster_3")
sc.tl.umap(ad_hs_gaba, init_pos="paga", maxiter=1000)

cluster_anno = {
    "LHX6+LAMP5+ GABA": ["2", "6"],
    "VIP+ GABA": ["1"],
    "CALB2+ GABA": ["0", "10"],
    "Medium_spiny_neuron": ["7"],
    "Purkinje_neuron": ["3", "9"],
    "Thalamic_Glu": ["4"],
    "Thalamic_GABA": ["8"],
    "Oligodendrocyte_precursor_committed": ["5"],
}

cluster_anno_rev = {}
for type_i in cluster_anno.keys():
    for cluster_i in cluster_anno[type_i]:
        cluster_anno_rev[cluster_i] = type_i

ad_hs_gaba.obs["type_2"] = ad_hs_gaba.obs["cluster_3"].map(cluster_anno_rev)
ad_hs_gaba.obs["type_2"] = ad_hs_gaba.obs["type_2"].astype("category")

abbreviation = {
    "LHX6+LAMP5+ GABA": "LHX6+LAMP5+ GABA",
    "VIP+ GABA": "VIP+ GABA",
    "CALB2+ GABA": "CALB2+ GABA",
    "Medium_spiny_neuron": "MSN",
    "Purkinje_neuron": "PC",
    "Thalamic_Glu": "Th Glu",
    "Thalamic_GABA": "Th GABA",
    "Oligodendrocyte_precursor_committed": "COPC",
}
ad_hs_gaba.obs["type_2_abbr"] = ad_hs_gaba.obs["type_2"].map(abbreviation)
ad_hs_gaba.obs["type_2_abbr"] = ad_hs_gaba.obs["type_2_abbr"].astype("category")

ad_hs_gaba.obs["type_1"] = ad_hs_gaba.obs["type_1"].astype(str)
ad_hs_gaba.obs["type_group"] = ad_hs_gaba.obs["type_group"].astype(str)
ad_hs_gaba.obs["type_1"][ad_hs_gaba.obs["type_2"] == "Thalamic_Glu"] = "Glutamatergic_neuron"
ad_hs_gaba.obs["type_1"][
    ad_hs_gaba.obs["type_2"] == "Oligodendrocyte_precursor_committed"
] = "Glial_precursor"
ad_hs_gaba.obs["type_group"][
    ad_hs_gaba.obs["type_2"] == "Oligodendrocyte_precursor_committed"
] = "Progenitors"
ad_hs_gaba.obs["type_1"] = ad_hs_gaba.obs["type_1"].astype("category")
ad_hs_gaba.obs["type_group"] = ad_hs_gaba.obs["type_group"].astype("category")

# Write refined labels back into the full object
cols_to_update = ["type_group", "type_1", "type_2", "type_2_abbr"]
glu_idx = ad_hs_glu.obs.index
gaba_idx = ad_hs_gaba.obs.index
ad_hs.obs["type_2_abbr"] = ad_hs.obs["type_2_abbr"].astype(str)
ad_hs.obs["type_2"] = ad_hs.obs["type_2"].astype(str)
ad_hs.obs["type_1"] = ad_hs.obs["type_1"].astype(str)
ad_hs.obs["type_group"] = ad_hs.obs["type_group"].astype(str)
ad_hs.obs.loc[glu_idx, cols_to_update] = ad_hs_glu.obs[cols_to_update]
ad_hs.obs.loc[gaba_idx, cols_to_update] = ad_hs_gaba.obs[cols_to_update]

# Save result
ad_hs.write_h5ad(
    filename=os.path.join(proj_dir, "output/Annotation_hvg5000/L2/hs_anno_2_Others.h5ad")
)
print("Save as <output/Annotation_hvg5000/L2/hs_anno_2_Others.h5ad>")

cell_anno_info = ad_hs.obs[["type_group", "type_1", "type_2", "type_2_abbr"]]
cell_anno_info.insert(0, "cell_name", cell_anno_info.index)
cell_anno_info.to_csv(
    os.path.join(proj_dir, "output/Annotation_hvg5000/L2/Others_cell_metadata.csv"),
    index=False,
)
