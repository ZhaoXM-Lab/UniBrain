#!/usr/bin/env python3
"""
=============================================================================
File      : 14_L3_Glu_step2.py
Directory : 05_annotation
Purpose   : Level 3 annotation assigning glutamatergic subtype labels to refined clusters.
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
type_i = "Glu"
ad_hs = sc.read_h5ad(
    os.path.join(
        proj_dir,
        "output/Annotation_hvg5000/L3/hs_anno_3_%s_cluster.h5ad" % type_i,
    )
)
print(ad_hs)

# Cluster compactness across resolutions (guides the resolution choice)
cp_index = []
resolution = np.linspace(0.1, 1.5, 15)
for res_i in resolution:
    key_name = "leiden_refine_" + ("%.1f" % res_i)
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
ad_hs.obs["cluster_3"] = ad_hs.obs["leiden_refine_1.5"]
ad_hs.obsm["X_umap_all"] = ad_hs.obsm["X_umap"]
sc.tl.paga(ad_hs, groups="cluster_3")
sc.tl.umap(ad_hs, init_pos="paga", maxiter=1000)

# Marker genes used to interpret each cluster (canonical literature markers)
marker_genes_dict = {
    # Glutamatergic subtypes
    "Glutamatergic": ["SLC17A6", "SLC17A7", "NEUROD2"],
    # Cortical layers (PMID: 32915944, 34615802)
    "L2/3-CUX2": ["CUX2", "SATB2"],
    "L4-RORB": ["RORB", "NTNG2"],
    "L5/6-THEMIS": ["THEMIS", "FEZF2"],
    "L5/6-TLE4": ["TLE4", "BCL11B"],

    # Thalamic / hypothalamic / hippocampal / amygdala
    "Thalamic Glu": ["NTNG1", "PCP4", "FSTL4"],
    "Hypothalamic Glu": ["SIM1", "POMC", "NPY2R", "SLC17A6"],
    "Amygdala Glu": ["ENC1", "DSP", "TAFA2"],
    "Hippocampal_CA1-3 Glu": ["COL24A1", "COL5A2", "CCBE1"],
    "Hippocampal_CA4 Glu": ["SHISA6", "NRP1", "TRHDE"],
    "Hippocampal_dentate_gyrus Glu": ["SEMA5A", "ADRA1A", "ALDH1A2"],
    # Cerebellum (PMID: 32296178)
    "Granule": ["ZIC4", "GABRA6", "CNPY1"],

    # GABAergic subtypes (for cross-checking)
    "GABAergic": ["GAD1", "GAD2", "SLC32A1"],
    "MGE-dev": ["NKX2-1", "LHX6", "NXPH1"],  # PMID: 30737435, 32350463
    "PVALB+ GABA": ["PVALB", "SST", "CALB1", "KCNC1"],  # Fast-spiking
    "SST+ GABA": ["SST", "NPY", "LHX6", "LAMP5"],  # Martinotti/Non-Martinotti
    "CGE-dev": ["NFIX", "PROX1", "NR2F2"],
    "VIP+ GABA": ["VIP", "CCK", "CALB2", "CRH"],  # Vasoactive intestinal peptide
    "CALB2+ GABA": ["CALB2", "RELN", "SCUBE3", "HTR3A"],
    "Reelin+ GABA": ["RELN", "CALB2", "CXCL12"],
    "LGE-dev": ["ZFHX3", "FOXP1", "EBF1"],
    "Medium_spiny": ["DRD1", "ISL1", "DRD2", "PENK"],  # Striatal medium spiny
    "OB Interneuron": ["CHD7", "ID2", "PAX6"],  # Olfactory bulb interneuron
    "Purkinje": ["CALB1", "GRM1", "ITPR1", "CA8"],  # Purkinje cells
    "Thalamic GABA": ["LHX5", "TAL1", "GATA3"],
    "Midbrain-derived GABA": ["GPRIN2", "NOS1", "ANKRD34B"],  # Substantia nigra

    # Other neurons
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
}

# Keep only clusters with enough cells for differential expression
valid_clusters = ad_hs.obs["cluster_3"].value_counts()[
    ad_hs.obs["cluster_3"].value_counts() > 2
].index
ad_hs = ad_hs[ad_hs.obs["cluster_3"].isin(valid_clusters)]

# Differential expression between clusters (Wilcoxon, Bonferroni corrected)
sc.tl.rank_genes_groups(
    ad_hs,
    groupby="cluster_3",
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
    os.path.join(
        proj_dir, "output/Annotation_hvg5000/L3/clusters_3_deg_%s.csv" % type_i
    ),
    index=False,
)

# Map L3 clusters to glutamatergic subtype labels
cluster_anno = {
    "L2/3-CUX2 Glu": ["0", "5", "13", "21", "27", "29", "32", "39"],
    "L2/3-SATB2 Glu": ["1", "2", "4", "40"],
    "L4-RORB Glu": ["8", "15"],
    "L5-BCL11B Glu": ["7", "22"],
    "L5/6-THEMIS Glu": ["17", "23"],
    "L5/6-TLE4 Glu": ["14", "24"],
    "Thalamic_Glu": ["6", "12", "35", "41", "43", "48"],
    "Hypothalamic_Glu": ["31", "34"],
    "Hippocampal_Glu": ["16", "19", "38", "45", "46", "56"],
    "Granule_neuron": ["10", "58"],
    "Medium_spiny_neuron": ["3", "49"],
    "LHX6+LAMP5+ GABA": ["9", "26"],
    "CALB2+ GABA": ["18", "25", "53"],
    "Midbrain-derived_GABA": ["44"],
    "Purkinje_neuron": ["30", "51"],
    "Cholinergic": ["55"],
    "Radial_glia": ["20"],
    "Neuroblast": ["11", "33"],
    "Remove": [
        "28", "36", "37", "42", "47", "50", "52", "57", "54", "59",
        "60", "61", "62",
    ],
}

cluster_anno_rev = {}
for type_i_2 in cluster_anno.keys():
    for cluster_i in cluster_anno[type_i_2]:
        cluster_anno_rev[cluster_i] = type_i_2

ad_hs.obs["type_2"] = ad_hs.obs["cluster_3"].map(cluster_anno_rev)
ad_hs.obs["type_2"] = ad_hs.obs["type_2"].astype("category")

# Cell type abbreviations
abbreviation = {
    "L2/3-CUX2 Glu": "L2/3-CUX2 Glu",
    "L2/3-SATB2 Glu": "L2/3-SATB2 Glu",
    "L4-RORB Glu": "L4-RORB Glu",
    "L5-BCL11B Glu": "L5-BCL11B Glu",
    "L5/6-THEMIS Glu": "L5/6-THEMIS Glu",
    "L5/6-TLE4 Glu": "L5/6-TLE4 Glu",
    "Thalamic_Glu": "TH Glu",
    "Hypothalamic_Glu": "HT Glu",
    "Hippocampal_Glu": "HIP Glu",
    "Granule_neuron": "Granule",
    "Medium_spiny_neuron": "MSN",
    "LHX6+LAMP5+ GABA": "LHX6+LAMP5+ GABA",
    "CALB2+ GABA": "CALB2+ GABA",
    "Midbrain-derived_GABA": "MB GABA",
    "Purkinje_neuron": "PC",
    "Cholinergic": "ACh",
    "Radial_glia": "RG",
    "Neuroblast": "NB",
    "Remove": "Remove",
}
ad_hs.obs["type_2_abbr"] = ad_hs.obs["type_2"].map(abbreviation)
ad_hs.obs["type_2_abbr"] = ad_hs.obs["type_2_abbr"].astype("category")

# Reassign broad type groups / level-1 labels for re-classified clusters
ad_hs.obs["type_group"] = ad_hs.obs["type_group"].astype(str)
ad_hs.obs["type_group"][ad_hs.obs["type_2"] == "Neuroblast"] = "Progenitors"
ad_hs.obs["type_group"][ad_hs.obs["type_2"] == "Radial_glia"] = "Progenitors"
ad_hs.obs["type_group"] = ad_hs.obs["type_group"].astype("category")

ad_hs.obs["type_1"] = ad_hs.obs["type_1"].astype(str)
ad_hs.obs["type_1"][ad_hs.obs["type_2"] == "Medium_spiny_neuron"] = "GABAergic_neuron"
ad_hs.obs["type_1"][ad_hs.obs["type_2"] == "LHX6+LAMP5+ GABA"] = "GABAergic_neuron"
ad_hs.obs["type_1"][ad_hs.obs["type_2"] == "CALB2+ GABA"] = "GABAergic_neuron"
ad_hs.obs["type_1"][ad_hs.obs["type_2"] == "Midbrain-derived_GABA"] = "GABAergic_neuron"
ad_hs.obs["type_1"][ad_hs.obs["type_2"] == "Purkinje_neuron"] = "GABAergic_neuron"
ad_hs.obs["type_1"][ad_hs.obs["type_2"] == "Cholinergic"] = "Other neuron"
ad_hs.obs["type_1"][ad_hs.obs["type_2"] == "Radial_glia"] = "Radial_glia"
ad_hs.obs["type_1"][ad_hs.obs["type_2"] == "Neuroblast"] = "Neuronal_precursor"
ad_hs.obs["type_1"] = ad_hs.obs["type_1"].astype("category")

# Save result
ad_hs.write_h5ad(
    filename=os.path.join(
        proj_dir, "output/Annotation_hvg5000/L3/hs_anno_3_%s.h5ad" % type_i
    )
)
print("Save as <output/Annotation_hvg5000/L3/hs_anno_3_%s.h5ad>" % type_i)

cell_anno_info = ad_hs.obs[["type_group", "type_1", "type_2", "type_2_abbr"]]
cell_anno_info.insert(0, "cell_name", cell_anno_info.index)
cell_anno_info.to_csv(
    os.path.join(
        proj_dir,
        "output/Annotation_hvg5000/L3/%s_cell_metadata.csv" % type_i,
    ),
    index=False,
)
