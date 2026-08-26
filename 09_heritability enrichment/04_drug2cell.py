#!/usr/bin/env python3
"""
=============================================================================
File      : 04_drug2cell.py
Directory : 09_heritability enrichment
Purpose   : Score drugs against single-cell expression using drug2cell and link enriched drugs to targets.
=============================================================================
"""

import os

import numpy as np
import pandas as pd
import scanpy as sc
import drug2cell as d2c
from rapidfuzz import process, fuzz

## Configuration ##
# Set the base directory for input/output data below
# proj_dir must point to the pipeline output root <project>/data/human/output
BASE_DIR = os.environ.get("PROJ_DIR", "path/to/your/project")  # <-- MODIFY THIS
DRUG_DIR = os.path.join(BASE_DIR, "Analysis", "disease", "drug")  # drug2cell working directory
AD_GENE_DIR = os.path.join(DRUG_DIR, "AD-GENE")  # AD gene-set reference files
MIC_DIR = os.path.join(DRUG_DIR, "mic")  # microglia-specific outputs
MIC_DEG_FILE = os.path.join(BASE_DIR, "Analysis", "immune", "validation", "Mic", "mic_functional_cluster6_marker_new.csv")

# Drug metadata (ChEMBL merged, gene symbols, human)
DRUG_META_FILE = os.path.join(DRUG_DIR, "chembl_30_merged_genesymbols_humans.pkl")
# All-cell-type metacell object
METACELL_FILE = os.path.join(BASE_DIR, "Analysis", "metacell", "human_all_celltypes_metacell_raw_k100.h5ad")
# Microglia subcluster object
MIC_FILE = os.path.join(BASE_DIR, "Annotation_hvg5000", "L4", "hs_anno_4_Mic.h5ad")


# -----------------------------------------------------------------------------
# 1. Load drug metadata; build ATC mapping; filter by quality criteria
# -----------------------------------------------------------------------------
drug_df = pd.read_pickle(DRUG_META_FILE)

# ATC classification mapping per drug (ChEMBL ID -> ATC info)
atc_drugs = drug_df[
    drug_df['atc_classification|level1'].notna() |
    drug_df['atc_classification|level2'].notna()
]
drug_atc_mapping = atc_drugs.groupby('molecule_dictionary|chembl_id').agg({
    'atc_classification|level1': 'first',
    'atc_classification|level2': 'first',
    'atc_classification|level1_description': 'first',
    'atc_classification|level2_description': 'first',
    'molecule_dictionary|pref_name': 'first'
}).to_dict('index')

# Approved drugs (max clinical phase == 4)
approved_drugs = drug_df[drug_df['molecule_dictionary|max_phase'] == 4]
approved_chembl_ids = approved_drugs['molecule_dictionary|chembl_id'].unique()

# High-confidence assays (minimum confidence score >= 8)
drug_df_valid_conf = drug_df.dropna(subset=['assays|confidence_score'])
chembl_conf_min = drug_df_valid_conf.groupby('molecule_dictionary|chembl_id')['assays|confidence_score'].min()
high_conf_chembl_ids = chembl_conf_min[chembl_conf_min >= 8].index.tolist()
confidence_drugs = drug_df[drug_df['molecule_dictionary|chembl_id'].isin(high_conf_chembl_ids)]
confidence_chembl_ids = confidence_drugs['molecule_dictionary|chembl_id'].unique()

# Small-molecule drugs only
small_molecule_drugs = drug_df[drug_df['molecule_dictionary|molecule_type'] == 'Small molecule']
small_molecule_chembl_ids = small_molecule_drugs['molecule_dictionary|chembl_id'].unique()


# -----------------------------------------------------------------------------
# 2. Score drugs on the all-cell-type metacell object; annotate with ATC
# -----------------------------------------------------------------------------
ad_hs = sc.read_h5ad(METACELL_FILE)

d2c.score(ad_hs, use_raw=False)
ad_drug = ad_hs.uns['drug2cell']


def extract_chembl_id(index_name):
    """Extract the ChEMBL ID from a drug index name of the form 'CHEMBLxxx|NAME'."""
    if '|' in index_name:
        return index_name.split('|')[0]
    return index_name


ad_drug.var['chembl_id'] = ad_drug.var.index.map(extract_chembl_id)
ad_drug.var['pref_name'] = ad_drug.var.index.map(lambda x: x.split('|')[1] if '|' in x else x)

# Annotate each drug with ATC classification (Unknown if not in the mapping)
atc_info_list = []
for chembl_id in ad_drug.var['chembl_id']:
    if chembl_id in drug_atc_mapping:
        atc_info = drug_atc_mapping[chembl_id]
        atc_info_list.append({
            'atc_level1': atc_info.get('atc_classification|level1', 'Unknown'),
            'atc_level2': atc_info.get('atc_classification|level2', 'Unknown'),
            'atc_level1_desc': atc_info.get('atc_classification|level1_description', 'Unknown'),
            'atc_level2_desc': atc_info.get('atc_classification|level2_description', 'Unknown')
        })
    else:
        atc_info_list.append({
            'atc_level1': 'Unknown',
            'atc_level2': 'Unknown',
            'atc_level1_desc': 'Unknown',
            'atc_level2_desc': 'Unknown'
        })

atc_info_df = pd.DataFrame(atc_info_list, index=ad_drug.var.index)
ad_drug.var = pd.concat([ad_drug.var, atc_info_df], axis=1)

ad_drug.write_h5ad(os.path.join(DRUG_DIR, "ad_drug2cell_human_all_celltypes_metacell_k100_with_atc.h5ad"))


# -----------------------------------------------------------------------------
# 3. Differential expression of drugs across functional clusters
# -----------------------------------------------------------------------------
sc.tl.rank_genes_groups(
    ad_drug,
    method="wilcoxon",
    groupby="function_cluster",
    pts=True,
    corr_method="benjamini-hochberg"
)

result = ad_drug.uns["rank_genes_groups"]
groups = result["names"].dtype.names
dfs_list = []

for cluster in groups:
    # Build the DEG table for a single cluster
    df = pd.DataFrame({
        "gene": result["names"][cluster],
        "log2fc": result["logfoldchanges"][cluster],
        "pvals": result["pvals"][cluster],
        "qvals": result["pvals_adj"][cluster],
        "expr_frac": result["pts"][cluster]
    })
    df["cluster"] = cluster
    dfs_list.append(df)

all_deg_df = pd.concat(dfs_list, ignore_index=True)
filtered_deg_df = all_deg_df[
    (all_deg_df["qvals"] < 0.01) &
    (all_deg_df["log2fc"] > 0.5) &
    (all_deg_df["expr_frac"] >= 0.2)
]

all_deg_df.to_csv(os.path.join(MIC_DIR, "drug2cell_all_deg_mic_func6_FDR_results_new.csv"), index=False)
filtered_deg_df.to_csv(os.path.join(MIC_DIR, "drug2cell_deg_mic_func6_FDR_results_new.csv"), index=False)

# Map each cell type to a coarse cell class (annotation for downstream grouping)
celltype_group = {
    # Glutamatergic neurons (excitatory)
    "L2/3-SATB2 Glu": "Glutamatergic neuron",
    "L2/3-CUX2 Glu": "Glutamatergic neuron",
    "L4-RORB Glu": "Glutamatergic neuron",
    "L5-BCL11B Glu": "Glutamatergic neuron",
    "L5/6-TLE4 Glu": "Glutamatergic neuron",
    "L5/6-THEMIS Glu": "Glutamatergic neuron",
    "Thalamic_Glu": "Glutamatergic neuron",
    "Hippocampal_Glu": "Glutamatergic neuron",
    "Hypothalamic_Glu": "Glutamatergic neuron",

    # GABAergic neurons (inhibitory)
    "SST+ GABA": "GABAergic neuron",
    "CALB2+ GABA": "GABAergic neuron",
    "LHX6+LAMP5+ GABA": "GABAergic neuron",
    "Midbrain-derived_GABA": "GABAergic neuron",
    "VIP+ GABA": "GABAergic neuron",
    "RELN+ GABA": "GABAergic neuron",
    "PVALB+ GABA": "GABAergic neuron",
    "Thalamic_GABA": "GABAergic neuron",

    # Other neurons mapped to the excitatory/inhibitory classes
    "Medium_spiny_neuron": "GABAergic neuron",  # striatal GABA neuron
    "Purkinje_neuron": "GABAergic neuron",      # inhibitory
    "Granule_neuron": "Glutamatergic neuron",   # cerebellar granule cell (excitatory)

    # Glia
    "Oligodendrocyte": "Glia",
    "Astrocyte": "Glia",

    # Neural precursor cells
    "Radial_glia": "Neural precursor cell",
    "Neuroblast": "Neural precursor cell",
    "Neuronal_IPC": "Neural precursor cell",
    "Glioblast": "Neural precursor cell",
    "Oligodendrocyte_precursor": "Neural precursor cell",
    "Oligodendrocyte_precursor_committed": "Neural precursor cell",

    # Non-neural cell types
    "Microglia": "Non-neural",
    "Fibroblast": "Non-neural",
    "Endothelial_cell": "Non-neural",
    "Pericyte": "Non-neural",
    "Choroid_plexus_epithelial": "Non-neural",
    "Ependymal": "Non-neural",
    "Macrophage": "Non-neural",
    "Vascular_smooth_muscle_cell": "Non-neural",
    "Cholinergic": "Non-neural",  # neuron-like but grouped as non-neural here
    "Serotonergic": "Non-neural"  # neuron-like but grouped as non-neural here
}

ad_drug.obs["cell_class"] = ad_drug.obs["type_2"].map(celltype_group)


# -----------------------------------------------------------------------------
# 4. Microglia DEG overlap with AD literature and MAGMA gene sets
#    (generates mic_paper1_overlap_genes.csv used in section 5)
# -----------------------------------------------------------------------------
# Paper 1: single-cell multiregion dissection of Alzheimer's disease (microglia)
paper1_data = pd.read_csv(os.path.join(AD_GENE_DIR, 'aggregated_fullset.Mic_Immune_Mic-Single-cell multiregion dissection of Alzheimer’s disease.tsv'),
                          sep='\t', header=0)
# Paper 2: eQTL meta-analysis linking genetic risk to brain disorders
paper2_data = pd.read_excel(os.path.join(AD_GENE_DIR, 'AD-Mic-eQTL-A meta-analysis of single-nucleus expression quantitative trait loci linking genetic risk to brain disorders.xlsx'),
                            header=0)

# Our microglia functional-cluster marker genes
mic_deg_data = pd.read_csv(MIC_DEG_FILE, header=0)

# MAGMA AD association results and NCBI gene-location table (Entrez ID -> symbol)
magma_data = pd.read_csv(os.path.join(AD_GENE_DIR, "AD_magma.genes.out"), sep='\s+', header=0)
ncbi_data = pd.read_csv(os.path.join(AD_GENE_DIR, "NCBI38.gene.loc"), sep='\s+', header=None,
                        names=["GENE", "CHR", "START", "STOP", "STRAND", "SYMBOL"])

# Map Entrez gene IDs to gene symbols
magma_data['GENE'] = magma_data['GENE'].astype(str)
ncbi_data['GENE'] = ncbi_data['GENE'].astype(str)
gene_symbol_map = dict(zip(ncbi_data['GENE'], ncbi_data['SYMBOL']))
magma_data['gene symbol'] = magma_data['GENE'].map(gene_symbol_map)

# MAGMA: significant AD genes with positive association
magma_data_sub = magma_data[
    (magma_data['P'] <= 0.05) &
    (magma_data['gene symbol'].notna()) &
    (magma_data['ZSTAT'] >= 0)
]
magma_data_gene = magma_data_sub['gene symbol'].drop_duplicates().tolist()

# Paper 1: significant genes across all brain regions, consistent direction
paper1_data_sub = paper1_data[
    (paper1_data['region'] == 'allregions') &  # all brain regions
    (paper1_data['log10p_nm'] >= 1.3) &        # FDR <= 0.05
    (paper1_data['logFC_nb'] * paper1_data['coef_mast'] > 0)
]
paper1_data_gene = paper1_data_sub['gene'].drop_duplicates().tolist()

# Paper 2: eQTL genes
paper2_data_gene = paper2_data['QTL gene symbol'].drop_duplicates().tolist()

# Our microglia DEGs: Enhanced redox cluster, up-regulated markers
mic_deg_data_sub = mic_deg_data[
    (mic_deg_data['cluster'] == 'Enhanced redox') &
    (mic_deg_data['p_val_adj'] <= 0.05) &
    (mic_deg_data['avg_log2FC'] > 0) &
    (mic_deg_data['pct.1'] > 0.1)
]
mic_deg_data_gene = mic_deg_data_sub['gene'].tolist()

# Union of gene sets with per-set membership indicators
all_genes = list(set(paper1_data_gene + mic_deg_data_gene + magma_data_gene))

gene_df = pd.DataFrame({
    'gene': all_genes,
    'in_mic_deg': [1 if g in mic_deg_data_gene else 0 for g in all_genes],
    'in_paper1': [1 if g in paper1_data_gene else 0 for g in all_genes],
    'in_magma': [1 if g in magma_data_gene else 0 for g in all_genes]
})
gene_df['total_count'] = gene_df[['in_mic_deg', 'in_paper1', 'in_magma']].sum(axis=1)
gene_df = gene_df.sort_values(by=['total_count', 'gene'], ascending=[False, True]).reset_index(drop=True)

gene_df.to_csv(os.path.join(AD_GENE_DIR, "mic_ad_gene_set.csv"))

# Set-size and overlap statistics
set1 = set(mic_deg_data_gene)
set2 = set(paper1_data_gene)
set3 = set(magma_data_gene)

set_stats = {
    'mic_deg unique': len(set1),
    'paper1 unique': len(set2),
    'magma unique': len(set3),
    'mic_deg & paper1 overlap': len(set1 & set2),
    'mic_deg & magma overlap': len(set1 & set3),
    'paper1 & magma overlap': len(set2 & set3),
    'mic_deg & paper1 union': len(set1 | set2),
    'mic_deg & magma union': len(set1 | set3),
    'paper1 & magma union': len(set2 | set3),
    'three-way overlap': len(set1 & set2 & set3),
    'total union': len(set1 | set2 | set3)
}
print(set_stats)

# Save pairwise and three-way overlap gene lists
gene1 = list(set1 & set2)
gene1_df = pd.DataFrame({"gene": gene1})
gene1_df.to_csv(os.path.join(AD_GENE_DIR, "mic_paper1_overlap_genes.csv"), index=False, header=True)

gene2 = list(set1 & set3)
gene2_df = pd.DataFrame({"gene": gene2})
gene2_df.to_csv(os.path.join(AD_GENE_DIR, "mic_magma_overlap_genes.csv"), index=False, header=True)

gene3 = list(set(gene1 + gene2))
gene3_df = pd.DataFrame({"gene": gene3})
gene3_df.to_csv(os.path.join(AD_GENE_DIR, "mic_paper1_magma_overlap_genes.csv"), index=False, header=True)


# -----------------------------------------------------------------------------
# 5. Enhanced redox (microglia energy metabolism) drug-target linking
# -----------------------------------------------------------------------------
# Significantly enriched drugs in the Enhanced redox cluster
filtered_deg_df_EM = filtered_deg_df[filtered_deg_df['cluster'] == 'Enhanced redox']
filtered_deg_df_EM = filtered_deg_df_EM.sort_values(by='qvals', ascending=True)
ad_drug_sub = ad_drug[:, filtered_deg_df_EM["gene"]].copy()
unique_mask = ~ad_drug_sub.var['pref_name'].duplicated()
ad_drug_sub = ad_drug_sub[:, unique_mask].copy()
ad_drug_sub.var['pref_name'] = ad_drug_sub.var['pref_name'].astype(str).apply(lambda x: x.capitalize())
ad_drug_sub.var_names = ad_drug_sub.var['pref_name'].values
ad_drug_sub.var_names_make_unique()
# Keep only approved, high-confidence and small-molecule drugs
ad_drug_sub = ad_drug_sub[:, ad_drug_sub.var['chembl_id'].isin(approved_chembl_ids)]
ad_drug_sub = ad_drug_sub[:, ad_drug_sub.var['chembl_id'].isin(confidence_chembl_ids)]
ad_drug_sub = ad_drug_sub[:, ad_drug_sub.var['chembl_id'].isin(small_molecule_chembl_ids)]

# Energy-metabolism target set: microglia genes shared with the AD literature
energy_genes = pd.read_csv(os.path.join(AD_GENE_DIR, "mic_paper1_overlap_genes.csv"))["gene"].tolist()

# Build the drug x target matrix: 1 if the drug's targets include an
# energy-metabolism gene
rows = []
for drug, targets in ad_drug_sub.var["all_genes"].items():
    if pd.isna(targets):
        continue
    target_genes = [g.strip() for g in targets.split(",")]
    for g in target_genes:
        if g in energy_genes:
            rows.append({"drug": drug, "gene": g, "value": 1})

drug_gene_df = pd.DataFrame(rows)
matrix = drug_gene_df.pivot_table(index="drug", columns="gene", values="value", fill_value=0)

# Order genes alphabetically and drugs by their first target position
gene_order = sorted(matrix.columns.tolist())
gene_order = [g for g in gene_order if g in matrix.columns]


def first_hit_index(row, gene_order):
    """Return the position of the first gene in gene_order targeted by this drug row."""
    hits = [gene_order.index(g) for g in gene_order if row[g] == 1]
    return min(hits) if len(hits) > 0 else np.inf


drug_order = (
    matrix
    .assign(sort_key=matrix.apply(first_hit_index, axis=1, gene_order=gene_order))
    .sort_values("sort_key")
    .index
)

matrix = matrix.loc[drug_order, gene_order]
matrix = matrix.loc[matrix.sum(axis=1) > 0]


# -----------------------------------------------------------------------------
# 6. Per cell type: identify significantly enriched drugs with ATC annotation
# -----------------------------------------------------------------------------
all_cell_types = ad_drug.obs['function_cluster'].cat.categories.tolist()
per_celltype_drugs = {}

for target_cell_type in all_cell_types:
    # Differential expression of drugs against the target cell type
    sc.tl.rank_genes_groups(
        ad_drug,
        groupby='function_cluster',
        method='wilcoxon',
        groups=[target_cell_type],
        pts=True,
        corr_method='bonferroni'
    )

    de_results = sc.get.rank_genes_groups_df(ad_drug, group=target_cell_type)

    # Significance thresholds for enriched drugs
    log2fc_threshold = 0.5
    pval_threshold = 0.05
    pts_threshold = 0.2

    significant_drugs = de_results[
        (de_results['logfoldchanges'] > log2fc_threshold) &
        (de_results['pvals_adj'] < pval_threshold) &
        (de_results['pct_nz_group'] >= pts_threshold)
    ]

    if len(significant_drugs) == 0:
        print(f"  {target_cell_type}: no significant drugs, skipped")
        continue

    significant_drug_names = significant_drugs['names'].tolist()

    # Subset the drug object and keep only drugs with a known ATC classification
    ad_drug_subset = ad_drug[:, significant_drug_names].copy()
    known_atc_mask = (ad_drug_subset.var['atc_level1'] != 'Unknown') & (ad_drug_subset.var['atc_level1_desc'] != 'Unknown')
    ad_drug_subset = ad_drug_subset[:, known_atc_mask].copy()

    if len(ad_drug_subset.var_names) == 0:
        print(f"  {target_cell_type}: no drugs with known ATC classification, skipped")
        continue

    # Sort drugs by ATC level1/level2 classification
    ad_drug_subset.var['sort_key'] = ad_drug_subset.var['atc_level1'].astype(str) + '_' + ad_drug_subset.var['atc_level2'].astype(str)
    ad_drug_subset = ad_drug_subset[:, ad_drug_subset.var.sort_values('sort_key').index]

    # Store the enriched drugs (with ATC info) for the target cell type
    per_celltype_drugs[target_cell_type] = ad_drug_subset.var[
        ['chembl_id', 'pref_name', 'atc_level1', 'atc_level1_desc', 'atc_level2', 'atc_level2_desc']
    ].reset_index()
    print(f"  {target_cell_type}: {len(ad_drug_subset.var_names)} enriched drugs with ATC info")


# -----------------------------------------------------------------------------
# 7. Microglia functional-cluster modules: drug2cell scoring
# -----------------------------------------------------------------------------
ad_hs_mic = sc.read_h5ad(MIC_FILE)

# Drop intermediate-resolution clustering columns
ad_hs_mic.obs = ad_hs_mic.obs.drop(columns=[
    'leiden_l4_0.1', 'leiden_l4_0.2', 'leiden_l4_0.3', 'leiden_l4_0.4',
    'leiden_l4_0.5', 'leiden_l4_0.6', 'leiden_l4_0.7', 'leiden_l4_0.8',
    'leiden_l4_0.9', 'leiden_l4_1.0', 'leiden_l4_1.1', 'leiden_l4_1.2',
    'leiden_l4_1.3', 'leiden_l4_1.4', 'leiden_l4_1.5'
])

# Map microglia subclusters to functional modules
module_mapping = {
    "Proliferative": ["Microglia-9", "Microglia-16"],
    "Enhanced redox": ["Microglia-1", "Microglia-15"],
    "Inflammatory": ["Microglia-6", "Microglia-11"],
    "Lipid associated": ["Microglia-3", "Microglia-4", "Microglia-5", "Microglia-7", "Microglia-8", "Microglia-17"],
    "Surveilling": ["Microglia-2", "Microglia-10", "Microglia-12", "Microglia-14"],
    "Reacting": ["Microglia-13"]
}

for module_name, microglia_list in module_mapping.items():
    # Assign the functional module name to matching microglia subclusters
    mask = ad_hs_mic.obs["type_3"].isin(microglia_list)
    ad_hs_mic.obs.loc[mask, "function_cluster"] = module_name

# Score drugs on the microglia object and save results
d2c.score(ad_hs_mic, use_raw=True)
ad_drug_mic = ad_hs_mic.uns['drug2cell']

ad_hs_mic.write_h5ad(os.path.join(DRUG_DIR, "ad_drug2cell_human_mic_func_cluster_all_new.h5ad"))
ad_drug_mic.write_h5ad(os.path.join(DRUG_DIR, "ad_drug2cell_human_mic_func6_cluster_new.h5ad"))
