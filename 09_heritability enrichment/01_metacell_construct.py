#!/usr/bin/env python3
"""
=============================================================================
File      : 01_metacell_construct.py
Directory : 09_heritability enrichment
Purpose   : Construct pseudobulk metacells from single-cell RNA-seq by KNN aggregation and integrate with harmony/scVI.
=============================================================================
"""

import os

# Limit BLAS / parallel library thread counts (memory and speed control)
os.environ["OPENBLAS_NUM_THREADS"] = "8"
os.environ["MKL_NUM_THREADS"] = "8"
os.environ["OMP_NUM_THREADS"] = "8"
os.environ["NUMBA_NUM_THREADS"] = "8"
os.environ["VECLIB_MAXIMUM_THREADS"] = "8"
os.environ["NUMEXPR_NUM_THREADS"] = "8"

## Configuration ##
# Set the base directory for input/output data below.
# proj_dir must point to the directory containing "output/" and "Analysis/"
BASE_DIR = os.environ.get("PROJ_DIR", "path/to/your/project")  # <-- MODIFY THIS

# Path to the GTF annotation file used to extract protein-coding genes
# (located at <project>/data/ref/gencode.v42.annotation.gtf)
GTF_FILE = os.path.join(BASE_DIR, "..", "..", "ref", "gencode.v42.annotation.gtf")

# Derived directories
METACELL_DIR = os.path.join(BASE_DIR, "Analysis", "metacell")
SCVI_DIR = os.path.join(METACELL_DIR, "scvi")
ANNOTATION_DIR = os.path.join(BASE_DIR, "Annotation_hvg5000")

import re
import time
import warnings
from itertools import combinations
from typing import Optional, List, Union, Dict, Tuple

import anndata as ad
import scipy.io
import scipy.sparse as sparse
import numpy as np
import scanpy as sc
import pandas as pd
from sklearn.neighbors import NearestNeighbors

import scvi

os.makedirs(SCVI_DIR, exist_ok=True)


def load_processed_adata(cell_type, base_path=os.path.join(ANNOTATION_DIR, "L4")):
    """
    Load a processed AnnData object for one cell type from saved mtx files.

    Parameters
    ----------
    cell_type : str
        Cell type name (directory name under <base_path>/R/).
    base_path : str
        Base directory containing the per-cell-type R subfolders.

    Returns
    -------
    AnnData
        Processed AnnData object with raw counts, metadata and harmony
        coordinates.
    """
    print(f"Loading cell type: {cell_type}")

    # Read saved mtx files
    dir_path = os.path.join(base_path, "R", cell_type)
    if not os.path.exists(dir_path):
        raise FileNotFoundError(f"Directory not found: {dir_path}")

    counts_matrix_path = os.path.join(dir_path, "counts_matrix.mtx")
    cells_path = os.path.join(dir_path, "counts_cells.csv")
    genes_path = os.path.join(dir_path, "counts_genes.csv")
    harmony_path = os.path.join(dir_path, "pca_harmony.csv")
    qc_genes_path = os.path.join(dir_path, "raw_genes.csv")
    meta_path = os.path.join(dir_path, "cells.csv")

    if not all(os.path.exists(p) for p in [counts_matrix_path, cells_path, genes_path]):
        raise FileNotFoundError(f"Necessary files not found in {dir_path}")

    counts_matrix = scipy.io.mmread(counts_matrix_path)
    cells_df = pd.read_csv(cells_path, index_col=0)
    genes_df = pd.read_csv(genes_path, index_col=0)
    harmony = pd.read_csv(harmony_path, index_col=0)
    qc_genes = pd.read_csv(qc_genes_path, index_col=0).index
    meta = pd.read_csv(meta_path, index_col=0)

    # Transpose matrix to match AnnData format (cells x genes)
    if sparse.issparse(counts_matrix):
        counts_matrix = counts_matrix.T.tocsr()
    else:
        counts_matrix = counts_matrix.T

    adata = sc.AnnData(X=counts_matrix, obs=cells_df, var=genes_df)
    common_genes = qc_genes.intersection(adata.var_names)
    adata = adata[:, common_genes]
    adata.obs = meta.loc[adata.obs_names]

    # Add harmony coordinates if available
    adata.obsm['X_harmony'] = harmony

    # Drop unneeded clustering columns
    columns_to_drop = [
        'leiden_l4_0.1', 'leiden_l4_0.2', 'leiden_l4_0.3', 'leiden_l4_0.4',
        'leiden_l4_0.5', 'leiden_l4_0.6', 'leiden_l4_0.7', 'leiden_l4_0.8',
        'leiden_l4_0.9', 'leiden_l4_1.0', 'leiden_l4_1.1', 'leiden_l4_1.2',
        'leiden_l4_1.3', 'leiden_l4_1.4', 'leiden_l4_1.5',
        'cluster_4', 'cluster_4_merge'
    ]
    existing_columns = [col for col in columns_to_drop if col in adata.obs.columns]
    adata.obs = adata.obs.drop(columns=existing_columns, errors='ignore')

    print(f"Loaded cell type: {cell_type} ({adata.n_obs} cells, {adata.n_vars} genes)")
    return adata


def construct_metacells(
    adata: ad.AnnData,
    k: int = 25,
    name: str = 'agg',
    reduction: str = 'X_pca',
    dims: Optional[List[int]] = None,
    obs_key: Optional[str] = None,
    use_raw: bool = False,
    mode: str = 'average',
    max_shared: int = 15,
    target_metacells: int = 1000,
    max_iter: int = 5000,
    verbose: bool = False
) -> Optional[ad.AnnData]:
    """
    Construct metacells from single-cell data using KNN aggregation.

    The algorithm iteratively picks seed cells such that no two metacells
    share more than `max_shared` cells, then aggregates the expression of
    each metacell by averaging (or summing) its k neighbors.

    Parameters
    ----------
    adata : AnnData
        Annotated data object.
    k : int
        Number of nearest neighbors aggregated into one metacell.
    name : str
        Prefix for metacell names.
    reduction : str
        Key in adata.obsm used for KNN (e.g., 'X_pca', 'X_harmony').
    dims : list, optional
        Dimensions to use from the reduction.
    obs_key : str, optional
        obs column used to store grouping information.
    use_raw : bool
        Whether to use adata.raw for expression data.
    mode : str
        'average' or 'sum' for aggregating expression.
    max_shared : int
        Maximum number of cells shared between metacells.
    target_metacells : int
        Target number of metacells to construct.
    max_iter : int
        Maximum iterations for the algorithm.
    verbose : bool
        Print progress information.

    Returns
    -------
    AnnData or None
        Metacell AnnData object, or None if construction failed.
    """
    print("Constructing metacells...")

    if reduction not in adata.obsm:
        raise ValueError(f"Reduction {reduction} not found in adata.obsm")
    if mode not in ['average', 'sum']:
        raise ValueError("mode must be 'average' or 'sum'")

    # Get reduction coordinates
    reduced_coords = adata.obsm[reduction]
    if dims is not None:
        reduced_coords = reduced_coords[:, dims]

    # Run KNN
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='auto')
    nbrs.fit(reduced_coords)
    nn_map = nbrs.kneighbors(return_distance=False)

    # Add self as neighbor
    nn_map = np.column_stack([nn_map, np.arange(len(nn_map))])

    # Greedy seed selection with a shared-cell constraint
    good_choices = np.arange(len(nn_map))
    chosen = []

    if len(good_choices) > 0:
        choice = np.random.choice(len(good_choices), size=1, replace=False)[0]
        chosen = [good_choices[choice]]
        good_choices = good_choices[good_choices != good_choices[choice]]

    it = 0
    k2 = k * 2

    def get_shared(other_idx, this_choice, cell_sample):
        return k2 - len(np.union1d(cell_sample[other_idx], this_choice))

    while len(good_choices) > 0 and len(chosen) < target_metacells and it < max_iter:
        it += 1
        choice = np.random.choice(len(good_choices), size=1, replace=False)[0]
        new_chosen = chosen + [good_choices[choice]]
        good_choices = good_choices[good_choices != good_choices[choice]]

        cell_sample = nn_map[new_chosen]
        this_choice = cell_sample[-1]
        others = np.arange(len(cell_sample) - 1)

        shared = [get_shared(other, this_choice, cell_sample) for other in others]
        if max(shared) <= max_shared:
            chosen = new_chosen

    if len(chosen) <= 1:
        warnings.warn('Metacell construction failed')
        return None

    cell_sample = nn_map[chosen]

    # Overlap statistics between metacells
    combs = list(combinations(range(len(cell_sample)), 2))
    shared = [k2 - len(np.unique(cell_sample[list(comb_idx)])) for comb_idx in combs]

    if verbose:
        print(f"Overlap QC metrics:\n"
              f"Cells per bin: {k}\n"
              f"Max shared cells: {max(shared)}\n"
              f"Mean shared cells: {np.mean(shared):.2f}\n"
              f"Median shared cells: {np.median(shared):.2f}")
        if np.mean(shared) / k > 0.1:
            warnings.warn("More than 10% of cells are shared between metacells on average")

    # Get expression matrix
    if use_raw and adata.raw is not None:
        exprs_matrix = adata.raw.X
    else:
        exprs_matrix = adata.X

    # Build sparse aggregation matrix (cells x metacells)
    n_cells = adata.n_obs
    n_metacells = len(cell_sample)

    row_ind, col_ind, data = [], [], []
    for i, cells in enumerate(cell_sample):
        for cell_idx in cells:
            row_ind.append(cell_idx)
            col_ind.append(i)
            data.append(1.0)

    agg_matrix = sparse.csr_matrix((data, (row_ind, col_ind)), shape=(n_cells, n_metacells))

    # Aggregate expression
    if sparse.issparse(exprs_matrix):
        new_exprs = exprs_matrix.T.dot(agg_matrix)
    else:
        new_exprs = exprs_matrix.T.dot(agg_matrix.toarray())

    if mode == 'average':
        new_exprs = new_exprs / k
    new_exprs = new_exprs.T

    # Create metacell AnnData object
    metacell_obs = pd.DataFrame(index=[f"{name}_{i+1}" for i in range(n_metacells)])
    metacell_obs['cells_merged'] = [','.join(adata.obs_names[cells]) for cells in cell_sample]
    if obs_key and obs_key in adata.obs:
        metacell_obs[obs_key] = adata.obs[obs_key].iloc[0]

    metacell_adata = ad.AnnData(
        X=new_exprs,
        obs=metacell_obs,
        var=adata.var.copy()
    )

    # Add metadata
    metacell_adata.uns['metacell_stats'] = {
        'name': name,
        'max_shared': max(shared),
        'mean_shared': np.mean(shared),
        'median_shared': np.median(shared),
        'n_metacells': n_metacells,
        'k': k
    }
    metacell_adata.uns['cell_sample'] = cell_sample
    metacell_adata.uns['original_cell_indices'] = [
        adata.obs_names[cells].tolist() for cells in cell_sample
    ]

    return metacell_adata


def metacells_by_groups(
    adata: ad.AnnData,
    groupby: Union[str, List[str]],
    k: int = 25,
    reduction: str = 'X_pca',
    dims: Optional[List[int]] = None,
    use_raw: bool = False,
    mode: str = 'average',
    min_cells: int = 100,
    max_shared: int = 15,
    target_metacells: int = 1000,
    max_iter: int = 5000,
    verbose: bool = False
) -> ad.AnnData:
    """
    Construct metacells separately for each group in the data.

    Parameters
    ----------
    adata : AnnData
        Annotated data object.
    groupby : str or list
        Column name(s) in adata.obs to group by (combined into one key).
    k : int
        Number of nearest neighbors to aggregate.
    reduction : str
        Key in adata.obsm used for KNN.
    dims : list, optional
        Dimensions to use from the reduction.
    use_raw : bool
        Whether to use adata.raw for expression data.
    mode : str
        'average' or 'sum' for aggregation.
    min_cells : int
        Minimum cells required per group.
    max_shared : int
        Maximum shared cells between metacells.
    target_metacells : int
        Target number of metacells.
    max_iter : int
        Maximum iterations.
    verbose : bool
        Print progress information.

    Returns
    -------
    AnnData
        Combined metacell AnnData object for all valid groups.
    """
    print("Constructing metacells by group...")

    if isinstance(groupby, str):
        groupby = [groupby]

    # Create combined grouping column
    grouping_col = '_'.join(groupby)
    adata.obs[grouping_col] = adata.obs[groupby].astype(str).agg('_'.join, axis=1)

    # Keep groups with enough cells
    group_counts = adata.obs[grouping_col].value_counts()
    valid_groups = group_counts[group_counts >= min_cells].index.tolist()
    if len(valid_groups) == 0:
        raise ValueError("No groups meet the min_cells requirement")

    metacell_adatas = []
    stats_list = []

    for group in valid_groups:
        if verbose:
            print(f"Processing group: {group}")

        group_adata = adata[adata.obs[grouping_col] == group].copy()

        metacell_adata = construct_metacells(
            group_adata,
            k=k,
            name=group,
            reduction=reduction,
            dims=dims,
            obs_key=grouping_col,
            use_raw=use_raw,
            mode=mode,
            max_shared=max_shared,
            target_metacells=target_metacells,
            max_iter=max_iter,
            verbose=verbose
        )

        if metacell_adata is not None:
            for gb_col in groupby:
                metacell_adata.obs[gb_col] = group_adata.obs[gb_col].iloc[0]
            metacell_adatas.append(metacell_adata)
            stats_list.append(metacell_adata.uns['metacell_stats'])

    if len(metacell_adatas) == 0:
        raise ValueError("No metacells were successfully constructed")

    # Combine all metacell objects
    if len(metacell_adatas) == 1:
        combined_adata = metacell_adatas[0]
    else:
        combined_adata = ad.concat(metacell_adatas, join='outer', index_unique='-')

    combined_adata.uns['metacell_stats'] = pd.DataFrame(stats_list)
    combined_adata.uns['metacell_params'] = {
        'k': k,
        'reduction': reduction,
        'mode': mode,
        'groupby': groupby,
        'min_cells': min_cells
    }

    return combined_adata


# ---------------------------------------------------------------------------
# Metacell processing utilities
# ---------------------------------------------------------------------------
def normalize_metacells(metacell_adata: ad.AnnData, **kwargs) -> ad.AnnData:
    """Normalize metacell data using scanpy's normalize_total and log1p."""
    adata = metacell_adata.copy()
    sc.pp.normalize_total(adata, **kwargs)
    sc.pp.log1p(adata)
    return adata


def scale_metacells(metacell_adata: ad.AnnData, **kwargs) -> ad.AnnData:
    """Scale metacell data."""
    adata = metacell_adata.copy()
    sc.pp.scale(adata, **kwargs)
    return adata


def pca_metacells(metacell_adata: ad.AnnData, **kwargs) -> ad.AnnData:
    """Run PCA on metacell data."""
    adata = metacell_adata.copy()
    sc.tl.pca(adata, **kwargs)
    return adata


def umap_metacells(metacell_adata: ad.AnnData, **kwargs) -> ad.AnnData:
    """Compute UMAP embedding of metacell data."""
    adata = metacell_adata.copy()
    sc.pp.neighbors(adata, **kwargs)
    sc.tl.umap(adata)
    return adata


def extract_protein_coding_genes(gtf_file_path: str) -> set:
    """
    Extract the set of protein-coding gene names from a GTF annotation file.

    Parameters
    ----------
    gtf_file_path : str
        Path to the GTF file (gene_biotype or gene_type attribute).

    Returns
    -------
    set
        Protein-coding gene names.
    """
    protein_coding_genes = set()
    with open(gtf_file_path, 'r') as f:
        for line in f:
            if line.startswith('#'):
                continue
            parts = line.strip().split('\t')
            if len(parts) < 9:
                continue
            if parts[2] != 'gene':
                continue

            gene_info = {}
            for attr in parts[8].split(';'):
                attr = attr.strip()
                if ' ' in attr:
                    key, value = attr.split(' ', 1)
                    gene_info[key] = value.strip('"')

            if (gene_info.get('gene_biotype') == 'protein_coding' or
                    gene_info.get('gene_type') == 'protein_coding'):
                gene_name = gene_info.get('gene_name')
                if gene_name:
                    protein_coding_genes.add(gene_name)

    return protein_coding_genes


# ===========================================================================
# Part 1: Construct metacells per cell type (k = 50, on harmony embedding)
# ===========================================================================
cell_type_list = [
    'COPC', 'VIP+ GABA', 'EC', 'Epc', 'Per', 'HIP Glu',
    '5-HT', 'Ast', 'CALB2+ GABA', 'CP', 'Fib', 'GB', 'Granule', 'HT Glu',
    'IPC', 'L2/3-CUX2 Glu', 'L2/3-SATB2 Glu', 'L4-RORB Glu', 'L5-BCL11B Glu',
    'L5/6-THEMIS Glu', 'L5/6-TLE4 Glu', 'LHX6+LAMP5+ GABA', 'MB GABA',
    'MSN', 'Mac', 'Mic', 'NB', 'OPC', 'Oli', 'PC', 'PVALB+ GABA',
    'RELN+ GABA', 'RG', 'SST+ GABA', 'Th Glu', 'VSMC'
]


def sanitize_cell_type(cell_type_i: str) -> str:
    """Sanitize a cell type name into a valid file-name token."""
    cell_type = re.sub(r'[\\/:*?"<>|+ ]', '_', cell_type_i)
    if cell_type and cell_type[0] in '0123456789!@#$%^&*()-=+':
        cell_type = 'type_' + cell_type
    return cell_type


for cell_type_i in cell_type_list:
    start_time = time.time()

    cell_type = sanitize_cell_type(cell_type_i)
    adata = load_processed_adata(cell_type)

    cell_num = adata.n_obs
    target_metacell = int(cell_num / 50)

    # Construct metacells grouped by cell-type x sample
    metacell_adata = metacells_by_groups(
        adata,
        groupby=['type_2', 'sample_new'],
        k=50,
        use_raw=False,
        reduction='X_harmony',
        min_cells=200,
        mode='average',
        max_shared=20,
        target_metacells=target_metacell,
        max_iter=5000,
        verbose=True
    )

    # Save metacell AnnData
    output_path = os.path.join(METACELL_DIR, f"{cell_type}_metacell_raw_k50.h5ad")
    metacell_adata.write_h5ad(output_path)

    end_time = time.time()
    print(f"Cell type {cell_type} done in {end_time - start_time:.1f}s")
    print("-" * 50)

    del adata, metacell_adata

# ===========================================================================
# Part 2: Merge metacells, normalize, DEG analysis
# ===========================================================================
adatas_list = []
for cell_type_i in cell_type_list:
    cell_type = sanitize_cell_type(cell_type_i)
    file_path = os.path.join(METACELL_DIR, f"{cell_type}_metacell_raw_k50.h5ad")

    if os.path.exists(file_path):
        adata = ad.read_h5ad(file_path)
        adata.obs_names_make_unique()  # avoid index conflicts when concatenating
        adatas_list.append(adata)
    else:
        print(f"Warning: file not found - {file_path}")

# Merge all metacell objects
merged_adata = ad.concat(adatas_list, join='outer')

# Store as sparse float32 to save memory
if not sparse.issparse(merged_adata.X):
    merged_adata.X = sparse.csr_matrix(merged_adata.X, dtype=np.float32)
elif merged_adata.X.dtype != np.float32:
    merged_adata.X = merged_adata.X.astype(np.float32)

merged_adata.write_h5ad(os.path.join(METACELL_DIR, "human_all_celltypes_metacell_raw_k50.h5ad"))

# Normalize metacell expression
merged_adata.obs = merged_adata.obs.drop(columns=['cells_merged', 'type_2_sample_new'])
merged_adata.raw = merged_adata
sc.pp.normalize_total(merged_adata, target_sum=1e4)
sc.pp.log1p(merged_adata)

# Drop low-representation cell types
excluded_types = [
    "Serotonergic", "Vascular_smooth_muscle_cell", "Macrophage",
    "Choroid_plexus_epithelial", "Pericyte", "Ependymal"
]
merged_adata = merged_adata[~merged_adata.obs['type_2'].isin(excluded_types), :]

merged_adata.write_h5ad(os.path.join(METACELL_DIR, "human_all_celltypes_metacell_norm_k100.h5ad"))

# Merge with sample metadata (stage, region, etc.)
meta_path = os.path.join(ANNOTATION_DIR, "final", "hs_all_metadata_update.csv")
meta = pd.read_csv(meta_path, index_col=0)
meta = meta[meta["type_2"] != "Remove"]
meta['stage_abbr'] = 'S' + meta['stage'].str.split('-').str[0]
meta = meta[~meta["dataset"].isin(["hs09", "hs11", "hs12", "hs14"])]
meta.rename(columns={'sample': 'samples'}, inplace=True)
meta = meta.drop(columns=["type_group", "type_1", "type_2", 'type_2_abbr', "type_3"])
meta_unique = meta.drop_duplicates(subset='samples', keep='first')

obs = merged_adata.obs
obs.rename(columns={'sample_new': 'samples'}, inplace=True)
obs_merged = obs.merge(
    meta_unique,
    on='samples',
    how='left',
    suffixes=('', '_meta')
)

# Export merged metacell matrix for downstream R analysis
r_dir = os.path.join(METACELL_DIR, "R_k100")
os.makedirs(r_dir, exist_ok=True)
scipy.io.mmwrite(os.path.join(r_dir, "matrix.mtx"), merged_adata.X.T)
obs_merged.to_csv(os.path.join(r_dir, "cells.csv"))
merged_adata.var.to_csv(os.path.join(r_dir, "genes.csv"))

# ---------------------------------------------------------------------------
# Differential expression analysis per cell type (wilcoxon, Bonferroni)
# ---------------------------------------------------------------------------
sc.tl.rank_genes_groups(
    merged_adata,
    groupby="type_2",
    method="wilcoxon",
    use_raw=False,
    pts=True,  # compute fraction of cells expressing each gene per cluster
    corr_method="bonferroni",
)

result = merged_adata.uns["rank_genes_groups"]
groups = result["names"].dtype.names
dfs_list = []

for cluster in groups:
    df = pd.DataFrame({
        "gene": result["names"][cluster],
        "log2fc": result["logfoldchanges"][cluster],
        "pvals": result["pvals"][cluster],
        "qvals": result["pvals_adj"][cluster],
        "expr_frac": result["pts"][cluster]
    })
    df["cluster"] = cluster
    dfs_list.append(df)

merged_df = pd.concat(dfs_list, axis=0, ignore_index=True)
merged_df.sort_values(by=["cluster", "log2fc"], ascending=[True, False], inplace=True)
merged_df.to_csv(os.path.join(METACELL_DIR, "deg_type2_metacell_k100.csv"), index=False)


def filter_deg_top_half(group: pd.DataFrame) -> pd.DataFrame:
    """Apply significance filters and keep the top 50% by log2fc per cluster."""
    filtered = group[
        (group['qvals'] < 0.05) &
        (group['expr_frac'] >= 0.2) &
        (group['log2fc'] > 0)
    ].copy()

    if filtered.empty:
        return filtered

    filtered_sorted = filtered.sort_values('log2fc', ascending=False)
    n_top_genes = max(1, int(len(filtered_sorted) * 0.5))
    return filtered_sorted.head(n_top_genes)


final_filtered_df = merged_df.groupby('cluster').apply(filter_deg_top_half).reset_index(drop=True)
final_filtered_df.sort_values(by=['cluster', 'log2fc'], ascending=[True, False], inplace=True)
final_filtered_df.to_csv(os.path.join(METACELL_DIR, "deg_type2_metacell_k100_top05.csv"), index=False)

# ---------------------------------------------------------------------------
# Restrict DEGs to protein-coding genes
# ---------------------------------------------------------------------------
protein_coding_genes = extract_protein_coding_genes(GTF_FILE)
print(f"Found {len(protein_coding_genes)} protein-coding genes")

sig_pc = merged_df[
    (merged_df['qvals'] <= 0.05) &
    (merged_df['expr_frac'] >= 0.2) &
    (merged_df['log2fc'] > 0) &
    merged_df['gene'].isin(protein_coding_genes)
]
sig_pc.to_csv(os.path.join(METACELL_DIR, "deg_type2_metacell_k100_sig_pc.csv"), index=False)

# ---------------------------------------------------------------------------
# Microglia functional-cluster marker genes, restricted to protein coding
# ---------------------------------------------------------------------------
mic_marker_file = os.path.join(
    BASE_DIR, "Analysis", "immune", "validation", "Mic", "mic_functional_cluster6_marker_new.csv"
)
mic_deg = pd.read_csv(mic_marker_file, index_col=0)
mic_deg_pc = mic_deg[mic_deg['gene'].isin(protein_coding_genes)]


def filter_mic_deg(group: pd.DataFrame) -> pd.DataFrame:
    """Apply significance filters to microglia functional-cluster markers."""
    return group[
        (group['p_val_adj'] < 0.05) &
        (group['avg_log2FC'] > 0) &
        (group['pct.1'] >= 0.1)
    ]


mic_filtered = mic_deg_pc.groupby('cluster').apply(filter_mic_deg).reset_index(drop=True)
mic_filtered.sort_values(by=['cluster', 'avg_log2FC'], ascending=[True, False], inplace=True)
deg_dir = os.path.join(BASE_DIR, "Analysis", "disease", "deg_files")
os.makedirs(deg_dir, exist_ok=True)
mic_filtered.to_csv(os.path.join(deg_dir, "deg_mic_func6_new.csv"), index=False)

# ===========================================================================
# Part 3: Harmony integration of metacell expression
# ===========================================================================
sc.pp.highly_variable_genes(merged_adata, n_top_genes=2000, flavor="seurat")
merged_adata = merged_adata[:, merged_adata.var.highly_variable]
# total_counts is required by regress_out below but was never computed for the
# metacell object (metacell obs only carries type_2/sample annotations)
sc.pp.calculate_qc_metrics(merged_adata, percent_top=None, log1p=False, inplace=True)
sc.pp.regress_out(merged_adata, keys=["total_counts"], n_jobs=8)
sc.pp.scale(merged_adata)
sc.tl.pca(merged_adata, svd_solver="arpack", n_comps=50)

sc.external.pp.harmony_integrate(
    merged_adata, key="samples", basis="X_pca", max_iter_harmony=30
)

sc.pp.neighbors(merged_adata, n_pcs=50, n_neighbors=30, use_rep='X_pca_harmony')
# sc.pl.paga(plot=False) only computes the PAGA layout used for UMAP
# initialization below; no figure is rendered.
sc.tl.paga(merged_adata, groups='type_2')
sc.pl.paga(merged_adata, plot=False)
sc.tl.umap(merged_adata, init_pos='paga', maxiter=1000)

merged_adata.write_h5ad(os.path.join(METACELL_DIR, "human_all_celltypes_metacell_inter_k100.h5ad"))

# ===========================================================================
# Part 4: scVI / scanVI latent integration
# ===========================================================================
merged_adata.obs['sample_new_encoded'] = pd.Categorical(
    merged_adata.obs['samples']  # renamed from 'sample_new' above
).codes.astype(float).astype("category")


def compute_scvi_model(
    adata: sc.AnnData,
    n_latent: int = 5,
    n_epochs: int = 100,
    lr: float = 1e-3,
    batch_key: str = "sample_new_encoded",
    use_batches: bool = True,
    use_cuda: bool = True,
) -> Tuple[scvi.model.SCVI, np.ndarray]:
    """Train and return a scVI model and its latent representation."""
    device = "cuda" if use_cuda else "cpu"

    scvi.model.SCVI.setup_anndata(
        adata,
        batch_key=batch_key if use_batches else None
    )

    model = scvi.model.SCVI(
        adata,
        n_latent=n_latent,
        encode_covariates=True,
        dropout_rate=0.2,
    )

    model.train(
        max_epochs=n_epochs,
        plan_kwargs={"lr": lr},
    )

    latent = model.get_latent_representation()
    return model, latent


use_cuda = True
scvi_model, scvi_latent = compute_scvi_model(
    merged_adata,
    n_epochs=100,
    n_latent=50,
    use_cuda=use_cuda,
    batch_key="sample_new_encoded"
)
scvi_model.save(os.path.join(SCVI_DIR, "scvi"), overwrite=True)
np.savetxt(
    os.path.join(SCVI_DIR, "metacell_k100_hvg_scvi_latent50_epoch100.csv"),
    scvi_latent,
    delimiter=','
)

# Semi-supervised fine-tuning with scanVI
scanvi_model = scvi.model.SCANVI.from_scvi_model(
    scvi_model,
    adata=merged_adata,
    labels_key="type_2",
    unlabeled_category="Unknown"
)
scanvi_model.train(
    max_epochs=20,
    gradient_clip_val=1.0,
    n_samples_per_label=100
)

SCANVI_LATENT_KEY = "X_scANVI"
merged_adata.obsm[SCANVI_LATENT_KEY] = scanvi_model.get_latent_representation(merged_adata)
np.savetxt(
    os.path.join(SCVI_DIR, "metacell_k100_hvg_scvi_latent50_epoch100_scanvi_epoch20.csv"),
    merged_adata.obsm[SCANVI_LATENT_KEY],
    delimiter=','
)

# Save scanVI-based UMAP coordinates for downstream analysis
sc.pp.neighbors(merged_adata, use_rep=SCANVI_LATENT_KEY, n_neighbors=30, n_pcs=50)
sc.tl.umap(merged_adata)

umap_data = pd.DataFrame(
    data=merged_adata.obsm["X_umap"],
    index=merged_adata.obs_names,
    columns=["UMAP_1", "UMAP_2"]
)
umap_data.to_csv(os.path.join(SCVI_DIR, "umap_data_hvg_scvi100_scanvi20.csv"))
