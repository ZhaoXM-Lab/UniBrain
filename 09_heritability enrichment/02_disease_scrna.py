#!/usr/bin/env python3
"""
=============================================================================
File      : 02_disease_scrna.py
Directory : 09_heritability enrichment
Purpose   : Compute cell-type-specific gene specificity scores and generate LDSC-ready BED tracks.
=============================================================================
"""

import os

import numpy as np
import pandas as pd

## Configuration ##
# Set the base directory for input/output data below
# proj_dir must point to the data root <project>/data/human
BASE_DIR = os.environ.get("PROJ_DIR", "path/to/your/project")  # <-- MODIFY THIS
# reference files live under <project>/data/ref
REF_DIR = os.path.join(BASE_DIR, "..", "ref")

# gene_coordinates_v42.bed was pre-extracted from gencode.v42.annotation.gtf
GENE_COORDS_FILE = os.path.join(REF_DIR, "gene_coordinates_v42.bed")

DEG_DIR = os.path.join(BASE_DIR, "output", "Analysis", "disease", "deg_files")
LDSC_DIR = os.path.join(BASE_DIR, "output", "Analysis", "disease", "ldsc_files")

# Cell-type DEG results (microglia functional clusters, metacell-based)
DEG_FILE = os.path.join(DEG_DIR, "deg_mic_func6_new.csv")
SCORES_SIG_FILE = os.path.join(DEG_DIR, "gene_specificity_scores_mic_sig.csv")
SCORES_FUNC6_FILE = os.path.join(DEG_DIR, "gene_specificity_scores_mic_func6_new.csv")

# Output directories for LDSC BED tracks
BEDS_FUNC6_DIR = os.path.join(LDSC_DIR, "celltype_mic_beds_func6_new")
BEDS_SIG_DIR = os.path.join(LDSC_DIR, "celltype_mic_beds_sig_50kb")


def calculate_gene_scores(df):
    """Compute a cell-type-specific gene specificity score.

    For each cluster the raw evidence score is -2*log(adjusted p-value);
    non-positive adjusted p-values are replaced by the smallest positive
    value within the cluster. Scores are min-max normalized to 0-1, so genes
    with stronger differential-expression evidence get scores closer to 1.

    Args:
        df: DataFrame with columns 'gene', 'cluster' and 'p_val_adj'.

    Returns:
        The input DataFrame with an added 'gene_score' column.
    """
    results = []
    clusters = df['cluster'].unique()

    for cluster in clusters:
        cluster_genes = df[df['cluster'] == cluster].copy()

        # Raw evidence score: -2*log(adjusted p-value)
        valid_qvals = cluster_genes['p_val_adj'].copy()
        min_nonzero = valid_qvals[valid_qvals > 0].min() if any(valid_qvals > 0) else 1e-300
        valid_qvals[valid_qvals <= 0] = min_nonzero
        cluster_genes['neg2logq'] = -2 * np.log(valid_qvals)

        # Min-max normalization to 0-1
        if len(cluster_genes) > 0 and not cluster_genes['neg2logq'].isnull().all():
            min_val = cluster_genes['neg2logq'].min()
            max_val = cluster_genes['neg2logq'].max()
            if max_val == min_val:
                cluster_genes['gene_score'] = 0.5
            else:
                cluster_genes['gene_score'] = (cluster_genes['neg2logq'] - min_val) / (max_val - min_val)
        else:
            cluster_genes['gene_score'] = 0.0

        results.append(cluster_genes)

    return pd.concat(results, ignore_index=True)


def create_celltype_bed_files(gene_scores_file, gene_coords_file, output_dir):
    """Create one BED file per cell type with gene-score tracks.

    Gene coordinates are extended to +/-50 kb around each gene (100 kb total),
    matching the format of the reference psychoENCODE enhancer BED file.

    Args:
        gene_scores_file: CSV with gene/cluster/gene_score columns.
        gene_coords_file: TSV with gene/chr/start/end (no header).
        output_dir: Directory for the generated BED files.

    Returns:
        Array of cell types for which BED files were created.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Load gene scores and gene coordinates
    gene_scores = pd.read_csv(gene_scores_file)
    gene_coords = pd.read_csv(gene_coords_file, sep='\t',
                             names=['gene', 'chr', 'start', 'end'],
                             header=None)

    # Merge scores with coordinates
    merged = gene_scores.merge(gene_coords, on='gene', how='inner')

    # Create one BED file per cell type
    cell_types = merged['cluster'].unique()

    for cell_type in cell_types:
        safe_cell_type = str(cell_type).replace('/', '.').replace('+', '.').replace(' ', '_').replace('\n', '_')
        cell_data = merged[merged['cluster'] == cell_type].copy()

        # Extend gene region to +/-50 kb (100 kb total)
        cell_data['start_ext'] = np.maximum(0, cell_data['start'] - 50000)
        cell_data['end_ext'] = cell_data['end'] + 50000

        # Write BED: chr, start, end, score
        bed_data = cell_data[['chr', 'start_ext', 'end_ext', 'gene_score']]
        bed_data.columns = ['chr', 'start', 'end', 'score']

        bed_file = os.path.join(output_dir, f"{safe_cell_type}.bed")
        bed_data.to_csv(bed_file, sep='\t', index=False, header=False)
        print(f"Created {bed_file} with {len(bed_data)} regions")

    return cell_types


def create_control_bed_file(gene_scores_file, gene_coords_file, output_dir):
    """Create a control BED file for the union of all cell types.

    For each gene the maximum gene score across all cell types is used.

    Args:
        gene_scores_file: CSV with gene/cluster/gene_score columns.
        gene_coords_file: TSV with gene/chr/start/end (no header).
        output_dir: Directory for the generated control BED file.

    Returns:
        Path to the generated control BED file.
    """
    gene_scores = pd.read_csv(gene_scores_file)
    gene_coords = pd.read_csv(gene_coords_file, sep='\t',
                             names=['gene', 'chr', 'start', 'end'],
                             header=None)

    merged = gene_scores.merge(gene_coords, on='gene', how='inner')

    # Take the maximum score per gene across all cell types
    max_scores = merged.groupby('gene').agg({
        'gene_score': 'max',
        'chr': 'first',
        'start': 'first',
        'end': 'first'
    }).reset_index()

    # Extend gene region to +/-50 kb (100 kb total)
    max_scores['start_ext'] = np.maximum(0, max_scores['start'] - 50000)
    max_scores['end_ext'] = max_scores['end'] + 50000

    bed_data = max_scores[['chr', 'start_ext', 'end_ext', 'gene_score']]
    bed_data.columns = ['chr', 'start', 'end', 'score']

    control_file = os.path.join(output_dir, "control.bed")
    bed_data.to_csv(control_file, sep='\t', index=False, header=False)
    print(f"Created control file: {control_file}")

    return control_file


# -----------------------------------------------------------------------------
# Main workflow
# -----------------------------------------------------------------------------
# Load cell-type DEG results (microglia functional clusters, metacell log2FC)
deg_df = pd.read_csv(DEG_FILE)

# Compute cell-type-specific gene specificity scores
gene_scores_df = calculate_gene_scores(deg_df)
# The DEG input (deg_mic_func6_new.csv) follows the Seurat-style schema
# (avg_log2FC / p_val / p_val_adj); the old 'log2fc/pvals/qvals' column names
# do not exist in this table.
gene_scores_df[['gene', 'cluster', 'gene_score', 'avg_log2FC', 'p_val', 'p_val_adj']].to_csv(SCORES_SIG_FILE, index=False)
gene_scores_df[['gene', 'cluster', 'gene_score', 'avg_log2FC', 'p_val', 'p_val_adj']].to_csv(SCORES_FUNC6_FILE, index=False)

# Create cell-type-specific BED files for S-LDSC
create_celltype_bed_files(
    SCORES_FUNC6_FILE,
    GENE_COORDS_FILE,
    BEDS_FUNC6_DIR
)

# Create control BED file (union of all cell types)
create_control_bed_file(
    SCORES_SIG_FILE,
    GENE_COORDS_FILE,
    BEDS_SIG_DIR
)
