#!/usr/bin/env python3
"""
=============================================================================
File      : 01_qc_doublets.py
Directory : 02_qc
Purpose   : Run Scrublet doublet detection on every sample directory and save per-cell doublet scores and predictions.
=============================================================================
"""

import os
import random

import numpy as np
import pandas as pd
import scipy.io as io
import scrublet as scr

# ============================================================================
# Configuration: Set the project root directory before running
# ============================================================================
proj_dir = "/path/to/your/project"  # <-- MODIFY THIS
qc_dir = os.path.join(proj_dir, "data", "human", "qc")

random.seed(0)
np.random.seed(0)

os.chdir(path=qc_dir)

# List all sample directories
dir_list = []
for file_i in os.listdir():
    dir_i = os.path.join(qc_dir, file_i)
    if os.path.isdir(dir_i):
        dir_list.append(file_i)

dir_list.sort(reverse=True)

# Run Scrublet doublet detection on each sample
for dir_i in dir_list:
    counts_i = io.mmread(
        os.path.join(qc_dir, dir_i, "matrix.mtx")
    ).T.tocsc()
    genes_i = np.array(
        scr.load_genes(
            os.path.join(qc_dir, dir_i, "genes.tsv"),
            delimiter="\t", column=1
        )
    )
    cells_i = pd.read_csv(
        os.path.join(qc_dir, dir_i, "barcodes.tsv"),
        header=None
    )
    scrub_i = scr.Scrublet(
        counts_i, expected_doublet_rate=0.05
    )
    doublet_scores, predicted_doublets = scrub_i.scrub_doublets(
        min_counts=2, min_cells=3,
        min_gene_variability_pctl=85, n_prin_comps=30
    )
    results_i = pd.DataFrame(
        {
            "doublet_score": doublet_scores,
            "doublet_predict": predicted_doublets
        },
        index=cells_i.values[:, 0]
    )
    results_i.to_csv(
        os.path.join(qc_dir, dir_i, "doublets.csv"),
        sep="\t", header=False
    )
    print(dir_i + " finished!\n")
