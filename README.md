# UniBrain: A Spatiotemporal Single-Cell Atlas of the Human Brain Across the Lifespan

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org/)
[![R](https://img.shields.io/badge/R-4.2%2B-276DC3)](https://www.r-project.org)

Analysis pipeline for constructing a comprehensive single-cell transcriptomic atlas of the developing human brain, spanning from early gestation to late adulthood. The project integrates 30+ public scRNA-seq datasets, performs hierarchical cell type annotation across five resolution levels, and conducts systematic downstream analyses including lineage tracing, gene regulatory network inference, cell–cell communication, disease heritability enrichment, and an LLM-based brain atlas agent (UniBrain).

## Repository Structure

```
.
├── 00_reference/                                   # Reference metadata
│   ├── cell_type_description.json                  # Cell type nomenclature (L1–L5)
│   └── brain_region_L2_description.json            # Brain region hierarchy
│
├── 01_data_preprocessing/                          # Raw data → Seurat/AnnData
│   └── 01_rd2mtx.R                                 # RDS → 10X-format MTX conversion
│
├── 02_qc/                                          # Quality control
│   ├── 01_qc_doublets.py                           # Scrublet doublet detection
│   ├── 02_qc_merge.py                              # Metadata assembly + merge
│   └── 03_qc_threshold.py                          # MAD-based per-sample filtering
│
├── 03_integration/                                 # Batch integration
│   ├── 01_integrate_preprocess.py                  # HVG selection, regress-out, PCA
│   ├── 02_integrate_harmony.py                     # Harmony integration
│   ├── 03_integrate_scanorama.py                   # Scanorama integration
│   ├── 04_integrate_scalex.py                      # SCALEX integration
│   ├── 05_integrate_scvi.py                        # scVI variational autoencoder
│   └── 06_integrate_evaluation.py                  # iLISI / batch-silhouette evaluation
│
├── 04_umap/                                        # Dimensionality reduction
│   ├── 01_run_umap.py                              # UMAP for all integration methods
│   └── 02_save_umap_data.py                        # Assemble final UMAP AnnData
│
├── 05_annotation/                                  # Hierarchical cell type annotation
│   ├── 01_L1_leiden_cluster.py                     # L1 coarse clustering (multi-resolution)
│   ├── 02_L1_annotation.py                         # L1 broad category annotation
│   ├── 03-06_L2_*_step1.py                        # L2 subset clustering (glia/progenitor/neuron/others)
│   ├── 07-10_L2_*_step2.py                        # L2 differential expression + annotation
│   ├── 11-12_L3_GABA_Glu_step1.py                 # L3 GABAergic/Glutamatergic clustering
│   ├── 13-14_L3_GABA_Glu_step2.py                 # L3 annotation refinement
│   ├── 15_L4_subcluster_integration.py             # L4 per-lineage integration
│   ├── 16_L4_cluster.py                            # L4 consensus clustering
│   ├── 17_L4_tree.py                               # L4 hierarchical tree construction
│   ├── 18_L5_cluster.py                            # L5 fine clustering
│   ├── 19_L5_tree.py                               # L5 hierarchical tree
│   └── 20_final_anndata.py                         # Final atlas AnnData assembly
│
├── 06_cellular phylogeny & regulatory factors/     # Downstream analysis: lineage & gene regulation
│   ├── 01_lineage_time.py                          # Temporal lineage tree (KNN bootstrap)
│   ├── 02_lineage_regulator.py                     # RBP / TF / combined regulator screening
│   └── 03_hippo_npc.py                             # Hippocampal NPC composition analysis
│
├── 07_cell cell communication/                     # Downstream analysis: intercellular signaling
│   ├── 01_cellchat.R                               # CellChat ligand–receptor inference
│   └── 02_hormone2cell.py                          # Hormone → cell type receptor mapping
│
├── 08_functional subtypes/                         # Downstream analysis: immune functional subtypes
│   ├── 01_immune_macrophage.R                      # Macrophage DE + functional annotation
│   ├── 02_immune_microglia.R                       # Microglia DE + enrichment + clustering
│   └── 03_wgcna.R                                  # hdWGCNA co-expression per cell type
│
├── 09_heritability enrichment/                     # Downstream analysis: disease heritability & drug prioritization
│   ├── 01_metacell_construct.py                    # Metacell construction + DEG analysis
│   ├── 02_disease_scrna.py                         # Cell-type specificity for GWAS traits
│   ├── 03_sldsc_analysis.R                         # Stratified LD score regression
│   └── 04_drug2cell.py                             # Drug–cell type target prediction
│
├── 10_uniBrain agent/                              # Generative brain atlas model
│   ├── 00_vqvae_module.py                          # Shared VQVAE components (imported by 01/02)
│   ├── 01_inference.py                             # VQVAE encoding + LLM inference pipeline
│   ├── 02_vqvae.py                                 # VQVAE training (tokenizing gene expression)
│   ├── 03_corpus_construction.py                   # Build Q&A + single-cell training corpus
│   ├── 04_update_tokenizer.py                      # Add VQ code tokens to LLM tokenizer
│   ├── 05_lora_config.yaml                         # LoRA fine-tuning configuration
│   ├── 06_marker_gene_trajectory.py                # Marker gene trajectory validation
│   └── 07_data_completion.py                       # Atlas coverage matrix computation
│
└── 11_figures/                                     # Main-figure generation scripts
    ├── 01_Fig1.R                                   # Fig1: circos overview + circle-UMAP
    ├── 02_Fig2.py                                  # Fig2: lineage tree + regulator barplot
    ├── 03_Fig3.ipynb                               # Fig3: CellChat A–F (R) + hormone heatmap G (Python)
    ├── 04_Fig4.ipynb                               # Fig4: microglia UMAP A (Python) + B–F WGCNA/heatmaps (R)
    ├── 05_Fig5.ipynb                               # Fig5: SLDSC heatmaps A–B (R) + drug dotplot C (Python)
    └── 06_Fig6.py                                  # Fig6: UniBrain coverage/confusion/dotplot/trajectory
```

## Pipeline Overview

### 1. Data Preprocessing

Raw single-cell data from public repositories (GEO, EBI, etc.) are converted to a unified 10X-format directory structure: `matrix.mtx`, `genes.tsv`, `barcodes.tsv`. Each dataset (`hsXX` / `mmXX`) is assigned standardized metadata (species, brain region, developmental stage, sex, diagnosis).

### 2. Quality Control

Doublets are detected per sample using Scrublet. Samples are merged with unified region/platform/dataset mapping tables. Cells failing MAD-based thresholds (UMI count, gene count, mitochondrial fraction, doublet score) are removed per sample.

### 3. Batch Integration

After normalization (10K UMI target, log1p), highly variable gene selection (top 5,000 protein-coding genes), cell-cycle regression, and PCA, multiple integration methods are applied and evaluated:

| Method    | Type                    | Evaluation Metric       |
| --------- | ----------------------- | ----------------------- |
| Harmony   | Linear correction       | iLISI, batch silhouette |
| Scanorama | Panoramic stitching     | iLISI, batch silhouette |
| SCALEX    | Deep autoencoder        | iLISI, batch silhouette |
| scVI      | Variational autoencoder | iLISI (on latent space) |

### 4. Hierarchical Annotation (L1 → L5)

UniBrain uses a five-level ontology that combines coarse ontology-based labels with iterative, consensus clustering at finer levels:

- **L1 — cell category**: four broad groups — glia, neural precursor cell, neuron, and non-neural cell.
- **L2 — cell superclass**: 14 superclasses — astroependymal cells, oligodendrocytes, glial precursors, neuronal precursors, radial glia, glutamatergic neurons, GABAergic neurons, other neurons, immune cells, fibroblasts, endothelial cells, pericytes, choroid plexus epithelial cells, and vascular smooth muscle cells.
- **L3 — cell class**: 38 classes, resolving the L2 superclasses (for example, astrocytes versus ependymal cells; glioblasts, OPCs, and COPCs; and canonical glutamatergic and GABAergic neuronal classes).
- **L4 — cell type**: 766 fine-grained cell types obtained from lineage-specific consensus Leiden clustering and hierarchical refinement.
- **L5 — cell population**: 5,818 cell populations, named using lineage, marker-gene expression, and anatomical context.

For L4–L5, clustering is performed in the Harmony embedding, with resolution selection, 100-run consensus clustering, hierarchical-tree construction, Wilcoxon marker testing, and iterative pruning based on marker support, cluster size, and donor composition.

### 5. Downstream Analyses

- **Cellular phylogeny and regulatory factors**: reconstruct a stage-resolved phylogeny of 423 cell states across 38 cell classes and 15 developmental stages, then prioritize TFs and RBPs associated with cell-class specification.
- **Cell–cell communication**: infer CellChat communication networks for the 12 most abundant cell classes in prenatal, postnatal-developmental, and adult/aged phases; characterize glutamate ligand–receptor interactions and hormone-mediated producer–receiver signaling across 15 stages.
- **Functional diversification of brain-resident immune cells**: group 17 microglial cell types into six functional subtypes and define six macrophage functional subtypes; assess their spatiotemporal enrichment, differential expression, and functional programmes.
- **Gene co-expression programmes**: identify WGCNA modules in microglia and additional cell classes, and relate module activity to functional subtype and developmental dynamics.
- **Disease heritability and drug prioritization**: construct metacells, perform partitioned s-LDSC heritability enrichment across 23 brain-related traits, and use drug2cell to prioritize clinically approved, CNS-accessible candidate drugs for the enhanced-redox microglial subtype.

### 6. UniBrain Agent

A VQVAE + LLM framework that tokenizes single-cell gene expression into discrete codes and fine-tunes a language model (Qwen3) to:

- **Describe** cell types from expression profiles
- **Generate** synthetic single-cell data for unseen (stage, region, cell type) combinations
- **Complete** the developing brain atlas by predicting missing tissue compartments

### 7. Figures

Main-figure scripts read pre-computed results and generate publication-ready PDFs:

| Figure | Panels | Content                                                                                                                                                                               | Format               |
| ------ | ------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------- |
| Fig1   | —     | Atlas UMAP (L3 cell classes) with concentric tracks for L1–L5 annotations, dominant stage, brain region, donor sex, and population count                                             | R                    |
| Fig2   | A–C   | Cross-stage cellular phylogeny; candidate TF/RBP counts by cell class; phylogeny annotated with top candidate TFs                                                                     | Python               |
| Fig3   | A–G   | Phase-specific cell–cell communication networks; pathway information flow; glutamate ligand–receptor and sender–receiver interactions; lifespan ligand–receptor signaling heatmap | Jupyter (R + Python) |
| Fig4   | A–F   | Microglial-type UMAP; functional-subtype composition and regional enrichment; subtype DEGs/GO terms; WGCNA module–subtype associations and developmental trajectories                | Jupyter (R + Python) |
| Fig5   | A–C   | Cell-class s-LDSC, regional, and developmental-stage enrichment; microglial-subtype enrichment for AD/PD/IS; candidate drugs and target genes                                         | Jupyter (R + Python) |
| Fig6   | A–F   | UniBrain Agent architecture; atlas coverage completion; superclass-annotation accuracy; generated hippocampal expression, UMAP, and lifelong microglial trajectories                  | Python               |

## Dependencies

### Python (≥3.9)

```
scanpy, anndata, scvi-tools, scrublet, harmonypy, scalex, scanorama
numpy, scipy, pandas, scikit-learn, matplotlib, seaborn
torch (for scVI / UniBrain), tqdm, jupyter
```

### R (≥4.2)

```
Seurat (v5), hdWGCNA, WGCNA, CellChat, ComplexHeatmap
enrichR, UCell, dplyr, tidyverse, ggplot2, patchwork, data.table, Matrix
IRkernel (for Jupyter R cells)
```

### Create the Scanpy environment

Create and activate the tested Scanpy environment from the repository root:

```bash
conda env create -f conda-environment.yml
conda activate unibrain
```

## Usage

1. **Set the project root:**

   ```bash
   export PROJ_DIR=/path/to/your/project
   ```
2. **Run sequentially** within each directory (files are numbered by execution order):

   ```bash
   cd 02_qc
   python 01_qc_doublets.py
   python 02_qc_merge.py
   python 03_qc_threshold.py
   ```
3. **Intermediate outputs** (`.h5ad`, `.csv`, `.RData`) are saved under `$PROJ_DIR/data/human/output/`. Reference files (GTF annotations, gene lists) should be placed under `$PROJ_DIR/data/ref/`.

## Configuration

Every script includes a configuration block at the top:

```python
proj_dir = os.environ.get("PROJ_DIR", "path/to/your/project")
```

Set the `PROJ_DIR` environment variable or edit this line to point to your project root before running.

GPU selection is controlled outside the Python scripts. For example, on a
login shell use `CUDA_VISIBLE_DEVICES`, or request GPUs through the cluster
scheduler. `UNIBRAIN_DEVICE` selects a logical PyTorch device and defaults to
`auto`:

```bash
export CUDA_VISIBLE_DEVICES=0
export UNIBRAIN_DEVICE=auto       # or cuda:0 / cpu
python "10_uniBrain agent/02_vqvae.py"
```

## Data and model availability

Raw data, intermediate analysis objects, generated figures, and model weights are not included in this repository. Place downloaded source data under `$PROJ_DIR/data/`; scripts write derived outputs below `$PROJ_DIR/data/human/output/`. The repository `.gitignore` excludes these large or local artifacts, together with local environment files.

## Citation

If you use this code in your research, please cite the corresponding publication (in preparation) and the original data sources listed in `00_reference/`.

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
