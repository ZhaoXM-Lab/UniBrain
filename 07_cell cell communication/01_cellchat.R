#!/usr/bin/env Rscript
# =============================================================================
# File      : 01_cellchat.R
# Directory : 07_cell cell communication
# Purpose   : Infer cell-cell communication from metacell expression data using CellChat.
# =============================================================================

## Configuration ##
# Set the base directory for input/output data below.
# proj_dir must point to the UniBrain project root
BASE_DIR <- Sys.getenv("PROJ_DIR", unset = "path/to/your/project")  # <-- MODIFY THIS

# Input: metacell count matrix (matrix.mtx) with gene/cell annotations
DATA_DIR <- file.path(BASE_DIR, "data/human/output/Analysis/metacell/R_k100")

# Output: CellChat objects
OUTPUT_DIR <- file.path(BASE_DIR, "data/human/output/Analysis/ccc/all_cell")
PERIOD_OUTDIR <- file.path(OUTPUT_DIR, "by_phase")   # per period, grouped by type_2
PHASE_OUTDIR <- file.path(OUTPUT_DIR, "superclass")  # per phase, grouped by superclass

## Libraries ##
library(CellChat)
library(data.table)
library(Matrix)

# Enable parallel computation and raise the size limit for future globals
options(future.globals.maxSize = 40 * 1024^3)
future::plan("multisession", workers = 8)

## Load metacell count matrix ------------------------------------------------
matrix_data <- fread(file.path(DATA_DIR, "matrix.mtx"), skip = 2, header = FALSE)
genes <- read.csv(file.path(DATA_DIR, "genes.csv"), row.names = 1)
cells <- read.csv(file.path(DATA_DIR, "cells.csv"), row.names = 1)

n_genes <- as.numeric(matrix_data[1, "V1"])
n_cells <- as.numeric(matrix_data[1, "V2"])
matrix_data <- matrix_data[-1, ]
counts <- sparseMatrix(i = matrix_data$V1,
                       j = matrix_data$V2,
                       x = matrix_data$V3,
                       dims = c(n_genes, n_cells))
rownames(counts) <- rownames(genes)
colnames(counts) <- rownames(cells)
cells$samples <- as.factor(cells$samples)

## Filter cells with missing annotations --------------------------------------
na_rows <- which(apply(cells, 1, function(x) any(is.na(x))))
if (length(na_rows) > 0) {
  message("Removing ", length(na_rows), " cells with NA annotations")
  cells <- cells[complete.cases(cells), ]
  counts <- counts[, rownames(cells)]
} else {
  message("No cells with NA annotations found")
}

## Cell annotations: superclass, developmental period and phase ----------------
# Map fine-grained cell types (type_2) to lineage superclasses
superclass_mapping <- list(
  'Glutamatergic neuron' = c('L2/3-CUX2 Glu', 'L2/3-SATB2 Glu', 'L4-RORB Glu',
                             'L5-BCL11B Glu', 'L5/6-TLE4 Glu', 'L5/6-THEMIS Glu',
                             'Thalamic_Glu', 'Hippocampal_Glu', 'Granule_neuron',
                             'Hypothalamic_Glu'),
  'GABAergic neuron' = c('PVALB+ GABA', 'SST+ GABA', 'LHX6+LAMP5+ GABA',
                         'CALB2+ GABA', 'VIP+ GABA', 'RELN+ GABA',
                         'Medium_spiny_neuron', 'Purkinje_neuron',
                         'Midbrain-derived_GABA'),
  'Astrocyte' = 'Astrocyte',
  'Oligodendrocyte' = 'Oligodendrocyte',
  'Microglia' = 'Microglia',
  'Fibroblast' = 'Fibroblast',
  'Endothelial cell' = 'Endothelial_cell',
  'Glial precursor' = c('Glioblast', 'Oligodendrocyte_precursor_committed', 'Oligodendrocyte_precursor'),
  'Neuronal precursor' = c('Neuroblast', 'Neuronal_IPC'),
  'Radial glia' = 'Radial_glia'
)

cells$superclass <- sapply(cells$type_2, function(x) {
  for (lineage in names(superclass_mapping)) {
    if (x %in% superclass_mapping[[lineage]]) {
      return(lineage)
    }
  }
  return(NA)
})
cells$superclass <- factor(cells$superclass)

# Broad developmental period (3 groups)
cells$period <- dplyr::case_when(
  cells$stage_abbr %in% c("S01", "S02", "S03", "S04", "S05", "S06", "S07") ~ "Prenatal",
  cells$stage_abbr %in% c("S08", "S09", "S10", "S11", "S12") ~ "Postnatal developmental",
  cells$stage_abbr %in% c("S13", "S14", "S15") ~ "Adult and aging"
)
cells$period <- factor(cells$period,
                       levels = c("Prenatal", "Postnatal developmental", "Adult and aging"))

# Fine-grained developmental phase (8 groups)
cells$phase <- dplyr::case_when(
  cells$stage_abbr %in% c("S01", "S02") ~ "First trimester",
  cells$stage_abbr %in% c("S03", "S04", "S05") ~ "Second trimester",
  cells$stage_abbr %in% c("S06", "S07") ~ "Third trimester",
  cells$stage_abbr %in% c("S08", "S09") ~ "Infancy",
  cells$stage_abbr %in% c("S10", "S11") ~ "Childhood",
  cells$stage_abbr == "S12" ~ "Adolescence",
  cells$stage_abbr %in% c("S13", "S14") ~ "Adulthood",
  cells$stage_abbr == "S15" ~ "Aged"
)
cells$phase <- factor(cells$phase,
                      levels = c("First trimester", "Second trimester", "Third trimester",
                                 "Infancy", "Childhood", "Adolescence",
                                 "Adulthood", "Aged"))


## Per-period CellChat objects (grouped by type_2) -----------------------------
# Build one object per broad developmental period (Prenatal / Postnatal / Adult)
phases <- levels(cells$period)
dir.create(PERIOD_OUTDIR, recursive = TRUE, showWarnings = FALSE)
CellChatDB <- CellChatDB.human

for (ph in phases) {
  message("Building object for period: ", ph)
  cell.use <- rownames(cells)[cells$period == ph]

  if (length(cell.use) < 50) {
    message("Skip ", ph, ": too few cells (", length(cell.use), ")")
    next
  }
  counts.use <- counts[, cell.use]
  meta.use <- cells[cell.use, ]

  # Keep only the top-12 cell types when more are present
  celltype_counts <- table(meta.use$type_2)
  if (length(celltype_counts) > 12) {
    top_celltypes <- names(sort(celltype_counts, decreasing = TRUE)[1:12])
    cells_to_keep <- rownames(meta.use)[meta.use$type_2 %in% top_celltypes]
    counts.use <- counts.use[, cells_to_keep]
    meta.use <- meta.use[cells_to_keep, ]
  }

  cellchat <- createCellChat(object = counts.use, meta = meta.use, group.by = "type_2")
  cellchat <- setIdent(cellchat, ident.use = "type_2")
  cellchat@DB <- CellChatDB

  cellchat <- subsetData(cellchat)
  saveRDS(cellchat, file.path(PERIOD_OUTDIR, paste0("cellchat_", gsub(" ", "_", ph), "_raw.rds")))

  # identifyOverExpressedGenes runs on cellchat@data, which must still be
  # present here (the *_raw_light.rds object is re-read below and passed to
  # computeCommunProb)
  cellchat <- identifyOverExpressedGenes(cellchat)
  cellchat <- identifyOverExpressedInteractions(cellchat)
  saveRDS(cellchat, file.path(PERIOD_OUTDIR, paste0("cellchat_", gsub(" ", "_", ph), "_raw_light.rds")))
}

for (ph in phases) {
  message("Computing probabilities for period: ", ph)
  cellchat <- readRDS(file.path(PERIOD_OUTDIR, paste0("cellchat_", gsub(" ", "_", ph), "_raw_light.rds")))

  cellchat <- computeCommunProb(cellchat, type = "triMean")
  cellchat <- filterCommunication(cellchat, min.cells = 50)
  cellchat <- computeCommunProbPathway(cellchat)
  cellchat <- aggregateNet(cellchat)

  # drop the raw expression slot only after the computations above
  cellchat@data <- matrix()

  saveRDS(cellchat, file.path(PERIOD_OUTDIR, paste0("cellchat_", gsub(" ", "_", ph), "_ccc.rds")))
  rm(cellchat)
  gc()
}

## Per-phase CellChat objects (grouped by superclass) --------------------------
# Build one object per fine-grained developmental phase (8 phases)
phases <- c("First trimester", "Second trimester", "Third trimester",
            "Infancy", "Childhood", "Adolescence",
            "Adulthood", "Aged")
dir.create(PHASE_OUTDIR, recursive = TRUE, showWarnings = FALSE)
CellChatDB <- CellChatDB.human

for (ph in phases) {
  message("Building object for phase: ", ph)
  cell.use <- rownames(cells)[cells$phase == ph]

  if (length(cell.use) < 500) {
    message("Skip ", ph, ": too few cells (", length(cell.use), ")")
    next
  }
  counts.use <- counts[, cell.use]
  meta.use <- cells[cell.use, ]

  cellchat <- createCellChat(object = counts.use, meta = meta.use, group.by = "superclass")
  cellchat <- setIdent(cellchat, ident.use = "superclass")
  cellchat@DB <- CellChatDB

  cellchat <- subsetData(cellchat)
  saveRDS(cellchat, file.path(PHASE_OUTDIR, paste0("cellchat_", gsub(" ", "_", ph), "_raw.rds")))

  # identifyOverExpressedGenes runs on cellchat@data, which must still be
  # present here (the *_raw_light.rds object is re-read below and passed to
  # computeCommunProb)
  cellchat <- identifyOverExpressedGenes(cellchat)
  cellchat <- identifyOverExpressedInteractions(cellchat)
  saveRDS(cellchat, file.path(PHASE_OUTDIR, paste0("cellchat_", gsub(" ", "_", ph), "_raw_light.rds")))
}

for (ph in phases) {
  message("Computing probabilities for phase: ", ph)
  cellchat <- readRDS(file.path(PHASE_OUTDIR, paste0("cellchat_", gsub(" ", "_", ph), "_raw_light.rds")))

  cellchat@idents <- droplevels(cellchat@idents)
  cellchat <- computeCommunProb(cellchat, type = "triMean")
  cellchat <- filterCommunication(cellchat, min.cells = 50)
  cellchat <- computeCommunProbPathway(cellchat)
  cellchat <- aggregateNet(cellchat)

  # drop the raw expression slot only after the computations above
  cellchat@data <- matrix()

  saveRDS(cellchat, file.path(PHASE_OUTDIR, paste0("cellchat_", gsub(" ", "_", ph), "_ccc.rds")))
  rm(cellchat)
  gc()
}
