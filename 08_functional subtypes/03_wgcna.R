#!/usr/bin/env Rscript
# =============================================================================
# File      : 03_wgcna.R
# Directory : 08_functional subtypes
# Purpose   : Weighted gene co-expression network analysis (hdWGCNA) per cell type.
# =============================================================================

# ============================================================================
# Configuration: Set the project root directory before running
# ============================================================================
# proj_dir <- "path/to/your/project"  # <-- MODIFY THIS
# setwd(proj_dir)
proj_dir <- Sys.getenv("PROJ_DIR", unset = "path/to/your/project")
setwd(proj_dir)

# ============================================================================
# Libraries (only those actually used; ggplot2/patchwork/cowplot removed)
# ============================================================================
library(hdWGCNA)
library(Seurat)
library(Matrix)
library(dplyr)
library(data.table)
library(WGCNA)
library(future)
library(UCell)

# ============================================================================
# Global setup
# ============================================================================
set.seed(12345)
enableWGCNAThreads(nThreads = 8)
# note: the sources also set plan("multicore"), which fails on Windows; only
# multisession is kept here
plan("multisession", workers = 8)
options(future.globals.maxSize = 200 * 1024^3)

# ============================================================================
# Paths (relative to proj_dir)
# ============================================================================
data_dir <- "output/Annotation_hvg5000/L4/R/"  # per-cell-type matrices and reductions
wgcna_out <- "output/Analysis/WGCNA"           # WGCNA output root

# ============================================================================
# Cell types
# ============================================================================
cell_types <- c("Mic", "Mac", "OPC", "COPC", "IPC", "RG", "NB", "GB", "NPC")

# ============================================================================
# Per-cell-type parameters (extracted from the source scripts; values differ
# between cell types and are preserved via named vectors, NOT unified)
# ============================================================================
# fraction of cells a gene must be expressed in to be kept (SetupForWGCNA)
#   OPC = 0.20, all others = 0.05
fractions <- c(
  "Mic" = 0.05, "Mac" = 0.05, "OPC" = 0.20,
  "COPC" = 0.05, "IPC" = 0.05, "RG" = 0.05, "NB" = 0.05, "GB" = 0.05
)
# chosen soft power after TestSoftPowers (ConstructNetwork)
#   Mic = 14, Mac = 5, OPC = 16, COPC = 7, IPC = 8, RG = 8, NB = 7, GB = 7
soft_powers <- c(
  "Mic" = 14, "Mac" = 5, "OPC" = 16,
  "COPC" = 7, "IPC" = 8, "RG" = 8, "NB" = 7, "GB" = 7
)
# suffix of the TOM output directory (ConstructNetwork tom_outdir)
#   all "005" except OPC "020"
tom_suffix <- c(
  "Mic" = "005", "Mac" = "005", "OPC" = "020",
  "COPC" = "005", "IPC" = "005", "RG" = "005", "NB" = "005", "GB" = "005"
)
# suffix of the WGCNA_metacell RData saved after ConstructNetwork
#   Mic = 015, Mac = 010, OPC = 020, others = 005
rdata_suffix <- c(
  "Mic" = "015", "Mac" = "010", "OPC" = "020",
  "COPC" = "005", "IPC" = "005", "RG" = "005", "NB" = "005", "GB" = "005"
)
# suffix of the final hdWGCNA object file (matches the files actually read
# downstream by 5.4_WGCNA_NPC.R: Mic = 015, Mac = 010, all others = 005)
obj_suffix <- c(
  "Mic" = "015", "Mac" = "010", "OPC" = "005",
  "COPC" = "005", "IPC" = "005", "RG" = "005", "NB" = "005", "GB" = "005"
)
# n_genes for ModuleExprScore (UCell): Mac = 50, all others = 25
expr_n_genes <- c(
  "Mic" = 25, "Mac" = 50, "OPC" = 25,
  "COPC" = 25, "IPC" = 25, "RG" = 25, "NB" = 25, "GB" = 25
)

# ============================================================================
# Developmental stage -> numeric pseudotime mapping (used for
# ModuleTraitCorrelation, same mapping in all source scripts)
# ============================================================================
stage_mapping <- c(
  "01-Embryonic" = 1,
  "02-Early fetal" = 2,
  "03-Early fetal" = 3,
  "04-Early midfetal" = 4,
  "05-Early midfetal" = 5,
  "06-Late midfetal" = 6,
  "07-Late fetal" = 7,
  "08-Neonatal and early infancy" = 8,
  "09-Late infancy" = 9,
  "10-Early childhood" = 10,
  "11-Middle and late childhood" = 11,
  "12-Adolescence" = 12,
  "13-Young adulthood" = 13,
  "14-Middle adulthood" = 14,
  "15-Late adulthood" = 15
)

# ============================================================================
# Main loop: full hdWGCNA pipeline for each cell type
# SetupForWGCNA -> MetacellsByGroups -> NormalizeMetacells -> SetDatExpr ->
# TestSoftPowers -> ConstructNetwork -> ModuleEigengenes -> ModuleConnectivity ->
# ResetModuleNames -> GetHubGenes -> ModuleExprScore -> ModuleTraitCorrelation
# ============================================================================
for (ct in cell_types) {
  cat("Processing:", ct, "\n")

  # NPC has no per-cell-type WGCNA pipeline in the sources: 5.4_WGCNA_NPC.R is
  # a downstream module-stage regression / heatmap analysis of the other cell
  # types' hdWGCNA objects, so there is nothing to run here.
  if (ct == "NPC") {
    cat("  Skipping NPC: no WGCNA pipeline in sources\n")
    next
  }

  out_dir <- file.path(wgcna_out, ct)
  dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
  ct_dir <- file.path(data_dir, ct)

  # ---- 1. Load counts / normalized matrices and build the Seurat object ----
  matrix_data <- fread(file.path(ct_dir, "counts_matrix.mtx"), skip = 2, header = FALSE)
  matrix_norm_data <- fread(file.path(ct_dir, "raw_matrix.mtx"), skip = 2, header = FALSE)
  norm_genes <- read.csv(file.path(ct_dir, "raw_genes.csv"), row.names = 1)
  genes <- read.csv(file.path(ct_dir, "counts_genes.csv"), row.names = 1)
  norm_cells <- read.csv(file.path(ct_dir, "cells.csv"), row.names = 1)
  cells <- read.csv(file.path(ct_dir, "counts_cells.csv"), row.names = 1)
  hvg <- read.csv(file.path(ct_dir, "hvg.csv"), row.names = 1)

  n_r <- as.numeric(matrix_data[1, "V1"])
  n_c <- as.numeric(matrix_data[1, "V2"])
  matrix_data <- matrix_data[-1, ]
  counts <- sparseMatrix(
    i = matrix_data$V1,
    j = matrix_data$V2,
    x = matrix_data$V3,
    dims = c(n_r, n_c)
  )
  rownames(counts) <- rownames(genes)
  colnames(counts) <- rownames(cells)

  obj <- CreateSeuratObject(
    counts = counts,
    project = ct,
    meta.data = cells,
    min.cells = 0,
    min.features = 0
  )
  obj <- subset(obj, features = rownames(norm_genes))

  n_r <- as.numeric(matrix_norm_data[1, "V1"])
  n_c <- as.numeric(matrix_norm_data[1, "V2"])
  matrix_norm_data <- matrix_norm_data[-1, ]
  norm_counts <- sparseMatrix(
    i = matrix_norm_data$V1,
    j = matrix_norm_data$V2,
    x = matrix_norm_data$V3,
    dims = c(n_r, n_c)
  )
  rownames(norm_counts) <- rownames(norm_genes)
  colnames(norm_counts) <- rownames(norm_cells)
  obj@assays$RNA@data <- norm_counts

  # ---- 2. Add PCA / harmony PCA / UMAP reductions ----
  pca_df <- read.csv(file.path(ct_dir, "pca.csv"), row.names = 1)
  pca_harmony_df <- read.csv(file.path(ct_dir, "pca_harmony.csv"), row.names = 1)
  umap_df <- read.csv(file.path(ct_dir, "umap.csv"), row.names = 1)

  if (all(rownames(pca_df) %in% colnames(obj))) {
    obj[["pca"]] <- CreateDimReducObject(
      embeddings = as.matrix(pca_df[colnames(obj), ]),
      key = "PC_",
      assay = "RNA"
    )
  } else {
    warning("Cell names in PCA file do not match Seurat object.")
  }

  if (all(rownames(pca_harmony_df) %in% colnames(obj))) {
    obj[["pca_harmony"]] <- CreateDimReducObject(
      embeddings = as.matrix(pca_harmony_df[colnames(obj), ]),
      key = "harmonyPC_",
      assay = "RNA"
    )
  } else {
    warning("Cell names in Harmony PCA file do not match Seurat object.")
  }

  colnames(umap_df) <- c("UMAP_1", "UMAP_2")
  if (all(rownames(umap_df) %in% colnames(obj))) {
    obj[["umap"]] <- CreateDimReducObject(
      embeddings = as.matrix(umap_df[colnames(obj), ]),
      assay = "RNA"
    )
  } else {
    warning("Cell names in UMAP file do not match Seurat object.")
  }

  hvg$highly_variable <- as.logical(hvg$highly_variable)
  VariableFeatures(obj) <- rownames(hvg)[hvg$highly_variable]
  Idents(obj) <- obj$type_2

  saveRDS(obj, file = file.path(out_dir, "seurat_obj.rds"))

  # ---- 3. Keep the top 50% most expressed genes (by total counts) ----
  data_matrix <- GetAssayData(object = obj, slot = "counts")
  gene_sums <- Matrix::rowSums(data_matrix)
  sorted_genes <- names(sort(gene_sums, decreasing = TRUE))
  n_top <- ceiling(length(sorted_genes) * 0.50)
  obj_filtered <- subset(obj, features = sorted_genes[1:n_top])

  # ---- 4. hdWGCNA: metacell construction and normalization ----
  obj_wgcna <- SetupForWGCNA(
    obj_filtered,
    gene_select = "fraction", # the gene selection approach
    fraction = fractions[ct], # fraction of cells a gene must be expressed in
    wgcna_name = ct           # the name of the hdWGCNA experiment (= cell type)
  )

  obj_wgcna <- MetacellsByGroups(
    seurat_obj = obj_wgcna,
    group.by = c("type_2_abbr", "sample"), # columns to group metacells by
    reduction = "pca_harmony",             # reduction used for KNN
    k = 25,                                # nearest-neighbors parameter
    max_shared = 10,                       # max shared cells between two metacells
    ident.group = "type_2_abbr"            # Idents of the metacell seurat object
  )

  obj_wgcna <- NormalizeMetacells(obj_wgcna)

  # ---- 5. Co-expression network construction ----
  obj_wgcna <- SetDatExpr(
    obj_wgcna,
    group_name = ct,          # the group of interest in the group.by column
    group.by = "type_2_abbr", # metadata column containing the cell type info
    assay = "RNA",            # using RNA assay
    slot = "data"             # using normalized data
  )

  obj_wgcna <- TestSoftPowers(obj_wgcna, networkType = "signed")

  obj_wgcna <- ConstructNetwork(
    obj_wgcna,
    soft_power = soft_powers[ct],
    tom_outdir = file.path(out_dir, paste0("TOM_frac", tom_suffix[ct])),
    tom_name = ct, # name of the topological overlap matrix written to disk
    overwrite_tom = TRUE
  )
  save(obj_wgcna, file = file.path(out_dir, paste0("WGCNA_metacell_frac", rdata_suffix[ct], ".RData")))

  # ---- 6. Module eigengenes (harmonized across samples) ----
  # ScaleData is required first or else harmony throws an error
  all.genes <- rownames(obj_wgcna)
  obj_wgcna <- ScaleData(obj_wgcna, features = all.genes)

  # compute all MEs in the full single-cell dataset
  obj_wgcna <- ModuleEigengenes(
    obj_wgcna,
    group.by.vars = "sample", # metadata column containing the sample info
    wgcna_name = ct
  )

  # add harmonized MEs to the meta.data for downstream use
  hMEs <- GetMEs(obj_wgcna, harmonized = TRUE)
  obj_wgcna@meta.data <- cbind(obj_wgcna@meta.data, hMEs)

  # ---- 7. Module connectivity, renaming and hub genes ----
  obj_wgcna <- ModuleConnectivity(
    obj_wgcna,
    group.by = "type_2_abbr",
    group_name = ct
  )
  # rename the modules (e.g. "Mic-M1")
  obj_wgcna <- ResetModuleNames(obj_wgcna, new_name = paste0(ct, "-M"))

  # module assignment table (grey removed)
  modules <- GetModules(obj_wgcna) %>% subset(module != "grey")
  write.csv(modules, file = file.path(out_dir, "modules.csv"), row.names = FALSE)

  # hub genes
  hub_df <- GetHubGenes(obj_wgcna, n_hubs = 10)
  write.csv(hub_df, file = file.path(out_dir, "hub_genes.csv"), row.names = FALSE)

  saveRDS(obj_wgcna, file = file.path(out_dir, paste0("hdWGCNA_obj_frac", obj_suffix[ct], ".rds")))

  # ---- 8. Hub gene signature scores (UCell) ----
  obj_wgcna <- ModuleExprScore(
    obj_wgcna,
    n_genes = expr_n_genes[ct], # top N hub genes by kME for each module
    method = "UCell"
  )
  score_cols <- grep(" score$", colnames(obj_wgcna@meta.data), value = TRUE)
  if (length(score_cols) > 0) {
    write.csv(
      obj_wgcna@meta.data[, score_cols, drop = FALSE],
      file = file.path(out_dir, "module_expr_scores.csv"),
      row.names = TRUE
    )
  }

  # ---- 9. Differential module eigengenes (DME) ----
  # only Mac runs FindAllDMEs in the sources (group.by = 'type_3_abbr')
  if (ct == "Mac") {
    DMEs_all <- FindAllDMEs(
      obj_wgcna,
      group.by = "type_3_abbr",
      wgcna_name = ct
    )
    write.csv(DMEs_all, file = file.path(out_dir, "DMEs_all.csv"), row.names = FALSE)
  }

  # ---- 10. Module-trait correlations ----
  # map developmental stage to numeric pseudotime
  obj_wgcna@meta.data$stage_num <- stage_mapping[obj_wgcna@meta.data$stage]
  obj_wgcna@meta.data$stage_num <- as.numeric(as.character(obj_wgcna@meta.data$stage_num))
  obj_wgcna$platform <- as.factor(obj_wgcna$platform)
  obj_wgcna$sex <- as.factor(obj_wgcna$sex)
  obj_wgcna$region_L2 <- as.factor(obj_wgcna$region_L2)
  # trait metadata columns used in all sources
  traits <- c("stage_num", "region_L2", "platform", "sex")

  obj_wgcna <- ModuleTraitCorrelation(
    obj_wgcna,
    traits = traits,
    group.by = "type_3" # group by subtype
  )

  # extract and save the correlation results
  cor_results <- GetModuleTraitCorrelation(obj_wgcna)
  saveRDS(cor_results, file = file.path(out_dir, "Trait_Correlation_Results.rds"))

  if (!is.null(cor_results$cor)) {
    if (is.list(cor_results$cor)) {
      cor_list <- lapply(names(cor_results$cor), function(g) {
        m <- as.data.frame(cor_results$cor[[g]])
        m$trait <- rownames(m)
        m$group <- g
        m
      })
      cor_tab <- do.call(rbind, cor_list)
    } else {
      cor_tab <- as.data.frame(cor_results$cor)
    }
    write.csv(
      cor_tab,
      file = file.path(out_dir, "module_trait_correlations.csv"),
      row.names = FALSE
    )
  }

  # final object with all results
  saveRDS(obj_wgcna, file = file.path(out_dir, paste0("hdWGCNA_obj_frac", obj_suffix[ct], ".rds")))
  cat("  Saved results to:", out_dir, "\n")
}

# ============================================================================
# Export module gene tables (mirrors 5.4_WGCNA_save.R)
# For each cell type: read the hdWGCNA object, extract the module assignment,
# and write a padded gene-list table per module with fwrite.
# NOTE: 5.4_WGCNA_save.R reads "hdWGCNA_obj_frac015.rds" for every cell type,
# which is wrong for most of them; the per-cell-type obj_suffix above is used
# here instead (matches the files produced by the pipeline loop and the
# frac_mapping in 5.4_WGCNA_NPC.R).
# ============================================================================
for (ct in cell_types) {
  if (ct == "NPC") next
  cat("Processing:", ct, "\n")

  obj_wgcna <- readRDS(file.path(wgcna_out, ct, paste0("hdWGCNA_obj_frac", obj_suffix[ct], ".rds")))

  # extract module dataframe, remove grey module
  module_df <- obj_wgcna@misc[[ct]]$wgcna_modules
  module_df <- module_df[module_df$module != "grey", ]

  modules <- sort(unique(module_df$module))
  if (length(modules) == 0) next

  # gene list for each module
  module_gene_list <- lapply(modules, function(m) {
    module_df$gene_name[module_df$module == m]
  })
  names(module_gene_list) <- modules

  # pad to equal length so the table can be written as a data frame
  max_len <- max(sapply(module_gene_list, length))
  module_gene_list <- lapply(module_gene_list, function(x) {
    length(x) <- max_len
    x
  })
  module_gene_df <- as.data.frame(module_gene_list, check.names = FALSE)

  out_file <- file.path(wgcna_out, paste0(ct, "_module_genes.csv"))
  fwrite(module_gene_df, file = out_file, na = "")
  cat("Saved:", out_file, "\n")
}
