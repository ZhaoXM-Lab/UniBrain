#!/usr/bin/env Rscript
# =============================================================================
# File      : 01_rd2mtx.R
# Directory : 01_data_preprocessing
# Purpose   : Convert pre-processed Seurat objects into 10X-format sample directories (matrix.mtx/genes.tsv/barcodes.tsv) with reference-gene filtering and combined metadata output.
# =============================================================================

# ============================================================================
# Configuration: Set the project root directory before running
# ============================================================================
proj_dir <- "/path/to/your/project"  # <-- MODIFY THIS

library(Seurat)
library(Matrix)

setwd(file.path(proj_dir, "data"))
rm(list = ls())

#' Load the list of unique reference genes
gtf.unique <- read.table("ref/human_reference_unique.txt")
rownames(gtf.unique) <- gtf.unique$V2

#' Keep only genes present in the reference gene list
#'
#' @param mat Gene x cell count matrix
#' @return Count matrix restricted to reference genes
filter.gene <- function(mat) {
  ava.gene <- rownames(mat) %in% gtf.unique$V2
  return(mat[ava.gene, ])
}

#' Write a Seurat object as a 10X-format directory (matrix.mtx, genes.tsv, barcodes.tsv)
#'
#' @param dir.name Output directory name
#' @param data Seurat object to convert
save.mtx <- function(dir.name, data) {
  counts <- filter.gene(data@assays$RNA@counts)
  gene.info <- gtf.unique[rownames(counts), ]
  cell.info <- colnames(counts)

  writeMM(counts, file = paste0(dir.name, "/matrix.mtx"))
  write.table(
    gene.info, file = paste0(dir.name, "/genes.tsv"),
    sep = "\t", quote = FALSE,
    row.names = FALSE, col.names = FALSE
  )
  write.table(
    cell.info, file = paste0(dir.name, "/barcodes.tsv"),
    sep = "\t", quote = FALSE,
    row.names = FALSE, col.names = FALSE
  )
}

# Convert each pre-processed dataset to 10X-format sample directories
rd.list <- list.files(path = "pre_processed/", pattern = "^hs.*\\.RData$")
meta.list <- list()
cell.num <- c()
for (rd.i in rd.list) {
  load(paste0("pre_processed/", rd.i))

  data$sample <- gsub("/", "_", data$sample)
  cell.num[data$dataset[1]] <- nrow(data@meta.data)
  data <- subset(data, subset = nFeature_RNA > 10)

  meta.list <- c(meta.list, list(data@meta.data))

  if (data$platform[1] %in% c("STRT-seq", "Fluidigm C1")) {
    # Non-10X platforms: keep the whole dataset in a single directory
    dir.name <- paste0(
      "human/qc/",
      strsplit(rd.i, "[.]")[[1]][1]
    )
    dir.create(dir.name, recursive = TRUE)
    save.mtx(dir.name, data)
  } else {
    # 10X platforms: split by sample
    data.list <- SplitObject(data, split.by = "sample")
    for (data.i in names(data.list)) {
      dir.name <- paste0(
        "human/qc/", data.i
      )
      dir.create(dir.name, recursive = TRUE)
      save.mtx(dir.name, data.list[[data.i]])
    }
  }
  print(paste0(rd.i, " finished!"))
}

# Build the combined metadata table and cell-number summary
meta.list.new <- lapply(meta.list, function(df) df[, !names(df) %in% "percent.mt"])
meta.list.new <- lapply(meta.list.new, function(df) df[, !names(df) %in% "cellname"])

meta.table <- do.call("rbind", meta.list.new)
write.csv(
  meta.table, file = "human/qc/stab2_meta.csv"
)
write.table(
  cell.num, file = "human/qc/cell_num.csv",
  row.names = TRUE, col.names = FALSE, sep = "\t"
)
