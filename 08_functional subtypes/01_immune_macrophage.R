#!/usr/bin/env Rscript
# =============================================================================
# File      : 01_immune_macrophage.R
# Directory : 08_functional subtypes
# Purpose   : Macrophage subtype differential expression and enrichment analysis.
# =============================================================================

# ============================================================================
# Configuration: Set the project root directory before running
# ============================================================================
# proj_dir <- "path/to/your/project"  # <-- MODIFY THIS
# setwd(proj_dir)
proj_dir <- Sys.getenv("PROJ_DIR", unset = "path/to/your/project")
setwd(proj_dir)

library(Seurat)
library(Matrix)
library(data.table)
library(dplyr)
library(enrichR)

# Cell type to analyse
ct <- "Mac"

# Directories relative to proj_dir (under data/human/)
data_dir <- "output/Annotation_hvg5000/L4/R/"
out_dir  <- paste0("output/Analysis/immune/validation/", ct, "/")

# ----------------------------------------------------------------------------
# 1. Load count matrices and construct the Seurat object
# ----------------------------------------------------------------------------
matrix_data <- fread(paste0("data/human/", data_dir, ct, "/counts_matrix.mtx"), skip = 2, header = FALSE)
matrix_norm_data <- fread(paste0("data/human/", data_dir, ct, "/raw_matrix.mtx"), skip = 2, header = FALSE)
norm_genes <- read.csv(paste0("data/human/", data_dir, ct, "/raw_genes.csv"), row.names = 1)
genes <- read.csv(paste0("data/human/", data_dir, ct, "/counts_genes.csv"), row.names = 1)
norm_cells <- read.csv(paste0("data/human/", data_dir, ct, "/cells_new.csv"), row.names = 1)
cells <- read.csv(paste0("data/human/", data_dir, ct, "/counts_cells.csv"), row.names = 1)
hvg <- read.csv(paste0("data/human/", data_dir, ct, "/hvg.csv"), row.names = 1)

# Build sparse count matrix (first line of the .mtx holds dimensions)
nrow <- as.numeric(matrix_data[1, "V1"])
ncol <- as.numeric(matrix_data[1, "V2"])
matrix_data <- matrix_data[-1, ]
counts <- sparseMatrix(
    i = matrix_data$V1,
    j = matrix_data$V2,
    x = matrix_data$V3,
    dims = c(nrow, ncol)
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

# Keep only genes present in the normalised matrix
obj <- subset(obj, features = rownames(norm_genes))

# Load the normalised (raw) matrix into the data slot
nrow <- as.numeric(matrix_norm_data[1, "V1"])
ncol <- as.numeric(matrix_norm_data[1, "V2"])
matrix_norm_data <- matrix_norm_data[-1, ]
norm_counts <- sparseMatrix(
    i = matrix_norm_data$V1,
    j = matrix_norm_data$V2,
    x = matrix_norm_data$V3,
    dims = c(nrow, ncol)
)
rownames(norm_counts) <- rownames(norm_genes)
colnames(norm_counts) <- rownames(norm_cells)
obj@assays$RNA@data <- norm_counts

# ----------------------------------------------------------------------------
# 2. Attach pre-computed PCA / Harmony / UMAP embeddings
# ----------------------------------------------------------------------------
pca_df <- read.csv(paste0("data/human/", data_dir, ct, "/pca.csv"), row.names = 1)
pca_harmony_df <- read.csv(paste0("data/human/", data_dir, ct, "/pca_harmony.csv"), row.names = 1)
umap_df <- read.csv(paste0("data/human/", data_dir, ct, "/umap.csv"), row.names = 1)

if (all(rownames(pca_df) %in% colnames(obj))) {
  pca_matrix <- as.matrix(pca_df[colnames(obj), ])  # align cell order
  obj[["pca"]] <- CreateDimReducObject(embeddings = pca_matrix, key = "PC_", assay = "RNA")
} else {
  warning("Cell names in PCA file do not match Seurat object.")
}

if (all(rownames(pca_harmony_df) %in% colnames(obj))) {
  pca_harmony_matrix <- as.matrix(pca_harmony_df[colnames(obj), ])
  obj[["pca_harmony"]] <- CreateDimReducObject(embeddings = pca_harmony_matrix, key = "harmonyPC_", assay = "RNA")
} else {
  warning("Cell names in Harmony PCA file do not match Seurat object.")
}

colnames(umap_df) <- c("UMAP_1", "UMAP_2")
if (all(rownames(umap_df) %in% colnames(obj))) {
  umap_matrix <- as.matrix(umap_df[colnames(obj), ])
  obj[["umap"]] <- CreateDimReducObject(embeddings = umap_matrix, assay = "RNA")
} else {
  warning("Cell names in UMAP file do not match Seurat object.")
}

hvg$highly_variable <- as.logical(hvg$highly_variable)
VariableFeatures(obj) <- rownames(hvg)[hvg$highly_variable]

# Add subtype annotations from the normalised cell table
obj$type_3 <- norm_cells[colnames(obj), "type_3"]
obj$type_3_abbr <- norm_cells[colnames(obj), "type_3_abbr"]

saveRDS(obj, file = paste0("data/human/", out_dir, "seurat_obj_new.rds"))
obj <- subset(obj, subset = type_3 != "Remove")
saveRDS(obj, file = paste0("data/human/", out_dir, "seurat_obj_sub.rds"))

# ----------------------------------------------------------------------------
# 3. Differential expression between subtypes (type_3)
# ----------------------------------------------------------------------------
Idents(obj) <- obj$type_3
all.markers <- FindAllMarkers(
  object = obj,
  assay = "RNA",
  slot = "data",
  only.pos = FALSE,
  logfc.threshold = 0,
  test.use = "wilcox"
)
write.csv(all.markers, file = paste0("data/human/", out_dir, "mac_all_markers_new.csv"))
all.markers <- read.csv(paste0("data/human/", out_dir, "mac_all_markers_new.csv"))

# Significant DEGs used for enrichment analysis
all.deg <- all.markers %>%
  filter(p_val_adj < 0.05) %>%
  filter(avg_log2FC > 0.4)

# ----------------------------------------------------------------------------
# 4. Enrichr functional enrichment per subtype
# ----------------------------------------------------------------------------
setEnrichrSite("Enrichr")
dbs <- listEnrichrDbs()
head(dbs$libraryName)

# Run enrichR for each subtype using its DEGs
RunEnrichment <- function(deg_df,
                          pval_cutoff = 0.05,
                          logfc_cutoff = 0.5,
                          databases = c("GO_Biological_Process_2023",
                                        "GO_Cellular_Component_2023",
                                        "GO_Molecular_Function_2023",
                                        "KEGG_2021_Human")) {
  clusters <- unique(deg_df[, "cluster"])
  results <- list()
  for (cluster_i in clusters) {
    cluster_genes <- deg_df %>%
      filter(cluster == cluster_i) %>%
      pull(gene)
    if (length(cluster_genes) > 0) {
      enriched <- tryCatch({
        enrichr(cluster_genes, databases)
      }, error = function(e) {
        return(NULL)
      })
      if (!is.null(enriched)) {
        results[[cluster_i]] <- enriched
      }
    }
  }
  return(results)
}

enrichment_results <- RunEnrichment(all.deg)

# ----------------------------------------------------------------------------
# 5. Assign functional clusters (from hierarchical clustering of pathway
#    similarity) and save the annotated object
# ----------------------------------------------------------------------------
module_mapping <- list(
  "Enhanced redox"               = c("Macrophage-2"),
  "Synaptic interacting"         = c("Macrophage-3"),
  "Inflammatory"                 = c("Macrophage-7"),
  "Neurodegenerative associated" = c("Macrophage-4", "Macrophage-6"),
  "Stress responsive"            = c("Macrophage-5"),
  "Lipid associated"             = c("Macrophage-1")
)

obj@meta.data$function_cluster <- NA
for (module_name in names(module_mapping)) {
  obj@meta.data$function_cluster[obj@meta.data$type_3 %in% module_mapping[[module_name]]] <- module_name
}
Idents(obj) <- obj@meta.data$function_cluster
saveRDS(obj, file = paste0("data/human/", out_dir, "seurat_obj_function.rds"))
