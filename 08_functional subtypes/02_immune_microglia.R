#!/usr/bin/env Rscript
# =============================================================================
# File      : 02_immune_microglia.R
# Directory : 08_functional subtypes
# Purpose   : Microglia differential expression, functional scoring and GWAS enrichment.
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
library(tidyr)
library(purrr)
library(enrichR)

# Cell type to analyse
ct <- "Mic"

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
norm_cells <- read.csv(paste0("data/human/", data_dir, ct, "/cells.csv"), row.names = 1)
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

Idents(obj) <- obj$type_2
saveRDS(obj, file = paste0("data/human/", out_dir, "seurat_obj.rds"))

# ----------------------------------------------------------------------------
# 3. Score microglia subtypes with published microglia gene sets
# ----------------------------------------------------------------------------
# Literature gene sets (microglia subtype signatures)
green_2024_genes <- list(
  Proliferative_Green2024 = c("TOP2A", "DIAPH3", "EZH2", "BRIP1"),
  Surveilling_Green2024 = c("CX3CR1", "XXCT", "SFMBT2", "P2RY12", "ANKRD44", "GRID2", "XACT"),
  Reacting_Green2024 = c("TMEM163", "CD14", "IL4R", "FCGBP", "HIF1A", "SRGN", "ACSL1", "TGFBR1", "SH3RF3", "RUNX1", "CCDC26"),
  Enhanced_redox_Green2024 = c("FLT1", "FTL", "HLA-DPB1", "RGS10", "HLA-DRA", "RPS27", "TMSB4X"),
  Stress_response_Green2024 = c("HSPH1", "DNAJB1", "DNAJB6", "ATP2C1", "TGFBR2", "ITPR1", "ITPR2"),
  Lipid_associated_Green2024 = c("APOE", "ZNF804A", "NHSL1", "GPNMB", "SPP1", "PTPRG", "TPRG1", "GLDN"),
  Interferon_response_Green2024 = c("IFI6", "EPSTI1", "PARP14", "MX2", "IFI44L", "IFIT2", "IFIT3", "OAS1", "OAS2", "OAS3", "IFIH1", "PNPT1"),
  Inflammatory_Green2024 = c("CCL3", "CCL4", "NFKB1", "IL1B", "CD83", "CCL3L1", "ITGAX", "OLR1", "MSR1", "PIK3R1"),
  SERPINE1_expressing_Green2024 = c("SERPINE1", "ZBTB16", "CSGALNACT1", "CCND3")
)

kerenshaul_2017_genes <- list(
  DAM1_down_KerenShaul2017 = c("P2RY13", "CX3CR1", "P2RY12", "TMEM119", "TGFBR1", "GLUL", "TXNIP", "SERINC3"),
  DAM2_KerenShaul2017 = c("CLEC7A", "HIF1A", "CSF1", "ITGAX", "CD9", "SPP1", "AXL", "CADM1", "CST7", "SERPINE2", "GUSB", "CD52", "LPL", "CD68", "CD63", "CTSZ", "CTSL", "CTSA", "ANK1"),
  DAM1_KerenShaul2017 = c("HLA-C", "HLA-A", "HLA-B", "YZ", "B2M", "FTH1", "TYROBP", "CSTB", "CTSD", "TIMP2", "CTSB", "APOE")
)

gerrits_2021_genes <- list(
  Stress_response_Gerrits2021 = c("HSPA1ZB", "HSPA1A", "HSP90AA1", "HIST1H2AC", "FOS", "JUNB"),
  Proliferative_Gerrits2021 = c("MKI67", "TOP2A", "BRIP1", "FANSPCI"),
  Inflammatory_Gerrits2021 = c("CD83", "GPR183", "IL1B", "SRGN", "SLC2A3", "CCL4", "CCL3", "NFKB1"),
  Homeostatic_Gerrits2021 = c("SOX5", "CX3CR1", "P2RY12", "SH3RF3", "TMEM163"),
  AD2_tangles_associated_Gerrits2021 = c("ADGRB3", "DPP10", "NAV2", "GRID2", "FOXP2", "DDX17"),
  Abeta_associated_c9_Gerrits2021 = c("LPL", "ITGAX", "GLDN", "MYO1E", "PTPRG", "TPRG1", "DTNA", "SPATS2L"),
  Abeta_associated_c7_Gerrits2021 = c("DSCAM", "SOCS6", "IPCEF1", "ADAMTS17"),
  Abeta_associated_c10_Gerrits2021 = c("EYA2", "STARD13", "CPM", "GPNMB", "APOE")
)

cain_2023_genes <- list(
  Anti_inflamm_Cain2023 = c("SPP1", "TMEM163", "SLC2A3", "DDIT4"),
  Interferon_resp_Cain2023 = c("PTPRG", "IFI6", "IFI44L", "IFIT3"),
  Enhanced_redox_Cain2023 = c("C1QA", "C1QB", "C1QC", "RPS19", "FTH1", "RPS6", "RPL19", "FTL", "APOE"),
  Proliferative_type_Cain2023 = c("TOP2A", "HMMR", "ARHGAP11B"),
  Homeostatic_Cain2023 = c("CX3CR1", "SYNDIG1", "P2RY12", "FRMD4A")
)

li_2022_genes <- list(
  Original_Li2022 = c("LYVE1", "HBZ", "MRC1", "DAB2"),
  Proliferation_Li2022 = c("MKI67", "TK1", "TOP2A", "TYMS"),
  Immune_related_Li2022 = c("APOE", "FCGR3A", "FCGR1A"),
  Neuronal_Li2022 = c("STMN2", "SOX11", "MEG3", "TUBB2A", "MAP1B")
)

# Merge all literature gene sets and keep genes present in the object
all_literature_genes <- c(green_2024_genes, kerenshaul_2017_genes,
                          gerrits_2021_genes, cain_2023_genes, li_2022_genes)
all_literature_genes <- lapply(all_literature_genes, function(x) x[x %in% rownames(obj)])
all_literature_genes <- all_literature_genes[sapply(all_literature_genes, length) > 0]

# Score each cell for each literature gene set
obj <- AddModuleScore(
  object = obj,
  features = all_literature_genes,
  name = names(all_literature_genes),
  ctrl = 100,
  assay = "RNA"
)

# ----------------------------------------------------------------------------
# 4. Differential expression between subtypes (type_2)
# ----------------------------------------------------------------------------
all.markers <- FindAllMarkers(
  object = obj,
  assay = "RNA",
  slot = "data",
  only.pos = FALSE,
  logfc.threshold = 0,
  test.use = "wilcox"
)
write.csv(all.markers, file = paste0("data/human/", out_dir, "mic_all_markers.csv"))
all.markers <- read.csv(paste0("data/human/", out_dir, "mic_all_markers.csv"))

# Significant DEGs used for enrichment analysis
all.deg <- all.markers %>%
  filter(p_val_adj < 0.05) %>%
  filter(avg_log2FC > 0.4)
write.csv(all.deg, file = paste0("data/human/", out_dir, "mic_all_deg_fc04.csv"))

# ----------------------------------------------------------------------------
# 5. Enrichr functional enrichment per subtype
# ----------------------------------------------------------------------------
setEnrichrSite("Enrichr")
dbs <- listEnrichrDbs()
head(dbs$libraryName)

# Run enrichR for each subtype using its DEGs
RunEnrichment <- function(deg_df,
                          pval_cutoff = 0.05,
                          logfc_cutoff = 0.25,
                          databases = c("GO_Biological_Process_2023",
                                        "GO_Cellular_Component_2023",
                                        "GO_Molecular_Function_2023",
                                        "KEGG_2021_Human")) {
  deg_df <- as.data.frame(deg_df) %>% ungroup()
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
# 6. Assign functional clusters (from hierarchical clustering of pathway
#    similarity) and save the annotated object
# ----------------------------------------------------------------------------
module_mapping <- list(
  "Proliferative"  = c("Microglia-9", "Microglia-16"),
  "Enhanced redox" = c("Microglia-1", "Microglia-15"),
  "Inflammatory"   = c("Microglia-6", "Microglia-11"),
  "Lipid associated" = c("Microglia-3", "Microglia-4", "Microglia-5", "Microglia-7", "Microglia-8", "Microglia-17"),
  "Surveilling"    = c("Microglia-2", "Microglia-10", "Microglia-12", "Microglia-14"),
  "Reacting"       = c("Microglia-13")
)

obj@meta.data$function_cluster <- NA
for (module_name in names(module_mapping)) {
  obj@meta.data$function_cluster[obj@meta.data$type_3 %in% module_mapping[[module_name]]] <- module_name
}
Idents(obj) <- obj@meta.data$function_cluster
saveRDS(obj, file = paste0("data/human/", out_dir, "seurat_obj_function_new.rds"))

# ----------------------------------------------------------------------------
# 7. Differential expression between functional clusters
# ----------------------------------------------------------------------------
Idents(obj) <- "function_cluster"
all.module.markers <- FindAllMarkers(
  object = obj,
  assay = "RNA",
  slot = "data",
  only.pos = FALSE,
  logfc.threshold = 0,
  test.use = "wilcox"
)
write.csv(all.module.markers, file = paste0("data/human/", out_dir, "mic_functional_cluster6_marker_new.csv"))

# Significant cluster markers used for GWAS enrichment below
celltype_markers <- all.module.markers %>%
  filter(p_val_adj < 0.05) %>%
  filter(avg_log2FC > 0.5)
total_gene <- 31953

