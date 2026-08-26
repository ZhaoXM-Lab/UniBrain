#!/usr/bin/env Rscript
# =============================================================================
# File      : 01_Fig1.R
# Directory : 11_figures
# Purpose   : Fig1 — Atlas overview: circos plot of cell type metadata +
#            circle-UMAP with density contours.
# Input     : Annotation_hvg5000/final/meta_sample2w.csv
#             Annotation_hvg5000/final/hs_all_metadata_final.csv
#             Visualization/umap_data_scvi100_scanvi50.csv
# Output    : Visualization/Fig1/single_cell_circos7.pdf
#             Visualization/Fig1/all_legends_transparent.pdf
#             Visualization/Fig1/circle_umap_with_contour4.pdf
# =============================================================================

library(circlize)
library(ComplexHeatmap)
library(dplyr)
library(ggplot2)
library(Seurat)
library(Matrix)
library(MASS)

# ============================================================================
# Configuration
# ============================================================================
proj_dir <- Sys.getenv("PROJ_DIR", unset = "path/to/your/project")  # <-- MODIFY THIS
setwd(file.path(proj_dir, "data", "human", "output"))

dir.create("Visualization/Fig1", recursive = TRUE, showWarnings = FALSE)

# ============================================================================
# Shared colour palettes (used by both circos and UMAP panels)
# ============================================================================
col_stage <- c(
  "01-Early first trimester" = "#1A318B", "02-Late first trimester" = "#2E4CB5",
  "03-Early second trimester" = "#496CCE", "04-Mid second trimester" = "#658AD7",
  "05-Late second trimester" = "#82AAE7", "06-Early third trimester" = "#A2C0EC",
  "07-Late third trimester" = "#B9D2F3", "08-Neonatal and early infancy" = "#CFE0F9",
  "09-Mid infancy" = "#E6F0FE", "10-Early childhood" = "#F0E8F1",
  "11-Middle and late childhood" = "#F9DBE5", "12-Adolescence" = "#F8BED6",
  "13-Young adulthood" = "#F7A6BF", "14-Middle adulthood" = "#EA89A0",
  "15-Late adulthood" = "#9A133D"
)

col_region <- c(
  "Brain" = "#5c8e9b", "Diencephalon" = "#85b76e", "Mesencephalon" = "#decf98",
  "Metencephalon " = "#874e94", "Myelencephalon " = "#e97c17",
  "Prosencephalon" = "#da84b6", "Rhombencephalon" = "#4f9446",
  "Spinal cord" = "#2a71aa", "Telencephalon" = "#ae94d7"
)

col_sex <- c("M" = "#FE9376", "F" = "#0099FF", "unknown" = "#D0CECE")

col_type_group <- c(
  "Glia" = "#BBD642", "Progenitors" = "#7AA8D9",
  "Neuron" = "#EBD0C8", "Others" = "#8E8E90"
)

col_type_1 <- c(
  "AstroEpen" = "#9CB1B8", "Oligodendrocyte" = "#DC143C", "Glial_precursor" = "#8DD3C7",
  "Neuronal_precursor" = "#D3EBB7", "Radial_glia" = "#4682B4",
  "Glutamatergic_neuron" = "#DC9720", "GABAergic_neuron" = "#EA7075",
  "Other neuron" = "#FFA500", "Immune" = "#FFED6F", "Fibroblast" = "#D2B48C",
  "Endothelial_cell" = "#1E90FF", "Pericyte" = "#6495ED",
  "Choroid_plexus_epithelial" = "#FF6347", "Vascular_smooth_muscle_cell" = "#FF8C00"
)

col_type_2 <- c(
  "Astrocyte" = "#2E8B57", "CALB2+ GABA" = "#FF1493", "Cholinergic" = "#FFFF00",
  "Choroid_plexus_epithelial" = "#FF6347", "Endothelial_cell" = "#1E90FF",
  "Ependymal" = "#FF69B4", "Fibroblast" = "#D2B48C", "Glioblast" = "#90EE90",
  "Granule_neuron" = "#B0C4DE", "Hippocampal_Glu" = "#6A5ACD",
  "Hypothalamic_Glu" = "#5F9EA0", "L2/3-CUX2 Glu" = "#87CEEB",
  "L2/3-SATB2 Glu" = "#87CEFA", "L4-RORB Glu" = "#6495ED",
  "L5-BCL11B Glu" = "#4169E1", "L5/6-THEMIS Glu" = "#0000CD",
  "L5/6-TLE4 Glu" = "#ABC8E5", "LHX6+LAMP5+ GABA" = "#DC143C",
  "Macrophage" = "#A0522D", "Medium_spiny_neuron" = "#9400D3",
  "Microglia" = "#8B4513", "Midbrain-derived_GABA" = "#800080",
  "Neuroblast" = "#CAA2F4", "Neuronal_IPC" = "#FC9A9A",
  "Oligodendrocyte" = "#DC143C", "Oligodendrocyte_precursor" = "#32CD32",
  "Oligodendrocyte_precursor_committed" = "#ADFF2F", "Pericyte" = "#6495ED",
  "Purkinje_neuron" = "#BA55D3", "PVALB+ GABA" = "#CA81A9",
  "Radial_glia" = "#4682B4", "RELN+ GABA" = "#EF82B2", "Serotonergic" = "#FFA500",
  "SST+ GABA" = "#FF4500", "Thalamic_GABA" = "#9932CC", "Thalamic_Glu" = "#7B68EE",
  "Vascular_smooth_muscle_cell" = "#FF8C00", "VIP+ GABA" = "#C71585"
)

# ============================================================================
# Panel A: Circos plot of cell type metadata
# ============================================================================
meta <- read.csv("Annotation_hvg5000/final/meta_sample2w.csv", header = TRUE)
rownames(meta) <- meta$cell_name
meta <- meta[meta$type_2 != "Remove", ]

# Summarise per (type_2, type_3) combination
cell_info <- meta %>%
  group_by(type_2, type_3) %>%
  summarise(
    n_cell = n(),
    stage = names(sort(table(stage), decreasing = TRUE))[1],
    region_L1 = names(sort(table(region_L1), decreasing = TRUE))[1],
    sex = names(sort(table(sex), decreasing = TRUE))[1],
    n_sample = n_distinct(sample),
    n_type4 = n_distinct(type_4),
    .groups = "drop"
  )

# Merge hierarchical labels
type_mapping <- meta %>%
  group_by(type_2) %>%
  summarise(type_1 = unique(type_1), type_group = unique(type_group), .groups = "drop")
cell_info <- cell_info %>%
  left_join(type_mapping, by = "type_2") %>%
  arrange(type_group, type_1, type_2, type_3) %>%
  group_by(type_2) %>%
  mutate(x = row_number()) %>%
  ungroup()

type2_info <- cell_info %>%
  group_by(type_2) %>%
  summarise(n_type3 = max(x), type_1 = unique(type_1),
            type_group = unique(type_group), .groups = "drop")

cell_info$type_2     <- factor(cell_info$type_2, levels = unique(cell_info$type_2))
cell_info$stage      <- factor(cell_info$stage)
cell_info$region_L1  <- factor(cell_info$region_L1)
cell_info$sex        <- factor(cell_info$sex)
cell_info$type_1     <- factor(cell_info$type_1)
cell_info$type_group <- factor(cell_info$type_group)

# Continuous colour maps
col_sample <- colorRamp2(c(0, max(cell_info$n_sample)), c("#f7f7f7", "#9A4942"))
col_type4  <- colorRamp2(c(0, max(cell_info$n_type4)),  c("#f7f7f7", "#6DA7CA"))

# ---- Draw circos ----
pdf("Visualization/Fig1/single_cell_circos7.pdf", width = 16, height = 16)
circos.clear()
circos.par(
  start.degree = 90,
  gap.degree = c(rep(0.1, length(unique(cell_info$type_2)) - 1), 15),
  track.margin = c(0.005, 0.005),
  cell.padding = c(0, 0, 0, 0),
  clock.wise = TRUE,
  canvas.xlim = c(-1, 1), canvas.ylim = c(-1, 1),
  points.overflow.warning = FALSE
)

sectors <- cell_info %>%
  group_by(type_2) %>%
  summarise(max_x = max(x, na.rm = TRUE), .groups = "drop") %>%
  tibble::deframe()
circos.initialize(factors = cell_info$type_2, xlim = cbind(0, sectors))

# Track 1: n_sample
circos.track(track.index = 1, ylim = c(0, max(cell_info$n_sample)),
  panel.fun = function(x, y) {
    d <- cell_info %>% filter(type_2 == CELL_META$sector.index)
    circos.rect(xleft = d$x - 0.5, xright = d$x + 0.5, ybottom = 0,
                ytop = d$n_sample, col = col_sample(d$n_sample), border = NA)
  }, track.height = 0.06, bg.border = NA)

# Track 2: n_type4
circos.track(track.index = 2, ylim = c(0, max(cell_info$n_type4)),
  panel.fun = function(x, y) {
    d <- cell_info %>% filter(type_2 == CELL_META$sector.index)
    circos.rect(xleft = d$x - 0.5, xright = d$x + 0.5, ybottom = 0,
                ytop = d$n_type4, col = col_type4(d$n_type4), border = NA)
  }, track.height = 0.06, bg.border = NA)

# Track 3: sex (dot)
circos.track(track.index = 3, ylim = c(0, 1),
  panel.fun = function(x, y) {
    d <- cell_info %>% filter(type_2 == CELL_META$sector.index)
    circos.points(x = d$x, y = rep(0.5, nrow(d)), pch = 16, cex = 0.7,
                  col = col_sex[as.character(d$sex)])
  }, track.height = mm_h(0.3), bg.border = NA)

# Track 4: region_L1 (dot)
circos.track(track.index = 4, ylim = c(0, 1),
  panel.fun = function(x, y) {
    d <- cell_info %>% filter(type_2 == CELL_META$sector.index)
    circos.points(x = d$x, y = rep(0.5, nrow(d)), pch = 16, cex = 0.7,
                  col = col_region[as.character(d$region_L1)])
  }, track.height = mm_h(0.3), bg.border = NA)

# Track 5: stage (dot)
circos.track(track.index = 5, ylim = c(0, 1),
  panel.fun = function(x, y) {
    d <- cell_info %>% filter(type_2 == CELL_META$sector.index)
    circos.points(x = d$x, y = rep(0.5, nrow(d)), pch = 16, cex = 0.7,
                  col = col_stage[as.character(d$stage)])
  }, track.height = mm_h(0.3), bg.border = NA)

# ---- Hierarchical colour bands (type_2, type_1, type_group) ----
track5_inner <- get.cell.meta.data("cell.bottom.radius", track.index = 5)
type2_outer <- track5_inner - 0.01
type2_inner <- type2_outer - 0.03
type1_outer <- type2_inner - 0.01
type1_inner <- type1_outer - 0.03
type_group_outer <- type1_inner - 0.01
type_group_inner <- type_group_outer - 0.03

sector_index <- get.all.sector.index()
sector_info <- data.frame(
  type_2 = as.character(sector_index),
  start_degree = sapply(sector_index, function(s)
    get.cell.meta.data("cell.start.degree", sector.index = s, track.index = 5)),
  end_degree = sapply(sector_index, function(s)
    get.cell.meta.data("cell.end.degree", sector.index = s, track.index = 5))
) %>%
  left_join(type2_info %>% mutate(type_2 = as.character(type_2)) %>%
              select(type_2, type_1, type_group), by = "type_2")

build_angle <- function(df, key) {
  df %>% filter(!is.na(.data[[key]])) %>%
    group_by(.data[[key]]) %>%
    summarise(start_degree = max(start_degree), end_degree = min(end_degree),
              .groups = "drop")
}

type2_angle   <- build_angle(sector_info, "type_2")
type1_angle   <- build_angle(sector_info, "type_1")
type_group_angle <- build_angle(sector_info, "type_group")

for (i in seq_len(nrow(type2_angle)))
  draw.sector(type2_angle$start_degree[i], type2_angle$end_degree[i],
              rou1 = type2_inner, rou2 = type2_outer,
              col = col_type_2[type2_angle$type_2[i]], border = NA)
for (i in seq_len(nrow(type1_angle)))
  draw.sector(type1_angle$start_degree[i], type1_angle$end_degree[i],
              rou1 = type1_inner, rou2 = type1_outer,
              col = col_type_1[type1_angle$type_1[i]], border = NA)
for (i in seq_len(nrow(type_group_angle)))
  draw.sector(type_group_angle$start_degree[i], type_group_angle$end_degree[i],
              rou1 = type_group_inner, rou2 = type_group_outer,
              col = col_type_group[type_group_angle$type_group[i]], border = NA)
dev.off()

# ---- Combined legend ----
make_dot_legend <- function(labels, title, cols, ncol = 1, size = 5, ...) {
  Legend(labels = labels, title = title, type = "points",
         background = "transparent",
         legend_gp = gpar(col = cols, fill = cols), pch = 19,
         size = unit(size, "mm"), ncol = ncol, ...)
}

lgd_stage  <- make_dot_legend(names(col_stage),  "Developmental Stage",  col_stage,  ncol = 3)
lgd_region <- make_dot_legend(names(col_region), "Brain Region",         col_region)
lgd_sex    <- make_dot_legend(names(col_sex),    "Sex",                  col_sex)
lgd_group  <- make_dot_legend(names(col_type_group), "Cell Category",    col_type_group)
lgd_t1     <- make_dot_legend(names(col_type_1), "Cell Superclass",      col_type_1, ncol = 3,
                              title_gp = gpar(fontsize = 12, fontface = "bold"),
                              labels_gp = gpar(fontsize = 9))
lgd_t2     <- make_dot_legend(names(col_type_2), "Cell Class",           col_type_2, ncol = 9,
                              title_gp = gpar(fontsize = 12, fontface = "bold"),
                              labels_gp = gpar(fontsize = 8))

lgd_sample <- Legend(title = "Number of Samples", col_fun = col_sample,
                     type = "gradient", background = "transparent",
                     legend_width = unit(4, "cm"), direction = "horizontal")
lgd_type4  <- Legend(title = "Number of Cell Populations", col_fun = col_type4,
                     type = "gradient", background = "transparent",
                     legend_width = unit(4, "cm"), direction = "horizontal")

combined_legend <- packLegend(
  packLegend(lgd_stage, lgd_region, lgd_sex, lgd_group, lgd_t1, lgd_t2,
             direction = "vertical", gap = unit(1, "cm")),
  packLegend(lgd_sample, lgd_type4, direction = "horizontal", gap = unit(2, "cm")),
  direction = "vertical", gap = unit(2, "cm")
)

pdf("Visualization/Fig1/all_legends_transparent.pdf", width = 18, height = 22)
draw(combined_legend)
dev.off()

# ============================================================================
# Panel B: Circle-UMAP with density contours (plot1cell-style)
# ============================================================================

# ---- plot1cell helper functions ----
transform_coordinates <- function(coord_data, zoom) {
  center_data <- coord_data - mean(c(min(coord_data), max(coord_data)))
  center_data * zoom / max(center_data)
}

get_metadata <- function(seu_obj, reductions = "umap", coord_scale = 0.8, color) {
  metadata <- seu_obj@meta.data
  metadata$Cluster <- seu_obj@active.ident
  metadata$dim1 <- as.numeric(seu_obj[[reductions]]@cell.embeddings[, 1])
  metadata$dim2 <- as.numeric(seu_obj[[reductions]]@cell.embeddings[, 2])
  metadata$x <- transform_coordinates(metadata$dim1, zoom = coord_scale)
  metadata$y <- transform_coordinates(metadata$dim2, zoom = coord_scale)
  color_df <- data.frame(Cluster = levels(seu_obj), Colors = color)
  cellnames <- rownames(metadata)
  metadata$cells <- rownames(metadata)
  metadata <- merge(metadata, color_df, by = "Cluster")
  rownames(metadata) <- metadata$cells
  metadata[cellnames, ]
}

cell_order <- function(dat) {
  celltypes <- names(table(dat$Cluster))
  new_dat <- lapply(celltypes, function(ct) {
    d <- dat[dat$Cluster == ct, ]
    d$x_polar <- seq_len(nrow(d))
    d
  })
  do.call("rbind", new_dat)
}

prepare_circlize_data <- function(seu_obj, scale = 0.8) {
  celltypes <- levels(seu_obj)
  cell_colors <- scales::hue_pal()(length(celltypes))
  data_plot <- get_metadata(seu_obj, color = cell_colors, coord_scale = scale)
  data_plot <- cell_order(data_plot)
  data_plot$x_polar2 <- log10(data_plot$x_polar)
  data_plot
}

plot_circlize <- function(data_plot, do.label = TRUE, contour.levels = c(0.2, 0.3),
                          pt.size = 0.5, kde2d.n = 1000, bg.color = "#F9F2E4",
                          col.use = NULL, label.cex = 0.5, repel = FALSE) {
  centers <- data_plot %>%
    dplyr::group_by(Cluster) %>%
    summarise(x = median(x), y = median(y), .groups = "drop")

  z <- MASS::kde2d(data_plot$x, data_plot$y, n = kde2d.n)
  celltypes <- names(table(data_plot$Cluster))

  if (!is.null(col.use)) {
    col_df <- data.frame(Cluster = celltypes, color2 = col.use)
    cells_order <- rownames(data_plot)
    data_plot <- merge(data_plot, col_df, by = "Cluster")
    rownames(data_plot) <- data_plot$cells
    data_plot <- data_plot[cells_order, ]
    data_plot$Colors <- data_plot$color2
  }

  circos.clear()
  par(bg = bg.color, mar = c(0, 0, 0, 0))
  plot.new()

  x_range <- max(data_plot$x) - min(data_plot$x)
  y_range <- max(data_plot$y) - min(data_plot$y)
  x_buf <- x_range * 0.15; y_buf <- y_range * 0.15
  plot.window(xlim = c(min(data_plot$x) - x_buf, max(data_plot$x) + x_buf),
              ylim = c(min(data_plot$y) - y_buf, max(data_plot$y) + y_buf), asp = 1)

  points(data_plot$x, data_plot$y, pch = 19,
         col = alpha(data_plot$Colors, 0.2), cex = pt.size)
  contour(z, drawlabels = FALSE, nlevels = 100, levels = contour.levels,
          col = "#ae9c76", lwd = 1.5, add = TRUE)

  if (do.label) {
    if (repel) {
      textplot(x = centers$x, y = centers$y, words = centers$Cluster,
               cex = label.cex, new = FALSE, show.lines = FALSE)
    } else {
      text(centers$x, centers$y, labels = centers$Cluster, cex = label.cex, col = "black")
    }
  }
}

# ---- Load UMAP data ----
umap_df <- read.csv("Visualization/umap_data_scvi100_scanvi50.csv")
colnames(umap_df)[1] <- "cell_name"

meta_umap <- read.csv("Annotation_hvg5000/final/hs_all_metadata_final.csv", header = TRUE)
plot_df <- inner_join(meta_umap, umap_df, by = "cell_name")

set.seed(123)
plot_df <- plot_df %>% slice_sample(n = 1000000, replace = FALSE)

# Build dummy Seurat object for plot1cell pipeline
expr_matrix <- Matrix(data = 0, nrow = 5, ncol = nrow(plot_df), sparse = TRUE)
rownames(expr_matrix) <- paste0("dummy_gene_", 1:5)
colnames(expr_matrix) <- plot_df$cell_name

seurat_obj <- CreateSeuratObject(counts = expr_matrix, meta.data = plot_df)
umap_embeddings <- as.matrix(plot_df[, c("UMAP_1", "UMAP_2")])
rownames(umap_embeddings) <- plot_df$cell_name
seurat_obj[["umap"]] <- CreateDimReducObject(embeddings = umap_embeddings,
                                             key = "UMAP_", assay = "RNA")
Idents(seurat_obj) <- "type_2"

# ---- Draw circle-UMAP ----
circ_data <- prepare_circlize_data(seurat_obj, scale = 0.8)

pdf(file.path("Visualization/Fig1", "circle_umap_with_contour4.pdf"),
    width = 15, height = 15)
plot_circlize(circ_data, do.label = FALSE, label.cex = 1.0, pt.size = 0.01,
              contour.levels = c(0.2, 0.3), col.use = col_type_2,
              bg.color = "white", kde2d.n = 200)
dev.off()
