#!/usr/bin/env Rscript
# =============================================================================
# File      : 03_sldsc_analysis.R
# Directory : 09_heritability enrichment
# Purpose   : Merge LDSC results and compute FDR-corrected cell-type enrichment across brain regions and stages.
# =============================================================================

## Configuration ##
# Set the base directory for input/output data below.
# proj_dir must point to the UniBrain project root
BASE_DIR <- Sys.getenv("PROJ_DIR", unset = "path/to/your/project")  # <-- MODIFY THIS

# Input: root directory whose subfolders contain per-cell-type LDSC .results files
SLDSC_INPUT_DIR <- file.path(BASE_DIR,
    "data/human/output/Analysis/disease/output/mic_func6_new/celltype_sldsc")

# Output: directory for merged and filtered LDSC tables
SLDSC_OUTPUT_DIR <- file.path(BASE_DIR,
    "data/human/output/Analysis/disease/output/celltype_sldsc_analysis")

# Input: cell metadata table (columns: cell_name, type_2, region_L2, stage, ...)
META_FILE <- file.path(BASE_DIR,
    "data/human/output/Annotation_hvg5000/final/hs_all_metadata_final.csv")

# Output files
MERGED_OUTPUT_FILE <- file.path(SLDSC_OUTPUT_DIR, "merged_sldsc_mic_func6_results_new.tsv")
FILTERED_OUTPUT_FILE <- file.path(SLDSC_OUTPUT_DIR, "sldsc_mic_sig_pc_50kb_filtered_new.csv")
REGION_ENRICHMENT_FILE <- file.path(SLDSC_OUTPUT_DIR, "enrichment_region_by_celltype.csv")
STAGE_ENRICHMENT_FILE <- file.path(SLDSC_OUTPUT_DIR, "enrichment_stage_by_celltype.csv")

## Libraries ##
library(dplyr)
library(readr)
library(purrr)

## Parameters ##
# Cell types (microglia functional states) to process
target_celltypes <- c('Enhanced_redox', 'Proliferative', 'Lipid_associated',
                      'Inflammatory', 'Surveilling', 'Reacting')

# GWAS disease traits to keep
target_diseases <- c("ADHD", "AD", "ALS", "ANX", "ASD", "ASP",
                     "AUD", "BIP", "CI", "DPW", "EA", "EPI", "INS",
                     "INT", "IS", "MDD", "MS", "PD", "PTSD", "SC",
                     "SCZ", "SI", "TS")

## Merge per-cell-type LDSC results ------------------------------------------
#' Merge per-cell-type LDSC .results files into a single table
#'
#' @param base_dir         Root directory whose immediate subdirectories are cell types.
#' @param output_file      Path of the merged TSV output file.
#' @param target_celltypes Character vector or NULL. Cell types to keep; all if NULL.
#' @param target_diseases  Character vector. Keep only result files whose name
#'                         contains one of these disease codes.
#' @return data.frame of merged LDSC results annotated with Cell_Type, Disease
#'         and Source_File columns.
merge_sldsc_results <- function(base_dir, output_file, target_celltypes = NULL, target_diseases) {

  # List cell type directories (one subdirectory per cell type)
  all_cell_dirs <- list.dirs(base_dir, full.names = TRUE, recursive = FALSE)
  if (!is.null(target_celltypes)) {
    all_cell_dirs <- all_cell_dirs[basename(all_cell_dirs) %in% target_celltypes]
  }
  if (length(all_cell_dirs) == 0) {
    stop("No cell type directories found. Check base_dir or target_celltypes.")
  }

  all_data_list <- list()

  for (cell_dir in all_cell_dirs) {
    cell_type <- basename(cell_dir)

    # Find all .results files for this cell type, restricted to target diseases
    result_files <- list.files(cell_dir, pattern = ".*\\.results$", full.names = TRUE)
    result_files <- result_files[grepl(paste(target_diseases, collapse = "|"), result_files)]

    for (file_path in result_files) {
      # Extract the disease code from the file name (format: <disease>_<celltype>_ldsc.results)
      file_name <- basename(file_path)
      disease_name <- sub(paste0("_", cell_type, "_ldsc\\.results$"), "", file_name)

      # Detect and skip header/comment lines starting with "#"
      file_lines <- readLines(file_path, n = 10)
      data_start_line <- which.max(!grepl("^#", file_lines))
      skip_lines <- ifelse(data_start_line > 1, data_start_line - 1, 0)

      file_data <- tryCatch({
        read_tsv(file_path,
                 skip = skip_lines,
                 col_types = cols(.default = "c"),  # read as character to avoid format issues
                 show_col_types = FALSE)
      }, error = function(e) {
        warning("Failed to read file: ", file_path, " - error: ", e$message)
        return(NULL)
      })

      if (!is.null(file_data) && nrow(file_data) > 0) {
        file_data$Cell_Type <- cell_type
        file_data$Disease <- disease_name
        file_data$Source_File <- file_name
        all_data_list[[length(all_data_list) + 1]] <- file_data
      }
    }
  }

  if (length(all_data_list) == 0) {
    stop("No result files could be read. Check file format and paths.")
  }

  # Row-bind all tables (bind_rows handles mismatched column names)
  final_merged_data <- bind_rows(all_data_list) %>%
    select(Cell_Type, Disease, Source_File, everything())

  write_tsv(final_merged_data, output_file)
  message("\nMerged ", length(all_data_list), " files -> ", output_file,
          " (", nrow(final_merged_data), " rows x ", ncol(final_merged_data), " cols)")

  return(final_merged_data)
}

dir.create(SLDSC_OUTPUT_DIR, recursive = TRUE, showWarnings = FALSE)

merged_results <- merge_sldsc_results(SLDSC_INPUT_DIR, MERGED_OUTPUT_FILE,
                                      target_celltypes, target_diseases)

## Filter and annotate enrichments -------------------------------------------
# Keep the L2_0 annotation level and coerce enrichment columns to numeric
sldsc_filtered <- merged_results %>%
  filter(Category == "L2_0") %>%
  mutate(Enrichment_num = as.numeric(Enrichment),
         Enrichment_p = as.numeric(Enrichment_p))

# Cell lineage mapping: standard cell type -> lineage category
lineage_mapping <- list(
  'Glutamatergic neuron' = c('L2/3-CUX2 Glu', 'L2/3-SATB2 Glu', 'L4-RORB Glu',
                             'L5-BCL11B Glu', 'L5/6-TLE4 Glu', 'L5/6-THEMIS Glu',
                             'Thalamic_Glu', 'Hippocampal_Glu', 'Granule_neuron',
                             'Hypothalamic_Glu'),
  'GABAergic neuron' = c('PVALB+ GABA', 'SST+ GABA', 'LHX6+LAMP5+ GABA',
                         'CALB2+ GABA', 'VIP+ GABA', 'RELN+ GABA',
                         'Medium_spiny_neuron', 'Purkinje_neuron', 'Thalamic_GABA',
                         'Midbrain-derived_GABA'),
  'Glia' = c('Astrocyte', 'Ependymal', 'Oligodendrocyte'),
  'Non-neural' = c('Microglia', 'Macrophage', 'Fibroblast', 'Endothelial_cell',
                   'Pericyte', 'Choroid_plexus_epithelial', 'Vascular_smooth_muscle_cell'),
  'Neural precursor cell' = c('Glioblast', 'Neuroblast', 'Neuronal_IPC', 'Radial_glia',
                              'Oligodendrocyte_precursor_committed', 'Oligodendrocyte_precursor'),
  'Other neuron' = c('Serotonergic', 'Cholinergic')
)

# Disease category mapping
disease_mapping <- list(
  'Behavioral-cognitive phenotype' = c('CI', 'EA', 'INT', 'NEU', 'ASP', 'DPW', 'GRT', 'SC', 'SI'),
  'Neurological disorder' = c('INS', 'EPI', 'IS', 'AD', 'ALS', 'MS', 'PD'),
  'Psychiatric disorder' = c('OCD', 'TS', 'ANX', 'MDD', 'ADHD', 'AUD', 'PTSD', 'BIP', 'SCZ', 'ASD')
)

# File-level cell type names -> standard names
cell_type_conversion <- c(
  'Astrocyte' = 'Astrocyte',
  'Choroid_plexus_epithelial' = 'Choroid_plexus_epithelial',
  'L5.6-THEMIS_Glu' = 'L5/6-THEMIS Glu',
  'RELN._GABA' = 'RELN+ GABA',
  'Pericyte' = 'Pericyte',
  'Glioblast' = 'Glioblast',
  'Granule_neuron' = 'Granule_neuron',
  'Hypothalamic_Glu' = 'Hypothalamic_Glu',
  'Oligodendrocyte' = 'Oligodendrocyte',
  'Fibroblast' = 'Fibroblast',
  'Hippocampal_Glu' = 'Hippocampal_Glu',
  'L5.6-TLE4_Glu' = 'L5/6-TLE4 Glu',
  'Midbrain-derived_GABA' = 'Midbrain-derived_GABA',
  'VIP._GABA' = 'VIP+ GABA',
  'PVALB._GABA' = 'PVALB+ GABA',
  'Endothelial_cell' = 'Endothelial_cell',
  'Ependymal' = 'Ependymal',
  'Cholinergic' = 'Cholinergic',
  'L2.3-CUX2_Glu' = 'L2/3-CUX2 Glu',
  'Medium_spiny_neuron' = 'Medium_spiny_neuron',
  'Oligodendrocyte_precursor' = 'Oligodendrocyte_precursor',
  'Radial_glia' = 'Radial_glia',
  'Neuronal_IPC' = 'Neuronal_IPC',
  'Microglia' = 'Microglia',
  'L4-RORB_Glu' = 'L4-RORB Glu',
  'LHX6.LAMP5._GABA' = 'LHX6+LAMP5+ GABA',
  'Macrophage' = 'Macrophage',
  'L5-BCL11B_Glu' = 'L5-BCL11B Glu',
  'Purkinje_neuron' = 'Purkinje_neuron',
  'SST._GABA' = 'SST+ GABA',
  'CALB2._GABA' = 'CALB2+ GABA',
  'Serotonergic' = 'Serotonergic',
  'Thalamic_Glu' = 'Thalamic_Glu',
  'Neuroblast' = 'Neuroblast',
  'Vascular_smooth_muscle_cell' = 'Vascular_smooth_muscle_cell',
  'Oligodendrocyte_precursor_committed' = 'Oligodendrocyte_precursor_committed',
  'L2.3-SATB2_Glu' = 'L2/3-SATB2 Glu'
)

# Map file-level names to standard names and assign lineage/disease categories
sldsc_filtered <- sldsc_filtered %>%
  mutate(Cell_Type_standard = recode(Cell_Type, !!!cell_type_conversion))

#' Return the lineage category of a standard cell type (NA if unknown)
get_cell_category <- function(cell_type) {
  for (category in names(lineage_mapping)) {
    if (cell_type %in% lineage_mapping[[category]]) {
      return(category)
    }
  }
  return(NA)
}

#' Return the category of a disease (NA if unknown)
get_disease_category <- function(disease) {
  for (category in names(disease_mapping)) {
    if (disease %in% disease_mapping[[category]]) {
      return(category)
    }
  }
  return(NA)
}

# Assign categories and compute FDR-corrected enrichment significance
sldsc_filtered <- sldsc_filtered %>%
  mutate(
    Cell_Category = sapply(Cell_Type_standard, get_cell_category),
    Disease_Category = sapply(Disease, get_disease_category),
    Enrichment_p = as.numeric(Enrichment_p),
    Enrichment_FDR = p.adjust(Enrichment_p, method = "BH"),
    log10FDR = -log10(Enrichment_FDR),
    log10p = -log10(Enrichment_p)
  )

write.csv(sldsc_filtered, file = FILTERED_OUTPUT_FILE, row.names = FALSE)
message("Filtered LDSC results written to: ", FILTERED_OUTPUT_FILE)

## Fisher exact enrichment of cell types across regions and stages -----------
# Load cell metadata and build group/type counts
meta <- read_csv(META_FILE, show_col_types = FALSE)
meta <- as.data.frame(meta)  # tibbles do not support row names
rownames(meta) <- meta$cell_name
meta <- meta[meta$type_2 != "Remove", ]
total_cells <- nrow(meta)

region_totals <- meta %>% count(region_L2) %>% rename(total_group = n)
stage_totals <- meta %>% count(stage) %>% rename(total_group = n)
type_totals <- meta %>% count(type_2) %>% rename(total_type = n)

#' One-sided Fisher exact test of cell-type enrichment within a grouping factor
#'
#' For each (cell type, group) pair, builds the 2x2 contingency table
#' (in group / outside group) x (in cell type / outside cell type) and tests
#' for over-representation (alternative = "greater").
#'
#' @param cluster_counts data.frame with columns type_2, <group_col> and count.
#' @param group_totals   data.frame with a <group_col> column and a total_group column.
#' @param type_totals    data.frame with per-cell-type totals.
#' @param group_col      Column name of the grouping factor (e.g. "region_L2" or "stage").
#' @param total_cells    Total number of cells.
#' @return data.frame with enrichment score (odds ratio), -log10 p-value and
#'         significance stars per (cell type, group) pair.
compute_fisher_enrichment <- function(cluster_counts, group_totals, type_totals, group_col, total_cells) {
  enrichment <- cluster_counts %>%
    left_join(group_totals, by = group_col) %>%
    left_join(type_totals, by = "type_2") %>%
    mutate(
      a = count,                    # cells of this type in this group
      b = total_type - count,       # cells of this type outside the group
      c = total_group - count,      # cells of other types in this group
      d = total_cells - a - b - c,  # cells of other types outside the group
      fisher_result = pmap(list(a, b, c, d), function(a, b, c, d) {
        fisher.test(matrix(c(a, b, c, d), nrow = 2), alternative = "greater")
      }),
      p_value = map_dbl(fisher_result, ~ .x$p.value),
      odds_ratio = map_dbl(fisher_result, ~ .x$estimate),
      enrichment_score = odds_ratio,
      neg_log_p = -log10(p_value),
      significance_star = case_when(
        p_value < 0.001 ~ "***",
        p_value < 0.01 ~ "**",
        p_value < 0.05 ~ "*",
        TRUE ~ ""
      )
    ) %>%
    select(type_2, all_of(group_col), enrichment_score, neg_log_p, significance_star)

  return(enrichment)
}

# Region-level enrichment
cluster_region_counts <- meta %>% count(type_2, region_L2) %>% rename(count = n)
enrichment_region <- compute_fisher_enrichment(
  cluster_region_counts, region_totals, type_totals, "region_L2", total_cells
)
write_csv(enrichment_region, REGION_ENRICHMENT_FILE)
message("Region enrichment written to: ", REGION_ENRICHMENT_FILE)

# Stage-level enrichment
cluster_stage_counts <- meta %>% count(type_2, stage) %>% rename(count = n)
enrichment_stage <- compute_fisher_enrichment(
  cluster_stage_counts, stage_totals, type_totals, "stage", total_cells
)
write_csv(enrichment_stage, STAGE_ENRICHMENT_FILE)
message("Stage enrichment written to: ", STAGE_ENRICHMENT_FILE)
