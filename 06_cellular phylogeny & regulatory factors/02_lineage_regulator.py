#!/usr/bin/env python3
"""
=============================================================================
File      : 02_lineage_regulator.py
Directory : 06_cellular phylogeny & regulatory factors
Purpose   : Lineage regulator (TF / RBP / TF+RBP) analysis across cell types.
=============================================================================
"""

import os
import re
import warnings

import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ============================================================================
# Configuration: Set the project root directory before running
# ============================================================================
# proj_dir = "path/to/your/project"  # <-- MODIFY THIS
proj_dir = os.environ.get("PROJ_DIR", "path/to/your/project")
os.chdir(proj_dir)
np.random.seed(123)

# Human data directory
data_dir = os.path.join(proj_dir, "data", "human")


# ============================================================================
# Shared subsampling helper
# ============================================================================
def create_subsampled_adata_full_genes(meta_path, h5ad_path, sample_cell_num=200000):
    """
    Subsample cells from the h5ad file while retaining all genes.

    Parameters
    ----------
    meta_path : str
        Path to the metadata CSV file.
    h5ad_path : str
        Path to the h5ad file.
    sample_cell_num : int
        Number of cells to sample.

    Returns
    -------
    AnnData, DataFrame
        Downsampled AnnData object (all genes) and sampled metadata.
    """
    print("Reading metadata...")
    meta = pd.read_csv(meta_path, index_col=0)
    meta = meta[meta["type_2"] != "Remove"]
    meta["stage_abbr"] = "S" + meta["stage"].str.split("-").str[0]

    print(f"Randomly sampling {sample_cell_num} cells from {len(meta)} cells...")
    if sample_cell_num < len(meta):
        sampled_indices = np.random.choice(meta.index, size=sample_cell_num, replace=False)
        sampled_meta = meta.loc[sampled_indices]
    else:
        sampled_meta = meta
        sampled_indices = meta.index

    print("Opening h5ad file in backed read-only mode...")
    ad_hs_orig = sc.read_h5ad(h5ad_path, backed="r")

    print("Reading all genes...")
    selected_genes = ad_hs_orig.var_names.tolist()
    print(f"Selected all {len(selected_genes)} genes")

    print("Matching cell indices...")
    available_cells = set(ad_hs_orig.obs_names) & set(sampled_indices)
    selected_cells = list(available_cells)
    selected_cell_indices = [ad_hs_orig.obs_names.get_loc(cell) for cell in selected_cells]
    print(f"Found {len(selected_cells)} matched cells")

    print("Reading expression matrix in chunks (all genes)...")
    n_cells_sampled = len(selected_cells)
    n_genes_selected = len(selected_genes)
    X_new = sp.lil_matrix((n_cells_sampled, n_genes_selected), dtype=ad_hs_orig.X.dtype)
    chunk_size = 20000
    selected_cell_indices_sorted = np.sort(selected_cell_indices)
    old_to_new_index = {old_idx: new_idx for new_idx, old_idx in enumerate(selected_cell_indices)}

    for start in tqdm(range(0, len(selected_cell_indices_sorted), chunk_size)):
        end = min(start + chunk_size, len(selected_cell_indices_sorted))
        batch_old_indices = selected_cell_indices_sorted[start:end]
        chunk_start = batch_old_indices[0]
        chunk_end = batch_old_indices[-1] + 1
        chunk_data = ad_hs_orig.X[chunk_start:chunk_end, :]

        for i, old_idx in enumerate(batch_old_indices):
            new_idx = old_to_new_index[old_idx]
            relative_idx_in_chunk = old_idx - chunk_start
            if sp.issparse(chunk_data):
                X_new[new_idx] = chunk_data[relative_idx_in_chunk].toarray().flatten()
            else:
                X_new[new_idx] = chunk_data[relative_idx_in_chunk]

    print("Converting to CSR format...")
    X_new_csr = X_new.tocsr()

    print("Building new AnnData object...")
    selected_obs = ad_hs_orig.obs.loc[selected_cells].copy()
    selected_var = ad_hs_orig.var.copy()
    ad_hs_sampled = sc.AnnData(
        X=X_new_csr, obs=selected_obs, var=selected_var, dtype=ad_hs_orig.X.dtype,
    )

    if "obsm" in ad_hs_orig.__dict__:
        for key in ad_hs_orig.obsm.keys():
            if key in ad_hs_orig.obsm:
                ad_hs_sampled.obsm[key] = ad_hs_orig.obsm[key][selected_cell_indices]
    if "uns" in ad_hs_orig.__dict__:
        ad_hs_sampled.uns = ad_hs_orig.uns.copy()

    ad_hs_orig.file.close()
    print(f"Done! New object shape: {ad_hs_sampled.shape}")
    print(f"Estimated memory usage: {X_new_csr.data.nbytes / 1024**2:.2f} MB")
    return ad_hs_sampled, sampled_meta


# ============================================================================
# Unified lineage regulator filter class
# ============================================================================
class LineageRegulatorFilter:
    """Filter for lineage-associated regulators (TFs, RBPs, or both).

    Identifies key regulators along the developmental lineage by:
    1. Finding each cell type's ancestor and sister cell types in the lineage tree.
    2. Differential expression vs. ancestor and vs. each sister (scanpy Wilcoxon).
    3. Three-criteria filtering: upregulated vs. ancestor, sufficient expression
       fraction, not upregulated in sisters.
    4. Composite z-score ranking.
    """

    # Broad lineage-level → specific cell type mapping
    LINEAGE_MAPPING = {
        "Glutamatergic neuron": [
            "L2/3-CUX2 Glu", "L2/3-SATB2 Glu", "L4-RORB Glu",
            "L5-BCL11B Glu", "L5/6-TLE4 Glu", "L5/6-THEMIS Glu",
            "Thalamic_Glu", "Hippocampal_Glu", "Granule_neuron",
            "Hypothalamic_Glu",
        ],
        "GABAergic neuron": [
            "PVALB+ GABA", "SST+ GABA", "LHX6+LAMP5+ GABA",
            "CALB2+ GABA", "VIP+ GABA", "RELN+ GABA",
            "Medium_spiny_neuron", "Purkinje_neuron", "Thalamic_GABA",
            "Midbrain-derived_GABA",
        ],
    }

    def __init__(self, final_tree_path, gene_list, adata):
        """
        Parameters
        ----------
        final_tree_path : str
            Path to the differentiation tree file (tab-separated: pre, nex, prob).
        gene_list : list of str
            Gene symbols to restrict the analysis to (e.g. TF list, RBP list,
            or the union of both).
        adata : AnnData
            Single-cell data (full or downsampled).
        """
        self.final_tree = pd.read_csv(
            final_tree_path, sep="\t", header=None,
            names=["pre", "nex", "prob"],
        )
        self.gene_list = gene_list
        self.adata = adata

        # Subset to genes present in the data
        genes_in_data = [g for g in gene_list if g in self.adata.var_names]
        self.adata_subset = adata[:, genes_in_data]

        # Parse state names: "S02:ExN" → pre_cell="ExN", pre_time="S02"
        self.final_tree["pre_cell"] = self.final_tree["pre"].str.split(":").str[1]
        self.final_tree["nex_cell"] = self.final_tree["nex"].str.split(":").str[1]
        self.final_tree["pre_time"] = self.final_tree["pre"].str.split(":").str[0]
        self.final_tree["nex_time"] = self.final_tree["nex"].str.split(":").str[0]

    # ------------------------------------------------------------------
    # Tree helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _extract_time_number(time_str):
        return float(time_str.replace("S", "")) if time_str.startswith("S") else float(time_str)

    def find_ancestor_and_sisters(self, target_cell_type):
        """Find the ancestor and sister cell types of a target cell type.

        Returns (ancestor: str | None, sisters: list of str).
        """
        target_edges = self.final_tree[self.final_tree["nex_cell"] == target_cell_type]
        if target_edges.empty:
            return None, []

        target_edges = target_edges.copy()
        target_edges["time_num"] = target_edges["nex_time"].apply(self._extract_time_number)
        earliest_time_num = target_edges["time_num"].min()
        earliest_edges = target_edges[target_edges["time_num"] == earliest_time_num]
        if earliest_edges.empty:
            return None, []

        main_edge = earliest_edges.loc[earliest_edges["prob"].idxmax()]
        ancestor = main_edge["pre_cell"]
        earliest_time_point = main_edge["nex_time"]

        print(f"Cell type '{target_cell_type}' first appears at {earliest_time_point}, "
              f"ancestor is '{ancestor}'")

        sister_edges = self.final_tree[
            (self.final_tree["pre_cell"] == ancestor)
            & (self.final_tree["nex_time"] == earliest_time_point)
            & (self.final_tree["nex_cell"] != target_cell_type)
            & (self.final_tree["nex_cell"] != ancestor)
        ]
        sisters = sister_edges["nex_cell"].unique().tolist()
        if sisters:
            print(f"  Sister cell types (same-time divergence): {sisters}")
        else:
            print("  No same-time sister cell types")
        return ancestor, sisters

    # ------------------------------------------------------------------
    # Differential expression
    # ------------------------------------------------------------------
    def _get_cell_indices(self, cell_type):
        """Resolve a lineage-level or specific cell type name to cell indices."""
        if cell_type in self.LINEAGE_MAPPING:
            specific_types = self.LINEAGE_MAPPING[cell_type]
            return self.adata_subset.obs[
                self.adata_subset.obs["type_2"].isin(specific_types)
            ].index
        return self.adata_subset.obs[
            self.adata_subset.obs["type_2"] == cell_type
        ].index

    def calculate_differential_expression(self, cell_type1, cell_type2,
                                          save_prefix=None):
        """Compute DE between two cell types using scanpy Wilcoxon test.

        Parameters
        ----------
        cell_type1 : str  — target cell type
        cell_type2 : str  — reference (ancestor or sister)
        save_prefix : str, optional  — if given, save full DE table to disk

        Returns DataFrame with columns: gene, log2fc, pval, pval_adj, pct1
        """
        # Root ancestor: all genes considered uniformly expressed
        if cell_type2 == "Root":
            return pd.DataFrame({
                "gene": self.gene_list,
                "log2fc": [1.0] * len(self.gene_list),
                "pval": [0.0] * len(self.gene_list),
                "pval_adj": [0.0] * len(self.gene_list),
                "pct1": [1.0] * len(self.gene_list),
            })

        ct1_cells = self._get_cell_indices(cell_type1)
        ct2_cells = self._get_cell_indices(cell_type2)

        if len(ct1_cells) == 0 or len(ct2_cells) == 0:
            print(f"Warning: not enough cells ({cell_type1}: {len(ct1_cells)}, "
                  f"{cell_type2}: {len(ct2_cells)})")
            return pd.DataFrame()

        temp_adata = self.adata_subset[ct1_cells.union(ct2_cells)].copy()
        temp_adata.obs["group"] = "other"
        temp_adata.obs.loc[ct2_cells, "group"] = "group2"
        temp_adata.obs.loc[ct1_cells, "group"] = "group1"

        sc.tl.rank_genes_groups(
            temp_adata, groupby="group", groups=["group1"],
            reference="group2", method="wilcoxon", pts=True,
            corr_method="bonferroni",
        )

        de_results = pd.DataFrame({
            "gene": temp_adata.uns["rank_genes_groups"]["names"]["group1"],
            "log2fc": temp_adata.uns["rank_genes_groups"]["logfoldchanges"]["group1"],
            "pval": temp_adata.uns["rank_genes_groups"]["pvals"]["group1"],
            "pval_adj": temp_adata.uns["rank_genes_groups"]["pvals_adj"]["group1"],
            "pct1": temp_adata.uns["rank_genes_groups"]["pts"]["group1"],
        })

        if save_prefix:
            self._save_de_results(de_results, cell_type1, cell_type2, save_prefix)

        del temp_adata
        return de_results

    @staticmethod
    def _clean_cell_type_name(type_name):
        """Sanitize cell type names by replacing illegal filename characters."""
        cleaned = re.sub(r'[\\/:*?"<>|+ ]', "_", type_name)
        if cleaned and cleaned[0] in "0123456789!@#$%^&*()-=+":
            cleaned = "type_" + cleaned
        return cleaned

    def _save_de_results(self, de_results, cell_type1, cell_type2, save_prefix):
        """Save per-comparison DE results to disk."""
        clean_ct1 = self._clean_cell_type_name(cell_type1)
        clean_ct2 = self._clean_cell_type_name(cell_type2)
        clean_prefix = self._clean_cell_type_name(save_prefix)
        save_dir = os.path.join(
            data_dir, "output", "Analysis", "lineage", "all_cell",
            "TF_RBP", clean_prefix,
        )
        os.makedirs(save_dir, exist_ok=True)
        filename = os.path.join(save_dir, f"{clean_ct1}_vs_{clean_ct2}_de_results.csv")
        de_results.to_csv(filename, index=False)
        print(f"DE results saved to: {filename}")

    # ------------------------------------------------------------------
    # Three-criteria filtering
    # ------------------------------------------------------------------
    def filter_by_criteria(self, target_cell_type, de_ancestor, de_sisters,
                           pval_threshold=0.05, pct_threshold=0.1,
                           log2fc_threshold=0.0, ancestor=None):
        """Filter regulators by three criteria.

        1. Upregulated vs. ancestor (log2fc > log2fc_threshold, pval_adj < pval_threshold).
        2. Expressed in ≥ pct_threshold fraction of target cells.
        3. Not significantly upregulated in any sister (vs. target).

        When the ancestor is "Root", criterion 3 requires significant
        upregulation in ≥50% of sisters instead.
        """
        if de_ancestor.empty:
            return []

        filtered = []
        counts = {"rule1": 0, "rule2": 0, "rule3": 0}
        is_root = (ancestor == "Root") if ancestor else False

        for _, row in de_ancestor.iterrows():
            gene = row["gene"]
            passed = True

            # Rule 1
            if not (row["log2fc"] > log2fc_threshold and row["pval_adj"] < pval_threshold):
                counts["rule1"] += 1
                passed = False
                continue

            # Rule 2
            if row["pct1"] < pct_threshold:
                counts["rule2"] += 1
                passed = False
                continue

            # Rule 3
            if is_root:
                if de_sisters:
                    up_sisters, valid_sisters = 0, 0
                    for sister, sister_de in de_sisters.items():
                        if sister_de.empty:
                            continue
                        sister_row = sister_de[sister_de["gene"] == gene]
                        if sister_row.empty:
                            continue
                        valid_sisters += 1
                        sr = sister_row.iloc[0]
                        if (sr["log2fc"] > 0 and sr["pval_adj"] < pval_threshold
                                and sr["pct1"] >= pct_threshold):
                            up_sisters += 1
                    if valid_sisters > 0:
                        if up_sisters / valid_sisters < 0.5:
                            passed = False
                    else:
                        passed = False
                # else: no sisters → automatically satisfied
            else:
                for sister, sister_de in de_sisters.items():
                    if sister_de.empty:
                        continue
                    sister_row = sister_de[sister_de["gene"] == gene]
                    if not sister_row.empty:
                        sr = sister_row.iloc[0]
                        if sr["log2fc"] < 0 and sr["pval_adj"] < pval_threshold:
                            passed = False
                            break

            if not passed:
                counts["rule3"] += 1
            else:
                filtered.append(gene)

        print(f"    Regulators failing rule 1: {counts['rule1']}")
        print(f"    Regulators failing rule 2: {counts['rule2']}")
        print(f"    Regulators failing rule 3: {counts['rule3']}")
        return filtered

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------
    @staticmethod
    def _safe_z_score(value, values_array):
        """Z-score with protection against zero standard deviation."""
        mean_val = np.mean(values_array)
        std_val = np.std(values_array)
        if std_val < 1e-10:
            return 0.0
        return (value - mean_val) / std_val

    def calculate_scores(self, filtered_genes, de_ancestor, de_sisters):
        """Compute composite z-scores for filtered regulators.

        Returns DataFrame with columns: tf, log2fc_ancestor, log2fc_sister_avg,
        z_ancestor, z_sister, composite_score.
        """
        all_ancestor_fc = de_ancestor["log2fc"].values
        all_sister_fc = []
        for sd in de_sisters.values():
            if not sd.empty:
                all_sister_fc.extend(sd["log2fc"].values)
        if not all_sister_fc:
            all_sister_fc = [0.0]

        scores = []
        for gene in filtered_genes:
            ancestor_row = de_ancestor[de_ancestor["gene"] == gene]
            if ancestor_row.empty:
                continue
            log2fc_ancestor = ancestor_row.iloc[0]["log2fc"]

            sister_log2fcs = []
            for sister, sister_de in de_sisters.items():
                if sister_de.empty:
                    continue
                sr = sister_de[sister_de["gene"] == gene]
                if not sr.empty:
                    sister_log2fcs.append(sr.iloc[0]["log2fc"])

            log2fc_sister_avg = np.mean(sister_log2fcs) if sister_log2fcs else 0.0
            z_ancestor = self._safe_z_score(log2fc_ancestor, all_ancestor_fc)
            z_sister = self._safe_z_score(log2fc_sister_avg, all_sister_fc)
            composite = z_ancestor if not sister_log2fcs else (z_ancestor + z_sister) / 2

            scores.append({
                "gene": gene,
                "log2fc_ancestor": log2fc_ancestor,
                "log2fc_sister_avg": log2fc_sister_avg,
                "z_ancestor": z_ancestor,
                "z_sister": z_sister,
                "composite_score": composite,
            })
        return pd.DataFrame(scores)

    # ------------------------------------------------------------------
    # Main run method
    # ------------------------------------------------------------------
    def run_analysis(self, pval_threshold=0.05, pct_threshold=0.1,
                     log2fc_threshold=0.0, top_n=5, save_de_results=False):
        """Run the full lineage regulator analysis.

        Returns (all_filtered_df, top_regulators_df).
        """
        all_results = []
        top_results = []

        all_cell_types = [ct for ct in self.final_tree["nex_cell"].unique()
                          if not ct.startswith("Root")]
        print(f"Analyzing {len(all_cell_types)} cell types...")

        for i, cell_type in enumerate(all_cell_types, 1):
            print(f"\nProcessing cell type {i}/{len(all_cell_types)}: {cell_type}")

            ancestor, sisters = self.find_ancestor_and_sisters(cell_type)
            if ancestor is None:
                print(f"  Warning: no ancestor found for {cell_type}")
                continue
            print(f"  Ancestor: {ancestor}, Sisters: {sisters}")

            # DE vs. ancestor
            save_prefix = f"de_results/{cell_type}" if save_de_results else None
            de_ancestor = self.calculate_differential_expression(
                cell_type, ancestor, save_prefix,
            )
            if de_ancestor.empty:
                print(f"  Warning: cannot compute DE for {cell_type} vs {ancestor}")
                continue

            # DE vs. each sister
            de_sisters = {}
            for sister in sisters:
                sp = f"de_results/{cell_type}" if save_de_results else None
                de_sisters[sister] = self.calculate_differential_expression(
                    cell_type, sister, sp,
                )

            # Filter
            filtered_genes = self.filter_by_criteria(
                cell_type, de_ancestor, de_sisters,
                pval_threshold, pct_threshold, log2fc_threshold, ancestor,
            )
            print(f"  Regulators passing criteria: {len(filtered_genes)}")
            if not filtered_genes:
                continue

            # Score
            gene_scores = self.calculate_scores(filtered_genes, de_ancestor, de_sisters)

            # Collect all passing regulators
            for _, row in gene_scores.iterrows():
                all_results.append({
                    "cell_type": cell_type, "ancestor": ancestor,
                    "sisters": ",".join(sisters),
                    "gene": row["gene"],
                    "log2fc_ancestor": row["log2fc_ancestor"],
                    "log2fc_sister_avg": row["log2fc_sister_avg"],
                    "composite_score": row["composite_score"],
                    "z_ancestor": row["z_ancestor"],
                    "z_sister": row["z_sister"],
                    "is_top": False,
                })

            # Top-N
            top_n_df = gene_scores.nlargest(top_n, "composite_score")
            for rank, (_, row) in enumerate(top_n_df.iterrows(), 1):
                top_results.append({
                    "cell_type": cell_type, "ancestor": ancestor,
                    "sisters": ",".join(sisters),
                    "gene": row["gene"],
                    "log2fc_ancestor": row["log2fc_ancestor"],
                    "log2fc_sister_avg": row["log2fc_sister_avg"],
                    "composite_score": row["composite_score"],
                    "z_ancestor": row["z_ancestor"],
                    "z_sister": row["z_sister"],
                    "rank": rank,
                })

            # Mark top genes
            top_gene_names = set(top_n_df["gene"].tolist())
            for rec in all_results:
                if rec["cell_type"] == cell_type and rec["gene"] in top_gene_names:
                    rec["is_top"] = True

            # Save per-cell-type tables when detailed DE is requested
            if save_de_results:
                clean_ct = self._clean_cell_type_name(cell_type)
                save_dir = os.path.join(
                    data_dir, "output", "Analysis", "lineage",
                    "all_cell", "TF_RBP", "de_results",
                )
                os.makedirs(save_dir, exist_ok=True)

                cell_all = [r for r in all_results if r["cell_type"] == cell_type]
                cell_top = [r for r in top_results if r["cell_type"] == cell_type]
                pd.DataFrame(cell_all).to_csv(
                    os.path.join(save_dir, f"{clean_ct}_all_filtered_regulator.csv"),
                    index=False,
                )
                pd.DataFrame(cell_top).to_csv(
                    os.path.join(save_dir, f"{clean_ct}_top{top_n}_regulator.csv"),
                    index=False,
                )

        return pd.DataFrame(all_results), pd.DataFrame(top_results)


# ============================================================================
# Helper: load gene lists
# ============================================================================
def load_tf_list(path):
    """Load TF symbols from a tab-separated file (column: 'Symbol')."""
    return pd.read_csv(path, sep="\t")["Symbol"].dropna().unique().tolist()


def load_rbp_list(path):
    """Load RBP names from a comma-separated file (column: 'Name')."""
    return pd.read_csv(path, sep=",")["Name"].dropna().unique().tolist()


def annotate_regulator_types(result_csv, tf_list, rbp_list, output_csv):
    """Add a 'regulator_type' column (TF / RBP / TF and RBP / Unknown)."""
    df = pd.read_csv(result_csv, encoding="utf-8")
    tf_set, rbp_set = set(tf_list), set(rbp_list)

    def classify(gene):
        if pd.isna(gene):
            return "Unknown"
        in_tf, in_rbp = gene in tf_set, gene in rbp_set
        if in_tf and in_rbp:
            return "TF and RBP"
        if in_tf:
            return "TF"
        if in_rbp:
            return "RBP"
        return "Unknown"

    df["regulator_type"] = df["gene"].apply(classify)
    df.to_csv(output_csv, index=False)
    print(f"Annotated regulator table saved to: {output_csv}")
    return df


# ============================================================================
# Main workflow
# ============================================================================
def main():
    # --- Shared inputs -------------------------------------------------------
    meta_path = os.path.join(
        data_dir, "output", "Annotation_hvg5000", "final",
        "hs_all_metadata_update.csv",
    )
    h5ad_path = os.path.join(data_dir, "output", "Integration", "hs_norm.h5ad")
    tree_path = os.path.join(
        data_dir, "output", "Analysis", "lineage", "all_cell",
        "human_edge_pca_k5.txt",
    )
    tf_file = os.path.join(
        data_dir, "output", "Analysis", "lineage", "sample20w", "TF",
        "Homo_sapiens_TF.txt",
    )
    rbp_file = os.path.join(
        data_dir, "output", "Analysis", "lineage", "RBP", "RBP.csv",
    )

    # --- Load data -----------------------------------------------------------
    ad_hs = sc.read(h5ad_path)
    meta = pd.read_csv(meta_path, index_col=0)
    meta = meta[meta["type_2"] != "Remove"]
    meta["stage_abbr"] = "S" + meta["stage"].str.split("-").str[0]

    ad_hs_sampled, sampled_meta = create_subsampled_adata_full_genes(
        meta_path=meta_path, h5ad_path=h5ad_path, sample_cell_num=200000,
    )

    # Attach annotation to both objects
    anno_cols = ["stage", "stage_abbr", "type_group", "type_1", "type_2",
                 "type_3", "region_L1", "region_L2", "region_L3"]
    for obj, src_meta in [(ad_hs_sampled, sampled_meta), (ad_hs, meta)]:
        obj.obs = obj.obs.drop(columns=["stage"], errors="ignore")
        obj.obs = obj.obs.join(src_meta[anno_cols], how="left")

    # Save downsampled object
    sample_dir = os.path.join(data_dir, "output", "Analysis", "lineage", "sample20w")
    os.makedirs(sample_dir, exist_ok=True)
    ad_hs_sampled.write_h5ad(os.path.join(sample_dir, "ad_sample20w.h5ad"))

    # --- Load gene lists -----------------------------------------------------
    tf_list = load_tf_list(tf_file)
    rbp_list = load_rbp_list(rbp_file)
    combined_list = list(set(tf_list).union(set(rbp_list)))
    print(f"Loaded {len(tf_list)} TFs, {len(rbp_list)} RBPs, "
          f"{len(combined_list)} combined")

    # ======================================================================
    # Analysis 1: RBP-only (full dataset)
    # ======================================================================
    print("\n" + "=" * 60)
    print("Analysis 1: RBP regulators (full dataset)")
    print("=" * 60)
    rbp_filter = LineageRegulatorFilter(tree_path, rbp_list, ad_hs)
    rbp_all, rbp_top = rbp_filter.run_analysis(
        pval_threshold=0.05, pct_threshold=0.1, top_n=8,
    )
    rbp_dir = os.path.join(data_dir, "output", "Analysis", "lineage", "all_cell", "RBP")
    os.makedirs(rbp_dir, exist_ok=True)
    rbp_all.to_csv(os.path.join(rbp_dir, "lineage_RBP.csv"), index=False)
    rbp_top.to_csv(os.path.join(rbp_dir, "lineage_RBP_top.csv"), index=False)

    # ======================================================================
    # Analysis 2: TF-only (sampled 200k, then full dataset)
    # ======================================================================
    print("\n" + "=" * 60)
    print("Analysis 2a: TF regulators (200k sampled cells)")
    print("=" * 60)
    tf_filter_sampled = LineageRegulatorFilter(tree_path, tf_list, ad_hs_sampled)
    tf_s_all, tf_s_top = tf_filter_sampled.run_analysis(
        pval_threshold=0.05, pct_threshold=0.1, top_n=5,
    )
    tf_s_dir = os.path.join(data_dir, "output", "Analysis", "lineage", "sample20w", "TF")
    os.makedirs(tf_s_dir, exist_ok=True)
    tf_s_all.to_csv(os.path.join(tf_s_dir, "lineage_TF.csv"), index=False)
    tf_s_top.to_csv(os.path.join(tf_s_dir, "lineage_TF_top.csv"), index=False)

    print("\n" + "=" * 60)
    print("Analysis 2b: TF regulators (full dataset)")
    print("=" * 60)
    tf_filter_full = LineageRegulatorFilter(tree_path, tf_list, ad_hs)
    tf_f_all, tf_f_top = tf_filter_full.run_analysis(
        pval_threshold=0.05, pct_threshold=0.1, top_n=8,
    )
    tf_f_dir = os.path.join(data_dir, "output", "Analysis", "lineage", "all_cell", "TF")
    os.makedirs(tf_f_dir, exist_ok=True)
    tf_f_all.to_csv(os.path.join(tf_f_dir, "lineage_TF.csv"), index=False)
    tf_f_top.to_csv(os.path.join(tf_f_dir, "lineage_TF_top.csv"), index=False)

    # ======================================================================
    # Analysis 3: TF+RBP combined (full dataset, with per-cell-type DE saves)
    # ======================================================================
    print("\n" + "=" * 60)
    print("Analysis 3: TF+RBP combined regulators (full dataset)")
    print("=" * 60)
    combined_filter = LineageRegulatorFilter(tree_path, combined_list, ad_hs)
    comb_all, comb_top = combined_filter.run_analysis(
        pval_threshold=0.05, pct_threshold=0.1, log2fc_threshold=0.5,
        top_n=20, save_de_results=True,
    )
    comb_dir = os.path.join(data_dir, "output", "Analysis", "lineage", "all_cell", "TF_RBP")
    os.makedirs(comb_dir, exist_ok=True)
    comb_all.to_csv(os.path.join(comb_dir, "lineage_regulator.csv"), index=False)
    comb_top.to_csv(os.path.join(comb_dir, "lineage_regulator_top.csv"), index=False)

    # Annotate regulator types in the combined results
    annotate_regulator_types(
        os.path.join(comb_dir, "lineage_regulator.csv"),
        tf_list, rbp_list,
        os.path.join(comb_dir, "lineage_regulator_info.csv"),
    )

    print("\nAll lineage regulator analyses complete.")


if __name__ == "__main__":
    main()
