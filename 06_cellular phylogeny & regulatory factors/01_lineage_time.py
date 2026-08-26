#!/usr/bin/env python3
"""
=============================================================================
File      : 01_lineage_time.py
Directory : 06_cellular phylogeny & regulatory factors
Purpose   : Temporal lineage analysis across developmental stages.
=============================================================================
"""

import os
import pickle
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from typing import List, Optional

# ============================================================================
# Configuration: Set the project root directory before running
# ============================================================================
# proj_dir = "path/to/your/project"  # <-- MODIFY THIS
proj_dir = os.environ.get("PROJ_DIR", "path/to/your/project")

# Data directory holding the input matrices and outputs (as in the original run)
data_dir = os.path.join(proj_dir, "data", "human")
os.chdir(data_dir)
np.random.seed(123)


def createLineage_Knn(emb: np.ndarray,
                      annotations_pd: pd.DataFrame,
                      reduction: str = "umap",
                      replication_times: int = 100,
                      removing_cells_ratio: float = 0.2,
                      k_neigh: int = 5,
                      random_seed: Optional[int] = None) -> List[pd.DataFrame]:
    """Reconstruct cell trajectories via k-NN and compute edge weights between cell states.

    Parameters
    ----------
    emb : np.ndarray
        Embedding matrix (n_cells x n_dims)
    annotations_pd : pd.DataFrame
        Cell annotations; must contain 'Anno' and 'relation' columns
    reduction : str
        Dimensionality reduction used ('umap' or 'pca')
    replication_times : int
        Number of bootstrap replicates
    removing_cells_ratio : float
        Fraction of cells excluded per bootstrap sample
    k_neigh : int
        k-NN parameter

    Returns
    -------
    List[pd.DataFrame]
        Edge weight matrices, one per bootstrap replicate
    """

    # Input validation
    if 'Anno' not in annotations_pd.columns or 'relation' not in annotations_pd.columns:
        raise ValueError("Error: pd must contain 'Anno' and 'relation' columns")
    if emb.shape[0] != len(annotations_pd):
        raise ValueError("Error: emb and pd row counts do not match")

    # Create a copy of the state column
    annotations_pd = annotations_pd.copy()
    annotations_pd['state'] = annotations_pd['Anno']

    results = []
    rep_i = 0

    while rep_i < replication_times:
        try:
            if random_seed is not None:
                # Unique but reproducible seed per replicate
                rep_seed = random_seed + rep_i
                rng = np.random.RandomState(rep_seed)
            else:
                rng = np.random  # no fixed seed

            # Bootstrap sampling: randomly keep 80% of cells
            n_cells = len(annotations_pd)
            n_sample = int(n_cells * (1 - removing_cells_ratio))
            sampling_indices = rng.choice(n_cells, n_sample, replace=False)

            emb_sub = emb.iloc[sampling_indices]
            pd_sub = annotations_pd.iloc[sampling_indices].copy()

            # Split data by time point
            pre_mask = pd_sub['relation'] == "pre"
            nex_mask = pd_sub['relation'] == "nex"

            emb_pre = emb_sub[pre_mask]
            emb_nex = emb_sub[nex_mask]
            pd_pre = pd_sub[pre_mask]
            pd_nex = pd_sub[nex_mask]

            if len(emb_pre) == 0 or len(emb_nex) == 0:
                print(f"Replicate {rep_i} skipped: empty split (pre: {len(emb_pre)}, nex: {len(emb_nex)})")
                rep_i += 1
                continue

            # Dynamically adjust k
            state_counts = pd_pre['state'].value_counts()
            if len(state_counts) == 0:
                print(f"Replicate {rep_i} skipped: no state counts in pd_pre")
                rep_i += 1
                continue

            pre_state_min = state_counts.min()
            current_k = k_neigh

            if k_neigh == 5:
                if pre_state_min < k_neigh:
                    if pre_state_min >= 3:  # need at least 3 samples to compute
                        current_k = min(pre_state_min, k_neigh)
                        print(f"Adjusted k from {k_neigh} to {current_k} (min state samples: {pre_state_min})")
                    elif pre_state_min >= 1:
                        current_k = 3
                    else:
                        print(f"Replicate {rep_i} skipped: too few state samples (min_samples: {pre_state_min})")
                        rep_i += 1
                        continue

            # Ensure k never exceeds the actual sample count
            current_k = min(current_k, len(emb_pre))
            if current_k < 1:
                print(f"Replicate {rep_i} skipped: invalid k after adjustment ({current_k})")
                rep_i += 1
                continue

            # k-NN nearest neighbor search
            nn = NearestNeighbors(n_neighbors=current_k, metric='euclidean')
            nn.fit(emb_pre)
            distances, indices = nn.kneighbors(emb_nex)

            # Build the state connection matrix
            state1 = pd_pre['state'].unique()
            state2 = pd_nex['state'].unique()

            neighbor_states = pd_pre['state'].values[indices]
            connection_matrix = np.zeros((len(state2), len(state1)))

            for i, target_state in enumerate(state2):
                target_mask = pd_nex['state'] == target_state
                if not target_mask.any():
                    continue

                target_neighbors = neighbor_states[target_mask]
                all_neighbor_states = target_neighbors.flatten()

                for j, source_state in enumerate(state1):
                    connection_count = np.sum(all_neighbor_states == source_state)
                    connection_matrix[i, j] = connection_count

            # Normalize to a transition-probability matrix (row sums = 1)
            row_sums = connection_matrix.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1  # avoid division by zero
            weight_matrix = connection_matrix / row_sums

            weight_df = pd.DataFrame(weight_matrix, index=state2, columns=state1)
            results.append(weight_df)
            rep_i += 1

            if rep_i % 20 == 0:
                print(f"Completed {rep_i}/{replication_times} replicates")

        except Exception as e:
            print(f"Replicate {rep_i} failed: {e}")
            rep_i += 1
            continue

    return results


def process_timepoints(work_path: str,
                       meta: pd.DataFrame,
                       emb_matrix: pd.DataFrame,
                       time_points: List[str]) -> None:
    """Run the lineage analysis for every pair of adjacent time points.

    Parameters
    ----------
    work_path : str
        Output directory
    meta : pd.DataFrame
        Cell metadata (must contain 'stage_abbr' and 'type_3')
    emb_matrix : pd.DataFrame
        Cell embedding, rows indexed by cell barcode
    time_points : List[str]
        Sorted list of stage labels
    """

    os.makedirs(work_path, exist_ok=True)

    for i in range(len(time_points) - 1):
        time_i = time_points[i]
        time_j = time_points[i + 1]
        print(f"\nProcessing time point pair: {time_i} -> {time_j}")

        type_i_index = meta.loc[meta['stage_abbr'] == time_i].index
        anno_i = pd.DataFrame({
            'relation': 'pre',
            'Anno': meta.loc[type_i_index, 'type_3'],
            'stage': time_i
        }, index=type_i_index)

        type_j_index = meta.loc[meta['stage_abbr'] == time_j].index
        anno_j = pd.DataFrame({
            'relation': 'nex',
            'Anno': meta.loc[type_j_index, 'type_3'],
            'stage': time_j
        }, index=type_j_index)

        anno_combined = pd.concat([anno_i, anno_j])
        emb_pre_nex = emb_matrix.loc[anno_combined.index]

        # Main k-NN analysis (k=5)
        results = createLineage_Knn(emb_pre_nex, anno_combined, k_neigh=5, random_seed=123)
        if not results:
            print(f"  No replicate succeeded for {time_i} -> {time_j}, skipping")
            continue

        # Save the bootstrap replicate weight matrices
        output_path = os.path.join(work_path, f"{time_i}_{time_j}_Knn_umap.rds")
        with open(output_path.replace('.rds', '.pkl'), 'wb') as f:
            pickle.dump(results, f)

        # Median weight matrix across replicates
        state1 = results[0].columns.tolist()
        state2 = results[0].index.tolist()

        median_matrix = np.zeros((len(state2), len(state1)))
        for i, state2_val in enumerate(state2):
            for j, state1_val in enumerate(state1):
                weights = []
                for result in results:
                    if state2_val in result.index and state1_val in result.columns:
                        weight_val = result.loc[state2_val, state1_val]
                        if not np.isnan(weight_val):
                            weights.append(weight_val)
                median_matrix[i, j] = np.median(weights) if weights else 0.0

        median_df = pd.DataFrame(median_matrix, index=state2, columns=state1)
        csv_path = os.path.join(work_path, f"{time_i}_{time_j}_Knn_umap.csv")
        median_df.to_csv(csv_path)
        print(f"Saved median weight matrix: {csv_path}")


# ------------------------------------------------------------
# All cells, UMAP embedding
# (results are written under "all_cell", which is the directory read by the
# LineageTreeBuilder below and by the downstream TF/RBP analysis scripts)
# ------------------------------------------------------------
work_path = os.path.join(data_dir, "output", "Analysis", "lineage", "all_cell")
time_points = [f"S{i:02d}" for i in range(1, 16)]

meta = pd.read_csv(os.path.join(data_dir, "output", "Annotation_hvg5000", "final",
                               "hs_all_metadata_update.csv"), index_col=0)
meta = meta[meta["type_2"] != "Remove"]
meta['stage_abbr'] = 'S' + meta['stage'].str.split('-').str[0]

emb_scanvi_umap = pd.read_csv(os.path.join(data_dir, "output", "Visualization", "Umap",
                                           "umap_data_scvi100_scanvi50.csv"), index_col=0)

process_timepoints(work_path, meta, emb_scanvi_umap, time_points)



#####################################################
### Step 2: Build the lineage tree ###
#####################################################

class LineageTreeBuilder:
    """Builder for the differentiation tree."""

    def __init__(self, work_path: str):
        self.work_path = work_path
        self.time_points = [f"S{i:02d}" for i in range(1, 16)]

    def load_knn_results(self, time_i: str, time_j: str) -> List[pd.DataFrame]:
        """Load the k-NN bootstrap result files."""
        file_path = os.path.join(self.work_path, f"{time_i}_{time_j}_Knn_umap.pkl")
        try:
            with open(file_path, 'rb') as f:
                results = pickle.load(f)
            return results
        except FileNotFoundError:
            print(f"Warning: file not found {file_path}")
            return []

    def calculate_median_probability_matrix(self, results: List[pd.DataFrame],
                                            time_i: str, time_j: str) -> pd.DataFrame:
        """Median probability matrix across bootstrap replicates."""
        if not results:
            return pd.DataFrame()

        state_1 = results[0].index.tolist()   # later states (nex)
        state_2 = results[0].columns.tolist()  # precursor states (pre)

        # Format state names with their time points
        state_1_formatted = [f"{time_j}:{self._format_state_name(state)}" for state in state_1]
        state_2_formatted = [f"{time_i}:{self._format_state_name(state)}" for state in state_2]

        median_matrix = np.zeros((len(state_1_formatted), len(state_2_formatted)))
        for i in range(len(state_1_formatted)):
            for j in range(len(state_2_formatted)):
                weights = []
                for result in results:
                    if state_1[i] in result.index and state_2[j] in result.columns:
                        weight_val = result.loc[state_1[i], state_2[j]]
                        if not np.isnan(weight_val):
                            weights.append(weight_val)
                median_matrix[i, j] = np.median(weights) if weights else 0.0

        return pd.DataFrame(median_matrix, index=state_1_formatted, columns=state_2_formatted)

    def _format_state_name(self, state_name: str) -> str:
        """Format state names (customize if needed)."""
        return state_name

    def combine_all_edges(self) -> pd.DataFrame:
        """Combine the edge data of all adjacent time point pairs."""
        all_edges = []

        for i in range(len(self.time_points) - 1):
            time_i = self.time_points[i]
            time_j = self.time_points[i + 1]
            print(f"Processing time point pair: {time_i} -> {time_j}")

            results = self.load_knn_results(time_i, time_j)
            if not results:
                continue

            median_df = self.calculate_median_probability_matrix(results, time_i, time_j)
            if median_df.empty:
                continue

            edges = median_df.stack().reset_index()
            edges.columns = ['nex', 'pre', 'prob']
            all_edges.append(edges)

        if not all_edges:
            raise ValueError("No valid edge data found")

        return pd.concat(all_edges, ignore_index=True)

    def manual_corrections(self, edges_df: pd.DataFrame) -> pd.DataFrame:
        """Manually correct specific edge relations."""
        edges = edges_df.copy()
        edges.loc[edges['pre'] == "S01:Thalamic_Glu", 'prob'] = 0.0
        return edges

    def add_metadata_columns(self, edges_df: pd.DataFrame) -> pd.DataFrame:
        """Add time point and cell type metadata columns."""
        edges = edges_df.copy()
        edges[['pre_time', 'pre_cell']] = edges['pre'].str.split(':', n=1, expand=True)
        edges[['nex_time', 'nex_cell']] = edges['nex'].str.split(':', n=1, expand=True)
        return edges

    def filter_edges_add_dummy_nodes(self, edges_df: pd.DataFrame,
                                     threshold: float = 0.2) -> pd.DataFrame:
        """Filter edges by probability threshold and add dummy nodes if needed."""
        filtered_edges = edges_df[edges_df['prob'] >= threshold].copy()
        filtered_edges = filtered_edges[['pre', 'nex', 'prob']]

        print(f"Edges after filtering: {len(filtered_edges)}")
        print(f"Nodes after filtering: {len(set(filtered_edges['pre']).union(set(filtered_edges['nex'])))}")

        dummy_edges = []  # add dummy nodes here if biologically required
        if dummy_edges:
            dummy_df = pd.DataFrame(dummy_edges, columns=['pre', 'nex', 'prob'])
            filtered_edges = pd.concat([filtered_edges, dummy_df], ignore_index=True)

        return filtered_edges

    def build_final_tree(self, edges_df: pd.DataFrame) -> pd.DataFrame:
        """For each later node, keep the edge with the highest probability."""
        tree_edges = []
        unique_nex = edges_df['nex'].unique()

        for nex_node in unique_nex:
            node_edges = edges_df[edges_df['nex'] == nex_node]
            if len(node_edges) == 1:
                tree_edges.append(node_edges.iloc[0])
            else:
                best_edge = node_edges.loc[node_edges['prob'].idxmax()]
                tree_edges.append(best_edge)

        tree_df = pd.DataFrame(tree_edges)
        return tree_df[['pre', 'nex']]

    def run_full_pipeline(self):
        """Run the full tree-building workflow."""
        print("Integrating differentiation tree results...")
        os.makedirs(self.work_path, exist_ok=True)

        all_edges = self.combine_all_edges()
        corrected_edges = self.manual_corrections(all_edges)
        edges_with_meta = self.add_metadata_columns(corrected_edges)

        edges_with_meta.to_pickle(os.path.join(self.work_path, "edge_all.pkl"))
        print("Saved full edge data")

        filtered_edges = self.filter_edges_add_dummy_nodes(edges_with_meta, threshold=0.2)
        final_tree = self.build_final_tree(filtered_edges)

        filtered_edges.to_csv(os.path.join(self.work_path, "edge_prob.txt"),
                              sep='\t', index=False, header=False)
        final_tree.to_csv(os.path.join(self.work_path, "human_edge.txt"),
                          sep='\t', index=False, header=False)
        print("Saved edge probabilities and final tree structure")

        print("Differentiation tree construction complete!")
        return edges_with_meta, filtered_edges, final_tree


work_path = os.path.join(data_dir, "output", "Analysis", "lineage", "all_cell")
builder = LineageTreeBuilder(work_path)
all_edges, filtered_edges, final_tree = builder.run_full_pipeline()
