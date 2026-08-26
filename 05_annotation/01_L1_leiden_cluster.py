#!/usr/bin/env python3
"""
=============================================================================
File      : 01_L1_leiden_cluster.py
Directory : 05_annotation
Purpose   : Level 1 coarse Leiden clustering of the integrated human brain atlas.
=============================================================================
"""

# ============================================================================
# Configuration: set PROJ_DIR to the project root before running
# ============================================================================
import os

proj_dir = os.path.join(
    os.environ.get("PROJ_DIR", "path/to/your/project"), "data", "human"
)
import random
from multiprocessing import Pool

import anndata
import pandas as pd
import scanpy as sc

random.seed(123)


def process(ad, res):
    """Run Leiden clustering at one resolution and return the cluster labels."""
    key_name = "leiden_coarse_" + ("%.1f" % res)
    sc.tl.leiden(ad, key_added=key_name, resolution=res)

    result_df = ad.obs.loc[:, key_name].to_frame()
    csv_path = os.path.join(
        proj_dir,
        "output/Annotation_hvg5000/L1/leiden",
        "leiden_r%.1f.csv" % res,
    )
    result_df.to_csv(csv_path, index=True)
    print("Finish leiden, resolution: %.1f" % res)
    return ad.obs.loc[:, key_name]


def run_batch(ad, resolutions, n_processes, batch_name):
    """Run Leiden clustering for a batch of resolutions in parallel."""
    result_dfs = []
    with Pool(processes=n_processes) as pool:
        async_results = [
            pool.apply_async(process, args=(ad, res_i)) for res_i in resolutions
        ]
        for i, ar in enumerate(async_results):
            try:
                result_dfs.append(ar.get())
                print(
                    "Batch %s: Processed resolution %.1f successfully"
                    % (batch_name, resolutions[i])
                )
            except Exception as e:
                print(
                    "Batch %s: Error processing resolution %.1f: %s"
                    % (batch_name, resolutions[i], str(e))
                )
    return result_dfs


if __name__ == "__main__":
    print("Scanpy version: %s" % sc.__version__)
    print("Anndata version: %s\n" % anndata.__version__)

    leiden_dir = os.path.join(proj_dir, "output/Annotation_hvg5000/L1/leiden")
    os.makedirs(leiden_dir, exist_ok=True)

    # Load dataset
    ad_hs = sc.read_h5ad(
        os.path.join(proj_dir, "output/Umap/hs_umap_hvg5000.h5ad")
    )
    print(ad_hs)

    # Build k-NN graph on the harmony-corrected PCA embedding
    ad_hs.obsm["X_pca_copy"] = ad_hs.obsm["X_pca"]
    ad_hs.obsm["X_pca"] = ad_hs.obsm["X_pca_harmony"]
    sc.pp.neighbors(ad_hs, n_neighbors=50, n_pcs=50)
    print("\nFinish KNN\n")

    ad_hs.write_h5ad(
        filename=os.path.join(
            proj_dir, "output/Annotation_hvg5000/L1/hs_anno_1_hvg5000_tmp.h5ad"
        )
    )

    # Drop unnecessary data to save memory
    del ad_hs.raw
    del ad_hs.X
    del ad_hs.obsm

    # Leiden clustering, batch 1: resolutions 0.1-1.0 (10 processes)
    res_batch1 = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    result_dfs = run_batch(ad_hs, res_batch1, 10, "1 (0.1-1.0)")

    # Leiden clustering, batch 2: resolutions 1.1-1.5 (5 processes)
    res_batch2 = [1.1, 1.2, 1.3, 1.4, 1.5]
    result_dfs += run_batch(ad_hs, res_batch2, 5, "2 (1.1-1.5)")

    if result_dfs:
        # Combine all results into a single DataFrame and save
        combined_df = pd.concat(result_dfs, axis=1)
        combined_path = os.path.join(
            proj_dir,
            "output/Annotation_hvg5000/L1/clustering_results_hvg5000_1.csv",
        )
        combined_df.to_csv(combined_path, index=True)
        print("Saved combined results to %s" % combined_path)

        # Update the AnnData object and save
        for col in combined_df.columns:
            ad_hs.obs[col] = combined_df[col]

        ad_hs.write_h5ad(
            filename=os.path.join(
                proj_dir,
                "output/Annotation_hvg5000/L1/hs_anno_1_hvg5000_cluster.h5ad",
            )
        )
        print("Saved final annotated object")
    else:
        print("No results to combine")
