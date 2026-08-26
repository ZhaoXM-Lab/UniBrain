#!/usr/bin/env python3
"""
=============================================================================
File      : 06_L2_neuron_step1.py
Directory : 05_annotation
Purpose   : Level 2 fine Leiden clustering of the Neuron subset over a resolution sweep.
=============================================================================
"""

# ============================================================================
# Configuration: set PROJ_DIR to the project root before running
# ============================================================================
import os

proj_dir = os.path.join(
    os.environ.get("PROJ_DIR", "path/to/your/project"), "data", "human"
)

# Limit the number of threads used by BLAS/parallel libraries
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["NUMBA_NUM_THREADS"] = "4"
os.environ["VECLIB_MAXIMUM_THREADS"] = "4"
os.environ["NUMEXPR_NUM_THREADS"] = "4"

from multiprocessing import Pool

import anndata
import numpy as np
import pandas as pd
import scanpy as sc

sc.settings.n_jobs = 4


def process(ad, res):
    """Run Leiden clustering at one resolution and return the cluster labels."""
    key_name = "leiden_fine_" + ("%.1f" % res)
    sc.tl.leiden(ad, key_added=key_name, resolution=res)

    result_df = ad.obs.loc[:, key_name].to_frame()
    csv_path = os.path.join(
        proj_dir,
        "output/Annotation_hvg5000/L2/leiden/Neuron",
        "leiden_r%.1f.csv" % res,
    )
    result_df.to_csv(csv_path, index=True)
    print("Finish leiden, resolution: %.1f" % res)
    return ad.obs.loc[:, key_name]


if __name__ == "__main__":
    print("Scanpy version: %s" % sc.__version__)
    print("Anndata version: %s\n" % anndata.__version__)

    leiden_dir = os.path.join(proj_dir, "output/Annotation_hvg5000/L2/leiden/Neuron")
    os.makedirs(leiden_dir, exist_ok=True)

    # Load dataset and reset to raw counts
    ad_hs = sc.read_h5ad(
        os.path.join(proj_dir, "output/Annotation_hvg5000/L1/hs_anno_1_Neuron.h5ad")
    )
    print(ad_hs)
    ad_hs = ad_hs.raw.to_adata()
    ad_hs.raw = ad_hs.copy()

    # HVG selection, regression, scaling and PCA
    sc.pp.highly_variable_genes(ad_hs, n_top_genes=5000, flavor="seurat")
    highly_variable = ad_hs.var.highly_variable
    ad_hs = ad_hs[:, highly_variable]
    sc.pp.regress_out(
        ad_hs, keys=["total_counts", "S_score", "G2M_score"], n_jobs=8
    )
    sc.pp.scale(ad_hs)
    sc.tl.pca(ad_hs, svd_solver="arpack", n_comps=50)

    # Harmony integration to correct for sample batch effects
    sc.external.pp.harmony_integrate(
        ad_hs, key="sample_new", basis="X_pca", max_iter_harmony=30
    )

    # Build k-NN graph on the harmony-corrected PCA embedding
    ad_hs.obsm["X_pca_copy"] = ad_hs.obsm["X_pca"]
    ad_hs.obsm["X_pca"] = ad_hs.obsm["X_pca_harmony"]
    sc.pp.neighbors(ad_hs, n_neighbors=30, n_pcs=50)
    print("\nFinish KNN\n")

    ad_hs.write_h5ad(
        filename=os.path.join(
            proj_dir,
            "output/Annotation_hvg5000/L2/hs_anno_2_Neuron_hvg5000_tmp.h5ad",
        )
    )

    # Drop unnecessary data to save memory
    del ad_hs.raw
    del ad_hs.X

    # Leiden clustering over resolutions 0.1-1.5 (10 processes)
    resolution = [
        0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,
        1.0, 1.1, 1.2, 1.3, 1.4, 1.5,
    ]
    result_dfs = []
    with Pool(processes=10) as pool:
        async_results = [
            pool.apply_async(process, args=(ad_hs, res_i)) for res_i in resolution
        ]
        for i, ar in enumerate(async_results):
            try:
                result_dfs.append(ar.get())
                print(
                    "Processed resolution %.1f successfully" % resolution[i]
                )
            except Exception as e:
                print(
                    "Error processing resolution %.1f: %s"
                    % (resolution[i], str(e))
                )

    if result_dfs:
        # Combine all results into a single DataFrame and save
        combined_df = pd.concat(result_dfs, axis=1)
        combined_path = os.path.join(
            proj_dir,
            "output/Annotation_hvg5000/L2/clustering_results_2_Neuron_hvg5000.csv",
        )
        combined_df.to_csv(combined_path, index=True)
        print("Saved combined results to %s" % combined_path)

        # Reload the tmp object (X/raw were dropped for memory) so the
        # clustered output keeps the expression matrix for the L3 steps
        ad_hs_out = sc.read_h5ad(
            os.path.join(
                proj_dir,
                "output/Annotation_hvg5000/L2/hs_anno_2_Neuron_hvg5000_tmp.h5ad",
            )
        )
        for col in combined_df.columns:
            ad_hs_out.obs[col] = combined_df[col]

        ad_hs_out.write_h5ad(
            filename=os.path.join(
                proj_dir,
                "output/Annotation_hvg5000/L2/hs_anno_2_Neuron_hvg5000_cluster.h5ad",
            )
        )
        print(
            "Saved final annotated object: "
            "output/Annotation_hvg5000/L2/hs_anno_2_Neuron_hvg5000_cluster.h5ad"
        )
    else:
        print("No results to combine")
