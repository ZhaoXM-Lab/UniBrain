#!/usr/bin/env python3
"""
=============================================================================
File      : 05_L2_progenitor_step1.py
Directory : 05_annotation
Purpose   : Level 2 fine Leiden clustering of the Progenitors subset over a resolution sweep.
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
import scanpy as sc

sc.settings.n_jobs = 4


def process(ad, res):
    """Run Leiden clustering at one resolution and return the cluster labels."""
    key_name = "leiden_fine_" + ("%.1f" % res)
    sc.tl.leiden(ad, key_added=key_name, resolution=res)

    result_df = ad.obs.loc[:, key_name].to_frame()
    csv_path = os.path.join(
        proj_dir,
        "output/Annotation_hvg5000/L2/leiden/Progenitors",
        "leiden_r%.1f.csv" % res,
    )
    result_df.to_csv(csv_path, index=True)
    print("Finish leiden, resolution: %.1f" % res)
    return ad.obs.loc[:, key_name]


if __name__ == "__main__":
    print("Scanpy version: %s" % sc.__version__)
    print("Anndata version: %s\n" % anndata.__version__)

    leiden_dir = os.path.join(
        proj_dir, "output/Annotation_hvg5000/L2/leiden/Progenitors"
    )
    os.makedirs(leiden_dir, exist_ok=True)

    # Load dataset and reset to raw counts
    ad_hs = sc.read_h5ad(
        os.path.join(
            proj_dir, "output/Annotation_hvg5000/L1/hs_anno_1_Progenitors.h5ad"
        )
    )
    print(ad_hs)
    ad_hs = ad_hs.raw.to_adata()
    ad_hs.raw = ad_hs.copy()

    # HVG selection, regression, scaling and PCA
    sc.pp.highly_variable_genes(ad_hs, n_top_genes=3000, flavor="seurat")
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
            proj_dir, "output/Annotation_hvg5000/L2/hs_anno_2_Progenitors_tmp.h5ad"
        )
    )

    # Leiden clustering over resolutions 0.1-1.5 (3 parallel batches of 5)
    resolution = [
        [0.1, 0.2, 0.3, 0.4, 0.5],
        [0.6, 0.7, 0.8, 0.9, 1.0],
        [1.1, 1.2, 1.3, 1.4, 1.5],
    ]
    result_list = []
    for chunk_i in range(3):
        pool = Pool(processes=5)
        for res_i in resolution[chunk_i]:
            result_list.append(pool.apply_async(process, args=(ad_hs, res_i)))
        pool.close()
        pool.join()

    result_list = [result_i.get().astype(str) for result_i in result_list]
    for result_i in result_list:
        ad_hs.obs[result_i.name] = result_i[ad_hs.obs_names]

    flattened_res = [res for sublist in resolution for res in sublist]
    columns = ["leiden_fine_%.1f" % res_i for res_i in flattened_res]
    results = ad_hs.obs.loc[:, columns]

    # Save result
    ad_hs.write_h5ad(
        filename=os.path.join(
            proj_dir,
            "output/Annotation_hvg5000/L2/hs_anno_2_Progenitors_cluster.h5ad",
        )
    )
    results.to_csv(
        os.path.join(
            proj_dir,
            "output/Annotation_hvg5000/L2/clustering_results_2_Progenitors.csv",
        )
    )
    print("Save as <output/Annotation_hvg5000/L2/hs_anno_2_Progenitors_cluster.h5ad>")
