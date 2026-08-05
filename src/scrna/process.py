import numpy as np
import pandas as pd
import scanpy as sc
import harmonypy

from .markers import ISLET_MARKERS


def load_counts(path, row_key="gene"):
    df = pd.read_csv(path, index_col=0)
    if "ensembl" in str(df.index.name).lower():
        df = df.loc[~df.index.duplicated()]
    adata = sc.AnnData(X=df.T.astype(np.float32))
    adata.var_names = df.index.astype(str)
    return adata


def qc_filter(adata, min_genes=200, max_genes=None, max_mito=0.15):
    adata.var["mt"] = adata.var_names.str.startswith("MT-")
    sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], percent_top=None, inplace=True)
    sc.pp.filter_cells(adata, min_genes=min_genes)
    if max_genes:
        sc.pp.filter_cells(adata, max_genes=max_genes)
    adata = adata[adata.obs["pct_counts_mt"] < max_mito * 100].copy()
    return adata


def normalize_and_harmony(adata, n_top_genes=3000, n_pcs=30, n_components=30):
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes, flavor="seurat")
    adata.raw = adata
    adata = adata[:, adata.var["highly_variable"]].copy()
    sc.tl.pca(adata, n_comps=n_pcs, svd_solver="arpack")
    if "batch" in adata.obs:
        try:
            hv = harmonypy.run_harmony(
                adata.obsm["X_pca"], adata.obs, "batch", n_components=n_components
            )
            adata.obsm["X_pca_harmony"] = hv.Z_corr.T
            use = "X_pca_harmony"
        except Exception:
            use = "X_pca"
    else:
        use = "X_pca"
    sc.pp.neighbors(adata, use_rep=use)
    sc.tl.umap(adata)
    sc.tl.leiden(adata, resolution=0.8)
    return adata


def score_celltypes(adata, markers=None):
    markers = markers or ISLET_MARKERS
    for ct, genes in markers.items():
        genes = [g for g in genes if g in adata.var_names]
        if genes:
            sc.tl.score_genes(adata, genes, score_name=f"score_{ct}")
    obs_cols = [f"score_{ct}" for ct in markers if f"score_{ct}" in adata.obs]
    score = adata.obs[obs_cols].fillna(0.0)
    adata.obs["cell_type_pred"] = score.idxmax(axis=1).str.replace("score_", "")
    return adata


def de_per_celltype(adata, condition_col="condition", min_cells=2):
    label = adata.obs["cell_type_pred"].astype(str) + "_" + adata.obs[condition_col].astype(str)
    adata.obs["ct_condition"] = label.astype("category")
    counts = adata.obs["ct_condition"].value_counts()
    groups = counts[counts >= min_cells].index.tolist()
    sc.tl.rank_genes_groups(
        adata, "ct_condition", groups=groups,
        method="wilcoxon", use_raw=True,
    )
    return adata.uns["rank_genes_groups"], groups
