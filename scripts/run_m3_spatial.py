import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
from scipy.optimize import nnls
from sklearn.cluster import KMeans

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src import ensure_dirs  # noqa: E402
from src.ingest.spatial import GATED_PANCREAS_VISIUM, VISIUM_DEMOS, download_visium  # noqa: E402

# Canonical mouse-brain cell-type markers (used only for the fallback demo tissue).
BRAIN_MARKERS = {
    "neuron_exc": ["Slc17a7", "Neurod6", "Tbr1", "Rorb"],
    "neuron_inh": ["Slc32a1", "Gad1", "Gad2", "Pvalb"],
    "oligo": ["Mbp", "Mog", "Plp1", "Mag"],
    "astro": ["Gfap", "Aqp4", "Slc1a3", "Aldh1l1"],
    "microglia": ["Cx3cr1", "Tmem119", "P2ry12", "C1qa"],
    "endothelial": ["Pecam1", "Cldn5", "Flt1"],
    "ependymal": ["Foxj1"],
}


def build_islet_signature(out_dir):
    """Part A: per-cell-type expression signature matrix from M2 islet scRNA."""
    processed = out_dir / "scrna"
    deg = pd.read_csv(processed / "deg_per_celltype.csv")
    adata = sc.read_h5ad(processed / "merged_islets.h5ad")
    markers = (
        deg.loc[deg["gene"].isin(adata.var_names)]
        .assign(absl2=deg["logfoldchanges"].abs())
        .sort_values("absl2", ascending=False)
        .groupby("cell_type")["gene"]
        .apply(lambda s: list(dict.fromkeys(s))[:30])
        .to_dict()
    )
    genes = list(dict.fromkeys(g for gs in markers.values() for g in gs))
    expr = adata[:, genes].to_df()
    means = expr.groupby(adata.obs["cell_type_pred"].astype(str)).mean().reindex(
        sorted(markers.keys())).fillna(0.0)
    means.to_csv(processed / "signature_matrix.csv")
    sig = pd.DataFrame({
        "cell_type": [ct for ct, gs in markers.items() for _ in gs],
        "gene": [g for gs in markers.values() for g in gs],
    })
    sig.to_csv(processed / "signature_markers.csv", index=False)
    print(f"      islet signature matrix: {means.shape} ({len(markers)} cell types x {len(genes)} genes)")
    return means, markers


def load_islet_signature(processed_dir, adata):
    """Part B: islet signature matrix restricted to genes present in the Visium data."""
    sig = pd.read_csv(processed_dir / "scrna" / "signature_matrix.csv", index_col=0)
    sig_genes = [g for g in sig.columns if g in adata.var_names]
    if len(sig_genes) < 10 or len(sig_genes) / len(sig.columns) < 0.1:
        raise RuntimeError(
            f"only {len(sig_genes)}/{len(sig.columns)} islet signature genes match the Visium "
            "data; tissue/species likely mismatched with the M2 islet scRNA (human pancreas)."
        )
    return sig.loc[:, sig_genes], sig_genes


def build_brain_signature(adata):
    """Part B: marker-weighted signature built from the fallback mouse-brain tissue."""
    present = {ct: [g for g in gs if g in adata.var_names] for ct, gs in BRAIN_MARKERS.items()}
    present = {ct: gs for ct, gs in present.items() if gs}
    sig_genes = list(dict.fromkeys(g for gs in present.values() for g in gs))
    gm = adata[:, sig_genes].to_df().mean(axis=0)
    sig = pd.DataFrame(0.0, index=list(present.keys()), columns=sig_genes)
    for ct, gs in present.items():
        sig.loc[ct, gs] = gm[gs].values
    return sig, sig_genes


def deconvolve_spots(adata, signature, signature_genes):
    """Per-spot NNLS against a column-L1-normalized signature matrix."""
    X = adata[:, signature_genes].to_df()
    A = signature.loc[:, signature_genes].values.T.copy()
    A = A / np.maximum(A.sum(axis=0, keepdims=True), 1e-12)
    props = np.zeros((X.shape[0], A.shape[1]), dtype=np.float32)
    for i, (_, b) in enumerate(X.iterrows()):
        x, _ = nnls(A, b.values.astype(np.float64))
        total = x.sum()
        props[i] = x / total if total > 0 else 0
    return pd.DataFrame(props, index=X.index, columns=signature.index)


def label_niches(proportions, n_niches=6, seed=0):
    km = KMeans(n_clusters=n_niches, random_state=seed, n_init=10)
    labels = km.fit_predict(proportions.values)
    niches = proportions.copy()
    niches["niche"] = labels
    order = niches.groupby("niche")[proportions.columns].mean().sum(axis=1).sort_values(
        ascending=False).index
    remap = {old: i for i, old in enumerate(order)}
    niches["niche"] = niches["niche"].map(remap)
    return niches


def main():
    ap = argparse.ArgumentParser(description="M3 spatial deconvolution + niche labeling")
    ap.add_argument("--visium", type=str, default=None,
                    help="Space Ranger out dir to deconvolve with the islet signature matrix "
                         "(e.g. pancreas Visium once gated access clears). Defaults to the "
                         "10x mouse-brain demo.")
    ap.add_argument("--n-niches", type=int, default=6)
    args = ap.parse_args()

    paths = ensure_dirs()
    raw_sp = paths["raw"] / "spatial"
    out = paths["results"] / "spatial"
    out.mkdir(parents=True, exist_ok=True)

    print("[M3] Part A: islet signature matrix from M2 scRNA")
    sig_iso, markers_iso = build_islet_signature(paths["processed"])

    custom = args.visium is not None
    if custom:
        print(f"[M3] Part B: Visium deconvolution + niche labeling (custom: {args.visium})")
        adata = sc.read_visium(args.visium)
        dataset = "custom_visium"
    else:
        print("[M3] Part B: Visium deconvolution + niche labeling (demo tissue)")
        dataset = "mouse_brain"
        data_dir = raw_sp / dataset
        download_visium(data_dir, dataset=dataset)
        adata = sc.read_visium(str(data_dir))
    adata.var_names_make_unique()
    print(f"      Visium: spots={adata.n_obs} genes={adata.n_vars}")

    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    if custom:
        sig_mat, sig_genes = load_islet_signature(paths["processed"], adata)
    else:
        sig_mat, sig_genes = build_brain_signature(adata)
    print(f"      deconv signature: {sig_mat.shape}")

    props = deconvolve_spots(adata, sig_mat, sig_genes)
    props.to_csv(out / "deconv.csv")
    print("      dominant cell type per spot:")
    dom = props.idxmax(axis=1).value_counts()
    print("      " + json.dumps(dom.to_dict()))

    niches = label_niches(props, n_niches=args.n_niches)
    niches.to_csv(out / "niches.csv")
    niche_comp = niches.groupby("niche")[props.columns].mean()
    niche_dom = niche_comp.idxmax(axis=1).to_dict()
    niche_sizes = niches["niche"].value_counts().sort_index().to_dict()
    print(f"      niches={len(niche_sizes)}; dominant cell type: {niche_dom}")
    print(f"      niche sizes: {niche_sizes}")

    summary = {
        "data": {"dataset": dataset, "visium_dir": args.visium},
        "gated_human_pancreas": GATED_PANCREAS_VISIUM,
        "spots": int(adata.n_obs),
        "signature_cell_types": list(sig_mat.index),
        "dominant_celltype_per_spot": dom.to_dict(),
        "niches": {"n": len(niche_sizes), "dominant_celltype": niche_dom, "sizes": niche_sizes},
        "islet_signature_matrix": sig_iso.shape,
        "islet_signature_genes_per_celltype": {k: len(v) for k, v in markers_iso.items()},
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print("[M3] done ->", out / "summary.json")


if __name__ == "__main__":
    main()
