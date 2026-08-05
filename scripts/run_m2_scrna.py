import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src import ensure_dirs
from src.scrna.process import load_counts, qc_filter, normalize_and_harmony, score_celltypes, de_per_celltype
from src.scrna.geneids import normalize_gene_ids


def load_gse86469(path):
    adata = load_counts(path)
    donor = adata.obs_names.str.split("_").str[0]
    condition = np.where(donor.isin(["10th", "11th", "12th"]), "T2D", "ND")
    adata.obs["donor"] = donor.values
    adata.obs["condition"] = condition
    adata.obs["study"] = "GSE86469"
    adata.obs["annotation_source"] = "marker_scored"
    return adata


def load_gse153855(counts_path, ann_path):
    counts = pd.read_csv(counts_path, sep="\t", header=None, dtype=str)
    genes = counts.iloc[:, 0].astype(str).values
    mat = counts.iloc[:, 1:].T.astype(np.float32)
    ann = pd.read_csv(ann_path, sep="\t", dtype=str)
    ann = ann.dropna(subset=[ann.columns[0]]).reset_index(drop=True)
    n = min(len(ann), mat.shape[0])
    mat, ann = mat.iloc[:n], ann.iloc[:n]
    assert mat.shape[1] == len(genes)
    cell_ids = ann.iloc[:, 0].astype(str).values
    adata = sc.AnnData(X=mat.values)
    adata.var_names = genes
    adata.obs_names = cell_ids
    adata.obs["donor"] = ann.iloc[:, 0].values
    adata.obs["condition"] = np.where(ann.iloc[:, 1].str.lower().str.contains("diabetes"), "T2D", "ND")
    adata.obs["cell_type_pred"] = ann.iloc[:, 2].values
    adata.obs["study"] = "GSE153855"
    adata.obs["annotation_source"] = "curated"
    return adata


DATASETS = {
    "GSE86469": load_gse86469,
    "GSE153855": load_gse153855,
}


def main():
    paths = ensure_dirs()
    raw = paths["raw"] / "scrna"
    out = paths["processed"] / "scrna"
    out.mkdir(parents=True, exist_ok=True)

    n_cap = 5000
    integrated = []
    for accession, loader in DATASETS.items():
        files = sorted(raw.glob(f"{accession}_*"))
        print(f"[M2] {accession}: loading {[f.name for f in files]}")
        if accession == "GSE153855":
            counts_f = next(f for f in files if "counts" in f.name)
            ann_f = next(f for f in files if "annotation" in f.name)
            adata = loader(counts_f, ann_f)
        else:
            adata = loader(files[0])
        print(f"      raw cells={adata.n_obs:,} genes={adata.n_vars:,}")
        adata = normalize_gene_ids(adata, cache_path=paths["external"] / "ensembl_to_symbol.csv")
        adata = qc_filter(adata, min_genes=500, max_genes=20000)
        if adata.n_obs > n_cap:
            sc.pp.subsample(adata, n_obs=n_cap, random_state=0)
        print(f"      after QC/subset: cells={adata.n_obs:,}")
        adata.obs["batch"] = adata.obs["donor"].astype(str)
        adata = normalize_and_harmony(adata)
        if "cell_type_pred" not in adata.obs:
            adata = score_celltypes(adata)
        integrated.append(adata)
        adata.write(out / f"{accession}.h5ad")

    print("[M2] merging + batch integration")
    merged = integrated[0].concatenate(integrated[1:], batch_key="study", index_unique=None) if len(integrated) > 1 else integrated[0]
    merged.obs["batch"] = merged.obs["study"].astype(str) + "_" + merged.obs["donor"].astype(str)
    merged = normalize_and_harmony(merged)
    merged = score_celltypes(merged)
    merged.write(out / "merged_islets.h5ad")

    print("[M2] cell-type DEG (T2D vs ND)")
    rgg, groups = de_per_celltype(merged)
    deg_rows = []
    for ct in merged.obs["cell_type_pred"].cat.categories:
        for cond in merged.obs["condition"].cat.categories:
            grp = f"{ct}_{cond}"
            if grp not in groups:
                continue
            for i, g in enumerate(rgg["names"][grp][:50]):
                deg_rows.append({
                    "cell_type": ct, "condition": cond,
                    "gene": g,
                    "logfoldchanges": float(rgg["logfoldchanges"][grp][i]),
                    "pvals_adj": float(rgg["pvals_adj"][grp][i]),
                })
    deg = pd.DataFrame(deg_rows)
    deg.to_csv(out / "deg_per_celltype.csv", index=False)

    print("[M2] per-study DEG + replicated disease signatures (v_disease)")
    vrows = []
    for study in merged.obs["study"].cat.categories:
        sub = merged[merged.obs["study"] == study].copy()
        if sub.obs["condition"].nunique() < 2:
            continue
        try:
            sub_rgg, sub_groups = de_per_celltype(sub)
            for ct in sub.obs["cell_type_pred"].cat.categories:
                for grp_name in ("T2D", "ND"):
                    grp = f"{ct}_{grp_name}"
                    if grp not in sub_groups:
                        continue
                    for i, g in enumerate(sub_rgg["names"][grp][:200]):
                        vrows.append({
                            "study": study, "cell_type": ct, "gene": g,
                            "group": grp_name,
                            "log2fc": float(sub_rgg["logfoldchanges"][grp][i]),
                            "padj": float(sub_rgg["pvals_adj"][grp][i]),
                        })
        except Exception as e:
            print(f"      DEG failed for {study}: {e}")

    vdeg = pd.DataFrame(vrows)
    vdeg.to_csv(out / "deg_per_study.csv", index=False)

    sig = []
    for (ct, gene, grp), sub in vdeg.groupby(["cell_type", "gene", "group"]):
        if sub["study"].nunique() < 2:
            continue
        sign = sub["log2fc"].apply(lambda x: np.sign(x) if abs(x) > 0.58 else 0)
        dirs = set(sign[sign != 0])
        if len(dirs) == 1:
            direction = 1 if grp == "T2D" else -1
            sig.append({
                "cell_type": ct, "gene": gene, "direction": direction,
                "mean_log2fc": float(sub["log2fc"].mean()),
                "min_padj": float(sub["padj"].min()),
                "n_studies": int(sub["study"].nunique()),
            })
    signature = pd.DataFrame(sig)
    signature.to_csv(out / "disease_signatures.csv", index=False)
    print(f"      replicated signature genes: {len(signature)}")
    print(signature["cell_type"].value_counts().to_dict())

    summary = {
        "cells_after_qc": {k: int(a.n_obs) for k, a in zip(DATASETS, integrated)},
        "merged_cells": int(merged.n_obs),
        "cell_types": merged.obs["cell_type_pred"].value_counts().to_dict(),
        "deg_rows": len(deg),
        "replicated_signature_genes": len(signature),
    }
    (paths["results"] / "scrna").mkdir(parents=True, exist_ok=True)
    (paths["results"] / "scrna" / "qc.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
