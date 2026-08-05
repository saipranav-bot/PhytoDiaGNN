import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.graph.build import build_hetero_graph, load_graph_inputs  # noqa: E402
from src.graph.model import (HeteroGraphSAGE, evaluate, sample_negatives,  # noqa: E402
                             train_epoch)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=str, default="results/m4_gnn")
    args = ap.parse_args()

    processed = ROOT / "data" / "processed"
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)

    print("[M4] assembling heterogeneous KG")
    compounds, targets, ct_edges, ppi, signatures, adata = load_graph_inputs({"processed": processed})
    data = build_hetero_graph(compounds, targets, ct_edges, ppi, signatures, adata)
    pos = data["compound", "therapeutic", "cell_type"].edge_index
    n_comp = data["compound"].x.shape[0]
    n_ct = data["cell_type"].x.shape[0]
    print(f"      nodes: compound={n_comp} protein={data['protein'].x.shape[0]} "
          f"cell_type={n_ct} edges(therapeutic)={pos.size(1)}")

    rng = np.random.RandomState(args.seed)
    idx = np.arange(pos.size(1))
    rng.shuffle(idx)
    fold_edges = np.array_split(idx, args.folds)
    in_feats = {nt: data[nt].x.shape[1] for nt in data.node_types}
    edge_types = [et for et in data.edge_types if "therapeutic" not in et[1]]

    summary = {"folds": []}
    models = []
    neg_pairs = sample_negatives(pos, n_comp, n_ct, rng, 1)

    for fold in range(args.folds):
        test_idx = np.concatenate([fold_edges[fold]])
        train_idx = np.concatenate([fold_edges[k] for k in range(args.folds) if k != fold])

        test_edge = pos[:, test_idx]
        train_edge = pos[:, train_idx]
        pos_sz = train_edge.size(1)
        neg_sz = pos_sz
        li = torch.cat([train_edge, neg_pairs[:neg_sz].T], dim=1)
        lb = torch.cat([torch.ones(pos_sz), torch.zeros(neg_sz)])
        test_li = torch.cat([test_edge, neg_pairs[pos_sz:pos_sz + test_edge.size(1)].T], dim=1)
        test_lb = torch.cat([torch.ones(test_edge.size(1)), torch.zeros(test_edge.size(1))])

        torch.manual_seed(args.seed + fold)
        model = HeteroGraphSAGE(in_feats, hidden=64, out=64, edge_types=edge_types)
        opt = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-4)
        best_auc = 0.0
        for epoch in range(args.epochs):
            loss = train_epoch(model, data, li, lb, opt)
            if epoch % 20 == 0 or epoch == args.epochs - 1:
                auc, ap = evaluate(model, data, test_li, test_lb)
                if auc > best_auc:
                    best_auc, best_ap = auc, ap
        auc, ap = evaluate(model, data, test_li, test_lb)
        summary["folds"].append({"fold": fold, "auc": auc, "ap": ap})
        models.append((model, test_li, test_lb))
        print(f"      fold {fold}: AUROC={auc:.3f} AUPRC={ap:.3f}")

    print("[M4] baselines")
    fp = np.stack([np.frombuffer(b.encode(), dtype="u1") for b in compounds["morgan1024"]])
    fp = fp.reshape(len(compounds), -1).astype(np.float32)
    ct_feat = data["cell_type"].x.numpy()
    all_edges = pos
    X = np.hstack([fp[all_edges[0].numpy()], ct_feat[all_edges[1].numpy()]])
    fold_auc_rf = []
    for fold in range(args.folds):
        test_idx = np.concatenate([fold_edges[fold]])
        train_idx = np.concatenate([fold_edges[k] for k in range(args.folds) if k != fold])
        tr_neg = neg_pairs[:len(train_idx)]
        te_neg = neg_pairs[len(train_idx):len(train_idx) + len(test_idx)]
        X_tr = np.vstack([X[train_idx], np.hstack([fp[tr_neg[:, 0]], ct_feat[tr_neg[:, 1]]])])
        y_tr = np.r_[np.ones(len(train_idx)), np.zeros(len(train_idx))]
        X_te = np.vstack([X[test_idx], np.hstack([fp[te_neg[:, 0]], ct_feat[te_neg[:, 1]]])])
        y_te = np.r_[np.ones(len(test_idx)), np.zeros(len(test_idx))]
        rf = RandomForestClassifier(n_estimators=120, random_state=args.seed, n_jobs=-1)
        rf.fit(X_tr, y_tr)
        p = rf.predict_proba(X_te)[:, 1]
        fold_auc_rf.append(roc_auc_score(y_te, p))
        summary["folds"][fold]["rf_auc"] = roc_auc_score(y_te, p)
    summary["mean_gnn_auc"] = float(np.mean([f["auc"] for f in summary["folds"]]))
    summary["mean_gnn_ap"] = float(np.mean([f["ap"] for f in summary["folds"]]))
    summary["mean_rf_auc"] = float(np.mean(fold_auc_rf))
    summary["random_auc"] = 0.5
    print(f"      RF AUROC={summary['mean_rf_auc']:.3f} | GNN AUROC={summary['mean_gnn_auc']:.3f}")

    print("[M4] benchmark recovery (top-K rank of 18 known actives)")
    bench = set(compounds[compounds["is_benchmark"]]["np_id"].astype(str))
    ct_map = data.ct_map
    all_candidates = []
    with torch.no_grad():
        model = models[0][0]
        model.eval()
        z = model.encode(data)
        cemb = z["compound"]
        ctemb = z["cell_type"]
    scores = {}
    for cid in range(n_comp):
        for ct in ct_map:
            s = float(torch.sigmoid(model.head(cemb[cid], ctemb[ct_map[ct]])).item())
            scores[(cid, ct)] = s
    ranked = sorted(scores.items(), key=lambda kv: -kv[1])
    rank_of_bench = [i for i, ((c, ct), _) in enumerate(ranked) if compounds["np_id"].iloc[c] in bench]
    summary["bench_top1_pct"] = float(len([r for r in rank_of_bench if r < len(ranked) * 0.01]) / len(rank_of_bench))
    summary["bench_top10_pct"] = float(len([r for r in rank_of_bench if r < len(ranked) * 0.10]) / len(rank_of_bench))
    print(f"      {len(rank_of_bench)} benchmark active pairs ranked; "
          f"top1%={summary['bench_top1_pct']:.2f} top10%={summary['bench_top10_pct']:.2f}")

    with open(out / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("[M4] done ->", out / "summary.json")


if __name__ == "__main__":
    main()
