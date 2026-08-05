import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.graph.build import build_hetero_graph, load_graph_inputs  # noqa: E402
from src.graph.model import HeteroGraphSAGE  # noqa: E402


def compute_srev(compounds, targets, ct_edges, signatures):
    sig = signatures.copy()
    sig = sig[sig["direction"] != 0]
    gene_dir = {g: int(sig.loc[sig["gene"] == g, "direction"].iloc[0]) for g in sig["gene"].unique()}
    tmap = dict(zip(targets["Target_ID"], targets["Gene_Symbol"]))
    tg_by_comp = ct_edges.groupby("compound_id")["target_id"].apply(set).to_dict()
    ct_sig = sig.groupby("cell_type")["gene"].apply(set).to_dict()

    rows = []
    for cid in compounds["np_id"]:
        for ct, genes in ct_sig.items():
            tgs = tg_by_comp.get(cid, set())
            up = down = 0
            hits = []
            for t in tgs:
                g = tmap.get(t)
                if g in gene_dir and g in genes:
                    d = gene_dir[g]
                    if d > 0:
                        up += 1
                    else:
                        down += 1
                    hits.append((g, d))
            n = len(hits)
            s_inh = up - down  # +1 per disease-UP hit (suppress = reversal), -1 per disease-DOWN hit
            rows.append({
                "compound_id": cid, "cell_type": ct,
                "s_rev_inh": s_inh, "s_rev_act": -s_inh,
                "n_hits": n, "n_up_hits": up, "n_down_hits": down,
                "coverage": n / len(genes) if genes else 0.0,
                "purity": s_inh / n if n else 0.0,
                "hit_genes": ";".join(h[0] for h in hits),
                "no_signal": int(n == 0),
            })
    return pd.DataFrame(rows)


def edge_attr_model(model, data, cid_idx, ct_idx, score_cache):
    def score():
        with torch.no_grad():
            z = model.encode(data)
            return float(torch.sigmoid(model.head(z["compound"][cid_idx], z["cell_type"][ct_idx])).item())
    return score


def main():
    processed = ROOT / "data" / "processed"
    out = ROOT / "results" / "m5_scoring"
    out.mkdir(parents=True, exist_ok=True)

    print("[M5] S_rev signature-reversal scoring")
    compounds, targets, ct_edges, ppi, signatures, adata = load_graph_inputs({"processed": processed})
    srev = compute_srev(compounds, targets, ct_edges, signatures)
    srev.to_csv(out / "s_rev_scores.csv", index=False)

    ranked = {}
    for ct in srev["cell_type"].unique():
        sub = srev[srev["cell_type"] == ct].sort_values(
            ["no_signal", "s_rev_inh", "n_hits"], ascending=[True, False, False])
        ranked[ct] = sub[["compound_id", "s_rev_inh", "s_rev_act", "n_hits",
                          "n_up_hits", "n_down_hits", "coverage", "purity", "no_signal"]].head(20)
    with pd.ExcelWriter(out / "top_per_celltype.xlsx") as xw:
        for ct, sub in ranked.items():
            sub.to_excel(xw, sheet_name=ct[:31], index=False)

    bench = compounds[compounds["is_benchmark"]]["np_id"].astype(str).drop_duplicates().tolist()
    bench_rank, bench_rank_hit = {}, {}
    for ct in srev["cell_type"].unique():
        sub = srev[srev["cell_type"] == ct].sort_values(
            ["no_signal", "s_rev_inh", "n_hits"], ascending=[True, False, False])
        sub_hit = sub[sub["n_hits"] > 0]
        for i, cid in enumerate(sub["compound_id"]):
            if cid in bench:
                bench_rank.setdefault(cid, {})[ct] = i + 1
        for i, cid in enumerate(sub_hit["compound_id"]):
            if cid in bench:
                bench_rank_hit.setdefault(cid, {})[ct] = i + 1
    print(f"      S_rev cells: {srev.shape[0]}, benchmark compounds scored: {len(bench_rank)}")
    print(f"      pairs engaging >=1 signature gene: {int((srev['n_hits'] > 0).sum())}")

    print("[M5] GNN edge-removal attribution on benchmark actives")
    data = build_hetero_graph(compounds, targets, ct_edges, ppi, signatures, adata)
    n_comp = data["compound"].x.shape[0]
    edge_types = [et for et in data.edge_types if "therapeutic" not in et[1]]
    in_feats = {nt: data[nt].x.shape[1] for nt in data.node_types}

    torch.manual_seed(42)
    model = HeteroGraphSAGE(in_feats, hidden=64, out=64, edge_types=edge_types)
    opt = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-4)
    pos = data["compound", "therapeutic", "cell_type"].edge_index
    rng = np.random.RandomState(42)
    perm = torch.randperm(pos.size(1))
    tr = perm[: int(0.8 * pos.size(1))]
    te = perm[int(0.8 * pos.size(1)):]
    from src.graph.model import evaluate, sample_negatives, train_epoch
    n_comp = data["compound"].x.shape[0]
    n_ct = data["cell_type"].x.shape[0]
    neg_pairs = sample_negatives(pos, n_comp, n_ct, rng, 1)
    li = torch.cat([pos[:, tr], neg_pairs[:tr.size(0)].T], dim=1)
    lb = torch.cat([torch.ones(tr.size(0)), torch.zeros(tr.size(0))])
    best = 0.0
    for epoch in range(120):
        loss = train_epoch(model, data, li, lb, opt)
        auc, ap = evaluate(model, data, torch.cat([pos[:, te], neg_pairs[tr.size(0):tr.size(0) + te.size(0)].T], dim=1),
                           torch.cat([torch.ones(te.size(0)), torch.zeros(te.size(0))]))
        if auc > best:
            best = auc
    print(f"      M5 model held-out AUROC: {best:.3f}")

    attr_rows = []
    comp_ids = compounds["np_id"].astype(str).tolist()
    prot_edges = data["compound", "targets", "protein"].edge_index
    comp_idx = {c: i for i, c in enumerate(comp_ids)}

    for bidx, cid in enumerate(bench):
        if cid not in comp_idx:
            continue
        c = comp_idx[cid]
        with torch.no_grad():
            z = model.encode(data)
            logits = model.head(
                z["compound"][c].unsqueeze(0).expand(z["cell_type"].shape[0], -1),
                z["cell_type"]).squeeze()
        scores = torch.sigmoid(logits)
        ct_rank = logits.argsort(descending=True)
        for cti in ct_rank[:3].tolist():
            ct_name = data.ct_list[cti]
            base = float(logits[cti])
            mask = prot_edges[0] == c
            if mask.sum() == 0:
                continue
            drops = []
            for tgt in np.unique(prot_edges[1][mask].numpy()).tolist():
                idx = (prot_edges[0] == c) & (prot_edges[1] == tgt)
                edge_sub = data.clone()
                keep = ~idx
                edge_sub["compound", "targets", "protein"].edge_index = prot_edges[:, keep]
                edge_sub["protein", "targeted_by", "compound"].edge_index = prot_edges[:, keep].flip(0)
                model.eval()
                with torch.no_grad():
                    zz = model.encode(edge_sub)
                    s2 = float(model.head(
                        zz["compound"][c].unsqueeze(0),
                        zz["cell_type"][cti].unsqueeze(0)).item())
                gene = data.prot_id2gene.get(data.prot_ids[tgt], "")
                attr_rows.append({"compound_id": cid, "cell_type": ct_name,
                                  "target": gene, "base_logit": base,
                                  "logit_wo_target": s2, "drop": base - s2})
        print(f"      {cid}: top={[f'{data.ct_list[i]}:{float(scores[i]):.3f}' for i in ct_rank[:3].tolist()]}")

    attr = pd.DataFrame(attr_rows)
    attr.to_csv(out / "gnn_attribution.csv", index=False)
    print(f"      attribution rows: {len(attr)}")

    summary = {
        "s_rev_convention": "CMap-style: s_rev_inh credits +1 per disease-UP gene hit "
                            "(target assumed suppressed) and -1 per disease-DOWN hit; "
                            "s_rev_act is the activation-assumption negation. no_signal rows "
                            "have zero signature-gene hits.",
        "pairs_engaging_signature": int((srev["n_hits"] > 0).sum()),
        "pairs_total": int(len(srev)),
        "s_rev_benchmark_top1_per_celltype": {c: r for c, r in bench_rank.items()},
        "s_rev_benchmark_top1_per_celltype_hits_only": {c: r for c, r in bench_rank_hit.items()},
        "attribution_rows": len(attr),
    }
    with open(out / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("[M5] done ->", out / "summary.json")


if __name__ == "__main__":
    main()
