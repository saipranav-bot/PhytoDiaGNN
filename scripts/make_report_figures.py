"""Regenerate all report figures (publication-quality v4).

Removed in v4:
  - fig_pipeline (workflow diagram) and fig_gnn_arch (GNN architecture
    schematic) deleted per user request.

Fixes applied vs v2:
  - Legend policy: every legend sits OUTSIDE the data area
    (right side, frameon=False) -- no legends inside any plot.
  - fig_srev_dist (a): log-axis value labels use a multiplicative
    offset (v * 1.35) so labels sit just above small bars.
  - fig_kg_graph: de-crowded to ~34 nodes (8 seeds + expressed targets
    only, capped), larger canvas, white-halo labels, legend outside.
  - fig_ppi: de-crowded to 6 seeds + top-12 neighbors (~18 nodes),
    halo labels, edge width by STRING score, legend outside.
  - NEW spatial overlay plots on the Visium tissue image:
      fig_spatial_dominant.png (spots colored by dominant cell type)
      fig_spatial_niche.png    (spots colored by niche cluster)
    using tissue_positions_list.csv + scalefactors_json.json +
    tissue_hires_image.png + deconv.csv / niches.csv.
  - fig_heatmap: sparse y-tick labels, light white gridlines, taller canvas.
  - fig_deconv bar chart dropped (replaced by the tissue overlays).
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patheffects import withStroke
import networkx as nx
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "results" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.bbox": "tight",
    "font.size": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.family": "DejaVu Sans",
})

CELL_ORDER = ["acinar", "alpha", "beta", "delta", "ductal",
              "endothelial", "epsilon", "gamma", "stellate"]
CT_COLORS = {
    "acinar": "#8C6BB1", "alpha": "#E6A23C", "beta": "#4C72B0",
    "delta": "#2CA02C", "ductal": "#C44E52", "endothelial": "#17BECF",
    "epsilon": "#F7B6D2", "gamma": "#7F7F7F", "stellate": "#BCBD22",
}
COND_COLORS = {"T2D": "#C44E52", "ND": "#4C72B0"}
KIND_COLORS = {"active": "#4C72B0", "decoys": "#C44E52", "candidate": "#2CA02C"}

BRAIN_CT = {
    "neuron_exc": "#4C72B0", "neuron_inh": "#DD8452", "oligo": "#C44E52",
    "astro": "#55A868", "microglia": "#E6A23C", "endothelial": "#17BECF",
    "ependymal": "#8C6BB1",
}
NICHE_COLORS = ["#4C72B0", "#DD8452", "#C44E52", "#55A868", "#E6A23C", "#17BECF"]

OUT_LEGEND = dict(loc="center left", bbox_to_anchor=(1.01, 0.5),
                  frameon=False)


def save(fig, name):
    fig.savefig(FIG / name)
    plt.close(fig)
    print("  wrote", name)


# --------------------------------------------------------------------------
# Figure 1. Cell-type composition
# --------------------------------------------------------------------------
def fig_celltypes(qc):
    ct = qc["cell_types"]
    order = [c for c in CELL_ORDER if c in ct]
    counts = [ct[c] for c in order]
    fig, ax = plt.subplots(figsize=(6.4, 3.5))
    bars = ax.bar(order, counts, color=[CT_COLORS[c] for c in order],
                  edgecolor="white", linewidth=0.6)
    for b, v in zip(bars, counts):
        ax.text(b.get_x() + b.get_width() / 2, v + 12, f"{v:,}",
                ha="center", fontsize=8)
    ax.set_ylabel("Number of cells")
    ax.set_title("Cell-type composition after integration and annotation")
    ax.text(0.985, 0.97, "Total cells = 3,880", transform=ax.transAxes,
            ha="right", va="top", fontsize=9, fontweight="bold")
    ax.tick_params(axis="x", rotation=45)
    ax.set_ylim(0, 900)
    save(fig, "fig_celltypes.png")


# --------------------------------------------------------------------------
# Figure 3. Direction-balanced disease signature (up = red, down = blue)
# --------------------------------------------------------------------------
def fig_signature_direction(sig):
    bal = sig.pivot_table(index="cell_type", columns="direction", values="gene",
                          aggfunc="count").reindex(CELL_ORDER).fillna(0)
    up, down = bal[1], bal[-1]
    fig, ax = plt.subplots(figsize=(6.4, 3.5))
    ax.bar(bal.index, up, color="#C44E52", label="Upregulated in T2D")
    ax.bar(bal.index, down, bottom=up, color="#4C72B0",
           label="Downregulated in T2D")
    ax.set_ylabel("Replicated genes")
    ax.set_title("Direction-balanced disease signature "
                 f"({int(sig['gene'].nunique()):,} replicated genes)")
    ax.tick_params(axis="x", rotation=45)
    ax.legend(fontsize=8, **OUT_LEGEND)
    ax.text(0.985, 0.97, f"{int(sig['direction'].eq(1).sum())} up  /  "
            f"{int(sig['direction'].eq(-1).sum())} down",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=9, fontweight="bold")
    save(fig, "fig_signature_direction.png")


# --------------------------------------------------------------------------
# Figures 4-5. UMAP by cell type and by condition
# --------------------------------------------------------------------------
def fig_umaps(adata):
    X = adata.obsm["X_umap"]
    obs = adata.obs
    fig, ax = plt.subplots(figsize=(5.8, 4.6))
    for ct in CELL_ORDER:
        m = obs["cell_type_pred"] == ct
        if m.sum():
            ax.scatter(X[m, 0], X[m, 1], s=4, color=CT_COLORS[ct],
                       label=ct, linewidths=0, alpha=0.8)
    ax.set_xlabel("UMAP-1"); ax.set_ylabel("UMAP-2")
    ax.set_title(f"Integrated human-islet single cells (n = {X.shape[0]:,})\n"
                 "colored by cell type")
    ax.legend(fontsize=7, markerscale=3, title="Cell type", title_fontsize=8,
              **OUT_LEGEND)
    save(fig, "fig_umap_celltype.png")

    fig, ax = plt.subplots(figsize=(5.6, 4.4))
    for cond in ["T2D", "ND"]:
        m = obs["condition"] == cond
        ax.scatter(X[m, 0], X[m, 1], s=5, color=COND_COLORS[cond],
                   label=cond, linewidths=0, alpha=0.65)
    ax.set_xlabel("UMAP-1"); ax.set_ylabel("UMAP-2")
    ax.set_title(f"Integrated human-islet single cells (n = {X.shape[0]:,})\n"
                 "colored by disease condition")
    ax.legend(fontsize=9, markerscale=3, **OUT_LEGEND)
    save(fig, "fig_umap_disease.png")


# --------------------------------------------------------------------------
# Figures 6-7. Spatial overlays on the Visium tissue (NEW)
# --------------------------------------------------------------------------
def _load_spatial():
    raw = ROOT / "data/raw/spatial/mouse_brain/spatial"
    pos = pd.read_csv(raw / "tissue_positions_list.csv", header=None)
    pos.columns = ["barcode", "in_tissue", "array_row", "array_col",
                   "pxl_row_fullres", "pxl_col_fullres"]
    pos = pos[pos["in_tissue"] == 1].set_index("barcode")
    sf = json.load(open(raw / "scalefactors_json.json"))
    img = matplotlib.image.imread(raw / "tissue_hires_image.png")
    deconv = pd.read_csv(ROOT / "results/spatial/deconv.csv", index_col=0)
    niches = pd.read_csv(ROOT / "results/spatial/niches.csv", index_col=0)
    pos = pos.loc[pos.index.intersection(deconv.index)]
    return pos, sf, img, deconv.loc[pos.index], niches.loc[pos.index]


def _spot_axes(img):
    fig, ax = plt.subplots(figsize=(7.8, 7.8))
    ax.imshow(img, origin="upper")
    ax.set_xlim(0, img.shape[1]); ax.set_ylim(img.shape[0], 0)
    ax.set_xticks([]); ax.set_yticks([])
    return fig, ax


def _spot_size(sf, fig_dpi=150):
    diam_hires = sf["spot_diameter_fullres"] * sf["tissue_hires_scalef"]
    r_pt = diam_hires / fig_dpi * 72 / 2
    return max(8.0, np.pi * r_pt ** 2)


def fig_spatial_dominant():
    pos, sf, img, deconv, _ = _load_spatial()
    dom = deconv.idxmax(axis=1)
    fig, ax = _spot_axes(img)
    x = pos["pxl_col_fullres"].values * sf["tissue_hires_scalef"]
    y = pos["pxl_row_fullres"].values * sf["tissue_hires_scalef"]
    s = _spot_size(sf)
    order = dom.value_counts().index
    handles = []
    for ct in order:
        m = dom == ct
        ax.scatter(x[m], y[m], s=s, color=BRAIN_CT[ct], linewidths=0,
                   alpha=0.75, label=ct)
        handles.append(mpatches.Patch(color=BRAIN_CT[ct], label=ct))
    ax.legend(handles=handles, fontsize=7.5, title="Dominant cell type",
              title_fontsize=8.5, **OUT_LEGEND)
    ax.set_title(f"Visium spot deconvolution (mouse brain, {len(dom):,} "
                 "spots)\nNNLS against brain reference, dominant cell type")
    save(fig, "fig_spatial_dominant.png")


def fig_spatial_niche():
    pos, sf, img, deconv, niches = _load_spatial()
    niche = niches["niche"].astype(int)
    fig, ax = _spot_axes(img)
    x = pos["pxl_col_fullres"].values * sf["tissue_hires_scalef"]
    y = pos["pxl_row_fullres"].values * sf["tissue_hires_scalef"]
    s = _spot_size(sf)
    handles = []
    for k in sorted(niche.unique()):
        m = niche == k
        ax.scatter(x[m], y[m], s=s, color=NICHE_COLORS[k], linewidths=0,
                   alpha=0.75, label=f"Niche {k}")
        handles.append(mpatches.Patch(color=NICHE_COLORS[k], label=f"Niche {k}"))
    ax.legend(handles=handles, fontsize=7.5, title="Niche cluster",
              title_fontsize=8.5, **OUT_LEGEND)
    ax.set_title(f"Visium niche clusters (mouse brain, {len(niche):,} spots)\n"
                 "KMeans on deconvolved cell-type proportions, k = 6")
    save(fig, "fig_spatial_niche.png")


# --------------------------------------------------------------------------
# Figure 8. AUROC comparison
# --------------------------------------------------------------------------
def fig_gnn_auc(m4):
    fig, ax = plt.subplots(figsize=(4.8, 3.4))
    labels = ["HeteroGNN", "RandomForest", "Random classifier"]
    vals = [m4["mean_gnn_auc"], m4["mean_rf_auc"], 0.5]
    ax.bar(labels, vals, color=["#4C72B0", "#55A868", "#C44E52"])
    for i, v in enumerate(vals):
        ax.text(i, v + 0.01, f"{v:.3f}", ha="center", fontsize=9)
    ax.set_ylim(0, 1.1); ax.set_ylabel("AUROC")
    ax.set_title("Link-prediction performance (5-fold edge-masked CV)")
    save(fig, "fig_gnn_auc.png")


# --------------------------------------------------------------------------
# Figure 9. Knowledge-graph subset (de-crowded)
# --------------------------------------------------------------------------
def fig_kg_graph():
    comp = pd.read_parquet(ROOT / "data/processed/compounds.parquet")
    ct_edges = pd.read_parquet(ROOT / "data/processed/compound_target_edges.parquet")
    tgt = pd.read_parquet(ROOT / "data/processed/targets.parquet")
    sig_m = pd.read_csv(ROOT / "data/processed/scrna/signature_matrix.csv",
                        index_col=0)

    seeds = ["NPC190454", "NPC461221", "NPC100742", "NPC71347",
             "NPC245229", "NPC164498", "NPC50861", "NPC279121"]
    sub_ct = (ct_edges[ct_edges["compound_id"].isin(seeds)]
              [["compound_id", "target_id"]].drop_duplicates())
    gene_of = dict(zip(tgt["Target_ID"], tgt["Gene_Symbol"]))
    sub_ct["gene"] = sub_ct["target_id"].map(gene_of)

    expr_genes = set(sig_m.columns)
    all_targets = list(dict.fromkeys(sub_ct["gene"].dropna()))
    if len(all_targets) > 16:
        keep = set(pd.Series(all_targets).value_counts()
                   .head(16).index)
        sub_ct = sub_ct[sub_ct["gene"].isin(keep)]
    targets = list(dict.fromkeys(sub_ct["gene"].dropna()))

    G = nx.Graph()
    G.add_nodes_from(seeds, kind="compound")
    G.add_nodes_from(targets, kind="protein")
    for _, r in sub_ct.iterrows():
        if r["gene"] in targets:
            G.add_edge(r["compound_id"], r["gene"])
    for g in targets:
        if g not in expr_genes:
            continue
        on_cts = sig_m.index[sig_m[g] > 0.5].tolist()
        for ct in on_cts:
            G.add_edge(g, ct)
    G.add_nodes_from(CELL_ORDER, kind="cell_type")
    G.add_node("T2D", kind="disease")
    for ct in CELL_ORDER:
        G.add_edge(ct, "T2D")

    pos = nx.spring_layout(G, seed=7, k=0.85, iterations=300)
    kind_color = {"compound": "#2CA02C", "protein": "#8C6BB1",
                  "cell_type": "#E6A23C", "disease": "#C44E52"}
    kind_size = {"compound": 620, "protein": 320, "cell_type": 520,
                 "disease": 900}
    fig, ax = plt.subplots(figsize=(9.2, 7.2))
    for k, col in kind_color.items():
        nodes = [n for n, d in G.nodes(data=True) if d["kind"] == k]
        nx.draw_networkx_nodes(G, pos, nodelist=nodes, node_size=kind_size[k],
                               node_color=col, alpha=0.9, ax=ax)
    nx.draw_networkx_edges(G, pos, alpha=0.30, edge_color="#888888",
                           width=0.8, ax=ax)
    halo = [withStroke(linewidth=2.2, foreground="white")]
    styles = {"compound": (9.5, True), "protein": (7.0, False),
              "cell_type": (8.0, False), "disease": (10.0, True)}
    for k, (fs, bold) in styles.items():
        nodes = [n for n, d in G.nodes(data=True) if d["kind"] == k]
        for n in nodes:
            ax.text(pos[n][0], pos[n][1], n, fontsize=fs,
                    fontweight="bold" if bold else "normal",
                    color="#1A1A1A", ha="center", va="center",
                    path_effects=halo)
    handles = [mpatches.Patch(color=c, label=k) for k, c in kind_color.items()]
    ax.legend(handles=handles, fontsize=8.5, title="Node type",
              title_fontsize=9, **OUT_LEGEND)
    ax.set_title("Knowledge graph (subset): candidate compounds, expressed "
                 "protein targets,\ncell types, and disease association "
                 f"({G.number_of_nodes()} nodes, {G.number_of_edges()} edges)")
    ax.axis("off")
    save(fig, "fig_kg_graph.png")


# --------------------------------------------------------------------------
# Figure 10. S_rev (engagement + signed scores)
# --------------------------------------------------------------------------
def fig_srev_dist(s):
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.5))
    ax = axes[0]
    eng = s[s["n_hits"] > 0]
    counts = [(s["n_hits"] == 0).sum(),
              (s["n_hits"] == 1).sum(),
              (s["n_hits"] == 2).sum(),
              (s["n_hits"] >= 3).sum()]
    labels = ["0 hits", "1 hit", "2 hits", "\u22653 hits"]
    bars = ax.bar(labels, counts, color=["#B0B0B0", "#4C72B0", "#E6A23C", "#C44E52"])
    for b, v in zip(bars, counts):
        ax.text(b.get_x() + b.get_width() / 2, v * 1.35, f"{v:,}",
                ha="center", fontsize=8)
    ax.set_yscale("log"); ax.set_ylabel("Pairs (log)")
    ax.set_title("(a) Signature-gene hits across 27,000 pairs")
    ax.set_ylim(ymax=max(counts) * 1.8)
    ax.text(0.985, 0.95, f"{len(eng):,} engaging pairs",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=9, fontweight="bold")

    ax = axes[1]
    bins = np.arange(-4, 5, 1)
    ax.hist(eng["s_rev_inh"], bins=bins, alpha=0.65, color="#C44E52",
            label="s_rev_inh (inhibition)")
    ax.hist(eng["s_rev_act"], bins=bins, alpha=0.65, color="#4C72B0",
            label="s_rev_act (activation)")
    ax.set_xlabel("Signed S_rev score"); ax.set_ylabel("Engaging pairs")
    ax.set_title("(b) Signed scores of engaging pairs only")
    ax.legend(fontsize=8, **OUT_LEGEND)
    fig.suptitle("S_rev reversal scoring (CMap convention)", y=1.0,
                 fontsize=11, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    save(fig, "fig_srev_dist.png")


# --------------------------------------------------------------------------
# Figure 12. Attribution (polished)
# --------------------------------------------------------------------------
def fig_attribution(a):
    top = (a.sort_values("drop", ascending=False)
            .drop_duplicates(["compound_id", "target"]).head(15))
    labels = [f"{c} / {t}" for c, t in zip(top["compound_id"], top["target"])]
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.barh(labels, top["drop"], color="#DD8452", edgecolor="white")
    ax.set_xlabel("Logit drop on edge removal")
    ax.set_title("GNN edge-removal attribution: top protein drivers")
    ax.invert_yaxis()
    save(fig, "fig_attribution.png")


# --------------------------------------------------------------------------
# Figure 13. Compound x cell-type S_rev heatmap
# --------------------------------------------------------------------------
def fig_heatmap(s):
    eng = s[s["n_hits"] > 0]
    piv = (eng.pivot_table(index="compound_id", columns="cell_type",
                           values="s_rev_inh", aggfunc="first")
           .reindex(columns=CELL_ORDER).fillna(0))
    order = piv.abs().sum(axis=1).sort_values(ascending=False).index
    piv = piv.loc[order]
    fig, ax = plt.subplots(figsize=(6.2, 9.6))
    vmax = max(2.5, np.abs(piv.values).max())
    im = ax.imshow(piv.values, cmap="RdBu_r", aspect="auto", vmin=-vmax,
                   vmax=vmax)
    ax.set_xticks(range(len(CELL_ORDER)))
    ax.set_xticklabels(CELL_ORDER, rotation=45, ha="right")
    step = max(1, round(len(piv) / 18))
    ytick = list(range(0, len(piv), step))
    ax.set_yticks(ytick)
    ax.set_yticklabels([piv.index[i] for i in ytick], fontsize=7)
    ax.set_ylabel("Compound")
    ax.set_xlabel("Cell type")
    ax.set_xticks(np.arange(len(CELL_ORDER)) - 0.5, minor=True)
    ax.set_yticks(np.arange(len(piv)) - 0.5, minor=True)
    ax.grid(which="minor", color="white", linewidth=0.6, alpha=0.7)
    ax.tick_params(which="minor", length=0)
    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("S_rev_inh")
    ax.set_title(f"Compound \u00d7 cell-type reversal scores "
                 f"({len(piv)} compounds, {len(eng)} engaging pairs)")
    save(fig, "fig_heatmap.png")


# --------------------------------------------------------------------------
# Figure 14. PPI subnetwork (de-crowded)
# --------------------------------------------------------------------------
def fig_ppi():
    ppi = pd.read_parquet(ROOT / "data/processed/ppi_edges.parquet")
    seeds = ["PLK1", "TTR", "ALOX5", "NPC1", "PPARG", "GSK3B"]
    G = nx.Graph()
    for _, r in ppi.iterrows():
        G.add_edge(r["gene_a"], r["gene_b"], weight=r["string_score"])
    sub = nx.Graph()
    sub.add_nodes_from(seeds)
    for seed in seeds:
        if seed not in G:
            continue
        for nb in G.neighbors(seed):
            sub.add_edge(seed, nb, weight=G[seed][nb]["weight"])
    if sub.number_of_nodes() > 18:
        deg = dict(sub.degree())
        keep = set(seeds) | set(sorted(deg, key=lambda n: -deg[n])[:12])
        sub = sub.subgraph(keep).copy()
    pos = nx.spring_layout(sub, seed=11, k=1.15, iterations=300)
    fig, ax = plt.subplots(figsize=(8.2, 6.6))
    node_colors = [("#C44E52" if n in seeds else "#8C6BB1")
                   for n in sub.nodes()]
    node_sizes = [720 if n in seeds else 260 for n in sub.nodes()]
    nx.draw_networkx_nodes(sub, pos, node_size=node_sizes,
                           node_color=node_colors, alpha=0.9, ax=ax)
    ws = [0.5 + 0.9 * sub[u][v]["weight"] / 1000.0 for u, v in sub.edges()]
    nx.draw_networkx_edges(sub, pos, width=ws, alpha=0.4,
                           edge_color="#888888", ax=ax)
    halo = [withStroke(linewidth=2.2, foreground="white")]
    for n in sub.nodes():
        ax.text(pos[n][0], pos[n][1], n,
                fontsize=9 if n in seeds else 7.5,
                fontweight="bold" if n in seeds else "normal",
                color="#1A1A1A", ha="center", va="center",
                path_effects=halo)
    ax.legend(handles=[mpatches.Patch(color="#C44E52", label="Attributed / benchmark target"),
                       mpatches.Patch(color="#8C6BB1", label="STRING interaction partner")],
              fontsize=8.5, **OUT_LEGEND)
    ax.set_title("STRING PPI subnetwork around key attributed targets "
                 f"(6 seeds + top-12 partners, {sub.number_of_edges()} edges)")
    ax.axis("off")
    save(fig, "fig_ppi.png")


# --------------------------------------------------------------------------
# Figures 15-16. Docking affinity + ROC
# --------------------------------------------------------------------------
def fig_docking(d, m6):
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.5), sharey=True)
    for ax, rec in zip(axes, ["GSK3B", "PPARG"]):
        sub = d[d["rec"] == rec]
        kinds = ["active", "decoys", "candidate"]
        pos = []
        for i, k in enumerate(kinds):
            vals = sub[sub["kind"] == k]["affinity"]
            pts = np.random.RandomState(3 + i).uniform(-0.18, 0.18, len(vals))
            ax.boxplot(vals, positions=[i], widths=0.5, showfliers=False,
                       patch_artist=True,
                       boxprops=dict(facecolor=KIND_COLORS[k], alpha=0.5),
                       medianprops=dict(color="black"))
            ax.scatter(i + pts, vals, s=11, color=KIND_COLORS[k], alpha=0.8,
                       edgecolor="white", linewidth=0.3)
            pos.append(np.median(vals))
        ax.set_xticks(range(len(kinds)))
        ax.set_xticklabels(["Known\nactives", "Decoys", "Novel\ncandidates"],
                           fontsize=8)
        for i, v in enumerate(pos):
            ax.text(i, v + 0.15, f"{v:.2f}", ha="center", fontsize=7.5)
        au = m6[rec]["AUROC"]
        ax.set_title(f"{rec}  (AUROC = {au:.3f})", fontsize=10)
        ax.axhline(0, color="gray", lw=0.8, ls=":")
    axes[0].set_ylabel("Vina affinity (kcal/mol)")
    fig.suptitle("Docking scores by compound kind and receptor "
                 "(median annotated)", fontsize=11, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    save(fig, "fig_docking_by_kind.png")


def fig_docking_roc(m6):
    from sklearn.metrics import roc_curve
    d = pd.read_csv(ROOT / "results/m6_docking/docking_results.csv")
    fig, ax = plt.subplots(figsize=(5.6, 4.4))
    for rec in ["GSK3B", "PPARG"]:
        sub = d[d["rec"] == rec]
        pos = sub[sub["kind"] == "active"]["affinity"].values
        neg = sub[sub["kind"] == "decoys"]["affinity"].values
        y = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
        x = np.concatenate([pos, neg])
        fpr, tpr, _ = roc_curve(y, -x)
        au = m6[rec]["AUROC"]
        ax.plot(fpr, tpr, lw=1.8, label=f"{rec} (AUROC = {au:.3f})")
    ax.plot([0, 1], [0, 1], ls="--", color="gray", lw=1.0)
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
    ax.set_title("Docking benchmark: known actives vs property-matched decoys")
    ax.legend(fontsize=9, **OUT_LEGEND)
    ax.text(0.985, 0.03,
            "14 actives vs 68 decoys per receptor", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=8, color="#555555")
    save(fig, "fig_docking_roc.png")


# --------------------------------------------------------------------------
# Figure 17. Top-20 novel candidates
# --------------------------------------------------------------------------
def fig_top20(d):
    cand = d[d["kind"] == "candidate"]
    best = cand.sort_values("affinity").groupby("compound_id").first().reset_index()
    best = best.nsmallest(20, "affinity")
    best = best.sort_values("affinity")
    colors = [KIND_COLORS["candidate"] if r == "GSK3B" else "#17BECF"
              for r in best["rec"]]
    fig, ax = plt.subplots(figsize=(6.6, 5.8))
    ax.barh(best["compound_id"], best["affinity"], color=colors,
            edgecolor="white")
    ax.invert_yaxis()
    ax.set_xlabel("Vina affinity (kcal/mol, more negative = stronger)")
    ax.set_title("Top-20 novel phytochemical candidates")
    ax.legend(handles=[mpatches.Patch(color=KIND_COLORS["candidate"], label="GSK3B"),
                       mpatches.Patch(color="#17BECF", label="PPARG")],
              fontsize=8, **OUT_LEGEND)
    save(fig, "fig_top20.png")


# --------------------------------------------------------------------------
# Figure 18. 2D structures of top candidates
# --------------------------------------------------------------------------
def fig_structures(d):
    from rdkit import Chem
    from rdkit.Chem import Draw
    cand = d[d["kind"] == "candidate"].drop_duplicates("compound_id")
    top = cand.sort_values("affinity").head(4)
    mols = []
    for _, r in top.iterrows():
        m = Chem.MolFromSmiles(r["smiles"])
        m.SetProp("_Name", f"{r['compound_id']}  ({r['affinity']:.1f} kcal/mol)")
        mols.append(m)
    img = Draw.MolsToGridImage(mols, molsPerRow=4, subImgSize=(420, 300))
    img.save(str(FIG / "fig_structures.png"))
    print("  wrote fig_structures.png")


# --------------------------------------------------------------------------
def main():
    qc = json.load(open(ROOT / "results/scrna/qc.json"))
    sig = pd.read_csv(ROOT / "data/processed/scrna/disease_signatures.csv")
    s = pd.read_csv(ROOT / "results/m5_scoring/s_rev_scores.csv")
    a = pd.read_csv(ROOT / "results/m5_scoring/gnn_attribution.csv")
    m4 = json.load(open(ROOT / "results/m4_gnn/summary.json"))
    m6 = json.load(open(ROOT / "results/m6_docking/summary.json"))
    d = pd.read_csv(ROOT / "results/m6_docking/docking_results.csv")

    import scanpy as sc
    adata = sc.read_h5ad(ROOT / "data/processed/scrna/merged_islets.h5ad")

    print("[figures] generating ...")
    fig_celltypes(qc)
    fig_signature_direction(sig)
    fig_umaps(adata)
    fig_spatial_dominant()
    fig_spatial_niche()
    fig_gnn_auc(m4)
    fig_kg_graph()
    fig_srev_dist(s)
    fig_attribution(a)
    fig_heatmap(s)
    fig_ppi()
    fig_docking(d, m6)
    fig_docking_roc(m6)
    fig_top20(d)
    fig_structures(d)
    print("done ->", FIG)


if __name__ == "__main__":
    main()
