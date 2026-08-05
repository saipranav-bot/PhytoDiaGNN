import numpy as np
import pandas as pd
import scanpy as sc
import torch
from torch_geometric.data import HeteroData
from sklearn.decomposition import PCA


EXPR_THRESH = 0.5


def load_graph_inputs(paths):
    compounds = pd.read_parquet(paths["processed"] / "compounds.parquet")
    targets = pd.read_parquet(paths["processed"] / "targets.parquet")
    ct_edges = pd.read_parquet(paths["processed"] / "compound_target_edges.parquet")
    ppi = pd.read_parquet(paths["processed"] / "ppi_edges.parquet")
    signatures = pd.read_csv(paths["processed"] / "scrna" / "disease_signatures.csv")
    adata = sc.read_h5ad(paths["processed"] / "scrna" / "merged_islets.h5ad")
    return compounds, targets, ct_edges, ppi, signatures, adata


def build_hetero_graph(compounds, targets, ct_edges, ppi, signatures, adata,
                       pca_dim=128, expr_thresh=EXPR_THRESH, seed=42):
    target_genes = sorted(set(targets["Gene_Symbol"].dropna()) & set(adata.var_names))
    ct_list = sorted(adata.obs["cell_type_pred"].astype(str).unique())
    ct_map = {ct: i for i, ct in enumerate(ct_list)}

    expr = adata[:, target_genes].to_df().groupby(
        adata.obs["cell_type_pred"].astype(str)).mean().reindex(ct_list).fillna(0.0)
    ct_feat = expr.values.astype(np.float32)
    on = expr > expr_thresh

    prot_ids = targets["Target_ID"].dropna().unique()
    prot_id2gene = dict(zip(targets["Target_ID"], targets["Gene_Symbol"]))
    prot_map = {pid: i for i, pid in enumerate(prot_ids)}
    gene2prot = {g: pid for pid, g in prot_id2gene.items()}

    prot_feat = np.zeros((len(prot_ids), len(ct_list)), dtype=np.float32)
    for i, pid in enumerate(prot_ids):
        g = prot_id2gene[pid]
        if g in expr.columns:
            prot_feat[i] = expr[g].values

    pca = PCA(n_components=pca_dim, random_state=seed, svd_solver="randomized")
    fp = np.stack([np.frombuffer(b.encode(), dtype="u1") for b in compounds["morgan1024"]])
    fp = fp.reshape(len(compounds), -1).astype(np.float32)
    pca.fit(fp)
    comp_feat = pca.transform(fp).astype(np.float32)
    comp_ids = compounds["np_id"].astype(str).tolist()
    comp_map = {c: i for i, c in enumerate(comp_ids)}

    edge_index_targets = []
    for _, r in ct_edges.iterrows():
        c, t = r["compound_id"], r["target_id"]
        if c in comp_map and t in prot_map:
            edge_index_targets.append([comp_map[c], prot_map[t]])
    edge_index_targets = torch.tensor(edge_index_targets, dtype=torch.long).T if edge_index_targets else torch.zeros((2, 0), dtype=torch.long)

    edge_index_ppi = []
    for _, r in ppi.iterrows():
        a, b = gene2prot.get(r["gene_a"]), gene2prot.get(r["gene_b"])
        if a in prot_map and b in prot_map:
            edge_index_ppi.append([prot_map[a], prot_map[b]])
    edge_index_ppi = torch.tensor(edge_index_ppi, dtype=torch.long).T if edge_index_ppi else torch.zeros((2, 0), dtype=torch.long)

    edge_index_expr = []
    for g in target_genes:
        pid = gene2prot.get(g)
        if pid in prot_map:
            for ct in on.index[on[g]]:
                edge_index_expr.append([prot_map[pid], ct_map[ct]])
    edge_index_expr = torch.tensor(edge_index_expr, dtype=torch.long).T if edge_index_expr else torch.zeros((2, 0), dtype=torch.long)

    edge_index_ctd = torch.tensor([[ct_map[ct] for ct in ct_list], [0] * len(ct_list)], dtype=torch.long)

    data = HeteroData()
    data["compound"].x = torch.tensor(comp_feat)
    data["protein"].x = torch.tensor(prot_feat)
    data["cell_type"].x = torch.tensor(ct_feat)
    data["disease"].x = torch.tensor([[1.0]])
    data["compound", "targets", "protein"].edge_index = edge_index_targets
    data["protein", "interacts", "protein"].edge_index = edge_index_ppi
    data["protein", "expressed_in", "cell_type"].edge_index = edge_index_expr
    data["cell_type", "associated_with", "disease"].edge_index = edge_index_ctd

    ther = set()
    ct_by_gene = {g: on.index[on[g]].tolist() for g in target_genes}
    for _, r in ct_edges.iterrows():
        c, t = r["compound_id"], r["target_id"]
        if c not in comp_map:
            continue
        g = prot_id2gene.get(t)
        if g in ct_by_gene:
            for ct in ct_by_gene[g]:
                ther.add((comp_map[c], ct_map[ct]))
    ther = sorted(ther)
    data["compound", "therapeutic", "cell_type"].edge_index = (
        torch.tensor(ther, dtype=torch.long).T if ther else torch.zeros((2, 0), dtype=torch.long))

    data["protein", "targeted_by", "compound"].edge_index = edge_index_targets.flip(0)
    data["cell_type", "expresses", "protein"].edge_index = edge_index_expr.flip(0)
    data["disease", "associates_with", "cell_type"].edge_index = edge_index_ctd.flip(0)
    data["cell_type", "therapeutic_for", "compound"].edge_index = (
        data["compound", "therapeutic", "cell_type"].edge_index.flip(0))

    data.ct_map = ct_map
    data.ct_list = ct_list
    data.comp_map = comp_map
    data.prot_ids = prot_ids
    data.prot_id2gene = prot_id2gene
    return data
