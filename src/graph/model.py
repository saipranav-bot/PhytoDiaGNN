import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, SAGEConv


class LinkPredictor(torch.nn.Module):
    def __init__(self, in_dim, hidden, out, num_layers=2, dropout=0.2):
        super().__init__()
        self.lin1 = torch.nn.Linear(2 * in_dim, hidden)
        self.lin2 = torch.nn.Linear(hidden, out)
        self.dropout = dropout

    def forward(self, x_src, x_dst):
        h = torch.cat([x_src, x_dst], dim=-1)
        h = F.relu(self.lin1(h))
        h = F.dropout(h, p=self.dropout, training=self.training)
        return self.lin2(h)


class HeteroGraphSAGE(torch.nn.Module):
    def __init__(self, in_feats, hidden=64, out=64, edge_types=None, num_layers=2, dropout=0.2):
        super().__init__()
        self.lins = torch.nn.ModuleDict()
        for ntype, dim in in_feats.items():
            self.lins[ntype] = torch.nn.Linear(dim, hidden)
        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            convs = {et: SAGEConv((hidden, hidden), hidden) for et in edge_types}
            self.convs.append(HeteroConv(convs, aggr="mean"))
        self.head = LinkPredictor(hidden, hidden, 1, num_layers=2, dropout=dropout)
        self.dropout = dropout

    def encode(self, data):
        x = {}
        for ntype, feats in data.x_dict.items():
            x[ntype] = F.relu(self.lins[ntype](feats))
        for conv in self.convs:
            x = conv(x, data.edge_index_dict)
            x = {k: F.relu(v) for k, v in x.items()}
            x = {k: F.dropout(v, p=self.dropout, training=self.training) for k, v in x.items()}
        return x

    def decode(self, z, edge_label_index):
        return self.head(z["compound"][edge_label_index[0]],
                         z["cell_type"][edge_label_index[1]])

    def forward(self, data, edge_label_index):
        z = self.encode(data)
        return self.decode(z, edge_label_index)


def train_epoch(model, data, edge_label_index, edge_label, optimizer, batch_size=1024):
    model.train()
    perm = torch.randperm(edge_label_index.size(1))
    total = 0.0
    n = 0
    for i in range(0, len(perm), batch_size):
        idx = perm[i:i + batch_size]
        optimizer.zero_grad()
        out = model(data, edge_label_index[:, idx]).squeeze()
        loss = F.binary_cross_entropy_with_logits(out, edge_label[idx])
        loss.backward()
        optimizer.step()
        total += loss.item() * len(idx)
        n += len(idx)
    return total / n


def evaluate(model, data, edge_label_index, edge_label):
    model.eval()
    with torch.no_grad():
        out = torch.sigmoid(model(data, edge_label_index).squeeze())
    from sklearn.metrics import roc_auc_score, average_precision_score
    y = edge_label.numpy()
    p = out.numpy()
    auc = roc_auc_score(y, p) if len(set(y)) > 1 else float("nan")
    ap = average_precision_score(y, p)
    return auc, ap


def sample_negatives(pos_index, n_compounds, n_celltypes, rng, n_neg):
    n = pos_index.size(1)
    neg = set()
    pos_set = set(map(tuple, pos_index.T.tolist()))
    while len(neg) < n_neg * n:
        ci = rng.randint(0, n_compounds, size=n_neg * n)
        ct = rng.randint(0, n_celltypes, size=n_neg * n)
        for c, t in zip(ci, ct):
            if (c, t) not in pos_set:
                neg.add((int(c), int(t)))
            if len(neg) >= n_neg * n:
                break
    neg = torch.tensor(list(neg)[:n_neg * n], dtype=torch.long)
    return neg
