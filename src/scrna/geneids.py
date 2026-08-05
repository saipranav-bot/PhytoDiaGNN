import json
import time
import requests
import pandas as pd

MYGENE_URL = "https://mygene.info/v3/query"


def _batch_query(ids, fields="symbol", species="human", chunk=1000, retries=3):
    out = {}
    for i in range(0, len(ids), chunk):
        batch = ids[i:i + chunk]
        payload = {"q": json.dumps(batch), "scopes": "ensembl.gene", "fields": fields, "species": species}
        ok = False
        for attempt in range(retries):
            try:
                r = requests.post(MYGENE_URL, data=payload, timeout=120)
                r.raise_for_status()
                for hit in r.json():
                    q = hit.get("query")
                    sym = hit.get("symbol")
                    out[q] = sym if isinstance(sym, str) else (sym[0] if isinstance(sym, list) and sym else q)
                ok = True
                break
            except Exception:
                time.sleep(2 * (attempt + 1))
        if not ok:
            for q in batch:
                out.setdefault(q, q)
    return out


def ensembl_to_symbol_map(ensembl_ids, cache_path=None):
    ids = sorted(set(str(i) for i in ensembl_ids if str(i).startswith("ENSG")))
    mapping = {}
    if cache_path is not None:
        import os
        if os.path.exists(cache_path):
            cached = pd.read_csv(cache_path, dtype=str)
            mapping = dict(zip(cached["ensembl"], cached["symbol"]))
            ids = [i for i in ids if i not in mapping]
    if ids:
        new = _batch_query(ids)
        mapping.update(new)
        if cache_path is not None:
            df = pd.DataFrame({"ensembl": list(mapping), "symbol": list(mapping.values())})
            df.to_csv(cache_path, index=False)
    return mapping


def normalize_gene_ids(adata, var_col="var_names", cache_path=None):
    mapping = ensembl_to_symbol_map(adata.var_names, cache_path=cache_path)
    mapped = pd.Series([mapping.get(v, v) for v in adata.var_names], index=adata.var_names)
    adata.var["original_id"] = adata.var_names.astype(str)
    adata.var_names = mapped.astype(str)
    adata = adata[:, ~adata.var_names.duplicated()].copy()
    return adata
