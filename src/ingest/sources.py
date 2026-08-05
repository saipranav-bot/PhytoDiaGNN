import gzip
import hashlib
import json
import os
import pandas as pd
import requests


def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


def download(url, dest, timeout=300):
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    if os.path.exists(dest):
        return dest
    with requests.get(url, stream=True, timeout=timeout) as r:
        r.raise_for_status()
        with open(dest, "wb") as fh:
            for chunk in r.iter_content(chunk_size=1 << 20):
                fh.write(chunk)
    return dest


def read_delimited(path, sep="\t"):
    if str(path).endswith(".gz"):
        with gzip.open(path, "rt") as fh:
            return pd.read_csv(fh, sep=sep, dtype=str, low_memory=False)
    return pd.read_csv(path, sep=sep, dtype=str, low_memory=False)


CMAUP_BASE = "https://bidd.group/CMAUP/downloadFiles"

CMAUP_FILES = {
    "ingredients": "CMAUPv2.0_download_Ingredients_All.txt",
    "targets": "CMAUPv2.0_download_Targets.txt",
    "ingredient_target": "CMAUPv2.0_download_Ingredient_Target_Associations_ActivityValues_References.txt",
    "plants": "CMAUPv2.0_download_Plants.txt",
    "plant_ingredient": ("CMAUPv2.0_download_Plant_Ingredient_Associations_allIngredients.txt",
                         ["Plant_ID", "Ingredient_ID"]),
}

DEFERRED_SOURCES = {
    "imppat": {"status": "deferred", "reason": "JS-gated download; CMAUP covers MVP"},
    "tcmsp": {"status": "deferred", "reason": "login-gated download"},
}


def fetch_cmaup(raw_dir):
    out = {}
    for key, val in CMAUP_FILES.items():
        if isinstance(val, tuple):
            fname, names = val
            path = download(f"{CMAUP_BASE}/{fname}", os.path.join(raw_dir, fname))
            df = pd.read_csv(path, sep="\t", dtype=str, header=None, low_memory=False)
            df.columns = names
            out[key] = df
        else:
            path = download(f"{CMAUP_BASE}/{val}", os.path.join(raw_dir, val))
            out[key] = read_delimited(path)
        out[key].attrs["source"] = fname if isinstance(val, tuple) else val
    return out


def fetch_all(raw_dir):
    cmaup = fetch_cmaup(raw_dir)
    manifest = {}
    for k, v in CMAUP_FILES.items():
        fname = v if isinstance(v, str) else v[0]
        manifest[k] = {"status": "downloaded", "sha256": sha256_file(os.path.join(raw_dir, fname))}
    manifest.update(DEFERRED_SOURCES)
    return cmaup, manifest


def write_manifest(manifest, path):
    with open(path, "w") as fh:
        json.dump(manifest, fh, indent=2)
