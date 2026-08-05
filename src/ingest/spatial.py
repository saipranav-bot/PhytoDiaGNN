import json
import os
from pathlib import Path

import requests

SPATIAL_BASE = "https://cf.10xgenomics.com/samples/spatial-exp"

VISIUM_DEMOS = {
    "mouse_brain": {
        "name": "V1_Mouse_Brain_Sagittal_Anterior",
        "version": "1.1.0",
        "source": "10x Genomics public demo (guaranteed download)",
        "organism": "mus_musculus",
        "tissue": "brain",
    },
}

# Human-pancreas Visium options evaluated during the M3 data gate (none open/parseable):
#  - GSE264331 (CollinsLab, NGT vs T2D): count matrices gated in dbGaP (phs001188.v3.p1);
#    GEO hosts only GSE264331_RAW.tar (22.3 GB images).
#  - Zenodo 15177311: .cloupe binaries only (proprietary Loupe format, not parseable here).
#  - CZ CELLxGENE Census (2025-11-17): 569,586 human-pancreas cells, no spatial (Visium) spots.
#  - 10x cf.10xgenomics.com: no human-pancreas Visium demo (all candidate paths 403).
GATED_PANCREAS_VISIUM = {
    "GSE264331": {
        "status": "gated",
        "reason": "count matrices in dbGaP phs001188.v3.p1; GEO only hosts 22.3 GB image tar",
        "note": "Howell et al., Sci Data 2025 — 3 NGT + 3 T2D donors, 10x Visium v1",
    },
    "zenodo_15177311": {
        "status": "unavailable",
        "reason": ".cloupe binaries only; no standard Space Ranger outs",
    },
}


def download_visium(dest_dir, dataset="mouse_brain"):
    """Download a Visium demo (Space Ranger outs: filtered h5 + spatial dir)."""
    meta = VISIUM_DEMOS[dataset]
    dest_dir = Path(dest_dir)
    base = f"{SPATIAL_BASE}/{meta['version']}/{meta['name']}"
    h5 = dest_dir / "filtered_feature_bc_matrix.h5"
    spatial_tgz = dest_dir / "spatial.tar.gz"
    os.makedirs(dest_dir, exist_ok=True)
    if not h5.exists():
        r = requests.get(f"{base}/{meta['name']}_filtered_feature_bc_matrix.h5", stream=True, timeout=300)
        r.raise_for_status()
        with open(h5, "wb") as fh:
            for chunk in r.iter_content(1 << 20):
                fh.write(chunk)
    if not (dest_dir / "spatial").exists():
        if not spatial_tgz.exists():
            r = requests.get(f"{base}/{meta['name']}_spatial.tar.gz", stream=True, timeout=300)
            r.raise_for_status()
            with open(spatial_tgz, "wb") as fh:
                for chunk in r.iter_content(1 << 20):
                    fh.write(chunk)
        import tarfile
        with tarfile.open(spatial_tgz) as tf:
            tf.extractall(dest_dir)
    return dest_dir


def record(source, dest, extra=None):
    manifest = {
        "dataset": source,
        "source_url": f"{SPATIAL_BASE}/...",
        "files": [p.name for p in Path(dest).iterdir() if p.is_file()],
        **({} if extra is None else extra),
    }
    return manifest
