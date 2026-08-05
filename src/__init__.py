import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def env_paths():
    return {
        "repo_root": REPO_ROOT,
        "raw": REPO_ROOT / "data" / "raw",
        "processed": REPO_ROOT / "data" / "processed",
        "external": REPO_ROOT / "data" / "external",
        "results": REPO_ROOT / "results",
        "configs": REPO_ROOT / "configs",
    }


def ensure_dirs():
    for p in env_paths().values():
        os.makedirs(p, exist_ok=True)
    return env_paths()
