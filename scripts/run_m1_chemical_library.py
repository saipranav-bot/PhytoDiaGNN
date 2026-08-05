import json
import sys
from pathlib import Path

import pandas as pd
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src import ensure_dirs
from src.ingest.sources import fetch_all, write_manifest
from src.chem.filters import apply_filters, add_fingerprints, standardize
from src.chem.diversity import select_diverse
from src.chem.targets import (
    build_compound_target_edges,
    build_plant_compound_edges,
    fetch_string_ppi,
)


def main():
    paths = ensure_dirs()
    cfg = yaml.safe_load((REPO / "configs" / "chemical.yaml").read_text())

    print("[1/6] Fetching CMAUP v2.0 tables...")
    cmaup, manifest = fetch_all(str(paths["raw"]))
    write_manifest(manifest, str(paths["raw"] / "manifest.json"))

    ingredients = cmaup["ingredients"]
    targets_tbl = cmaup["targets"]
    it = cmaup["ingredient_target"]
    plants = cmaup["plants"]
    pi = cmaup["plant_ingredient"]
    print(f"      ingredients={len(ingredients):,} targets={len(targets_tbl):,} "
          f"ingredient-target pairs={len(it):,}")

    print("[2/6] Standardizing and filtering compounds...")
    chem_cfg = cfg["chemical"]
    std_map = ingredients.assign(std_smiles=ingredients["SMILES"].map(standardize))
    std_map = std_map.dropna(subset=["std_smiles"])
    bench_names = {n.lower() for n in chem_cfg["filters"].get("benchmark_names", [])}
    name_mask = std_map["pref_name"].str.strip().str.lower().isin(bench_names)
    bench_std = set(std_map.loc[name_mask, "std_smiles"])
    print(f"      benchmark actives identified: {len(bench_std)}")
    std_map = std_map.drop_duplicates(subset="std_smiles", keep="first")

    smi_list = ingredients["SMILES"].tolist()
    filtered, res = apply_filters(smi_list, chem_cfg, whitelist=bench_std)
    compounds = pd.DataFrame(filtered)

    print("[3/6] Attaching metadata + computing fingerprints...")
    compounds = compounds.merge(
        std_map[["std_smiles", "np_id", "pref_name", "chembl_id", "pubchem_cid"]],
        left_on="smiles",
        right_on="std_smiles",
        how="left",
    ).drop(columns=["std_smiles"])
    compounds["is_benchmark"] = compounds["smiles"].isin(bench_std)
    compounds = add_fingerprints(compounds)

    print(f"[4/6] Diversity cap -> {chem_cfg['max_compounds']} compounds...")
    keep_idx = compounds["smiles"].isin(bench_std)
    forced = compounds[keep_idx]
    pool = compounds[~keep_idx]
    n_fill = chem_cfg["max_compounds"] - len(forced)
    if n_fill > 0:
        filled = select_diverse(pool, n=n_fill, seed=cfg["seed"])
    else:
        filled = pool.head(0)
    compounds = pd.concat([forced, filled]).reset_index(drop=True)

    print("[5/6] Building target & plant edges...")
    tgt_cfg = cfg["targets"]
    edges = build_compound_target_edges(it, targets_tbl, threshold_um=tgt_cfg["chembl"]["activity_threshold_um"])
    kept = set(compounds["np_id"])
    edges = edges[edges["compound_id"].isin(kept)].reset_index(drop=True)
    plant_edges = build_plant_compound_edges(pi, plants)
    plant_edges = plant_edges[plant_edges["compound_id"].isin(kept)].reset_index(drop=True)

    string_ppi = fetch_string_ppi(edges["Gene_Symbol"].dropna().unique()[:2000],
                                  required_score=int(tgt_cfg["string"]["confidence_min"] * 1000))
    print(f"      compound-target edges={len(edges):,} plant-compound edges={len(plant_edges):,} "
          f"string ppi edges={len(string_ppi):,}")

    print("[6/6] Writing outputs...")
    out_dir = paths["processed"]
    compounds.to_parquet(out_dir / "compounds.parquet", index=False)
    edges.to_parquet(out_dir / "compound_target_edges.parquet", index=False)
    plant_edges.to_parquet(out_dir / "plant_compound_edges.parquet", index=False)
    targets_tbl.to_parquet(out_dir / "targets.parquet", index=False)
    if len(string_ppi):
        string_ppi.to_parquet(out_dir / "ppi_edges.parquet", index=False)

    results_dir = paths["results"] / "chemical_library"
    results_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "input_compounds": res.n_input,
        "parsed_smiles": res.n_parsed,
        "druglike_filtered": res.n_final,
        "final_capped": len(compounds),
        "drop_reasons": res.drops,
        "compound_target_edges": len(edges),
        "plant_compound_edges": len(plant_edges),
        "string_ppi_edges": len(string_ppi),
        "unique_targets": int(edges["target_id"].nunique()),
        "unique_genes": int(edges["Gene_Symbol"].nunique()),
    }
    (results_dir / "qc.json").write_text(json.dumps(report, indent=2))
    compounds.describe().to_csv(results_dir / "props_summary.csv")

    print("\n=== M1 SUMMARY ===")
    print(json.dumps(report, indent=2))
    print(f"\nOutputs -> {out_dir}\nReport  -> {results_dir}")


if __name__ == "__main__":
    main()
