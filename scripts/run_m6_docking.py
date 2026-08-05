import json
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

VINA = "vina"
OBABEL = "obabel"
DOCK = ROOT / "data" / "docking"
EXHAUST = 8
BOX = 22.0
NWORKERS = 4


def centroid(pdb, residue):
    atoms = []
    for line in pdb.splitlines():
        if line.startswith("HETATM") and line[17:20].strip() == residue:
            atoms.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
    if not atoms:
        raise ValueError(f"ligand residue {residue} not found")
    return np.mean(atoms, axis=0)


def prep_receptor(pdb_path, name):
    lines = [l for l in pdb_path.read_text().splitlines() if l.startswith("ATOM")]
    clean = Path(DOCK) / f"{name}_protein.pdb"
    clean.write_text("\n".join(lines) + "\n")
    out = Path(DOCK) / f"{name}_rec.pdbqt"
    subprocess.run([OBABEL, str(clean), "-O", str(out), "-xr", "-p", "7.4"],
                   check=True, capture_output=True)
    return out


def prep_ligand(smiles, name, tmpdir):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    mol = Chem.AddHs(mol)
    ok = False
    for seed in (42, 7, 1337):
        rc = AllChem.EmbedMolecule(mol, randomSeed=seed)
        if rc >= 0:
            ok = True
            break
    if not ok:
        rc = AllChem.EmbedMolecule(mol, randomSeed=42, useRandomCoords=True)
        ok = rc >= 0
    if not ok:
        return None
    try:
        AllChem.MMFFOptimizeMolecule(mol)
    except Exception:
        pass
    sdf = tmpdir / f"{name}.sdf"
    Chem.MolToMolFile(mol, str(sdf))
    out = tmpdir / f"{name}.pdbqt"
    r = subprocess.run([OBABEL, str(sdf), "-O", str(out), "-p", "7.4"],
                       capture_output=True, text=True)
    if r.returncode != 0 or not out.exists() or out.stat().st_size == 0:
        return None
    return out


def dock(rec, lig, center):
    out = lig.with_suffix(".out.pdbqt")
    cmd = [VINA, "--receptor", str(rec), "--ligand", str(lig),
           "--center_x", f"{center[0]:.3f}", "--center_y", f"{center[1]:.3f}",
           "--center_z", f"{center[2]:.3f}",
           "--size_x", str(BOX), "--size_y", str(BOX), "--size_z", str(BOX),
           "--exhaustiveness", str(EXHAUST), "--num_modes", "1", "--cpu", "1",
           "--out", str(out)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        return None, r.stderr[-300:]
    score = None
    for line in out.read_text().splitlines():
        if "VINA RESULT" in line:
            try:
                score = float(line.split()[-3])
            except (IndexError, ValueError):
                pass
    return score, None


def main():
    n_cpu = 4
    actives = pd.read_parquet(ROOT / "data" / "processed" / "compounds.parquet")
    actives = actives[actives["is_benchmark"]].drop_duplicates("np_id").reset_index(drop=True)
    library = pd.read_parquet(ROOT / "data" / "processed" / "compounds.parquet")
    library = library[~library["is_benchmark"]].reset_index(drop=True)

    actives["mw"] = actives["smiles"].map(lambda s: Descriptors.MolWt(Chem.MolFromSmiles(s)))
    actives["logp"] = actives["smiles"].map(lambda s: Descriptors.MolLogP(Chem.MolFromSmiles(s)))
    library["mw"] = library["smiles"].map(lambda s: Descriptors.MolWt(Chem.MolFromSmiles(s)))
    library["logp"] = library["smiles"].map(lambda s: Descriptors.MolLogP(Chem.MolFromSmiles(s)))

    rng = np.random.RandomState(7)
    decoys = []
    for _, a in actives.iterrows():
        cand = library[(library["mw"].between(a["mw"] - 25, a["mw"] + 25)) &
                       (library["logp"].between(a["logp"] - 1, a["logp"] + 1))]
        if len(cand) == 0:
            cand = library
        pick = cand.sample(min(5, len(cand)), random_state=rng)
        decoys.append(pick)
    decoys = pd.concat(decoys).drop_duplicates("np_id").head(82).reset_index(drop=True)
    print(f"[M6] actives={len(actives)} property-matched decoys={len(decoys)}")

    receptors = {
        "PPARG": ("2prg.pdb", "BRL"),
        "GSK3B": ("1j1b.pdb", "ANP"),
    }
    preps = {}
    for name, (pdb, lig_res) in receptors.items():
        pdb_path = DOCK / pdb
        center = centroid(pdb_path.read_text(), lig_res)
        rec = prep_receptor(pdb_path, name)
        preps[name] = (rec, center)
        print(f"[M6] receptor {name}: box center {center.round(2)}")

    rows = []
    for rec_name, (rec, center) in preps.items():
        for label, df in [("active", actives), ("decoys", decoys)]:
            for _, r in df.iterrows():
                rows.append({"kind": label, "rec": rec_name,
                             "compound_id": r["np_id"], "smiles": r["smiles"]})
    candidates = pd.read_csv(ROOT / "results" / "m5_scoring" / "s_rev_scores.csv")
    candidates["best_srev"] = candidates.groupby("compound_id")["s_rev"].transform("max")
    top = candidates.drop_duplicates("compound_id").sort_values("best_srev", ascending=False)
    top = top[~top["compound_id"].isin(actives["np_id"])].head(20)
    for _, r in top.iterrows():
        for rec_name in receptors:
            m = library["np_id"] == r["compound_id"]
            smi = library.loc[m, "smiles"].iloc[0] if m.any() else ""
            rows.append({"kind": "candidate", "rec": rec_name,
                         "compound_id": r["compound_id"], "smiles": smi})

    with tempfile.TemporaryDirectory() as td:
        tmpdir = Path(td)
        results = []
        fails = []

        def one(row):
            rec, center = preps[row["rec"]]
            lig = prep_ligand(row["smiles"], f"l{row['idx']}", tmpdir)
            if lig is None:
                return row, None, "ligand prep failed"
            score, err = dock(rec, lig, center)
            return row, score, err

        jobs = [{**row, "idx": i} for i, row in enumerate(rows)]
        csv_path = ROOT / "results" / "m6_docking" / "docking_results.csv"
        checkpoint = pd.DataFrame()
        if csv_path.exists() and csv_path.stat().st_size > 0:
            checkpoint = pd.read_csv(csv_path)
            done_keys = set(zip(checkpoint["compound_id"], checkpoint["rec"], checkpoint["kind"]))
            jobs = [j for j in jobs if (j["compound_id"], j["rec"], j["kind"]) not in done_keys]
            print(f"[M6] resuming: {len(done_keys)} already docked, {len(jobs)} remaining")
        with ThreadPoolExecutor(max_workers=NWORKERS) as ex:
            for i, (row, score, err) in enumerate(ex.map(one, jobs)):
                if score is None:
                    fails.append((row["compound_id"], row["rec"], err))
                    continue
                results.append({k: row[k] for k in ("kind", "rec", "compound_id", "smiles")}
                               | {"affinity": score})
                if (i + 1) % 25 == 0:
                    print(f"[M6] {i+1}/{len(jobs)} docked")
                    pd.concat([checkpoint, pd.DataFrame(results)]).to_csv(csv_path, index=False)
        all_rows = pd.concat([checkpoint, pd.DataFrame(results)], ignore_index=True)
        all_rows.to_csv(csv_path, index=False)
        results = all_rows.to_dict("records")
        print(f"[M6] docked={len(results)} failed={len(fails)}")

    df = pd.DataFrame(results)
    df.to_csv(ROOT / "results" / "m6_docking" / "docking_results.csv", index=False)

    report = {}
    for rec_name in receptors:
        sub = df[df["rec"] == rec_name]
        act = sub[(sub["kind"] == "active")].sort_values("affinity")
        dec = sub[sub["kind"] == "decoys"]
        ranked = sub[sub["kind"].isin(["active", "decoys"])].sort_values("affinity")
        n_act = len(act)
        n_dec = len(dec)
        if n_act and n_dec:
            from sklearn.metrics import roc_auc_score
            ranked = sub[sub["kind"].isin(["active", "decoys"])].sort_values("affinity")
            n_top1 = max(1, int(0.01 * len(ranked)))
            n_top5 = max(1, int(0.05 * len(ranked)))
            ef1 = act[act["affinity"] <= ranked["affinity"].nsmallest(n_top1).iloc[-1]].shape[0] / n_top1
            ef5 = act[act["affinity"] <= ranked["affinity"].nsmallest(n_top5).iloc[-1]].shape[0] / n_top5
            auc = roc_auc_score(ranked["kind"] == "active", -ranked["affinity"])
            report[rec_name] = {"n_actives": n_act, "n_decoys": n_dec,
                                "median_active_affinity": float(act["affinity"].median()),
                                "median_decoy_affinity": float(dec["affinity"].median()),
                                "EF1%": float(ef1), "EF5%": float(ef5), "AUROC": float(auc)}
        else:
            report[rec_name] = {"n_actives": n_act, "n_decoys": n_dec}

    cand_out = df[df["kind"] == "candidate"].sort_values("affinity")
    report["top_candidates"] = cand_out[["compound_id", "rec", "affinity"]].to_dict("records")
    Path(ROOT / "results" / "m6_docking").mkdir(parents=True, exist_ok=True)
    with open(ROOT / "results" / "m6_docking" / "summary.json", "w") as f:
        json.dump(report, f, indent=2)
    for k, v in report.items():
        if isinstance(v, dict):
            print(f"[M6] {k}: {v}")
    print("[M6] done ->", ROOT / "results" / "m6_docking" / "summary.json")


if __name__ == "__main__":
    main()
