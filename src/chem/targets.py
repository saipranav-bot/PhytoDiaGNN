import numpy as np
import pandas as pd
import requests

UNIT_TO_UM = {"nM": 1e-3, "uM": 1.0, "um": 1.0, "µM": 1.0, "mM": 1e3}


def activity_to_um(value, unit):
    try:
        v = float(value)
    except (TypeError, ValueError):
        return np.nan
    u = str(unit).strip().lower()
    if u in ("nm",):
        return v * 1e-3
    if u in ("um", "µm", "μm", "uM"):
        return v
    if u in ("mm",):
        return v * 1e3
    return np.nan


def build_compound_target_edges(ingredient_target, targets, threshold_um=10.0):
    it = ingredient_target.copy()
    it["activity_um"] = it.apply(
        lambda r: activity_to_um(r["Activity_Value"], r["Activity_Unit"]), axis=1
    )
    it = it[it["activity_um"].notna() & (it["activity_um"] <= threshold_um)]
    it = it.merge(
        targets[["Target_ID", "Gene_Symbol", "Uniprot_ID", "ChEMBL_ID"]].drop_duplicates(),
        on="Target_ID",
        how="left",
    )
    it = it.rename(columns={"Ingredient_ID": "compound_id", "Target_ID": "target_id"})
    return it[["compound_id", "target_id", "Gene_Symbol", "Uniprot_ID",
               "Activity_Type", "Activity_Value", "Activity_Unit", "activity_um",
               "Reference_ID", "Reference_ID_Type"]].drop_duplicates()


def build_plant_compound_edges(plant_ingredient, plants):
    pi = plant_ingredient.rename(
        columns={"Plant_ID": "plant_id", "Ingredient_ID": "compound_id"}
    )
    pi = pi.merge(
        plants[["Plant_ID", "Plant_Name"]].drop_duplicates(),
        left_on="plant_id",
        right_on="Plant_ID",
        how="left",
    ).drop(columns=["Plant_ID"])
    return pi.rename(columns={"Plant_Name": "plant_name"})[["plant_id", "plant_name", "compound_id"]]


def fetch_string_ppi(gene_symbols, species=9606, required_score=700, timeout=120):
    ids = [g for g in gene_symbols if pd.notna(g)]
    if len(ids) > 4000:
        ids = ids[:4000]
    if not ids:
        return pd.DataFrame(columns=["gene_a", "gene_b", "score"])
    url = "https://string-db.org/api/tsv/network"
    payload = {"identifiers": "%0d".join(ids), "species": species, "required_score": required_score}
    try:
        r = requests.post(url, data=payload, timeout=timeout)
        r.raise_for_status()
        header = r.text.split("\n")[0]
        if "stringId_A" not in header:
            return pd.DataFrame(columns=["gene_a", "gene_b", "string_score"])
        df = pd.read_csv(pd.io.common.StringIO(r.text), sep="\t")
        return df.rename(columns={
            "preferredName_A": "gene_a",
            "preferredName_B": "gene_b",
            "score": "string_score",
        })[["gene_a", "gene_b", "string_score"]]
    except Exception:
        return pd.DataFrame(columns=["gene_a", "gene_b", "string_score"])
