import sys
from pathlib import Path

import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.chem.filters import (
    apply_filters,
    load_whitelist,
    standardize,
    is_pains,
    add_fingerprints,
)
from src.chem.diversity import select_diverse, tanimoto_similarity
from src.chem.targets import activity_to_um, build_compound_target_edges


BASE_CFG = {
    "lipinski": {"mw_max": 500.0, "logp_max": 5.0, "hbd_max": 5, "hba_max": 10},
    "veber": {"rotatable_bonds_max": 10, "tpsa_max": 140.0},
    "filters": {"remove_pains": True},
}

QUERCETIN = "Oc1cc(O)c2c(c1)oc(c(c2=O)O)c1ccc(c(c1)O)O"
BENZOQUINONE = "O=C1C=CC(=O)C=C1"


def test_quercetin_whitelisted_passes():
    whitelist = load_whitelist([QUERCETIN])
    rows, res = apply_filters([QUERCETIN], BASE_CFG, whitelist=whitelist)
    assert res.n_final == 1
    assert rows[0]["MW"] < 500
    assert rows[0]["MW"] == pytest.approx(302.24, abs=0.1)
    assert rows[0]["LogP"] < 5


def test_quercetin_without_whitelist_is_pains():
    rows, res = apply_filters([QUERCETIN], BASE_CFG)
    assert res.drops.get("pains", 0) == 1
    assert res.n_final == 0


def test_standardize_strips_salt():
    std = standardize("CC(=O)[O-].[Na+]")
    assert std == "CC(=O)O"


def test_pains_removed():
    rows, res = apply_filters([BENZOQUINONE], BASE_CFG)
    assert res.drops.get("pains", 0) == 1
    assert res.n_final == 0


def test_unparseable_dropped():
    rows, res = apply_filters(["this-is-not-a-smiles!!", "n.a."], BASE_CFG)
    assert res.drops["unparseable"] == 2


def test_fingerprint_length_and_diversity_selection():
    smis = [QUERCETIN,
            "CC(=O)Oc1ccccc1C(=O)O",
            "O=C(O)c1ccccc1O",
            "CCO",
            "CC(=O)Oc1ccccc1O",
            "O=C(C)Oc1ccccc1"]
    rows, _ = apply_filters(smis, BASE_CFG)
    df = add_fingerprints(pd.DataFrame(rows))
    assert len(df["morgan1024"].iloc[0]) == 1024
    small = select_diverse(df, n=1, seed=7)
    assert len(small) == 1
    assert tanimoto_similarity(df["morgan1024"].iloc[0], df["morgan1024"].iloc[0]) == 1.0


def test_activity_unit_conversion():
    assert activity_to_um("100", "nM") == 0.1
    assert activity_to_um("5", "µM") == 5.0
    assert activity_to_um("2", "uM") == 2.0
    assert pd.isna(activity_to_um("abc", "nM"))


def test_edge_build_threshold():
    it = pd.DataFrame({
        "Ingredient_ID": ["C1", "C1", "C2"],
        "Target_ID": ["T1", "T2", "T1"],
        "Activity_Type": ["Ki", "Ki", "IC50"],
        "Activity_Value": ["100", "20000", "5"],
        "Activity_Unit": ["nM", "nM", "µM"],
        "Reference_ID": ["1", "2", "3"],
        "Reference_ID_Type": ["PMID"] * 3,
    })
    targets = pd.DataFrame({
        "Target_ID": ["T1", "T2"],
        "Gene_Symbol": ["GENE1", "GENE2"],
        "Uniprot_ID": ["P1", "P2"],
        "ChEMBL_ID": ["CB1", "CB2"],
    })
    edges = build_compound_target_edges(it, targets, threshold_um=10.0)
    assert set(edges["compound_id"]) == {"C1", "C2"}
    assert "T2" not in set(edges["target_id"])
    assert edges["activity_um"].between(0, 10).all()
