import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from scripts.run_m5_scoring import compute_srev  # noqa: E402


def _fixture():
    compounds = pd.DataFrame({"np_id": ["C1", "C2", "C3"]})
    targets = pd.DataFrame({"Target_ID": ["T1", "T2", "T3"],
                            "Gene_Symbol": ["UP_G", "DOWN_G", "OTHER_G"]})
    ct_edges = pd.DataFrame({
        "compound_id": ["C1", "C1", "C2", "C3"],
        "target_id": ["T1", "T2", "T1", "T3"],
    })
    signatures = pd.DataFrame({
        "cell_type": ["beta", "beta"],
        "gene": ["UP_G", "DOWN_G"],
        "direction": [1, -1],
    })
    return compounds, targets, ct_edges, signatures


def test_compute_srev_signs_and_hits():
    compounds, targets, ct_edges, signatures = _fixture()
    out = compute_srev(compounds, targets, ct_edges, signatures).set_index("compound_id")
    c1 = out.loc["C1"]
    # C1 hits UP_G (+1 credit) and DOWN_G (-1) under inhibition convention -> 0, 2 hits
    assert c1["s_rev_inh"] == 0
    assert c1["s_rev_act"] == 0
    assert c1["n_hits"] == 2
    assert c1["n_up_hits"] == 1 and c1["n_down_hits"] == 1
    assert c1["no_signal"] == 0
    # C2 hits only UP_G -> +1, purity 1.0
    c2 = out.loc["C2"]
    assert c2["s_rev_inh"] == 1 and c2["s_rev_act"] == -1
    assert c2["n_hits"] == 1 and c2["purity"] == 1.0
    # C3 hits a gene not in the signature -> no signal
    c3 = out.loc["C3"]
    assert c3["n_hits"] == 0 and c3["no_signal"] == 1
    assert c3["coverage"] == 0.0
