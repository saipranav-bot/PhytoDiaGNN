import numpy as np
from rdkit import DataStructs


def _fp_from_bits(bitstr):
    return DataStructs.CreateFromBitString(bitstr)


def select_diverse(df, smiles_col="smiles", fp_col="morgan1024", n=3000, seed=42):
    """Greedy max-min (Tanimoto) diversity selection with O(k*n) distance calls."""
    if len(df) <= n:
        return df
    fps = [_fp_from_bits(b) for b in df[fp_col]]

    def dist(i, j):
        return float(1.0 - DataStructs.TanimotoSimilarity(fps[i], fps[j]))

    rng = np.random.RandomState(seed)
    selected = [int(rng.randint(0, len(fps)))]
    d = [dist(selected[0], j) for j in range(len(fps))]
    d[selected[0]] = -1.0
    for _ in range(n - 1):
        nxt = int(np.argmax(d))
        selected.append(nxt)
        for j in range(len(fps)):
            if d[j] >= 0:
                dj = dist(nxt, j)
                if dj < d[j]:
                    d[j] = dj
    return df.iloc[selected].reset_index(drop=True)


def tanimoto_similarity(a_bits, b_bits):
    a = _fp_from_bits(a_bits)
    b = _fp_from_bits(b_bits)
    return DataStructs.TanimotoSimilarity(a, b)
