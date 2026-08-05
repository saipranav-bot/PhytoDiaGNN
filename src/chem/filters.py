from dataclasses import dataclass, field
import pandas as pd
from rdkit import Chem
from rdkit import RDLogger
from rdkit.Chem import Descriptors, Crippen, Lipinski, rdMolDescriptors
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams
from rdkit.Chem import rdFingerprintGenerator

RDLogger.DisableLog("rdApp.*")


PAINS_PARAMS = FilterCatalogParams()
PAINS_PARAMS.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS)
PAINS_CATALOG = FilterCatalog(PAINS_PARAMS)

LARGEST_FRAGMENT = rdMolStandardize.LargestFragmentChooser()
UNCHARGER = rdMolStandardize.Uncharger()


def standardize(smi):
    if not smi or smi == "n.a.":
        return None
    try:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return None
        mol = LARGEST_FRAGMENT.choose(mol)
        mol = UNCHARGER.uncharge(mol)
        return Chem.MolToSmiles(mol, isomericSmiles=True)
    except Exception:
        return None


def props_from_smiles(smi):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    return {
        "MW": Descriptors.MolWt(mol),
        "LogP": Crippen.MolLogP(mol),
        "HBD": Lipinski.NumHDonors(mol),
        "HBA": Lipinski.NumHAcceptors(mol),
        "RotB": Lipinski.NumRotatableBonds(mol),
        "TPSA": rdMolDescriptors.CalcTPSA(mol),
    }


def lipinski_ok(p, mw_max=500.0, logp_max=5.0, hbd_max=5, hba_max=10):
    return all([
        p["MW"] <= mw_max,
        p["LogP"] <= logp_max,
        p["HBD"] <= hbd_max,
        p["HBA"] <= hba_max,
    ])


def veber_ok(p, rotatable_bonds_max=10, tpsa_max=140.0):
    return p["RotB"] <= rotatable_bonds_max and p["TPSA"] <= tpsa_max


def is_pains(smi):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return False
    return PAINS_CATALOG.HasMatch(mol)


def load_whitelist(smiles_list):
    return {standardize(s) for s in smiles_list if standardize(s) is not None}


@dataclass
class FilterResult:
    n_input: int = 0
    n_parsed: int = 0
    n_standardized: int = 0
    n_lipinski: int = 0
    n_veber: int = 0
    n_pains_removed: int = 0
    n_final: int = 0
    drops: dict = field(default_factory=dict)


def apply_filters(smiles_list, cfg, whitelist=None, verbose=True):
    res = FilterResult(n_input=len(smiles_list))
    out = []
    for smi in smiles_list:
        std = standardize(smi)
        if std is None:
            res.drops["unparseable"] = res.drops.get("unparseable", 0) + 1
            continue
        res.n_parsed += 1
        p = props_from_smiles(std)
        if p is None:
            res.drops["props_failed"] = res.drops.get("props_failed", 0) + 1
            continue
        if not lipinski_ok(p, **cfg["lipinski"]):
            res.drops["lipinski"] = res.drops.get("lipinski", 0) + 1
            continue
        if not veber_ok(p, **cfg["veber"]):
            res.drops["veber"] = res.drops.get("veber", 0) + 1
            continue
        if cfg["filters"]["remove_pains"] and not (whitelist and std in whitelist) and is_pains(std):
            res.drops["pains"] = res.drops.get("pains", 0) + 1
            continue
        out.append({"smiles": std, **p})
    res.n_standardized = res.n_parsed
    res.n_lipinski = res.n_parsed - res.drops.get("lipinski", 0) - res.drops.get("veber", 0) - res.drops.get("pains", 0)
    res.n_veber = res.n_parsed - res.drops.get("veber", 0) - res.drops.get("pains", 0)
    res.n_pains_removed = res.drops.get("pains", 0)
    res.n_final = len(out)
    return out, res


FINGERPRINT_GEN = rdFingerprintGenerator.GetMorganGenerator(
    fpSize=1024, radius=2
)


def morgan_bytes(smi):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    return FINGERPRINT_GEN.GetFingerprint(mol).ToBitString()


def add_fingerprints(df, bit_col="morgan1024"):
    df[bit_col] = df["smiles"].apply(morgan_bytes)
    return df
