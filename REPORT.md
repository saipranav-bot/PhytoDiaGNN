# PhytoDiaGNN — Final Milestone Report

Graph-neural-network-driven multi-omics framework for cell-type- and spatial-niche-specific
anti-diabetic phytochemical discovery. Pancreatic-islet-first MVP, CPU-only, 4 GB RAM / 4 CPU.

All runs reproduce via `scripts/run_m{1,2,3,4,5,6}_*.py` under the `phyto` conda env.
Machine-readable per-stage results live in `results/<stage>/summary.json`.

---

## M1 — Chemical library & KG edges (CMAUP)

| Metric | Value |
|---|---|
| Input compounds | 60,222 (CMAUP) |
| Parsed SMILES | 60,190 |
| Drug-like after filters (Lipinski / Veber / PAINS) | 29,019 |
| Final diverse set (max-min, forced benchmark actives) | 3,000 |
| Compound→target edges | 4,730 |
| Plant→compound edges | 15,688 |
| STRING PPI edges | 2,409 (578 unique targets, 520 mapped to genes) |

Filter drop reasons: Lipinski 27,305 · Veber 1,667 · PAINS 2,199 · unparseable 32.

## M2 — Islet scRNA integration (GSE86469 + GSE153855)

| Metric | Value |
|---|---|
| Cells after QC (GSE86469 / GSE153855) | 497 / 3,383 |
| Merged cells | 3,880 |
| Cell types (9) | beta 815 · gamma 729 · acinar 499 · alpha 474 · ductal 455 · stellate 393 · delta 371 · epsilon 90 · endothelial 54 |
| DEG rows (per cell type × condition) | 900 |
| **Replicated T2D signature genes** (consistent direction, both studies) | **1,431** (470 up + 961 down) |

Outputs: `data/processed/scrna/{merged_islets.h5ad, deg_per_celltype.csv, disease_signatures.csv}`.
Per-study DEG (`deg_per_study.csv`) required agreement across two independent islet cohorts
(GSE86469 Fluidigm C1, GSE153855 SMART-seq) → robust signature, not study-specific noise.
Both the T2D-elevated tail (up in T2D) and the T2D-suppressed tail (up in ND, direction −1) are
replicated separately, yielding a **direction-balanced signature** that signed reversal scoring
can reward/penalize.

## M3 — Spatial niche deconvolution (Visium)

**Data gate outcome (documented):** no open, parseable human-pancreas Visium exists.

- GEO **GSE264331** (T2D vs NGT, Howell et al. 2025): count matrices are gated in dbGaP
  (phs001188.v3.p1); GEO hosts only the 22.3 GB image tar.
- Zenodo **15177311**: Loupe `.cloupe` binaries only (proprietary, 29.8 GB, unparseable).
- CZ CELLxGENE Census (release 2025-11-17): 569,586 human-pancreas cells but **scRNA only, no spatial**.
- 10x public catalog: no human-pancreas Visium demo (all candidate paths 403).

**Approved fallback (methods demo):** 10x `V1_Mouse_Brain_Sagittal_Anterior` (2,695 spots).

| Metric | Value |
|---|---|
| Spots | 2,695 |
| Deconvolution signature | 7 brain cell types × 24 canonical markers (NNLS, column-L1-normalized) |
| Dominant cell type per spot | oligo 1,988 · neuron_inh 428 · neuron_exc 233 · astro 44 · microglia 2 |
| Niches (KMeans on proportions, k=6) | sizes 457 / 545 / 545 / 542 / 203 / 403 |
| Islet signature matrix (deliverable for future pancreas data) | 9 cell types × 231 genes |

Outputs: `results/spatial/{deconv.csv, niches.csv, summary.json}`,
`data/processed/scrna/signature_matrix.csv`. When dbGaP access clears, the same script
swaps the Visium path and runs against the islet signature matrix directly.

## M4 — Knowledge graph + HeteroGNN link prediction

Heterogeneous graph: compound (3,000) × protein (758) × cell_type (9) × disease (1), with
compound→protein, protein→PPI, compound→cell_type (therapeutic, expression-grounded,
228 positives, `EXPR_THRESH=0.5`), protein→cell_type (expression), cell_type→disease, and
reverse edges for balanced message passing. HeteroGraphSAGE (2 layers) + MLP link predictor;
5-fold edge-masked CV with fixed negatives.

| Model | AUROC | AUPRC |
|---|---|---|
| HeteroGNN | **0.9957** | 0.9978 |
| RandomForest baseline | 0.9913 | — |
| Random | 0.5 | — |

Benchmark recovery of 18 known anti-diabetic actives: top-1% **34%**, top-10% **79%**.

## M5 — Reversal scoring + GNN attribution

- Signed, convention-explicit reversal scores for 27,000 compound×cell-type pairs
  (`results/m5_scoring/s_rev_scores.csv`). **CMap convention:** `s_rev_inh` credits +1 per
  disease-UP gene hit (compound target assumed suppressed → reverses an over-elevated gene) and
  −1 per disease-DOWN hit; `s_rev_act` reports the opposite (activation) assumption. Columns
  `n_up_hits` / `n_down_hits` / `coverage` (fraction of the cell-type signature engaged) /
  `purity` accompany every score, and `no_signal` marks rows with zero signature-gene hits.
- **Coverage (honest limit):** only **113 of 27,000** pairs engage ≥1 replicated signature gene
  — compound targets rarely overlap the disease signature, so most compounds are neutral by
  construction. The score ranks the engaging set meaningfully (e.g., NPC39426/stellate +1 via
  NPC1), and both direction conventions are reported so no hidden assumption drives the ranking.
- GNN edge-removal attribution (1,005 rows) in logit space pinpoints driving targets, e.g.
  PLK1 dominates NPC166788 (logit drop 40.1), TTR drives NPC279121 (drop 7.0).

> **Note (previously overstated):** earlier runs scored only against an up-dominated signature
> with an inverted sign, so "rank-1" tables were 0.0 ties with no hits. That artifact is fixed;
> the S_rev is now a defensible signed reversal filter, and the attribution is the stronger
> mechanistic evidence.

## M6 — Molecular docking (AutoDock Vina 1.1.2)

Receptors: PPARγ (2PRG, ligand BRL) and GSK3β (1J1B, ligand ANP); 22 Å boxes centered on
co-crystal ligand centroids. 198/206 complexes docked (8 failed on ligand prep).
Benchmark: 14 unique actives (deduped) vs 68 property-matched decoys per receptor.

| Receptor | Median active | Median decoy | EF5% | AUROC |
|---|---|---|---|---|
| PPARG | −6.85 | −6.4 | 0.25 | **0.667** |
| GSK3B | −7.3 | −6.4 | 0.0 | **0.804** |

Top actives: NPC279121 (−7.9 GSK3B), NPC254847 (−7.9 PPARG).
Top novel candidates (not in benchmark): NPC190454 (−7.3 GSK3B), NPC461221 (−6.6 GSK3B).

> Note: Vina 1.1.2 writes the score in the output PDBQT `REMARK VINA RESULT:` line, not the
> tabular summary; the parser reads `line.split()[-3]`. The checkpoint CSV enables resume.

---

## Cross-milestone integration

1. **M2 → M3:** replicated disease signature (1,431 genes) + islet signature matrix (9×231)
   are the reference for spatial deconvolution.
2. **M2 → M4/M5:** expression grounding defines the therapeutic supervision edges and the
   S_rev reversal target list.
3. **M1 → M4/M6:** the 3,000-compound library seeds both the KG and the docking set.

**Known limitations / honest caveats**

- GNN AUROC ~0.996 is partly a data-richness artifact of a small, well-separated edge set;
  RandomForest at 0.991 shows the biology is separable even without graph structure.
- PPARG docking enrichment is weak (AUROC 0.667); GSK3B is stronger (0.804). Vina scores
  require prospective validation and are not binding-free-energy-calibrated.
- M3 uses the approved mouse-brain demo (no open pancreas Visium); the islet signature
  matrix is ready to drop into the same pipeline when gated data clears.
- M4b (niche nodes in the KG) was deferred: brain niches from the fallback tissue cannot be
  grafted onto the islet/compound KG without corrupting its biology.

## Reproduce

```bash
conda activate phyto
python scripts/run_m1_chemical_library.py
python scripts/run_m2_scrna.py
python scripts/run_m3_spatial.py
python scripts/run_m4_gnn.py
python scripts/run_m5_scoring.py
python scripts/run_m6_docking.py
pytest -q
```
