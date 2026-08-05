# PhytoDiaGNN — Comprehensive Project Report

## A Graph Neural Network–Driven Multi-Omics Framework for Cell-Type-Specific Discovery of Anti-Diabetic Phytochemicals

**Status:** M1–M6 complete · all scripts reproducible · 9/9 tests passing
**Compute profile:** CPU-only, 4 GB RAM, 4 cores (all caps chosen for this budget)

---

## 1. Executive Summary

Type 2 diabetes mellitus (T2DM) is a *multicellular systems disease*, not a single-gene disease:
thousands of genes become dysregulated across many cell types, and the pancreatic beta cell is the
central casualty. Most computational drug-discovery pipelines reduce this complexity to a single
question — *"which compound binds this protein?"* — and thereby ignore the cell type, the disease
biology, and the transcriptome entirely.

PhytoDiaGNN reframes the question as: **"which phytochemical reverses the molecular disease state of a
specific diabetic cell type?"** The answer is assembled from four complementary lines of evidence:

1. **Molecular truth** — a direction-balanced, cross-study replicated disease signature for each of
   9 pancreatic cell types, derived from 3,880 single-cell islets (M2).
2. **Spatial context** — where those cell states live in tissue, via NNLS deconvolution of Visium
   spots into niches (M3; pipeline demonstrated on the approved 10x mouse-brain fallback because no
   open human-pancreas Visium exists).
3. **Pharmacological plausibility** — a heterogeneous knowledge graph (compounds–proteins–cell
   types–disease) scored by a HeteroGraphSAGE link predictor, with an expression-grounding rule that
   forbids biologically impossible predictions (M4/M5).
4. **Physical plausibility** — AutoDock Vina validation against the diabetes-relevant receptors
   PPARγ and GSK3β (M6).

End-to-end, the framework recovers known anti-diabetic actives (GNN top-10% recovery 79%;
GSK3β docking AUROC 0.80) and produces a shortlist of novel candidates, the strongest being
**NPC190454 (−7.3 kcal/mol GSK3β)** and **NPC461221 (−6.6 kcal/mol GSK3β)**, with
mechanism-level explanations (e.g., PLK1 for NPC166788, TTR for NPC279121).

---

## 2. Biological Foundations

### 2.1 Diabetes is a multicellular systems disease

T2DM arises from simultaneous dysregulation across tissues:

| Cell type | Abnormality in T2DM |
|---|---|
| Pancreatic β-cells | Fail to produce/secrete enough insulin; undergo dysfunction and loss |
| Pancreatic α-cells | Secrete excess glucagon (paradoxical hyperglucagonemia) |
| Immune cells | Chronic, low-grade inflammation (macrophage infiltration of islets) |
| Endothelial cells | Vascular dysfunction, impaired perfusion |
| Adipocytes | Insulin resistance, altered adipokine secretion |
| Kidney podocytes | Damage leading to albuminuria |

Each cell type develops its **own molecular abnormalities**, so a drug that helps β-cells may be
useless or harmful elsewhere. Any discovery pipeline that ignores cell identity is answering the
wrong question.

### 2.2 The islets of Langerhans

The pancreas is a mixed organ: acinar/ductal cells make digestive enzymes, while the **islets of
Langerhans** — tiny endocrine clusters (~1–2% of the organ) — control glucose homeostasis. The
integrated islet atlas in this project covers 9 cell types:

| Cell type | Key marker(s) | Hormone / role | Role in T2D |
|---|---|---|---|
| β-cell | INS, MAFA, PDX1 | Insulin — lowers blood glucose | **The key failing cell** |
| α-cell | GCG | Glucagon — raises glucose | Hyperglucagonemia |
| δ-cell | SST | Somatostatin — paracrine brake | Dysregulated inhibition |
| ε-cell | GHRL | Ghrelin | Minor |
| γ-cell (PP) | PPY | Pancreatic polypeptide | Minor |
| Acinar | CELA3B, CPA2, CTRB1 | Digestive enzymes | Exocrine–endocrine interplay |
| Ductal | KRT19, CA2 | Duct lining / plasticity | Potential β-cell neogenesis source |
| Stellate | — | Fibrosis / myofibroblast-like | Islet scarring in T2D |
| Endothelial | PECAM1, CLDN5 | Vasculature | Impaired islet perfusion |

The atlas composition (β 815 · γ 729 · acinar 499 · α 474 · ductal 455 · stellate 393 · δ 371 ·
ε 90 · endothelial 54) defines the granularity at which every downstream prediction is made.

### 2.3 The disease signature concept

Comparing diabetic vs healthy cells per cell type yields **differentially expressed genes (DEGs)**.
Genes whose expression reproducibly shifts with disease constitute the *disease signature* — the
molecular fingerprint of diabetes in that cell type. A candidate drug is then judged on whether its
targets move that signature **back toward healthy**.

---

## 3. Central Hypothesis & Strategy

> **Instead of asking "which compound binds this protein?", ask "which phytochemical can reverse the
> disease state of a specific diabetic cell type?"**

This hypothesis changes the entire pipeline:

```text
Healthy cell → disease develops → gene expression changes → disease signature forms
→ identify responsible proteins → find phytochemicals targeting them
→ predict reversal per cell type → verify physical binding → rank candidates
```

Four design rules follow:

1. **Cell specificity.** Every score and prediction is made *per cell type*, never organ-averaged.
2. **Expression grounding.** A compound may only influence a cell type through proteins that the
   cell type actually transcribes.
3. **Evidence triangulation.** Statistical, mechanistic, and structural evidence must agree.
4. **Honesty by construction.** Assumptions are made explicit, coverage limits are reported, and
   results are reproduced on independent cohorts wherever possible.

---

## 4. Data Sources & Selection Rationale

### 4.1 Compound and target sources

| Source | Used? | Status / rationale |
|---|---|---|
| **CMAUP v2.0** (Chinese Medicinal Plant database, BIDD) | **Yes** | Sole compound source: 60,222 plant–compound–target records, open download, plant-centric. |
| **STRING** | **Yes** | Protein–protein interaction edges (2,409 at confidence ≥ 0.7) for the knowledge graph. |
| IMPPAT | No | **Deferred** — JS-gated download (documented in `src/ingest/sources.py:47`); CMAUP covers the MVP. |
| TCMSP | No | **Deferred** — login-gated download. |
| PubChem / ChEBI | No | Evaluated but not needed; identifiers resolved inside CMAUP. |
| ChEMBL | Config only | Activity threshold (≤10 µM, confidence ≥ 8) used to define compound–target edges from CMAUP associations. |

> **Note:** the pipeline in this repository used **CMAUP v2.0 for compounds and targets** and
> **STRING for PPI**. The M1 config lists IMPPAT/TCMSP/ChEMBL as sources; IMPPAT and TCMSP are
> deferred with documented reasons and were not run.

### 4.2 Single-cell RNA-seq datasets

| Dataset | Platform | Content | Selected? |
|---|---|---|---|
| **GSE86469** (Lawlor/Stitzel) | Fluidigm C1 | Human islets, T2D vs nondiabetic, 497 cells after QC | **Yes** |
| **GSE153855** (Wierup) | SMART-seq | 6 control + 5 T2D donors, 3,383 cells after QC | **Yes** |
| GSE81608 (Segerstolpe) | Smart-seq2 | Islets, **no T2D condition** | No |
| GSE85241 (Muraro) | CEL-Seq2 | Islets, **no T2D condition** | No |
| GSE124742 (Camunas-Soler) | 10x | Islets, very large | No (size) |
| GSE221156 (48-donor atlas) | 10x | 245k cells | No (4 GB RAM cap) |

**Why these two?**
- Both contain **human islets with T2D and control donors** — required to build a disease signature.
- Both fit the **4 GB RAM / CPU-only** compute budget after capping.
- They use **different technologies** (C1 vs SMART-seq), making them an **orthogonal replication
  test**: a gene is admitted to the disease signature only if its direction changes **consistently in
  both studies**. This removes platform-specific noise.

### 4.3 Spatial transcriptomics (Visium) — the data gate

**No open, parseable human-pancreas Visium dataset exists.** This was verified and documented:

| Candidate | Status | Reason |
|---|---|---|
| GEO **GSE264331** (Howell et al., 2025; NGT vs T2D, 6 donors) | **Gated** | Count matrices are in dbGaP (phs001188.v3.p1); GEO hosts only a 22.3 GB image tar. |
| Zenodo **15177311** | **Unavailable** | Only Loupe `.cloupe` binaries (29.8 GB, proprietary, unparseable). |
| CZ **CELLxGENE Census** (2025-11-17) | **scRNA only** | 569,586 human-pancreas cells, but no spatial (Visium) spots. |
| 10x public catalog | **No demo** | All candidate human-pancreas Visium paths return 403. |

**Approved fallback (methods demo):** 10x `V1_Mouse_Brain_Sagittal_Anterior` — guaranteed
downloadable, standard Space Ranger outs, **2,695 spots**. The full methods pipeline was proven on
this tissue, and the *islet* signature matrix (9×231) was built as the deliverable for real pancreas
data. The same script (`run_m3_spatial.py --visium <dir>`) will deconvolve human pancreas Visium
with **zero code changes** once dbGaP access clears.

### 4.4 Docking structures

| Receptor | PDB | Co-crystal ligand | Relevance |
|---|---|---|---|
| **PPARγ** | 2PRG | BRL (farglitazar) | Target of thiazolidinedione insulin sensitizers |
| **GSK3β** | 1J1B | ANP | Kinase that inhibits glycogen synthesis; impairs β-cell survival |

Both are central to diabetes pharmacology and both provide co-crystal ligands to define a **22 Å
docking box** centered on the true binding site (PPARγ center [54.74, −20.97, 30.48]; GSK3β center
[28.69, 3.96, −21.81]).

---

## 5. Stage-by-Stage Methodology, Tools, and Results

### 5.0 Tools inventory

| Tool | Version | Role |
|---|---|---|
| Python + conda env `phyto` | 3.11 | Runtime |
| RDKit | — | SMILES parsing, Morgan fingerprints, Lipinski/Veber/PAINS filters, ligand 3D embedding |
| Scanpy | — | scRNA normalization, HVG, PCA, UMAP, Leiden, Wilcoxon `rank_genes_groups` |
| harmonypy | — | Batch-effect correction across the two islet datasets |
| scikit-learn | — | PCA on fingerprints, KMeans niches, RandomForest baseline, max-min diversity |
| PyTorch / PyTorch Geometric | — | HeteroGraphSAGE + MLP link predictor |
| scipy | — | Non-negative least squares (NNLS) deconvolution |
| OpenBabel | 3.1.0 | PDB → PDBQT receptor prep, ligand format conversion |
| AutoDock Vina | 1.1.2 | Molecular docking + affinity scoring |
| pandas / numpy / openpyxl / requests / pytest | — | Wrangling, Excel output, downloads, tests |

---

### 5.1 Stage 1 (M1) — Chemical library

**Purpose:** build a high-quality, diverse, drug-like set of phytochemicals to screen — small enough
for 4 GB RAM, broad enough to explore chemical space.

**Method.** Every CMAUP molecule is stored as **SMILES** (the "programming language of chemistry";
e.g., water = `O`, ethanol = `CCO`, glucose = `C(C1C(C(C(C(O1)O)O)O)O)O`). RDKit parses SMILES into
a chemical graph and generates Morgan fingerprints (radius 2, 1024 bits) for similarity. Drug-likeness
filters are applied:

- **Lipinski Rule of Five** — oral bioavailability: MW ≤ 500, logP ≤ 5, HBD ≤ 5, HBA ≤ 10.
- **Veber** — flexibility: rotatable bonds ≤ 10, TPSA ≤ 140.
- **PAINS** — removes pan-assay interference compounds (molecules that "stick to everything" and
  create false positives).

**Results.**

| Metric | Value |
|---|---|
| Input compounds (CMAUP) | 60,222 |
| Parsed SMILES | 60,190 |
| Drug-like after filters | 29,019 |
| Filter drops | Lipinski 27,305 · Veber 1,667 · PAINS 2,199 · unparseable 32 |
| **Final diverse library (max-min)** | **3,000** (+ 18 benchmark anti-diabetic actives forced in) |
| Compound→target edges | 4,730 |
| Plant→compound edges | 15,688 |
| STRING PPI edges | 2,409 (578 unique targets, 520 genes) |

Max-min diversity selection keeps chemically *different* scaffolds so the AI is not biased toward one
chemical family. The 18 benchmark actives (quercetin, curcumin, resveratrol, berberine, emodin,
genistein, kaempferol, baicalein, apigenin, luteolin, naringenin, catechin, gallic acid, ellagic acid,
betulinic acid, ursolic acid, oleanolic acid, thymoquinone) bypass PAINS but still must pass
Lipinski/Veber — an intentional, documented exception to guarantee positive controls.

---

### 5.2 Stage 2 (M2) — Single-cell RNA-seq: the disease signature

**Purpose:** define, per cell type, the *replicated* molecular fingerprint of T2DM — the ground truth
everything downstream is judged against.

**Why single-cell?** Bulk RNA-seq averages expression across millions of cells; if 50% of β-cells have
high INS and 50% have none, bulk reports the average and the biology is lost. Single-cell reads every
cell individually.

**Method.**
1. **QC** — remove dead cells, doublets, empty droplets (gene count, mitochondrial fraction ≤ 15%).
2. **Integration** — normalize + log → top 3,000 highly variable genes → PCA (30 dims) → **harmonypy**
   batch correction across GSE86469 (C1) and GSE153855 (SMART-seq) → neighbors/UMAP/Leiden.
3. **Annotation** — clusters are named by **marker-gene scoring** (`src/scrna/markers.py`) using
   canonical markers (INS→β, GCG→α, SST→δ, KRT19→ductal, PECAM1→endothelial, etc.). Result: 9 typed
   cell types, 3,880 cells.
4. **DEG** — Wilcoxon `rank_genes_groups` (T2D vs ND) **within each study separately**, per cell type.
5. **Replication** — a gene enters the disease signature only if its direction is **consistent in both
   studies**. Both tails are captured separately: genes up in T2D (from the T2D group's top-200) and
   genes down in T2D = up in ND (from the ND group's top-200).

**Results.**

| Metric | Value |
|---|---|
| Cells after QC (GSE86469 / GSE153855) | 497 / 3,383 |
| Merged cells | 3,880 |
| Cell types | 9 (β 815 · γ 729 · acinar 499 · α 474 · ductal 455 · stellate 393 · δ 371 · ε 90 · endothelial 54) |
| **Balanced replicated disease signature** | **1,431 genes = 470 up in T2D + 961 down in T2D** |
| Islet signature matrix (per-cell-type expression) | 9 cell types × 231 genes |

**Biological meaning.** The signature is *study-robust* (two independent platforms must agree) and
*direction-balanced* (a real down-regulated tail exists and is captured). The `signature_matrix.csv`
reads correctly — acinar cells are defined by digestive-enzyme expression (CELA3B 4.24, CPA2 4.40)
while endocrine cells are near zero for those genes.

---

### 5.3 Stage 3 (M3) — Spatial transcriptomics: where the disease lives

**Purpose:** locate cell states and define **niches** — the spatial neighborhoods in which cell types
act on each other (e.g., inflamed islet borders where stellate cells drive β-cell fibrosis).

**Method (proven on the approved mouse-brain fallback).**
1. `sc.read_visium` loads 2,695 spots; genes deduplicated (`var_names_make_unique`).
2. Normalize + log.
3. **NNLS deconvolution** (`scipy.optimize.nnls`) per spot against a marker signature matrix — cell
   proportions cannot be negative, so NNLS (non-negative least squares) is the correct estimator.
4. **KMeans** on spot proportions → 6 spatial niches.

**Results (fallback demo).**

| Output | Value |
|---|---|
| Spots | 2,695 |
| Deconvolution signature | 7 brain cell types × 24 canonical markers |
| Dominant cell type per spot | oligo 1,988 · neuron_inh 428 · neuron_exc 233 · astro 44 · microglia 2 |
| Niches (k=6) | sizes 457 / 545 / 545 / 542 / 203 / 403 |
| **Islet signature matrix (deliverable)** | **9 cell types × 231 genes** |

**Caveat.** Deconvolution on the fallback used marker-weighted signatures and produced a plausible but
imperfect anatomy gradient (see §8). The *real* deliverable is the islet signature matrix plus a
`--visium`-ready pipeline: `python scripts/run_m3_spatial.py --visium <SpaceRanger_dir>` deconvolves
pancreas Visium against the 9 islet cell types directly. A mismatch guard raises a clear error if the
tissue/species does not match the signature (≥10 genes or ≥10% overlap required).

---

### 5.4 Stage 4 (M4) — Knowledge graph + HeteroGNN

**Purpose:** learn cell-type-specific compound compatibility from a graph, not from independent
samples. Biology is a network (proteins interact, cells communicate, compounds hit multiple targets),
so the model must be a graph neural network.

**The graph.** Nodes: 3,000 compounds × 758 proteins × 9 cell types × 1 disease. Edge types:

| Edge | Source | Meaning |
|---|---|---|
| compound → protein | M1 (4,730) | drug–target |
| protein – protein | STRING (2,409) | PPI |
| protein → cell_type | M2 expression | gene expressed in cell type |
| **compound → cell_type ("therapeutic")** | M2 expression grounding | drug affects cell type **only if** its target is expressed there (mean log1p > 0.5) → 228 positives |
| cell_type → disease | design | association |
| reverse edges | — | balanced message passing |

**Expression grounding** is the biologically critical rule: a compound cannot influence a β-cell
through a protein β-cells do not transcribe. This single rule prevents biologically impossible
predictions and is what makes predictions cell-type-specific.

**Model & training.** HeteroGraphSAGE (2 layers, 64-dim) + MLP link predictor; node features = PCA of
Morgan fingerprints (compounds), per-cell-type expression (proteins, cell types). Training uses
**5-fold edge-masked cross-validation** — therapeutic edges are masked so we test generalization to
unseen compound–cell-type pairs, with fixed negatives. Baselines: RandomForest on the same features,
random guess.

**Results.**

| Model | AUROC | AUPRC |
|---|---|---|
| **HeteroGNN (PhytoDiaGNN)** | **0.9957** | 0.9978 |
| RandomForest baseline | 0.9913 | — |
| Random | 0.5 | — |
| Benchmark recovery | top-1% 34% | top-10% 79% |

**Interpretation.** The biology is strongly separable (RF already reaches 0.991 — features alone carry
much of the signal), and the graph *generalizes* to unseen compound–cell-type pairs — which is what
makes the model useful for screening, not just classification.

---

### 5.5 Stage 5 (M5) — Reversal scoring + explainable AI

**Purpose:** (a) rank all compounds per cell type by predicted disease *reversal*, and (b) explain each
prediction by identifying the responsible protein targets.

**S_rev reversal score (CMap convention).** For each of 27,000 compound × cell-type pairs (3,000 × 9),
count the compound's targets that lie in that cell type's disease signature, with explicit sign:

- `s_rev_inh` = **+1** per disease-**UP** gene hit (target assumed suppressed → corrects an
  over-elevated gene) **−1** per disease-**DOWN** gene hit (suppressing an already-low gene worsens it).
- `s_rev_act` = the **activation-assumption negation** of `s_rev_inh`.

Companion columns: `n_hits`, `n_up_hits`, `n_down_hits`, `coverage` (n_hits / cell-type signature
size), `purity` (s_rev / n_hits), `no_signal`. Compound–target databases do not encode
inhibitor-vs-activator mode, so **both conventions are reported** and no assumption is hidden.

**GNN edge-removal attribution (XAI).** Remove one compound→protein edge at a time, re-run the model,
and measure the logit drop. Large drop ⇒ that protein drives the prediction.

**Results.** 113 of 27,000 pairs engage ≥1 signature gene (see §8 for the coverage limit). Signed
scores now rank the engaging set meaningfully (e.g., NPC39426/stellate +1 via NPC1; NPC109083/beta −1,
honestly reflecting its hit on the disease-DOWN gene ALOX5). Top attribution drivers:

| Compound | Cell type | Target | Logit drop |
|---|---|---|---|
| NPC166788 | endothelial/ductal/stellate | **PLK1** | 40.1 / 40.0 / 39.9 |
| NPC279121 | endothelial/ductal/stellate | **TTR** | 7.04 |

These are not arbitrary picks — **PLK1** is a stress kinase implicated in β-cell dysfunction, and
**TTR** (transthyretin) forms the toxic amyloid aggregates characteristic of T2D islets. The model
surfaces known diabetes mechanisms.

**Correction history (documented in `REPORT.md`).** The original S_rev had an inverted sign and an
up-only signature, so "rank-1" tables were 0.0 ties among compounds with zero hits. That artifact is
fixed: the signature is balanced (470 up + 961 down), the score is signed and assumption-explicit, and
the retraction is on record.

---

### 5.6 Stage 6 (M6) — Molecular docking

**Purpose:** physical validation — statistical plausibility is not physical possibility.

**Method.**
1. Receptors from PDB 2PRG (PPARγ) and 1J1B (GSK3β); co-crystal ligands define a **22 Å centroid
   box** at the true binding site.
2. OpenBabel prepares receptors (PDB → PDBQT, protonation pH 7.4) and ligands (SMILES → 3D → PDBQT
   with embedding-seed retries for robustness).
3. AutoDock Vina 1.1.2 searches binding poses; 4 threaded workers (`--cpu 1` each); checkpoint CSV
   enables resume.
4. Benchmark: 14 unique known actives (deduplicated) vs **68 property-matched decoys** (MW ±25,
   logP ±1) per receptor → enrichment and AUROC.

**Results.**

| Receptor | Median active (kcal/mol) | Median decoy | EF5% | AUROC |
|---|---|---|---|---|
| **GSK3β** | −7.3 | −6.4 | 0.0 | **0.804** |
| PPARγ | −6.85 | −6.4 | 0.25 | 0.667 |

Top actives: NPC279121 (−7.9 GSK3β), NPC254847 (−7.9 PPARγ).
**Top novel candidates:** **NPC190454 (−7.3 GSK3β)**, **NPC461221 (−6.6 GSK3β)**.

**Tool quirk solved:** Vina 1.1.2 writes the score in the output PDBQT `REMARK VINA RESULT:` line, not
the terminal table — the original parser missed it, so all dockings silently "failed." The parser now
reads `line.split()[-3]`; 198/206 complexes docked (8 ligand-prep failures tolerated).

**Interpretation.** Docking *recovers* known actives against GSK3β (AUROC 0.80), supporting the
pipeline's validity. PPARγ enrichment is weaker (0.67) — a limitation (§8).

---

## 6. Key Results & Insights (cross-milestone)

1. **The T2D islet signature is best defined per cell type** — 1,431 genes, study-replicated and
   direction-balanced.
2. **Expression grounding keeps the graph honest** — therapeutic edges exist only where biology allows
   them, making predictions cell-type-specific by construction.
3. **GNN generalization + mechanism** — 0.996 AUROC with generalization to unseen pairs, and
   attribution surfaces real diabetes mechanisms (PLK1, TTR).
4. **Physical validation works** — GSK3β docking recovers actives (AUROC 0.80); two novel candidates
   dock strongly (NPC190454, NPC461221).
5. **The honest bottleneck is coverage** — compound targets rarely overlap the disease signature
   (113/27,000), so S_rev is a *filter*, and the GNN + docking are the primary evidence.
6. **Spatial is ready but gated** — the islet signature matrix + `--visium` flag await dbGaP access.

---

## 7. Limitations & Drawbacks (honest assessment)

### 7.1 Data-level limitations

| Limitation | Impact |
|---|---|
| **No open human-pancreas Visium** | M3 was demonstrated on a mouse-brain fallback; the diabetes-relevant spatial deliverable is a *prepared pipeline + signature matrix*, not a result. |
| **GSE153855 annotation mismatch (fixed)** | Required dropping rows and aligning cell counts; a small residual risk of annotation drift. |
| **Marker-based cell typing** | Types are assigned by canonical-marker scoring, not fully unsupervised annotation; rare/atypical states may be missed. |
| **Subsampling caps (≤5,000 cells/dataset)** | GSE124742/GSE221156 excluded by the 4 GB RAM cap — the atlas is not maximally comprehensive. |
| **PAINS bypass for the 18 actives** | Documented exception so positive controls exist; they are intentionally less stringent. |
| **DEG lists are up-biased by construction** | Fixed in M2 by capturing both tails; the balanced signature (1,431) is the corrected artifact. |

### 7.2 Method-level limitations

| Limitation | Impact |
|---|---|
| **GNN AUROC 0.996 near saturation** | The edge set is small and well-separated; RandomForest reaches 0.991, so the *incremental* value of the graph is modest — the graph's real value is generalization to unseen pairs. |
| **S_rev coverage 113/27,000** | Compound targets rarely overlap the replicated signature; most compounds are neutral by construction. S_rev is a weak filter, not a ranking of all compounds. |
| **Inhibitor-vs-activator ambiguity** | Solved by reporting both conventions, but the ambiguity is not eliminated. |
| **PPARγ docking enrichment weak (0.67)** | GSK3β is the strong target; PPARγ predictions should carry lower confidence. |
| **Vina scores are not calibrated binding free energies** | Affinity values are scoring-function estimates, not ΔG; rank-ordering within a target is meaningful, cross-target comparison is not. |
| **Mouse-brain deconvolution anatomy gradient imperfect** | Marker-only signatures without matched scRNA; acceptable for a methods demo. |
| **M4b (niche nodes in the KG) deferred** | Brain niches cannot be grafted onto the islet/compound KG without corrupting its biology; deferred until matched pancreas spatial data exists. |
| **Docking only 2 receptors × ~206 complexes** | Coverage-limited by compute budget; not exhaustive. |

### 7.3 Validation-level limitations

- **No wet-lab validation.** All conclusions are computational; the shortlist (NPC190454, NPC461221,
  NPC279121, NPC254847) requires experimental confirmation (cell assays, islet organoids, or in vivo).
- **No prospective blind test.** Benchmark recovery is retrospective on known actives.

---

## 8. Further Analysis & Future Work

### 8.1 Immediate next steps (CPU-only, no new data)

| Analysis | Tool/approach | Question answered |
|---|---|---|
| **Per-cell-type pathway enrichment** | gseapy / gprofiler (GO, KEGG) on the 470-up vs 961-down sets | Which biological programs define each cell type's diabetic state? |
| **Ligand–receptor cell–cell communication** | CellChat / LIANA on merged islets | Which β-cell–macrophage–stellate crosstalk drives progression? |
| **Target-mode resolution** | ChEMBL bioactivity (agonist vs inhibitor) | Resolve the S_rev inhibitor/activator ambiguity with real annotations. |
| **ADMET/toxicity screening of the shortlist** | SwissADME / ADMETlab | Are the top candidates orally bioavailable and non-toxic? |
| **Independent external validation** | GSE124742 (subset) | Does the 1,431-gene signature and GNN transfer to a 3rd cohort? |
| **Uncertainty quantification** | Monte-Carlo dropout ensembles on the GNN | Which predictions are high-confidence? |
| **Additional XAI** | GNNExplainer / Integrated Gradients | Confirm PLK1/TTR attribution with independent methods. |
| **Full-library docking** | Vina on all 3,000 compounds vs GSK3β | Full virtual screen; MM-GBSA rescoring of top hits. |
| **Scaffold analysis** | Bemis–Murcko decomposition | Chemical-series diversity of the final shortlist. |

### 8.2 Data unlocks

| Unlock | Action |
|---|---|
| **dbGaP access to GSE264331** | Run `run_m3_spatial.py --visium <dir>` → real islet deconvolution + niches; then **M4b**: add niche nodes to the KG and re-run the GNN. |
| IMPPAT / TCMSP download access | Expand the compound library beyond CMAUP. |
| A matched pancreas spatial + scRNA pair | Replace the fallback demo entirely; cross-validate niches against the atlas. |

### 8.3 Experimental validation

- **β-cell assays:** INS-1E or EndoC-βH1 lines treated with NPC190454 / NPC461221; measure insulin
  secretion, proliferation, and stress markers under glucolipotoxic conditions.
- **Islet organoids / ex vivo islets:** confirm cell-type-specific signature reversal (RNA-seq before
  vs after treatment).
- **TTR-focused assays:** test NPC279121-class compounds against transthyretin amyloid aggregation.
- **In vivo (long term):** diabetic mouse models for the final validated shortlist.

---

## 9. Reproducibility

```bash
conda activate phyto                    # Python 3.11 env; see README Setup
python scripts/run_m1_chemical_library.py   # M1: 3,000 compounds + KG edges
python scripts/run_m2_scrna.py             # M2: 3,880 cells, 1,431 balanced signature
python scripts/run_m3_spatial.py           # M3: islet matrix + Visium deconvolution/niches
python scripts/run_m4_gnn.py              # M4: HeteroGNN + baselines (5-fold CV)
python scripts/run_m5_scoring.py          # M5: S_rev scores + edge-removal attribution
python scripts/run_m6_docking.py          # M6: Vina docking + benchmark recovery
pytest -q                                  # 9 tests
```

**Output tree (key files):**

```text
results/
  chemical_library/   qc.json, props_summary.csv
  scrna/              qc.json
  spatial/            deconv.csv, niches.csv, summary.json
  m4_gnn/             summary.json
  m5_scoring/         s_rev_scores.csv, top_per_celltype.xlsx, gnn_attribution.csv, summary.json
  m6_docking/         docking_results.csv, summary.json, run.log
  summary.json        (master, machine-readable)
data/processed/
  compounds.parquet, targets.parquet, compound_target_edges.parquet,
  plant_compound_edges.parquet, ppi_edges.parquet
  scrna/  merged_islets.h5ad, deg_per_celltype.csv, deg_per_study.csv,
          disease_signatures.csv, signature_matrix.csv, signature_markers.csv
```

---

## 10. Appendices

### A. Result inventory (per milestone)

| Milestone | Key number(s) |
|---|---|
| M1 | 3,000 diverse compounds · 4,730 C–T · 15,688 P–C · 2,409 PPI |
| M2 | 3,880 cells · 9 types · 1,431 signature (470↑/961↓) · 231-gene islet matrix |
| M3 | 2,695 spots · 7 brain types · 6 niches |
| M4 | GNN AUROC 0.9957 · RF 0.9913 · top-10% 79% |
| M5 | 27,000 pairs · 113 engage signature · PLK1 drop 40.1 · TTR drop 7.04 |
| M6 | GSK3β AUROC 0.804 · PPARγ 0.667 · NPC190454 −7.3 · NPC461221 −6.6 |

### B. Glossary

- **SMILES** — Simple Molecular Input Line Entry System; text representation of molecules.
- **Lipinski / Veber / PAINS** — drug-likeness and assay-interference filters.
- **logP / TPSA / HBD / HBA** — lipophilicity, polar surface area, hydrogen-bond donors/acceptors.
- **DEG** — differentially expressed gene.
- **Harmony** — batch-effect correction by iterative clustering.
- **NNLS** — non-negative least squares; enforces non-negative cell proportions.
- **HeteroGraphSAGE** — inductive GNN for heterogeneous (multi-type) node graphs.
- **AUROC / AUPRC / EF** — area under ROC/precision-recall; enrichment factor.
- **XAI** — explainable AI (edge-removal attribution here).
- **PDBQT** — AutoDock's molecule format (partial charges + torsion tree).
