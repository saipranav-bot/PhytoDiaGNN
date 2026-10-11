# PhytoDiaGNN

A graph neural network-driven, multi-omics framework for cell-type- and spatial-niche-specific
discovery of anti-diabetic phytochemicals using single-cell and spatial transcriptomics.

**Scope (MVP):** pancreatic-islet-first, CPU-only, conservative caps (4 GB RAM target).

## Pipeline stages

| Stage | Dir | Status |
|---|---|---|
| M1 Chemical library (CMAUP → filters → diverse 3,000) | `src/chem` | done |
| M2 Islet scRNA integration (GSE86469 + GSE153855) | `src/scrna` | done |
| M3 Spatial niche + signaling (Visium) | `src/spatial` | done |
| M4 Knowledge graph + HeteroGNN link prediction | `src/graph` | done |
| M5 Reversal scoring + GNN attribution | `src/graph` | done |
| M6 Docking (AutoDock Vina) + benchmark recovery | `src/chem` | done |

## Setup

```bash
conda env create -n phyto python=3.11 -y
conda activate phyto
pip install .            # installs the pinned dependencies from pyproject.toml
conda install -c conda-forge -c bioconda autodock-vina openbabel
```

## Run the pipeline

```bash
conda activate phyto
python scripts/run_m1_chemical_library.py   # M1: 3,000 diverse compounds + KG edges
python scripts/run_m2_scrna.py             # M2: 3,880 islet cells, 1,431 balanced signature genes
python scripts/run_m3_spatial.py           # M3: islet signature matrix + Visium deconvolution/niches
python scripts/run_m4_gnn.py              # M4: HeteroGNN link prediction + baselines
python scripts/run_m5_scoring.py          # M5: S_rev scores + edge-removal attribution
python scripts/run_m6_docking.py          # M6: Vina docking + benchmark recovery report
```

## Results summary

- **M1** — 60,222 CMAUP compounds → 29,019 drug-like → 3,000 max-min diverse; 18 benchmark
  actives forced in; 4,730 compound→target, 15,688 plant→compound, 2,409 STRING PPI edges.
- **M2** — GSE86469 (Fluidigm C1) + GSE153855 (SMART-seq) integrated; 3,880 cells, 9 islet cell
  types; per-study DEG yields 1,431 replicated T2D signature genes (470 up + 961 down,
  `disease_signatures.csv`).
- **M3** — islet signature matrix (9 cell types × 231 genes) ready for pancreas Visium;
  methods demo on 10x mouse-brain Visium (2,695 spots): NNLS deconvolution to 7 brain cell
  types + 6 spatial niches. No open human-pancreas Visium exists (dbGaP-gated / cloupe-only /
  scRNA-only) — documented in `REPORT.md`.
- **M4** — heterogeneous GraphSAGE (compound/protein/cell_type/disease nodes) predicts
  therapeutic compound→cell-type edges; 5-fold AUROC **0.996** vs RandomForest 0.991, random 0.5;
  benchmark actives recovered in top-1% (34%) / top-10% (79%).
- **M5** — signed, convention-explicit S_rev scores for 27,000 compound×cell-type pairs
  (`s_rev_inh` CMap / `s_rev_act` activation; 113 pairs engage the signature); GNN edge-removal
  attribution pinpoints driving targets (e.g., PLK1 for NPC166788).
- **M6** — AutoDock Vina 1.1.2 vs PPARγ (2PRG) and GSK3β (1J1B); enrichment factor + AUROC report.

Outputs land in `results/` (per-stage `summary.json`, master `results/summary.json`, tables) and
`data/processed/`. Consolidated write-up: `REPORT.md`.
