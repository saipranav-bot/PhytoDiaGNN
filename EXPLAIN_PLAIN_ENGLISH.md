# Plain-English Explanation (No Biology Background Needed)

## What this project does
This project takes publicly available data (microarrays/scRNA-seq/spatial/16S) and runs a standard data-analysis pipeline to produce reproducible tables and figures. Each step follows the same idea: clean data -> find patterns -> summarize.

## Steps followed (in order)
1. **Get data**: Downloaded/fetched public datasets (with accession numbers). Where raw data is large, scripts fetch it on demand.
2. **Quality check**: Removed low-quality samples/reads/spots (too few measurements, too much noise) to keep reliable data.
3. **Normalize**: Scaled measurements so different samples can be fairly compared.
4. **Find key features**: Identified genes/taxa that vary most across groups.
5. **Reduce dimensions/cluster**: Grouped similar samples/cells/spots to see structure.
6. **Compare groups**: Compared groups (e.g. disease vs control/tumor vs normal) to find differences.
7. **Summarize**: Generated tables (lists) and figures (plots) summarizing findings.
8. **Annotate/interpret**: Added readable labels where useful (data-driven where possible).

## Verification
- Scripts run end-to-end with fixed seeds where randomness matters.
- Key outputs (CSV tables) match expected structure; figures regenerated.
- Git history shows clean commits; large raw files not committed.

## How to run (quick)
See README.md in this repo. Run numbered scripts in order (or run_all.sh if present).

## Notes
- Reproducible: seeds set, inputs referenced by accession.
- Clean: old/superseded files removed; only final outputs kept.
- Best-effort applied for repos with gated/licensed tools.
