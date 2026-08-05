"""Build PhytoDiaGNN_Report.docx with uniform, paper-style formatting.

One font family and consistent sizes throughout: body 11 pt, headings
bold in the same family, tables 9.5 pt, captions 9.5 pt italic. No
monospace/terminal styling — inline code and code blocks are rendered
in the document font so the report reads as a single typographic voice.
"""

import json
import re
import sys
from pathlib import Path

import pandas as pd
from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

MD = ROOT / "PROJECT_REPORT.md"
FIG = ROOT / "results" / "figures"
OUT = ROOT / "PhytoDiaGNN_Report.docx"

FONT = "Cambria"
BODY = 11
TABLE = 9.5
CAP = 9.5
BLACK = RGBColor(0x00, 0x00, 0x00)

FIGURES = {
    "### 5.1 Stage 1 (M1)": ["fig_celltypes.png", "fig_signature_direction.png"],
    "### 5.2 Stage 2 (M2)": ["fig_umap_celltype.png", "fig_umap_disease.png"],
    "### 5.3 Stage 3 (M3) \u2014 Spatial transcriptomics: where the disease lives":
        ["fig_spatial_dominant.png", "fig_spatial_niche.png"],
    "### 5.4 Stage 4 (M4)": ["fig_gnn_auc.png", "fig_kg_graph.png"],
    "### 5.5 Stage 5 (M5)": ["fig_srev_dist.png", "fig_attribution.png", "fig_heatmap.png", "fig_ppi.png"],
    "### 5.6 Stage 6 (M6)": ["fig_docking_by_kind.png", "fig_docking_roc.png", "fig_top20.png", "fig_structures.png"],
}

CAPTIONS = {
    "fig_celltypes.png": "Figure 1. Cell-type composition after integration and annotation (Total cells = 3,880).",
    "fig_signature_direction.png": "Figure 2. Direction-balanced disease signature: 1,431 replicated genes "
                                   "(470 up / 961 down in T2D), shown per cell type.",
    "fig_umap_celltype.png": "Figure 3. UMAP of integrated human-islet single cells (n = 3,880), colored by cell type.",
    "fig_umap_disease.png": "Figure 4. UMAP of integrated human-islet single cells, colored by disease condition "
                            "(T2D vs nondiabetic).",
    "fig_spatial_dominant.png": "Figure 5. M3 \u2014 Visium spot deconvolution overlaid on the tissue image: "
                                "2,695 spots colored by dominant brain cell type (demonstration mouse-brain dataset).",
    "fig_spatial_niche.png": "Figure 6. M3 \u2014 Visium niche clusters (k = 6) overlaid on the tissue image, "
                             "derived by KMeans on deconvolved cell-type proportions.",
    "fig_gnn_auc.png": "Figure 7. M4 \u2014 link-prediction AUROC: HeteroGNN vs RandomForest vs random classifier "
                       "(5-fold edge-masked CV).",
    "fig_kg_graph.png": "Figure 8. M4 \u2014 knowledge-graph subset: candidate compounds, their protein targets, "
                        "cell types, and the disease association.",
    "fig_srev_dist.png": "Figure 9. M5 \u2014 S_rev scoring: (a) signature-gene hits across 27,000 "
                         "compound\u2013cell-type pairs; (b) signed scores of the 113 engaging pairs.",
    "fig_attribution.png": "Figure 10. M5 \u2014 GNN edge-removal attribution: top protein drivers.",
    "fig_heatmap.png": "Figure 11. M5 \u2014 compound \u00d7 cell-type reversal scores (S_rev_inh) for the "
                       "113 engaging pairs.",
    "fig_ppi.png": "Figure 12. M5 \u2014 STRING PPI subnetwork around key attributed and benchmark targets.",
    "fig_docking_by_kind.png": "Figure 13. M6 \u2014 Vina affinity by compound kind and receptor (medians annotated).",
    "fig_docking_roc.png": "Figure 14. M6 \u2014 docking benchmark ROC: known actives vs property-matched decoys "
                           "(GSK3B AUROC 0.804; PPARG 0.667).",
    "fig_top20.png": "Figure 15. M6 \u2014 top-20 novel phytochemical candidates by Vina affinity.",
    "fig_structures.png": "Figure 16. M6 \u2014 2D structures of the top four novel candidates.",
}

CODE_RE = re.compile(r"^```")


def set_font(run, size=BODY, bold=None, italic=None, color=None):
    run.font.name = FONT
    run.font.size = Pt(size)
    if bold is not None:
        run.font.bold = bold
    if italic is not None:
        run.font.italic = italic
    if color is not None:
        run.font.color.rgb = color
    rpr = run._element.get_or_add_rPr()
    rf = rpr.find(qn("w:rFonts"))
    if rf is None:
        rf = OxmlElement("w:rFonts")
        rpr.append(rf)
    for attr in ("w:ascii", "w:hAnsi", "w:cs"):
        rf.set(qn(attr), FONT)


def inline_runs(p, text, bold=False, italic=False, size=BODY):
    """Emit runs for `text`, toggling bold/italic across `**`, `*`, backtick
    markers. Toggle state persists between calls so emphasis spanning line
    breaks in the markdown source stays coherent. Backticks (inline code) are
    rendered in the document font — no monospace."""
    i = 0
    while i < len(text):
        if text.startswith("**", i):
            bold = not bold
            i += 2
        elif text[i] == "*":
            italic = not italic
            i += 1
        elif text[i] == "`":
            i += 1
        else:
            j = i
            while j < len(text) and not text.startswith("**", j) and text[j] != "*" and text[j] != "`":
                j += 1
            r = p.add_run(text[i:j])
            set_font(r, size=size, bold=bold or None, italic=italic or None)
            i = j
    return bold, italic


def add_inline(p, text, size=BODY, bold=False, italic=False):
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    return inline_runs(p, text, bold=bold, italic=italic, size=size)


def add_heading(doc, text, level):
    h = doc.add_heading("", level=level)
    h.paragraph_format.space_before = Pt({1: 6, 2: 14, 3: 10}[level])
    h.paragraph_format.space_after = Pt(4)
    h.paragraph_format.keep_with_next = True
    r = h.add_run(text)
    set_font(r, size={1: 17, 2: 14, 3: 12}[level], bold=True, color=BLACK)
    return h


def style_table(table, size=TABLE):
    for row in table.rows:
        for cell in row.cells:
            for p in cell.paragraphs:
                p.paragraph_format.space_after = Pt(1)
                p.paragraph_format.space_before = Pt(1)
                for r in p.runs:
                    set_font(r, size=size)


def clean_cell(val):
    return val.replace("**", "").replace("`", "").replace("*", "").strip()


def add_md_table(doc, lines):
    rows = []
    for ln in lines:
        if not ln.strip() or set(ln.strip()) <= set("-| "):
            continue
        cells = [clean_cell(c) for c in ln.strip().strip("|").split("|")]
        rows.append(cells)
    if not rows:
        return
    header, body = rows[0], rows[1:]
    n = len(header)
    table = doc.add_table(rows=1 + len(body), cols=n)
    table.style = "Table Grid"
    for j, h in enumerate(header):
        c = table.rows[0].cells[j].paragraphs[0]
        c.alignment = WD_ALIGN_PARAGRAPH.LEFT
        r = c.add_run(h)
        set_font(r, size=TABLE, bold=True)
    for i, row in enumerate(body, start=1):
        for j in range(n):
            val = row[j] if j < len(row) else ""
            c = table.rows[i].cells[j].paragraphs[0]
            c.alignment = WD_ALIGN_PARAGRAPH.LEFT
            r = c.add_run(val)
            set_font(r, size=TABLE)
    style_table(table)
    doc.add_paragraph()
    doc.paragraphs[-1].paragraph_format.space_after = Pt(4)


def add_data_table(doc, title, df, max_rows=None, max_cols=None):
    add_heading(doc, title, level=2)
    if df is None or df.empty:
        p = doc.add_paragraph("(empty)")
        set_font(p.runs[0], size=BODY) if p.runs else None
        return
    d = df.head(max_rows) if max_rows else df
    if max_cols:
        d = d.iloc[:, :max_cols]
    header = [str(c) for c in d.columns]
    table = doc.add_table(rows=1 + len(d), cols=len(header))
    table.style = "Table Grid"
    for j, h in enumerate(header):
        c = table.rows[0].cells[j].paragraphs[0]
        c.alignment = WD_ALIGN_PARAGRAPH.LEFT
        r = c.add_run(h)
        set_font(r, size=TABLE, bold=True)
    for i, (_, row) in enumerate(d.iterrows(), start=1):
        for j, h in enumerate(header):
            v = row[h]
            if isinstance(v, float):
                v = round(v, 4)
            c = table.rows[i].cells[j].paragraphs[0]
            c.alignment = WD_ALIGN_PARAGRAPH.LEFT
            r = c.add_run(str(v))
            set_font(r, size=TABLE)
    style_table(table)
    doc.add_paragraph()
    doc.paragraphs[-1].paragraph_format.space_after = Pt(4)


def add_figure(doc, f):
    path = FIG / f
    if not path.exists():
        return
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(8)
    p.paragraph_format.keep_with_next = True
    r = p.add_run()
    r.add_picture(str(path), width=Inches(5.6))
    c = doc.add_paragraph()
    c.alignment = WD_ALIGN_PARAGRAPH.CENTER
    c.paragraph_format.space_after = Pt(8)
    r = c.add_run(CAPTIONS.get(f, f))
    set_font(r, size=CAP, italic=True)


def add_page_number(section):
    p = section.footer.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run()
    set_font(r, size=9)
    f1 = OxmlElement("w:fldChar")
    f1.set(qn("w:fldCharType"), "begin")
    it = OxmlElement("w:instrText")
    it.set(qn("xml:space"), "preserve")
    it.text = "PAGE"
    f2 = OxmlElement("w:fldChar")
    f2.set(qn("w:fldCharType"), "end")
    r._r.append(f1)
    r._r.append(it)
    r._r.append(f2)


def main():
    doc = Document()
    sec = doc.sections[0]
    sec.left_margin = sec.right_margin = Inches(1)
    sec.top_margin = sec.bottom_margin = Inches(1)
    add_page_number(sec)

    normal = doc.styles["Normal"]
    normal.font.name = FONT
    normal.font.size = Pt(BODY)
    normal.font.color.rgb = BLACK
    normal.paragraph_format.line_spacing = 1.15
    normal.paragraph_format.space_after = Pt(6)
    rpr = normal.element.get_or_add_rPr()
    rf = rpr.find(qn("w:rFonts"))
    if rf is None:
        rf = OxmlElement("w:rFonts")
        rpr.append(rf)
    for attr in ("w:ascii", "w:hAnsi", "w:cs"):
        rf.set(qn(attr), FONT)

    for sname in ("List Bullet", "List Number"):
        st = doc.styles[sname]
        st.font.name = FONT
        st.font.size = Pt(BODY)
        st.font.color.rgb = BLACK

    lines = MD.read_text().splitlines()
    n = len(lines)
    i = 0
    title_done = False
    in_code = False
    bold_on = False
    italic_on = False

    def emit(p, text):
        nonlocal bold_on, italic_on
        bold_on, italic_on = inline_runs(p, text, bold=bold_on, italic=italic_on)

    while i < n:
        raw = lines[i].rstrip()
        if not raw.strip():
            i += 1
            continue
        if set(raw.strip()) <= set("-"):
            if not title_done:
                doc.add_page_break()
                title_done = True
            i += 1
            continue
        if CODE_RE.match(raw):
            in_code = not in_code
            i += 1
            continue
        if in_code:
            p = doc.add_paragraph()
            p.alignment = WD_ALIGN_PARAGRAPH.LEFT
            p.paragraph_format.left_indent = Inches(0.25)
            r = p.add_run(raw)
            set_font(r, size=10)
            i += 1
            continue
        if raw.startswith("### "):
            add_heading(doc, raw[4:], level=3)
            for key, figs in FIGURES.items():
                if raw.strip().startswith(key):
                    for fig in figs:
                        add_figure(doc, fig)
                    break
            i += 1
            continue
        if raw.startswith("## "):
            if not title_done:
                p = doc.add_paragraph()
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                r = p.add_run(raw[3:])
                set_font(r, size=13, italic=True)
            else:
                add_heading(doc, raw[3:], level=2)
            i += 1
            continue
        if raw.startswith("# "):
            if not title_done:
                p = doc.add_paragraph()
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                p.paragraph_format.space_before = Pt(60)
                r = p.add_run(raw[2:])
                set_font(r, size=20, bold=True)
            else:
                add_heading(doc, raw[2:], level=1)
            i += 1
            continue
        if raw.startswith("|"):
            tbl = []
            while i < n and lines[i].strip().startswith("|"):
                tbl.append(lines[i].strip())
                i += 1
            add_md_table(doc, tbl)
            continue
        if raw.startswith("- "):
            p = doc.add_paragraph(style="List Bullet")
            p.alignment = WD_ALIGN_PARAGRAPH.LEFT
            emit(p, raw[2:])
            p.alignment = WD_ALIGN_PARAGRAPH.LEFT
            i += 1
            continue
        if re.match(r"^\d+\. ", raw):
            p = doc.add_paragraph(style="List Number")
            p.alignment = WD_ALIGN_PARAGRAPH.LEFT
            emit(p, re.sub(r"^\d+\. ", "", raw))
            p.alignment = WD_ALIGN_PARAGRAPH.LEFT
            i += 1
            continue
        if raw.startswith("> "):
            p = doc.add_paragraph()
            p.paragraph_format.left_indent = Inches(0.35)
            p.paragraph_format.right_indent = Inches(0.35)
            emit(p, raw[2:])
            i += 1
            continue
        p = doc.add_paragraph()
        if not title_done:
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        emit(p, raw)
        if not title_done:
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        i += 1

    doc.add_page_break()
    add_heading(doc, "Appendix: Data Outputs", level=1)
    p = doc.add_paragraph(
        "Full machine-readable outputs live in results/ (per-stage summary.json and CSVs) "
        "and are copied next to this report. Tables below show the key data; large tables "
        "are truncated and marked.")
    set_font(p.runs[0], size=BODY)

    q1 = pd.read_json(ROOT / "results/chemical_library/qc.json")
    add_data_table(doc, "M1 — chemical library QC", pd.DataFrame(
        [{"metric": k, "value": v if not isinstance(v, dict) else json.dumps(v)} for k, v in q1.items()]), max_rows=20)
    add_data_table(doc, "M1 — compound property summary (count/mean/std)",
                   pd.read_csv(ROOT / "results/chemical_library/props_summary.csv").rename(columns={"Unnamed: 0": "stat"}),
                   max_rows=6, max_cols=7)

    m2q = json.load(open(ROOT / "results/scrna/qc.json"))
    add_data_table(doc, "M2 — scRNA QC", pd.DataFrame(
        [{"metric": k, "value": v if not isinstance(v, dict) else json.dumps(v)} for k, v in m2q.items()]), max_rows=20)
    add_data_table(doc, "M2 — cell-type counts",
                   pd.DataFrame({"cell_type": m2q["cell_types"].keys(), "cells": m2q["cell_types"].values()}))
    add_data_table(doc, "M2 — islet signature matrix (mean log1p per cell type; first 30 genes)",
                   pd.read_csv(ROOT / "data/processed/scrna/signature_matrix.csv", index_col=0), max_cols=30)
    add_data_table(doc, "M2 — disease signature (first 30 of 1,431 rows)",
                   pd.read_csv(ROOT / "data/processed/scrna/disease_signatures.csv"), max_rows=30)

    s3 = json.load(open(ROOT / "results/spatial/summary.json"))
    add_data_table(doc, "M3 — spatial summary", pd.DataFrame(
        [{"field": k, "value": str(v)} for k, v in s3.items()]), max_rows=20)
    add_data_table(doc, "M3 — deconvolution: dominant cell type per spot",
                   pd.read_csv(ROOT / "results/spatial/deconv.csv", index_col=0).idxmax(axis=1).value_counts().rename_axis("cell_type").reset_index(name="spots"))
    add_data_table(doc, "M3 — deconvolution sample (first 10 spots)",
                   pd.read_csv(ROOT / "results/spatial/deconv.csv", index_col=0).head(10).round(3))

    add_data_table(doc, "M4 — GNN summary", pd.read_json(ROOT / "results/m4_gnn/summary.json"), max_rows=10)

    m5s = json.load(open(ROOT / "results/m5_scoring/summary.json"))
    add_data_table(doc, "M5 — summary (key fields)",
                   pd.DataFrame([{"field": k, "value": json.dumps(v) if isinstance(v, (dict, list)) else str(v)}
                                 for k, v in m5s.items() if not isinstance(v, dict) or k == "s_rev_convention"]), max_rows=10)
    srev = pd.read_csv(ROOT / "results/m5_scoring/s_rev_scores.csv")
    add_data_table(doc, "M5 — S_rev scores, pairs engaging the signature (113 rows)",
                   srev[srev["n_hits"] > 0].sort_values(["s_rev_inh", "n_hits"], ascending=False))
    add_data_table(doc, "M5 — GNN attribution (top 40 of 1,005)",
                   pd.read_csv(ROOT / "results/m5_scoring/gnn_attribution.csv").sort_values("drop", ascending=False).head(40))

    add_data_table(doc, "M6 — docking results (all docked complexes, 198 rows)",
                   pd.read_csv(ROOT / "results/m6_docking/docking_results.csv").sort_values(["rec", "affinity"]))
    m6s = json.load(open(ROOT / "results/m6_docking/summary.json"))
    add_data_table(doc, "M6 — docking summary (PPARG / GSK3B)",
                   pd.DataFrame([{"receptor": rec, "n_actives": v["n_actives"], "n_decoys": v["n_decoys"],
                                  "median_active": v["median_active_affinity"], "median_decoy": v["median_decoy_affinity"],
                                  "EF5%": v["EF5%"], "AUROC": v["AUROC"]} for rec, v in m6s.items() if isinstance(v, dict)]))

    doc.save(OUT)
    print("saved", OUT, OUT.stat().st_size, "bytes")


if __name__ == "__main__":
    main()
