"""Stage 1 — extract + de-boilerplate the hospital PDFs into clean Markdown.

Fully deterministic: PDFs in ``data/raw/hospital/`` -> ``data/clean/*.md``.
"""
from __future__ import annotations

import re
import statistics
import sys

import fitz  # PyMuPDF

from .config import CLEAN_DIR, RAW_HOSPITAL, ROOT, SLUG_MAP

# Fonts that only ever carry boilerplate: Type3 (cursive signatures, decorative glyphs,
# seal), CourierNew (the PKI metadata table), Nirmala (Sinhala/Tamil subtitles), Georgia
# (the letterhead wordmark).
DROP_FONT_SUBSTR = ("Type3", "CourierNew", "Nirmala", "Georgia")

# The page-0 header is bounded relative to the unambiguous "Ref: SGH-..." anchor line
# rather than a brittle fixed y-cutoff: the title sits level with the ref block.
RE_REF_ANCHOR = re.compile(r"Ref:\s*SGH-")
REF_STEP = 18.0       # ref-block lines are ~12pt apart; the body starts after a wider gap
PAGE0_HEADER_CUTOFF = 200.0  # fallback if the ref anchor is absent

MIN_SPAN_SIZE = 6.0   # below this, spans are seal fragments ("COLOMBO", "2026" at ~5pt)

# Running footer + signature remnants that survive the font filter (they are plain Arial).
FOOTER_PATTERNS = [
    re.compile(r"^Page\s+\d+\s+of\s+\d+$", re.I),
    re.compile(r"^SGH-[A-Z]+-\d+\s*\|\s*Rev", re.I),
    re.compile(r"SERENDIB GENERAL HOSPITAL\s*·"),  # footer centre (· COLOMBO / · EMERGENCY / …)
    re.compile(r"^Prices subject to revision", re.I),
    re.compile(r"^Document integrity verified", re.I),
    re.compile(r"^Issued by:\s*SGH PKI", re.I),
    re.compile(r"^Certificate chain:", re.I),
    re.compile(r"^Digitally Signed$", re.I),
    re.compile(r"^\s*✓?\s*Digitally Signed", re.I),
]

# Ref-block field extractors. Dates may be split across two lines by a stray "|", e.g.
# "Valid Until: 30 | September 2026", so the date patterns tolerate an intervening pipe.
RE_DOCID = re.compile(r"Ref:\s*(SGH-[A-Z]+-\d+)")
RE_ISSUED = re.compile(r"Issued:\s*(\d{1,2}\s*\|?\s*\w+\s+\d{4})")
RE_REV = re.compile(r"Rev:\s*([\d.]+)")
RE_DEPT = re.compile(r"Dept:\s*([^|]+?)\s*(?:\||Valid Until|$)")
RE_VALID = re.compile(r"Valid Until:\s*(\d{1,2}\s*\|?\s*\w+\s+\d{4})")


def _span_kept(span: dict) -> bool:
    """True if a span carries real content (not boilerplate font / size / glyph)."""
    if span["size"] < MIN_SPAN_SIZE:
        return False
    if any(s in span["font"] for s in DROP_FONT_SUBSTR):
        return False
    if not span["text"].strip():
        return False
    if span["text"].strip() in {"✦", "✓", "•", "▪", ""}:
        return False
    return True


def _iter_lines(page: fitz.Page, pno: int):
    """Yield one record per kept text line, with geometry and style flags.

    These PDFs emit one text line per block, and wrapping/columns scatter a logical table
    row across several blocks — hence line granularity plus geometric re-clustering.
    """
    data = page.get_text("dict")
    for blk in data.get("blocks", []):
        if blk.get("type") != 0:  # skip images
            continue
        for line in blk["lines"]:
            kept = [sp for sp in line["spans"] if _span_kept(sp)]
            if not kept:
                continue
            txt = re.sub(r"\s+", " ", "".join(sp["text"] for sp in kept)).strip()
            if not txt:
                continue
            y0 = min(sp["bbox"][1] for sp in kept)
            y1 = max(sp["bbox"][3] for sp in kept)
            yield {
                "page": pno,
                "x0": min(sp["bbox"][0] for sp in kept),
                "y0": y0,
                "y1": y1,
                "h": y1 - y0,
                "text": txt,
                "bold": bool(kept[0]["flags"] & 16),
                "all_bold": all(sp["flags"] & 16 for sp in kept),
            }


def _extract_frontmatter(header_text: str) -> dict:
    def _first(rx):
        m = rx.search(header_text)
        if not m:
            return ""
        # collapse any stray "|" that split a value across two header lines
        return re.sub(r"\s*\|\s*", " ", m.group(1)).strip()

    return {
        "doc_id": _first(RE_DOCID),
        "issued": _first(RE_ISSUED),
        "rev": _first(RE_REV),
        "dept": _first(RE_DEPT),
        "valid_until": _first(RE_VALID),
    }


ROW_TOL = 5.0  # lines within this many points of y share a visual row


def _visual_rows(lines: list[dict]) -> list[dict]:
    """Cluster kept lines into visual rows (same page, near-equal y)."""
    lines = sorted(lines, key=lambda l: (l["page"], round(l["y0"], 1), l["x0"]))
    rows: list[dict] = []
    cur: list[dict] = []
    for l in lines:
        if cur and l["page"] == cur[0]["page"] and abs(l["y0"] - cur[0]["y0"]) <= ROW_TOL:
            cur.append(l)
        else:
            if cur:
                rows.append(_make_row(cur))
            cur = [l]
    if cur:
        rows.append(_make_row(cur))
    return rows


def _make_row(cells: list[dict]) -> dict:
    cells = sorted(cells, key=lambda c: c["x0"])
    return {
        "page": cells[0]["page"],
        "y0": min(c["y0"] for c in cells),
        "y1": max(c["y1"] for c in cells),
        "h": statistics.median(c["h"] for c in cells),
        "cells": cells,
        "n_cells": len(cells),
        "text": " ".join(c["text"] for c in cells),
        "all_bold": all(c["all_bold"] for c in cells),
    }


def _cluster_anchors(xs: list[float], tol: float = 22.0) -> list[float]:
    """1-D clustering of x-coordinates into column anchors."""
    xs = sorted(xs)
    clusters, cur = [], [xs[0]]
    for x in xs[1:]:
        if x - cur[-1] <= tol:
            cur.append(x)
        else:
            clusters.append(cur)
            cur = [x]
    clusters.append(cur)
    return [sum(c) / len(c) for c in clusters]


def _render_table(rows: list[dict]) -> list[str] | None:
    """Reconstruct a table section as a GitHub-flavoured Markdown table.

    Cells are assigned to columns by nearest x-anchor; a row less than a line-and-a-half
    below the previous one is a wrapped continuation and is merged up into it.
    """
    anchors = _cluster_anchors([c["x0"] for r in rows for c in r["cells"]])
    if len(anchors) < 2:
        return None
    cont_thresh = 1.6 * statistics.median(r["h"] for r in rows if r["h"] > 0)

    out: list[list[str]] = []
    prev: dict | None = None
    for r in rows:
        cols = [""] * len(anchors)
        for c in r["cells"]:
            j = min(range(len(anchors)), key=lambda k: abs(c["x0"] - anchors[k]))
            cols[j] = (cols[j] + " " + c["text"]).strip()
        is_cont = (prev is not None and out and r["page"] == prev["page"]
                   and (r["y0"] - prev["y0"]) < cont_thresh)
        if is_cont:
            for k in range(len(anchors)):
                if cols[k]:
                    out[-1][k] = (out[-1][k] + " " + cols[k]).strip()
        else:
            out.append(cols)
        prev = r

    ncol = len(anchors)
    md = ["", "| " + " | ".join(out[0]) + " |",
          "| " + " | ".join(["---"] * ncol) + " |"]
    md += ["| " + " | ".join(row) + " |" for row in out[1:]]
    md.append("")
    return md


def _render_prose(rows: list[dict], md: list[str]) -> None:
    """Merge wrapped lines into paragraphs; short standalone lines become bullets."""
    para: list[str] = []
    prev: dict | None = None

    def flush() -> None:
        nonlocal para, prev
        if para:
            md.extend(["", " ".join(para)])
        para, prev = [], None

    for r in rows:
        text = r["text"]
        last = text.rstrip()[-1:]
        if r["n_cells"] == 1 and len(text) < 90 and last not in ".:,;" and not r["all_bold"]:
            flush()
            md.append(f"- {text}")
        else:
            same = (prev is not None and not r["cells"][0]["bold"]
                    and r["page"] == prev["page"]
                    and (r["y0"] - prev["y1"]) < 0.9 * r["h"])
            if not same:
                flush()
            para.append(text)
            prev = r
    flush()


def _flush_section(rows: list[dict], md: list[str]) -> None:
    """Render a section between two headings as a table, prose, or a mix.

    A mostly-grid section often still carries an intro sentence or a trailing note (a
    single wide cell of prose). Those are peeled out and rendered as paragraphs so they
    don't become empty-celled table rows.
    """
    if not rows:
        return
    if sum(1 for r in rows if r["n_cells"] >= 2) < 2:
        _render_prose(rows, md)
        return

    table_buf: list[dict] = []
    prose_buf: list[dict] = []

    def flush_table() -> None:
        if len(table_buf) >= 2 and (t := _render_table(table_buf)):
            md.extend(t)
        elif table_buf:
            _render_prose(table_buf, md)
        table_buf.clear()

    def flush_prose() -> None:
        if prose_buf:
            _render_prose(prose_buf, md)
        prose_buf.clear()

    for r in rows:
        # A single wide cell of 60+ chars is a wrapped intro/note sentence, not a table
        # row — real continuation cells like "glucose"/"Widal" are <20 chars.
        if r["n_cells"] == 1 and len(r["text"]) > 60:
            flush_table()
            prose_buf.append(r)
        else:
            flush_prose()
            table_buf.append(r)
    flush_table()
    flush_prose()


def _title_from_filename(name: str) -> str:
    """Derive a clean, deterministic title from the source filename."""
    stem = re.sub(r"\.pdf$", "", name, flags=re.I)
    stem = re.sub(r"\s+", " ", stem).strip()
    stem = re.sub(r"^FAQ\s*-\s*", "FAQ — ", stem)
    return stem.replace(" - ", " — ")


def clean_pdf(pdf_path) -> tuple[dict, str]:
    doc = fitz.open(pdf_path)

    # --- pass 1: split into header (front-matter) and body lines ---
    page0 = list(_iter_lines(doc[0], 0))
    ref_hits = [l["y0"] for l in page0 if RE_REF_ANCHOR.search(l["text"])]

    # header_end_y = bottom of the letterhead + ref block. From the ref anchor, grow down
    # through closely-spaced lines (wrapped ref values included) and stop at the first
    # wide gap, which is the body.
    if ref_hits:
        ref_y = ref_hits[0]
        header_end_y, prev = ref_y, ref_y
        for l in sorted((x for x in page0 if x["y0"] >= ref_y - 1), key=lambda x: x["y0"]):
            if l["y0"] - prev < REF_STEP:
                prev = l["y0"]
                header_end_y = max(header_end_y, l["y1"])
            else:
                break
    else:
        header_end_y = PAGE0_HEADER_CUTOFF

    header_lines = [l for l in page0 if l["y0"] < header_end_y]
    body_lines = [l for l in page0 if l["y0"] >= header_end_y]
    for pno in range(1, len(doc)):
        body_lines.extend(_iter_lines(doc[pno], pno))

    body_lines = [l for l in body_lines
                  if not any(rx.search(l["text"]) for rx in FOOTER_PATTERNS)
                  and l["text"].strip().upper() != "SERENDIB GENERAL HOSPITAL"]

    # The letterhead is two-column: the ref block sits in the right ~2/3 (x0 ~ 296-540)
    # while the doc title is far left (x0 ~ 57). Rebuild the ref text from right-side
    # header lines only, so the title never corrupts the Rev/Valid regexes.
    ref_text = " ".join(l["text"] for l in sorted(
        (l for l in header_lines if l["x0"] > 200), key=lambda l: l["y0"]))
    fm = _extract_frontmatter(ref_text)
    fm["title"] = _title_from_filename(pdf_path.name)

    # --- pass 2: visual rows -> sections (delimited by headings) -> Markdown ---
    md_lines: list[str] = []
    section: list[dict] = []
    for row in _visual_rows(body_lines):
        is_heading = (row["n_cells"] == 1 and row["all_bold"]
                      and len(row["text"]) < 110)
        if is_heading:
            _flush_section(section, md_lines)
            section = []
            prefix = "###" if row["text"].rstrip().endswith("?") else "##"
            md_lines += ["", f"{prefix} {row['text']}"]
        else:
            section.append(row)
    _flush_section(section, md_lines)

    body_md = re.sub(r"\n{3,}", "\n\n", "\n".join(md_lines).strip())
    return fm, body_md


def build_markdown(pdf_name: str, fm: dict, body_md: str) -> str:
    def esc(v: str) -> str:
        return v.replace('"', "'")

    front = [
        "---",
        f'source_pdf: "{esc(pdf_name)}"',
        f'doc_id: "{esc(fm.get("doc_id", ""))}"',
        f'title: "{esc(fm.get("title", ""))}"',
        f'dept: "{esc(fm.get("dept", ""))}"',
        f'issued: "{esc(fm.get("issued", ""))}"',
        f'rev: "{esc(fm.get("rev", ""))}"',
        f'valid_until: "{esc(fm.get("valid_until", ""))}"',
        "---",
        "",
    ]
    heading = f"# {fm.get('title') or pdf_name}"
    return "\n".join(front) + heading + "\n\n" + body_md + "\n"


def cmd_clean(_args) -> None:
    CLEAN_DIR.mkdir(parents=True, exist_ok=True)
    pdfs = sorted(RAW_HOSPITAL.glob("*.pdf"))
    if not pdfs:
        sys.exit(f"No PDFs found in {RAW_HOSPITAL}")
    for pdf in pdfs:
        slug = SLUG_MAP.get(pdf.name)
        if not slug:
            print(f"  ! no slug for {pdf.name}, skipping")
            continue
        fm, body = clean_pdf(pdf)
        md = build_markdown(pdf.name, fm, body)
        out = CLEAN_DIR / f"{slug}.md"
        out.write_text(md, encoding="utf-8")
        print(f"  ✓ {pdf.name}  ->  {out.relative_to(ROOT)}  "
              f"[{fm.get('doc_id','?')}] ({len(body)} chars)")
    print(f"Cleaned {len(pdfs)} hospital docs into {CLEAN_DIR.relative_to(ROOT)}")
