"""Stage 2 — parse the clean Markdown back into structured blocks/sections.

``parse_clean`` and ``iter_sections`` are the shared reader used by both this
stage and the generation stage.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from .config import CLEAN_DIR, INTERIM_DIR, ROOT


def parse_clean(path: Path) -> tuple[dict, list[dict]]:
    """Parse a clean .md into (front-matter, ordered blocks)."""
    text = path.read_text(encoding="utf-8")
    fm: dict = {}
    body = text
    if text.startswith("---"):
        _, fmblock, body = text.split("---", 2)
        for line in fmblock.strip().splitlines():
            m = re.match(r'(\w+):\s*"(.*)"', line.strip())
            if m:
                fm[m.group(1)] = m.group(2)

    blocks: list[dict] = []
    lines = body.splitlines()
    i = 0
    while i < len(lines):
        ln = lines[i].rstrip()
        if not ln.strip():
            i += 1
        elif ln.startswith("### "):
            blocks.append({"kind": "heading", "level": 3, "text": ln[4:].strip()})
            i += 1
        elif ln.startswith("## "):
            blocks.append({"kind": "heading", "level": 2, "text": ln[3:].strip()})
            i += 1
        elif ln.startswith("# "):
            i += 1  # H1 title, skip
        elif ln.lstrip().startswith("|"):
            raw = []
            while i < len(lines) and lines[i].lstrip().startswith("|"):
                raw.append([c.strip() for c in lines[i].strip().strip("|").split("|")])
                i += 1
            rows = [r for r in raw if not all(set(c) <= set("-: ") and c for c in r)]
            if rows:
                blocks.append({"kind": "table", "header": rows[0], "rows": rows[1:]})
        elif ln.lstrip().startswith("- "):
            items = []
            while i < len(lines) and lines[i].lstrip().startswith("- "):
                items.append(lines[i].lstrip()[2:].strip())
                i += 1
            blocks.append({"kind": "bullets", "items": items})
        else:
            blocks.append({"kind": "para", "text": ln.strip()})
            i += 1
    return fm, blocks


def iter_sections(blocks: list[dict]):
    """Yield sections: a heading plus the content blocks up to the next heading."""
    parent = ""
    i = 0
    while i < len(blocks):
        b = blocks[i]
        if b["kind"] == "heading":
            if b["level"] == 2:
                parent = b["text"]
            content = []
            j = i + 1
            while j < len(blocks) and blocks[j]["kind"] != "heading":
                content.append(blocks[j])
                j += 1
            yield {"heading": b["text"], "level": b["level"],
                   "parent": parent if b["level"] == 3 else "", "content": content}
            i = j
        else:
            i += 1


def cmd_segment(_args) -> None:
    INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    out = INTERIM_DIR / "sections.jsonl"
    n = 0
    with out.open("w", encoding="utf-8") as fh:
        for md in sorted(CLEAN_DIR.glob("*.md")):
            fm, blocks = parse_clean(md)
            for sec in iter_sections(blocks):
                rec = {
                    "doc_id": fm.get("doc_id", ""), "source_doc": fm.get("source_pdf", ""),
                    "clean_path": str(md.relative_to(ROOT)), "title": fm.get("title", ""),
                    "heading": sec["heading"], "parent": sec["parent"], "level": sec["level"],
                    "content": sec["content"],
                }
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n += 1
    print(f"Segmented {len(list(CLEAN_DIR.glob('*.md')))} docs into {n} sections "
          f"-> {out.relative_to(ROOT)}")
