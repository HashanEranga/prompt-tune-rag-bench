# Phase A — Dataset Creation & Preparation (COMPLETE)

Everything the judge and the three contenders consume was built here, from the 13
fictional **Serendib General Hospital** PDFs only. The 15 real `data/raw/medical/`
PDFs are held back as Phase B RAG distractors — never a source of ground truth.

All of it is produced by the `build_dataset` package (`src/build_dataset/`), in
four deterministic steps.

---

## What was done, step by step

### 1. Cleaning — `clean`  → `data/clean/*.md` (13 files)
Raw hospital PDFs are extracted with PyMuPDF and de-boilerplated: letterheads,
digital-signature/PKI blocks, seals, Sinhala/Tamil subtitles, page footers
(`Page X of Y`, `SGH-… | Rev`) and sub-6pt seal fragments are stripped by
font/pattern. Header vs. body is split on the `Ref: SGH-…` anchor; tables are
reconstructed into GFM tables by clustering lines into rows and columns.
**Output:** one clean `.md` per doc with YAML front-matter + headings. Verified no
boilerplate residue and key facts intact (e.g. cataract 90,000–140,000 LKR/eye,
CABG 2.5M–4M, OPD ~1,200/day).

### 2. Segmentation — `segment`  → `data/interim/sections.jsonl`
Each clean doc is broken into **logical sections** by heading/topic, tagging each
as prose or table. These sections are the units Q&A pairs are drawn from.

### 3. Q&A creation — `generate`  → `data/qa/pool.jsonl` (**433 pairs**)
Grounded pairs are built three ways, each carrying a **verbatim `evidence` quote**
from its source section plus `doc_id`, `section`, `answer_type`, and provenance:
- **FAQ docs** → each `### question` + its answer prose becomes a pair.
- **Table docs** (pricing/OPD/lab/insurance/pharmacy) → row templates
  ("cost of X?", "which days is clinic Y?") filled from real cells.
- **Hand-authored** (`src/authored_pairs.py`, 155 pairs) → fine-grained facts.

Result: **278 auto + 155 authored = 433 pairs**, `0` flagged for missing figures.
A `verify_sheet.md` review checklist is emitted alongside.

### 4. Verification + split — `split`  → `train.jsonl` / `test.jsonl` / `MANIFEST.md`
**Verification gate** (all 433 checked against their clean source doc): evidence
tokens must be 100% present in the source, every answer figure must trace to both
the source **and** the evidence, with borderline paraphrases read individually.
**All 433 passed → `verified: true`; 0 failed.** `split` then refuses to run on
anything but verified pairs, drops same-fact near-duplicates (Jaccard ≥ 0.8 on
*both* question and answer), and writes a stratified, seed-42 split.

---

## The frozen Phase A deliverable

| Artifact | Count | Notes |
|---|---|---|
| `data/clean/*.md` | 13 | de-boilerplated source docs |
| `data/qa/pool.jsonl` | 433 | all `verified: true`, evidence-backed |
| `data/qa/train.jsonl` | 300 | OpenAI-FT chat format + metadata |
| `data/qa/test.jsonl` | **100** | the **locked** eval set, judge-ready |
| `data/qa/MANIFEST.md` | — | *"Frozen split of the human-verified pool"* |

- **Test set:** all 13 docs represented; mix = factual 39 · numeric 37 · procedural 24.
- **Integrity:** 0 train/test overlap; no same-fact leakage across the split; seed 42;
  33 near-duplicates dropped.
- **Freeze:** recorded via SHA-256 hashes in `MANIFEST.md` (on-disk hashes match).

## Reproduce from scratch
```bash
uv sync                            # creates .venv from pyproject.toml + uv.lock
cd src
uv run python -m build_dataset clean      # PDFs  -> data/clean/*.md
uv run python -m build_dataset segment    # clean -> data/interim/sections.jsonl
uv run python -m build_dataset generate   # -> data/qa/pool.jsonl (+ verify_sheet.md)
uv run python -m build_dataset split      # verified pool -> train/test + MANIFEST.md
```

> **Integrity rule (Rule #1):** train and test stay permanently separate — the 100
> test questions are never seen during fine-tuning. Ground truth is the source
> document, never the drafted answer.
