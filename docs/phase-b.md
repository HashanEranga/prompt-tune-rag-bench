# Phase B — The Three Contenders (NOT STARTED)

Answer the **100 locked test questions** (`data/qa/test.jsonl`) three ways, under
**identical conditions**, logging `question_id · method · model · answer · latency · cost`
for every answer so the judge (Phase C) can score them and the cost/speed columns
feed the final verdict.

> **Inputs frozen in Phase A:** `data/qa/test.jsonl` (100 eval Qs) and
> `data/qa/train.jsonl` (300 fine-tuning pairs). Never let a test question leak
> into training — Integrity Rule #1.

---

## Step 1 · Prompting — baseline  (`src/1_prompting.py`)
- [ ] Ask all 100 test questions **directly**, no documents as context.
- [ ] Run ≥2 frontier models + ≥2 small/open models (e.g. via Ollama).
- [ ] Log every answer + latency + cost.
- **Expectation:** should underperform on Serendib-specific facts (no model has seen them).
- **Output:** all answers logged → `results/answers.jsonl`.

## Step 2 · Fine-tuning — small → medium → large  (`src/2_finetune.py`)
- [ ] Fine-tune on `train.jsonl` (learn *behaviour/format*, not new facts).
- [ ] Train three sizes; test each **only** on the 100 unseen questions.
- [ ] Record **training time + cost per size** (goes into the conclusions).
- **Tools:** OpenAI FT API, or LoRA/PEFT on an open size-ladder.
- **Output:** three models' scored answers + a training-cost table.

## Step 3 · RAG — retrieve, then answer  (`src/3_rag.py`)
- [ ] Index the 13 hospital docs (`data/clean/`); optionally add `data/raw/medical/` as distractors.
- [ ] Retrieve top-k section(s) per question → inject as context → answer.
- [ ] **One fixed config** across all 100 (no per-question tuning).
- **Tools:** sentence-transformers / embeddings → FAISS or Chroma → an LLM.
- **Output:** retrieve-then-answer pipeline + all scored answers.

---

## Phase B deliverable
- [ ] `results/answers.jsonl` — every producer's answer to all 100 Qs, with latency + cost.
- [ ] Documented fine-tuning **time + cost per size**.
- [ ] A working, fixed-config RAG pipeline.

_Update the checkboxes and fill in real numbers/paths as each step completes._
