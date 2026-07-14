# Phase D — Retrieval Engineering (⬜ NOT STARTED · PROPOSED, NOT BUILT)

> **Nothing in this document has been executed.** No code exists for it, no index has been built,
> no answer has been produced and no money has been spent. Every *measured* number below comes from
> files already on disk (`results/master_table.csv`, `results/scores.csv`, `results/answers.jsonl`,
> `data/interim/sections.jsonl`). Every *projected* number is labelled as a projection.
>
> Phase C answered the project's question. Phase D is what the answer points at next.

---

## Why this phase exists

Phase C scored all 1,000 answers and RAG won decisively:

| Producer | Faithfulness | Accuracy | Completeness | Clarity | **Weighted** | Safety flags |
|---|---|---|---|---|---|---|
| `rag-gpt` | 4.58 | 4.03 | 3.53 | 4.97 | **4.244** | 3% |
| `rag-llama3.1-8b` | 4.15 | 3.83 | 3.39 | 4.82 | **3.969** | 5% |
| `ft-llama3.1-8b` | 2.14 | 2.01 | 2.96 | 4.78 | 2.529 | 12% |
| `prompt-llama3.1-8b` | 1.39 | 1.35 | 2.60 | 4.46 | 1.927 | **26%** |

The TRIAD — one model, three methods, one judge — reads **1.927 → 2.529 → 3.969**. Retrieval
roughly triples faithfulness. Fine-tuning improves format, speed and safety, but cannot install a
fact it was never given. That is the finding the project was built to produce, and it is now
evidence, not assertion.

**But the accuracy column is capped by the retriever, not by the models.** Splitting the RAG scores
by what the retriever actually put in front of the model — a cut only this pipeline can make,
because every RAG answer logged which chunks it retrieved:

| What reached the model | n | Faithfulness | **Accuracy** | **Completeness** |
|---|---|---|---|---|
| The gold section | 32 | 4.66 | **4.66** | **4.84** |
| Right doc, wrong section | 38 | 4.76 | 4.32 | 3.82 |
| **Neither** | **30** | 4.27 | **3.00** | **1.77** |

*(`rag-gpt`; `rag-llama3.1-8b` shows the same shape — 4.56 / 4.16 / 2.63 on accuracy.)*

**Read that table carefully, because it overturns the obvious conclusion.** When the right section
is retrieved, the models are already near-perfect: accuracy **4.66**, completeness **4.84**. The
generator is not ignoring good context. It is being handed bad context on 30 of 100 questions, and
those 30 drag the averages down. **The bottleneck is retrieval, and only retrieval.**

### The headroom, computed from questions we already got right

Take each RAG producer's scores **on the 32 questions where retrieval already succeeded**, and ask
what the producer would have scored if that had happened on all 100:

| | now | ceiling if recall were 100% | headroom |
|---|---|---|---|
| `rag-gpt` | 4.244 | **4.728** | **+0.48** |
| `rag-llama3.1-8b` | 3.969 | **4.569** | **+0.60** |

> ⚠️ These are **ceilings, not promises.** They assume perfect retrieval, which no retriever
> achieves. They are computed from real scores on real answers — but on the *easy* 32, which are
> plausibly easier for reasons beyond retrieval. Treat them as an upper bound on the prize, not a
> forecast.

The second row is the interesting one. **A fixed retriever would put the free, local 8B model
(4.569) above today's paid frontier RAG (4.244).** For scale: upgrading the *model* from
`llama3.1:8b` to `gpt-4.1` bought **+0.275**. Fixing the *retriever* is worth roughly twice that,
on hardware that costs nothing to run.

---

## Two root causes, both found in the data

### 1. `nomic-embed-text` is running without its task prefixes

`src/contenders/rag.py` defines a single `_embed()` (line 93) and calls it for **both** sides of
the retrieval problem:

- the corpus, in `build_index()` — line 118
- the query, in `Retriever.search()` — line 152

Neither call adds a prefix. But **Nomic Embed is trained for *asymmetric* retrieval** and expects
`search_document: ` on corpus text and `search_query: ` on the question. Without them, questions
and documents are embedded as if they were the same kind of object, and every lookup in the project
has been running in a degraded mode.

This is the fourth silent bug in this project, and it has the family resemblance: **it does not
crash, does not warn, and returns perfectly plausible results.** It just returns slightly wrong
ones, 100 times, and the damage only becomes visible three phases downstream in a completeness
score.

> **⚠️ This bug is deliberately NOT being fixed in `rag.py`.** The v1 retriever is the *control* for
> the very comparison Phase D proposes. Silently repairing it now would destroy the frozen baseline
> that makes the improvement measurable, and would quietly invalidate 200 already-judged answers.
> It is recorded here so it is never mistaken for an oversight. **Fix it in v2, alongside v1.**

### 2. Table rows are averaged into a single embedding vector

The 30 misses are not spread evenly across the 13 documents. **18 of them fall in the two most
table-dense docs:**

| Doc | Sections | With tables | Table rows | Questions missed |
|---|---|---|---|---|
| `SGH-SU-001` — surgical pricing | **3** | 1 | 15 | **9 / 12** |
| `SGH-LAB-001` — lab catalog | 5 | 4 | 24 | **9 / 13** |
| `SGH-FAQ-002…006` — FAQs, **no tables** | 14–16 each | 0 | 0 | **1 / 28** *(all five combined)* |

**The contrast is the whole diagnosis.** The two table-heavy docs miss **18 of 25** questions
(72%). The five table-free FAQ docs — many small, single-topic sections — miss **1 of 28** (4%).
Same embedder, same index, same query, same distractors. **The only variable is chunk shape.**

The entire surgical pricing document is **three sections**. One of them is a 1,204-character table
holding **15 different procedures**:

```
ESTIMATED PROCEDURE COSTS
Procedure | Specialty | Estimated Cost (LKR) | Typical Stay
Appendectomy, Laparoscopic  | General Surgery | 180,000 to 240,000 | 2 to 3 days
Cholecystectomy, Laparoscopic | General Surgery | 200,000 to 280,000 | 1 to 2 days
Hernia Repair, Inguinal (mesh) | General Surgery | 150,000 to 220,000 | 1 to 2 days
… 12 more rows …
```

**That whole table is one embedding vector.** Every pricing question — *"How much does a
laparoscopic cholecystectomy cost?"* — competes against all 15 procedures averaged together, so the
signal for any single row is diluted ~15×. Meanwhile the FAQ documents, which are 14–16 small
single-topic sections with no tables, barely fail at all.

**This is a chunking failure, and it is specific to tables.** The Phase A segmenter split by
heading, which is correct for prose and wrong for a price list.

### What is *not* the problem: the distractors

Worth stating, because it was the thing we designed for. Only **3% of retrieved chunks** (16 of
600) were CDC/ECDC distractors. The retriever is **not** fooled by the 704 chunks of unrelated
medical noise — it finds the right hospital document 70% of the time. It simply cannot resolve the
right *section inside* it. The haystack was never the hard part; the needle was.

---

## The proposed fix

### Integrity first — nothing already measured gets overwritten

| Rule | Why |
|---|---|
| `data/index/corpus.faiss` + `results/rag_config.json` stay **frozen** | v2 builds into `data/index/v2/`, freezes `results/rag2_config.json` |
| The `rag-*` producers, their 200 answers and their 200 scores are **untouched** | They are the control. A before/after needs a before. |
| Phase C's rubric is **not modified** | The same frozen judge scores the new answers. Changing the rubric would invalidate the comparison it is being used to make. |
| Ship the fix as **new producers** (`rag2-*`), not as an edit to the old ones | The registry becomes 12 producers = 1,200 answers, and `assert_complete()` enforces that automatically — it is already written against `PRODUCERS`. |

> **Disclose this honestly in the write-up:** `rag2` changes **four things at once**. It therefore
> measures *the whole retrieval stack*, not one variable. The free benchmark below reports
> recall@3 **and** recall@5 for both versions, which is what separates "better ranking" from
> "simply more slots" — report that breakdown rather than claiming a single cause.

### The four changes

| Fix | What it does | Targets |
|---|---|---|
| **Nomic task prefixes** | `search_document: ` on every corpus chunk at index time; `search_query: ` on the question at search time. | Every lookup, globally. Free. |
| **Table row-level chunking** | Alongside each section chunk, emit **one chunk per table row**, self-describing: `"{heading}\n{col}: {val} \| {col}: {val} …"`. Row chunks inherit the parent `doc_id` + `section`, so a row hit still counts as a gold-section hit in the eval and in `judge aggregate`. ~99 new chunks (126 → ~225 hospital chunks). | The 18 of 30 misses in `SGH-SU-001` + `SGH-LAB-001`. |
| **`top_k` 3 → 5** | More shots on goal, at the cost of a few hundred input tokens per answer. | Recall generally. |
| **Hybrid BM25 + dense** | A ~40-line pure-Python BM25 (no new dependency) over the same corpus, fused with the dense ranking by **Reciprocal Rank Fusion** (`score = Σ 1/(60 + rank)`). RRF needs no score normalisation and no tuning. | Exact-token questions the embedder misses — e.g. *"What is the Surgical Helpline number?"*, currently a miss. |

> **Do not tune any of this against the 100 test questions.** RRF's `k=60` is the standard default
> and stays there; `top_k=5` is chosen up front. Tuning hyperparameters against the test set is
> **Rule #1 in a subtler costume** — the split stops protecting you the moment you optimise against it.

### The tool that makes this cheap — a free retrieval benchmark

**The most important piece of Phase D, and the one with no LLM in it at all.**

Today, retrieval quality is only observable *after* 1,000 judge calls. `retrieval-eval` would make
it observable in seconds, for **$0**: for each of the 100 test questions, retrieve with v1 and v2
and score against the gold `(doc_id, section)` **already sitting in `test.jsonl`**.

- **recall@k** for k = 1, 3, 5, 10 — the headline metric
- **MRR**, plus the same 3-tier split (gold / right-doc-wrong-section / neither) that
  `judge aggregate` prints, so the two speak the same language
- a per-document breakdown, so the `SGH-SU-001` / `SGH-LAB-001` collapse is visible directly

The only cost is local Ollama embeddings. **No answers, no judge, no money.** Iterate here until
recall is good, and *only then* spend anything.

### Files Phase D would create or touch

| File | Change |
|---|---|
| `src/contenders/retrieval_eval.py` | **New.** The free benchmark above. CLI: `contenders retrieval-eval`. |
| `src/contenders/rag2.py` | **New.** The engineered retriever. Mirrors `rag.py` (`build_index` / `Retriever` / `cmd_rag2`) and reuses its `_section_text()` and `_distractor_chunks()` rather than re-implementing them. |
| `src/contenders/models.py` | Add `rag2-llama3.1-8b` and `rag2-gpt` (both `method = RAG`, so they flow into the existing RAG diagnosis cut with no other change). Add the v1↔v2 pairs. |
| `src/contenders/__main__.py` | Wire `rag2` and `retrieval-eval` exactly like `rag`. |
| `src/judge/aggregate.py` | One new cut: v1 vs v2 retrieval. The 3-tier diagnosis already groups by producer, so `rag2-*` appears in it for free. |
| `.idea/runConfigurations/` | `16_retrieval_eval` · `17_rag2_build_index` · `18_rag2`, continuing the ladder. |
| `src/contenders/rag.py` | **UNCHANGED — deliberately.** See the warning above. |

**Judging needs no new code at all.** `judge submit` iterates `PRODUCERS` and skips what is already
scored, so it would pick up exactly the 200 new answers and nothing else.

---

## Projected cost

| | |
|---|---|
| `retrieval-eval` + `rag2 --build-index` | **$0.00** — local Ollama + FAISS |
| `rag2-llama3.1-8b` — 100 answers | **$0.00** — local |
| `rag2-gpt` — 100 answers | ~$0.15 |
| Judge — 200 new answers, batched | ~$1.20 |
| **Total** | **≈ $1.35** |

For context: the project has spent **$4.39** (Phase B) plus the Phase C judge to date. Phase D is
the cheapest phase in the project, and it targets the largest remaining gap.

---

## The free gate — and the discipline it enforces

**`retrieval-eval` runs first, costs $0, and decides whether anything else happens.**

| | |
|---|---|
| Baseline to beat | **gold-section recall@3 = 32%**, 3-tier split **32 / 38 / 30** |
| Target | **recall@5 ≥ 60%**, with `SGH-SU-001` and `SGH-LAB-001` visibly recovering — they are 18 of the 30 misses |
| **If it does not improve** | **STOP.** Do not build the index, do not answer, do not judge. |

That last row is the whole point. A free benchmark exists so that a bad idea costs **nothing** to
discover. Phase B learned this the expensive way — 20 minutes of retries hiding a `TypeError`, and
100 answers produced by a model that wasn't the one we fine-tuned.

## Success criteria, if it is executed

1. **The "neither" bucket shrinks from 30.** This is the direct measure; everything else is a
   consequence of it.
2. **`rag2-llama3.1-8b` clears 4.244** — today's `rag-gpt` score. If a free, local, self-hosted 8B
   with a properly engineered retriever beats a paid frontier model with a naive one, that is the
   headline of the entire report, and the cost column is what proves it.
3. **Accuracy and completeness move most; faithfulness moves least.** Those were the two dimensions
   the misses were suppressing (accuracy 3.00 and completeness 1.77 in the miss bucket, against
   4.66 and 4.84 in the hit bucket). Faithfulness was never the broken one — it held up even on
   misses, because the model correctly refused to invent. **If faithfulness jumps but accuracy
   doesn't, the measurement is wrong, not the retriever.**

---

## The finding Phase D is really chasing

Phase C proved *RAG beats fine-tuning beats prompting*. That is the assignment, and it is done.

Phase D asks a sharper question that the data has already half-answered:

> **Is it cheaper to buy a better model, or to build a better retriever?**

Going `llama3.1:8b` → `gpt-4.1` inside the RAG pipeline bought **+0.275 weighted**, and costs money
on every single answer, forever. Fixing the retriever is worth up to **+0.60** on the free local
model, costs **$0.00 per answer**, and the fix is: *use the embedding model's documented prefixes,
and don't put fifteen prices in one vector.*

If that holds, the conclusion is not *"use RAG"* — it is:

> **Most of the accuracy people try to buy with a bigger model is sitting unclaimed in their
> retrieval layer.**
