# Phase B — The Three Contenders ✅ **COMPLETE**

Answer the **100 locked test questions** (`data/qa/test.jsonl`) three ways, under
**identical conditions**, logging `question_id · method · model · answer · latency · cost`
for every answer so the judge (Phase C) can score them and the cost/speed columns
feed the final verdict.

**All three steps have run.** `results/answers.jsonl` holds **1,000 answers — 10 producers ×
100 questions, 0 errors, 0 duplicates, one model per producer** (`assert_complete()` enforces
it). Total spend: **$0.3946 inference + $4.00 one-off training = $4.39.**

Phase C now scores every one of those answers → [phase-c.md](phase-c.md).

> **Inputs frozen in Phase A:** `data/qa/test.jsonl` (100 eval Qs) and
> `data/qa/train.jsonl` (300 fine-tuning pairs). Never let a test question leak
> into training — Integrity Rule #1, asserted in code before every training run.

---

## The answer-producer roster

**1,000 answers = 100 questions × 10 producers.** Declared in `src/contenders/models.py`,
which is the single source of truth for ids, providers and prices.

| # | Key | Method | Model | Cost |
|---|---|---|---|---|
| 1 | `prompt-gpt` | prompting · frontier | `gpt-4.1` | paid |
| 2 | `prompt-gemini` | prompting · frontier | `gemini-3.5-flash` | paid |
| 3 | `prompt-llama3.1-8b` | prompting · small open | `llama3.1:8b` (Ollama) | free |
| 4 | `prompt-qwen3.5-9b` | prompting · small open | `qwen3.5:9b` (Ollama) | free |
| 5–7 | `ft-local-{small,medium,large}` | fine-tuned · QLoRA | `Qwen2.5-{0.5B,1.5B,3B}` | free |
| 8 | `ft-llama3.1-8b` | fine-tuned · **Together** | `Llama-3.1-8B-Instruct` | **$4 once**, then free |
| 9 | `rag-llama3.1-8b` | RAG | `llama3.1:8b` + docs | free |
| 10 | `rag-gpt` | RAG | `gpt-4.1` + docs | paid |

### 🏆 The TRIAD — the report's headline

Producers **3, 8 and 9 are the same model.** All three of the project's "ways", measured on
one set of weights, scored by one judge. A score difference can therefore only be the
**method**:

```
llama3.1:8b  prompted    → X   (prompt-llama3.1-8b)
llama3.1:8b  fine-tuned  → Y   (ft-llama3.1-8b)
llama3.1:8b  + documents → Z   (rag-llama3.1-8b)
```

Without producer 8, *"fine-tuning vs RAG"* compares a **1.5B Qwen** against an **8B Llama**
and calls the difference "method" — which is not a comparison at all. Asserted in `models.py`
(`TRIAD`), alongside the two `AB_PAIRS`.

> **Honest caveat:** Ollama serves `llama3.1:8b` at **Q4_0**; the fine-tuned leg is **4-bit
> NF4** over the fp16 base. Same weights and lineage, different quantisation — a strong
> control, not a perfect one. Say so in the write-up rather than overclaim.

> ### 🔴 It was 12, then 9, now 10. The managed ladder died and came back different.
> Three `ft-openai-*` producers were removed on **2026-07-14**: OpenAI's fine-tuning API
> returns **`403 training_not_available`** — *"OpenAI is winding down the fine-tuning platform
> and your organization is no longer able to create new fine-tuning training jobs."* Revoked at
> the **organisation** level (this account has **zero** prior FT jobs, so it is permanently the
> wrong side of the 2026-05-07 cutoff). No base model, retry or key gets around it.
>
> Every alternative was checked: **Gemini** dropped tuning from the API entirely (Vertex only),
> **Groq**'s is closed beta, **OpenRouter** has no training stack. **Together** and **Fireworks**
> still offer it — and *why* is the finding:
>
> > **Every provider still offering fine-tuning tunes OPEN-WEIGHT models. Every provider
> > withdrawing it owns closed frontier weights.**
>
> The labs' own pitch is now *"don't fine-tune — use context and retrieval"*, which is
> **precisely this project's thesis**. Our Step 2 result (fluent, confidently invented answers)
> reproduces, at 0.5B on a laptop, the reasoning behind an industry-wide product retreat.
>
> The replacement is deliberately **not** a like-for-like managed size ladder. Re-training a
> Qwen rung on Together would pay $4 to reproduce what the laptop already did free. Instead the
> money bought the one producer the hardware **cannot** make: `ft-llama3.1-8b`.

> ⛔ **Claude is NOT on this list, deliberately.** It is the Phase C judge, and
> Integrity Rule #3 requires the judge be a different model family than every
> answerer. Using Claude here would invalidate the comparison.

### The two controlled A/B pairs — the report's headline

Producers 3 & 11 are the **same model**, and 1 & 12 are the **same model**, differing
only in whether documents were retrieved. Any lift is therefore *purely retrieval*:

```
llama3.1:8b  no docs → X ┐ lift = Y−X   (small model)
llama3.1:8b  + docs  → Y ┘
gpt-4.1      no docs → P ┐ lift = Q−P   (frontier model)
gpt-4.1      + docs  → Q ┘
```
Comparing the two lifts answers *"does retrieval help a small model more than a big
one?"* — which no single-RAG setup can. Asserted in `models.py` (`AB_PAIRS`).

### Local models — nothing to download ✅

`ollama serve` is all that's needed. Benchmarked on the RTX 4060 (8 GB):

| Model | Placement | Measured | Role |
|---|---|---|---|
| `llama3.1:8b` | 100% GPU (6.3 GB @ 8k ctx) | 3.9 s · 18 tok/s | producers 3 + 11 |
| `qwen3.5:9b` | 72% GPU / 28% CPU | 3.6 s · 21 tok/s | producer 4 |
| `nomic-embed-text` | — | 768-dim | RAG embeddings |

The QLoRA bases (`Qwen/Qwen2.5-{0.5B,1.5B,3B}-Instruct`, ~10 GB) download themselves
from Hugging Face on first training run and are **not gated** — `HF_TOKEN` isn't needed.

> ### 🔴 `think=False` is mandatory on every Ollama call
> Qwen3.x reasons by default. On this machine, one question took **>9 minutes and never
> completed** with thinking on, versus **3.6 s** with it off. It is also a *data-integrity*
> fix: reasoning traces leak into the answer text the judge scores and inflate the latency
> and token-count deliverables. Enforced unconditionally in `clients.py`.

---

## Step 0 · Price it before you run it — `estimate`

**Zero API calls.** Counts the real tokens in the real questions with `tiktoken`, multiplies
by the `PRICING` table, and charges every answer the full 300-token output cap — so the
figure is a **ceiling**, not a hopeful average.

Projected (⚠️ **prices still unverified** — confirm `PRICING` in `models.py` against the
source URLs it prints, then set `PRICES_VERIFIED = True`):

| | ceiling |
|---|---|
| Phase B — inference | $0.83 |
| Phase B — FT training (one-off) | $4.00 — Together's per-job floor, paid once |
| **Phase B total** (1,000 answers) | **$4.83** |
| Phase C — judge | $4.21 |
| **Project total** | **$9.04** |

> **Every paid answer in this project is a prompting or RAG answer.** The fine-tuning step —
> all four rungs, 400 answers — costs **$0.00 to run**. The only fine-tuning charge is the
> single $4.00 Together training job, and even that produces an artefact we keep.
>
> ⚠️ **The $4.00 is a *floor*, not a price.** Together bills $0.48/1M LoRA training tokens; our
> 77,559 trained tokens are worth **$0.04**. We pay the minimum — a **~100× markup on what we
> actually consume**. That belongs in the cost write-up.

---

## Step 1 · Prompting — baseline  ✅ **COMPLETE**  (`contenders prompt`)
- [x] Ask all 100 test questions **directly**, no documents as context.
- [x] 2 frontier + 2 small/open models, logged with latency + cost.
- [x] **400/400 answers · 0 errors · 0 duplicates · one model per producer.**

| Producer | Model | Answers | Spend |
|---|---|---|---|
| `prompt-gpt` | `gpt-4.1` | 100/100 | $0.1122 |
| `prompt-gemini` | `gemini-3.5-flash` | 100/100 | $0.1643 |
| `prompt-llama3.1-8b` | `llama3.1:8b` | 100/100 | $0.00 |
| `prompt-qwen3.5-9b` | `qwen3.5:9b` | 100/100 | $0.00 |
| | | **400** | **$0.2765** |

**The baseline failed as designed.** Asked the AEC registration fee (truth: **1,000 LKR**),
all four got it wrong and each wrong differently: llama said *"there is no fee"*, Qwen said
*"Rs. 500"*, Gemini said *"LKR 1,500"* — and both Qwen and Gemini **hallucinated what "AEC"
stands for**. These facts exist in no model's weights, which is exactly the point.

> ### ⚠️ Two model traps hit here — both now handled in code
> **1. `gemini-2.5-flash` is retired.** It still appears in `models.list()` but calling it
> returns `404 — no longer available to new users`. Switched to **`gemini-3.5-flash`**
> ($1.50/$9.00 per 1M, verified). The 21 answers produced by the old model were purged with
> `reset --producer prompt-gemini` — mixing two models under one producer key would have let
> the judge score the blend as a single contender.
>
> **2. Gemini 3.x reasons by default**, and Google bills thinking tokens at the *output* rate.
> `clients.py` now sets `thinking_config=ThinkingConfig(thinking_budget=0)` — the same fix as
> `think=False` on Ollama, for the same two reasons: unbounded cost, and reasoning traces
> leaking into the answer text the judge scores.

## Step 2a · Fine-tuning — the self-hosted ladder  ✅ **COMPLETE**  (`finetune --backend local`)
- [x] 4-bit QLoRA on the 4060; records GPU-hours + peak VRAM, cost $0.
- [x] Tested **only** on the 100 unseen questions. **300/300 answers · 0 errors · 0 duplicates.**
- [x] **Rule #1 asserted in code** before every training run — 300 train / 100 test, zero overlap.
- [~] `--backend openai` — **withdrawn**, see the roster note above. The flag now exits with the
      403 rather than silently doing nothing.

| Rung | Base | Train time | Peak VRAM | Cost |
|---|---|---|---|---|
| `ft-local-small` | `Qwen2.5-0.5B-Instruct` | 2.3 min | 1.69 GB | $0.00 |
| `ft-local-medium` | `Qwen2.5-1.5B-Instruct` | 3.7 min | 3.54 GB | $0.00 |
| `ft-local-large` | `Qwen2.5-3B-Instruct` | 7.2 min | 5.24 GB | $0.00 |

**The whole ladder trained inside 8 GB with room to spare** — the 3B rung peaked at 5.24 GB.
Total training: 13.2 minutes of GPU time, $0.00.

- **Expectation held:** fine-tuning taught *format/behaviour*, not facts. Asked the AEC
  registration fee (truth: **1,000 LKR**), the tuned 0.5B produced a fluent, well-shaped,
  **entirely invented** *"$500 to $1,000 per day"* — wrong currency, wrong unit, wrong number.
  Better-formed answers that are still factually wrong is exactly the predicted result, and
  it is what Step 3 exists to fix.
- ⚠️ Don't train while Ollama holds a model in VRAM (shared 8 GB) — `finetune --backend local`
  detects this and prints the `ollama stop` command.
- ⚠️ **Never run two local commands at once.** Both processes load the ladder into the same
  8 GB card; the loser gets `ValueError: Some modules are dispatched on the CPU or the disk`
  (bitsandbytes refuses to offload a 4-bit model), and the winner's latency numbers are
  contaminated by the contention. Latency is a deliverable — run local steps one at a time.

> ### 🔴 Three bugs hit here — all now fixed in `clients.py`
> **1. transformers v5 broke local inference.** `apply_chat_template()` returns a
> `BatchEncoding` dict in v5, not a tensor. The old code passed it straight into `generate()`,
> which died at `inputs_tensor.shape[0]` inside a bare `raise AttributeError` — producing
> **100 identical, empty `AttributeError:` rows** with the traceback discarded. Now uses
> `return_dict=True` + `generate(**enc)`, which also passes `attention_mask` through for the
> first time.
>
> **2. Code bugs were being retried as if transient.** `_is_transient()` returned `True` for
> anything unrecognised, so each of those 100 questions burned 4 attempts × ~14 s of backoff
> before recording the error — ~20 minutes spent hiding the traceback that named the line.
> `AttributeError`/`TypeError`/`NameError`/`ImportError` now fail immediately.
>
> **3. The adapter was served on the wrong base.** `_hf_pipeline` loaded the base in **bf16**,
> not the 4-bit NF4 it was *trained* under — a different model than the one that was tuned,
> and ~6.2 GB for the 3B rung. Now loads with the identical `BitsAndBytesConfig` as training.
>
> Also: the model load happened *inside* the timed section, billing the first question of every
> local producer ~5–7 s of load time. Latency is a deliverable, so the pipeline is now warmed
> before the clock starts.

---

## Step 2b · The rung this GPU cannot train  ✅ **COMPLETE**  (`finetune --backend together`)

- [x] Trained `Llama-3.1-8B-Instruct` LoRA on Together — **$4.00, once**. Job `ft-7d574f7b-9565`.
- [x] **Downloaded the adapter** → `models/ft-llama3.1-8b/`, answered **locally** for $0.
- [x] **100/100 answers · 0 errors.** Mean latency **1.77 s**, served at 4-bit NF4 on the 4060.
- [x] Completes the **TRIAD** (see roster above) — the report's headline.

### Why pay at all, when Step 2a was free?

Because of one asymmetry we measured directly:

| | Llama-3.1-8B | the RTX 4060 (8 GB) |
|---|---|---|
| **Fine-tuning** (QLoRA) | ~15 GB | ❌ **impossible** |
| **Inference** (4-bit) | ~5.5 GB | ✅ *already runs `llama3.1:8b` at 6.3 GB* |

Extrapolating our own peaks (1.69 GB @ 0.5B → 3.54 @ 1.5B → 5.24 @ 3B) puts an 8B **well past
8 GB to train** — but serving one at 4-bit is something this box does every day.

So **Together is a rented trainer, not a host.** We rent a big GPU for the ten minutes we
cannot do ourselves, **download the LoRA adapter**, and keep it. No dedicated endpoint, no
hourly meter ($5.49/hr H100), nothing to tear down. `clients._hf_pipeline` serves the
downloaded adapter on a 4-bit base through the *same code path* as the Qwen ladder.

**The $4 is paid once and buys a file.** Re-answer, re-evaluate and re-use it forever, free.

### What we deliberately did NOT buy

Re-training a **Qwen rung** on Together (mirroring `ft-local-medium`) was the obvious move and
the wrong one: same base, same data, same method means **paying $4 to reproduce what the laptop
did free in 3.7 minutes**. The money went to the one producer the hardware genuinely cannot
make — which is also the one that closes the triad.

- **Hyperparameters are identical to Step 2a** (`r=16, alpha=32, dropout=0.05, 3 epochs,
  lr=2e-4`). That is what makes managed-vs-self-hosted a *controlled* comparison rather than
  two unrelated runs.
- ⚠️ **The base is gated.** `meta-llama/Meta-Llama-3.1-8B-Instruct` returns `403
  GatedRepoError` without a granted Meta licence, so `models.py` points at
  **`NousResearch/Meta-Llama-3.1-8B-Instruct`** — an ungated mirror of the identical weights
  (verified: `llama`, 32 layers, 4096 hidden, 128256 vocab, GQA 8 KV heads). The adapter fits
  either; swap back if Meta grants you the licence.
- ⚠️ **Stop Ollama before answering** — the 8B 4-bit base needs ~5.5 GB of the same 8 GB.

## Step 3 · RAG — retrieve, then answer  ✅ **COMPLETE**  (`contenders rag`)
- [x] `--build-index` — embedded the 126 Phase A sections **+ the 15 medical PDFs as distractors**
      → FAISS `IndexFlatIP` (L2-normalised = exact cosine). Free and fully local.
- [x] Retrieved top-3 → injected → answered. **One fixed config** across all 100, frozen to
      `results/rag_config.json`. **200/200 answers · 0 errors.**
- [x] Every answer logs *which* chunks were retrieved and whether they were distractors —
      this is what lets Phase C tell **"the retriever missed"** from **"the model ignored
      good context"**. `judge aggregate` cuts the table on exactly this.
- **Premise validated:** given the real retrieved section, `llama3.1:8b` answers the
  AEC fee question **"1,000 LKR"** — exactly the gold answer it got wrong without documents.

### The frozen retrieval config (`results/rag_config.json`)

| | |
|---|---|
| Embeddings | `nomic-embed-text` (Ollama), 768-dim |
| Index | `faiss.IndexFlatIP`, L2-normalised = exact cosine |
| Chunks | **830 total** — 126 hospital + **704 distractor** |
| `top_k` | **3**, identical for all 100 questions |

**704 of 830 chunks are noise.** The right hospital section has to win against 5.6× its own
volume in unrelated CDC/ECDC medical text — a retrieval test that can actually fail, not a toy
index where nothing competes with the answer.

> ⚠️ **`data/raw/medical/` is NOT in git** (25 MB of public CDC/ECDC PDFs, gitignored to keep
> the repo lean). It's only needed for Step 3's distractors. Without it the index still builds
> from the 13 hospital docs alone — but retrieval will look better than it deserves to, because
> nothing competes with the right answer. Ground truth still comes **only** from the hospital docs.

---

## Execution ladder — free things first, money last

```bash
cd src
uv run python -m contenders estimate                    # $0 — approve the projected spend
uv run python -m contenders check                       # cents — do model ids + keys work?
uv run python -m contenders rag --build-index           # $0 — local FAISS + Ollama
uv run python -m contenders prompt --local-only --limit 3   # $0 — full pipeline smoke test
uv run python -m contenders prompt --limit 3            # pennies — last stop before the run
uv run python -m contenders prompt                      # Step 1
uv run python -m contenders finetune --backend local    # Step 2a — free (one at a time!)
uv run python -m contenders finetune --backend together --train-only    # Step 2b — $4, ONCE
ollama stop llama3.1:8b                                 # free the card for the 8B
uv run python -m contenders finetune --backend together --answer-only   # Step 2b — free, local
uv run python -m contenders rag                         # Step 3
uv run python -m contenders status                      # progress + spend
```
Every answering command **resumes**: rows already in `results/answers.jsonl` are skipped, so a
crash or Ctrl-C never pays for the same call twice. `--limit N` tries any step on N questions first.

**`--train-only` and `--answer-only` are split for Step 2b on purpose.** Training is the single
irreversible $4 step; answering is free, local and re-runnable. Keeping them on separate commands
(and separate PyCharm buttons) means the paid one cannot be triggered by reflex while iterating.

### Running it from PyCharm

`.idea/runConfigurations/` ships the same ladder as clickable configs — **`0_estimate` →
`9_status`** — and they are now **committed** (`.gitignore` ignores IDE *state* but not the run
configs: the pipeline is a deliverable). The dead `7 - STEP 2b finetune OPENAI` button is gone,
replaced by:

| Config | Runs | Cost |
|---|---|---|
| `7 - STEP 2b finetune TOGETHER · TRAIN` | `finetune --backend together --train-only` | **$4 · RUN ONCE** |
| `7b - STEP 2b finetune TOGETHER · ANSWER` | `finetune --backend together --answer-only` | free · local GPU |

> ### ⚠️ Use **Run ▸**, not **Debug ▸**
> Under PyCharm's debugger these exit **0** but print an alarming shutdown traceback —
> `AttributeError: module 'pydevd' has no attribute '__file__'` from `pydev_monkey.patch_args`
> racing `multiprocessing.resource_tracker` as the interpreter tears down. Every frame is inside
> `pydev/` or `multiprocessing/`; none is ours. It fires **after** all work has completed and
> been written. It is not a failure.

## Phase B deliverable ✅

- [x] `results/answers.jsonl` — **1,000 answers**, each with latency + cost. (`assert_complete()`
      enforces exactly 100 per producer, zero errored rows.) **1,000/1,000 · 0 errors.**
- [x] `results/finetune_jobs.json` — training **time + cost per rung**, both backends.
- [x] `results/rag_config.json` — the one frozen retrieval config.

### What actually ran — measured from `answers.jsonl`

| Producer | Method | Answers | Spend | Mean latency |
|---|---|---|---|---|
| `prompt-gpt` | prompting | 100 | $0.1122 | 2.27 s |
| `prompt-gemini` | prompting | 100 | $0.1643 | 1.99 s |
| `prompt-llama3.1-8b` | prompting | 100 | $0.00 | 2.43 s |
| `prompt-qwen3.5-9b` | prompting | 100 | $0.00 | 8.68 s |
| `ft-local-small` | fine-tuned | 100 | $0.00 | **0.94 s** |
| `ft-local-medium` | fine-tuned | 100 | $0.00 | 1.23 s |
| `ft-local-large` | fine-tuned | 100 | $0.00 | 1.32 s |
| `ft-llama3.1-8b` | fine-tuned | 100 | $0.00 *(+$4.00 training)* | 1.77 s |
| `rag-llama3.1-8b` | RAG | 100 | $0.00 | 1.20 s |
| `rag-gpt` | RAG | 100 | $0.1181 | 5.98 s |
| | | **1,000** | **$0.3946** | |

### Two findings are already visible — before the judge has scored anything

**1. Fine-tuning won on speed, and it is not close.** The four fine-tuned rungs occupy four of
the five fastest slots on the board; `ft-local-small` answers in **0.94 s**, roughly **9× faster
than `prompt-qwen3.5-9b`** (8.68 s) and 2.4× faster than the same-family `prompt-llama3.1-8b`.
Fine-tuning teaches *how to behave* — the tuned models emit a short, well-shaped answer
immediately instead of preambling their way toward one.

**2. Retrieval is not free, and the bill is latency.** `rag-gpt` costs **5.98 s** against
`prompt-gpt`'s **2.27 s** — the same model, **2.6× slower**, purely because it must embed, search
830 chunks, and read three injected sections before it may answer. That is the price of looking
things up, and it belongs in the final verdict next to whatever quality RAG buys.

Both of these are cost/speed findings that stand *independently of the judge*. Rule #4 exists
precisely because these columns can crown a different winner than the score column does.
