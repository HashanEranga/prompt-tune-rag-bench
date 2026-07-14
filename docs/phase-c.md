# Phase C — The Judge & Conclusions (BUILT · NOT YET RUN)

A **separate AI** scores every Phase B answer against the **source document**
(never the drafted answer), then the scores are aggregated into one master table
and a written verdict.

The `judge` package (`src/judge/`) is written and imports clean. **No judge call has been
made yet** — no API calls, no spend. See *Execution ladder* below.

> **Integrity Rule #3:** the judge must be a **different model family** than the
> answerers. A model judging its own family may favour answers that "sound like
> itself."
>
> **This project's assignment:** answerers are GPT · Gemini · Llama · Qwen
> (see the [Phase B roster](phase-b.md#the-answer-producer-roster)) → the judge is
> **Claude** (`ANTHROPIC_API_KEY`). Claude therefore never produces an answer in
> Phase B. Keep it that way.

**Judge model: `claude-sonnet-5`.** (`models.py` previously declared `claude-sonnet-4-5`, now a
legacy id; the old price row is kept so earlier estimates still reproduce.)

---

## The three guarantees, enforced in code

The rubric is not just a prompt — it is where three of the four rules stop being promises.
All three live in `src/judge/rubric.py`, and the smoke test verifies them.

| Guarantee | How it is enforced |
|---|---|
| **Rule #2 — the document is the authority** | `build_user()` receives the **full cleaned source document** and the candidate answer. It never receives `test.jsonl`'s `answer` field. The gold answer was drafted by an AI and can itself be wrong; scoring against it would compare answer-to-answer, not answer-to-document. |
| **Blind scoring** | No `producer`, `method`, `model` or retrieval trace reaches the judge. It scores an anonymous string, so it cannot favour "the RAG one" for knowing it is the RAG one. |
| **Consistent application** | One module-level prompt, one JSON schema, no per-method branching. `assert_complete()` additionally refuses to aggregate a `scores.csv` that mixes two judge models. |

**Why the full document, not just the tagged section:** the section is ~100 tokens, and an
answer that correctly draws on a *neighbouring* section of the same document would be marked
unfaithful for it — punishing the model for the dataset's own segmentation. The 13 cleaned
documents average 969 tokens; handing over the whole file removes the artefact for pennies.

---

## Step 4 · The Judge  (`src/judge/`)

For each of the 1,000 answers the judge reads **question + real source document +
candidate answer** and returns structured JSON:

| Dimension | Scale | Weight | Notes |
|---|---|---|---|
| Faithfulness to Source | 1–5 | **0.40** | **weighted highest** |
| Accuracy | 1–5 | 0.30 | prices / times / numbers correct? |
| Completeness | 1–5 | 0.20 | everything asked addressed? |
| Clarity | 1–5 | 0.10 | readable for patient/staff? |
| Safety Flag | Yes/No | — | could a wrong answer cause **physical** harm? |
| Reason | text | — | one-line verdict, pointing at the document |

- [x] Same judge, same rubric, same prompt applied to **every** method's answers.
- [x] Structured JSON forced by `output_config.format` — a constrained decode, not a request.
      Every verdict is then re-validated against a Pydantic model on the way out.
- [x] **Weights are stated, not hidden.** A composite score nobody can decompose is not evidence.
- [ ] **Output:** `results/scores.csv` — one row per answer.

### Two conventions worth knowing before you read the scores

**Abstention.** If an answer declines ("the documents don't say"), it invented nothing: it scores
faithfulness 5, accuracy 3, **completeness 1**. The penalty lands on completeness, where it
belongs. This is applied mechanically even when the document *does* contain the answer — keeping
the rule dumb keeps it consistent.

**Safety is narrow on purpose.** The flag means *physical* harm or delayed care — a wrong
emergency number, wrong triage advice, wrong dosage, wrong pre-op fasting instruction. A wrong
*fee* is an accuracy failure, not a safety one. A flag that fires on everything measures nothing.

### Two model traps, handled in `batch.py`

> **1. `temperature=0.0` is a 400 on Sonnet 5.** Phase B pinned every one of its ten answerers at
> `temperature=0.0` for reproducibility, so the reflex is to do the same for the judge. Claude
> Sonnet 5 **rejects non-default sampling parameters** — `temperature`, `top_p` and `top_k` all
> return `400`. The judge therefore sets none of them, and consistency is carried by the rubric
> and the constrained JSON schema instead.
>
> **2. Thinking is disabled by default.** Sonnet 5 runs adaptive thinking when `thinking` is
> omitted, which would roughly double the run's output tokens (**$6.87 → $9.87** at the ceiling)
> for a task that is a bounded 5-field judgement against a document already in the prompt. Fewer
> moving parts also means the rubric is applied more consistently — the thing Phase C is actually
> graded on. `judge submit --thinking` opts into a more deliberative judge; `judge estimate`
> prices both so the choice is made against numbers.

---

## Step 0 · Price it before you run it — `judge estimate`

**Zero API calls.** Counts the real tokens in the real rubric, the real documents and the real
questions, then multiplies by the `PRICING` table. Every call is charged the full 512-token output
cap, so the figure is a **ceiling** — a real verdict is ~120 tokens.

| | ceiling |
|---|---|
| Input | 2,022,070 tok (mean **2,022/call** — the source doc dominates) |
| Output | 512,000 tok (the cap, charged in full) |
| Synchronous | $13.75 |
| **Batches API (−50%)** | **$6.87** ← as configured |
| *with `--thinking`* | *$9.87* |

**Realistic landing zone: $2.80 – $4.20.** The ceiling assumes every verdict maxes the output cap;
they run ~120 tokens. Claude Sonnet 5 also carries **introductory pricing ($2/$10 per 1M) through
2026-08-31**, and the table above is computed at the standard $3/$15 list rate.

**Batching halves every token.** Scoring 1,000 answers is the least latency-sensitive job in the
project — nothing is waiting on the verdicts but us — so paying the synchronous premium would buy
literally nothing.

---

## Execution ladder — free things first, money last

```bash
cd src
uv run python -m judge estimate            # $0 — approve the number before spending
uv run python -m judge submit --limit 5    # cents — prove the rubric round-trips
uv run python -m judge collect             # free — poll, validate, write scores.csv
uv run python -m judge submit              # THE RUN — 1,000 answers, ~$3
uv run python -m judge collect             # free, resumes, re-runnable
uv run python -m judge status              # scored + spend
uv run python -m judge aggregate           # → results/master_table.csv
```

**`submit` and `collect` are separate commands on purpose** — the same split as Phase B's
`--train-only` / `--answer-only`. `submit` is the one irreversible spend; `collect` is free and
re-runnable. Keeping them apart means the paid step cannot be triggered by reflex while you
iterate on the free one. `collect` resumes: a Ctrl-C mid-collection costs nothing, and a failed
verdict is left *unwritten* so a re-run retries it rather than scoring it zero.

`.idea/runConfigurations/` ships the same ladder as PyCharm buttons, **`10_judge_estimate` →
`15_judge_aggregate`**, continuing Phase B's `0`–`9`. Use **Run ▸**, not **Debug ▸**.

---

## The final compilation & write-up  (Days 17–18)
- [ ] Aggregate into **one master table** → `results/master_table.csv`
      (weighted score, faithfulness highest, + cost/latency columns).
- [ ] `report/summary.md` — one page: *which method wins in which situation*, backed by numbers.
- [ ] `report/reflection.md` — mistakes watched for + what surprised you.
- [ ] Polish so a stranger can run the whole pipeline from scratch.

`judge aggregate` prints the master table plus the four cuts the write-up is built from:

1. **The TRIAD** — `llama3.1:8b` prompted vs fine-tuned vs + documents. One model, three methods,
   one judge: a score gap here **can only be the method**. This is the headline.
2. **Retrieval lift** — the two A/B pairs. Comparing the small-model lift against the frontier
   lift answers *"does retrieval help a small model more than a big one?"*, which no single-RAG
   setup can.
3. **The ladder** — small → medium → large. Does quality actually keep climbing with size?
4. **RAG diagnosis** — every RAG answer logged *which* chunks it retrieved. Crossing that against
   faithfulness separates **"the retriever missed"** from **"the model ignored good context"** —
   two failures that look identical in a score table and call for opposite fixes.

---

## Phase C deliverable
- [ ] `results/scores.csv` — judge output, one row per answer.
- [ ] `results/master_table.csv` — the aggregated comparison.
- [ ] `report/summary.md` + `report/reflection.md`.

> **Rule #4:** there is no single winner — name **which method wins at which task**,
> tracking cost and speed alongside quality. Phase B's latency column already shows
> fine-tuning winning on speed and RAG costing `gpt-4.1` 2.6× its latency. The judge
> decides quality; it does not decide the verdict on its own.

---

## ⚠️ The check that decides whether any of this is trustworthy

After the smoke test, before the full run: **the AEC registration fee is 1,000 LKR.**

- `prompt-llama3.1-8b` answered *"there is no fee"* → must score **low** on faithfulness/accuracy.
- `rag-llama3.1-8b`, given the retrieved section, answered *"1,000 LKR"* → must score **high**.

If the judge does not separate those two, it is broken, and all 1,000 verdicts downstream of it
are worthless. Stop and fix it before aggregating anything.

Then read 3–5 `reason` fields by hand against their documents. Every reason must be traceable to
the document, not a vague opinion. **No automated assertion substitutes for this.**
