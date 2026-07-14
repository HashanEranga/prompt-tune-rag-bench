# Reflection — what nearly broke this, and what surprised me

> Written from the record in `docs/phase-a.md` and `docs/phase-b.md`. Every item below actually
> happened. The Phase C section is completed after the judge runs.

---

## The mistakes that would have invalidated the results

These are the ones that mattered — not because they were hard to fix, but because each would have
produced a **plausible-looking number that was wrong**, which is the only kind of bug that
actually threatens a comparison project.

### 1. A silent model swap under one producer key

`gemini-2.5-flash` is retired. It still appears in `models.list()`, but calling it returns
`404 — no longer available to new users`. Switching to `gemini-3.5-flash` was routine; what was
**not** routine is that 21 answers from the old model were already sitting in `answers.jsonl`
under the key `prompt-gemini`.

Had those stayed, the judge would have scored a **blend of two different models as one
contender**. The fix was `reset --producer prompt-gemini` to purge them — and then a guard in
`runner.py` that refuses to resume a producer whose model id has changed since its existing
answers were written. **The bug is not the retired model. The bug is that resuming felt safe.**

### 2. 100 identical empty errors, with the traceback thrown away

`transformers` v5 changed `apply_chat_template()` to return a `BatchEncoding` dict rather than a
tensor. The old code passed it straight into `generate()`, which died inside a bare
`raise AttributeError` — producing **100 identical, empty `AttributeError:` rows** with the actual
traceback discarded.

Worse, `_is_transient()` returned `True` for anything it didn't recognise, so **each of those 100
questions burned 4 retry attempts and ~14 s of backoff** before recording the error. Roughly 20
minutes spent patiently retrying a bug in my own code, while hiding the line number that named it.

Two lessons, and the second is the real one:
- Retry transient *provider* failures. **Never retry a `TypeError`.** `AttributeError`,
  `TypeError`, `NameError` and `ImportError` now fail immediately.
- **A generic retry wrapper is a traceback shredder.** It converted a 5-minute fix into a 20-minute
  mystery, and it did so *quietly*.

### 3. The adapter was served on the wrong base model

`_hf_pipeline` loaded the base model in **bf16** — but the LoRA adapter had been *trained* under
**4-bit NF4**. The code ran fine. It produced answers. They were answers from a **different model
than the one that was fine-tuned**, and nothing anywhere would have told me.

This is the scariest bug in the project, because it has **no symptom**. It does not crash, does not
warn, and produces perfectly fluent output. It would simply have made the fine-tuning results
quietly meaningless. The fix — load with the identical `BitsAndBytesConfig` used in training — is
one line. Finding it required asking a question I nearly didn't ask: *"is the thing I am serving
actually the thing I trained?"*

### 4. Latency was a deliverable, and I was measuring the wrong thing

The model load was happening **inside** the timed section, billing the first question of every
local producer with ~5–7 s of load time. Latency is a *result* in this project — one of the three
axes the final verdict turns on — so the pipeline is now warmed before the clock starts. Related:
running two local commands at once puts both models on the same 8 GB card, and the contention
silently corrupts the latency numbers of whichever one survives.

### 5. Reasoning traces leaking into the scored answer

Qwen3.x reasons by default; so does Gemini 3.x. Left alone, two things happen: one question took
**over 9 minutes and never completed** (vs **3.6 s** with thinking off), and — the part that
actually matters — **reasoning traces leak into the answer text the judge scores**, and inflate
both the latency and token-count deliverables. `think=False` on Ollama and
`thinking_budget=0` on Gemini are enforced unconditionally. This is a *data-integrity* fix wearing
a performance fix's clothes.

---

## What genuinely surprised me

### The managed fine-tuning product disappeared underneath the project

Three `ft-openai-*` producers were designed, costed at $2.44, and approved. Submitting the first
training job returned:

```
403 training_not_available — "OpenAI is winding down the fine-tuning platform and your
organization is no longer able to create new fine-tuning training jobs."
```

Revoked at the **organisation** level. No base model, no retry, no key gets around it.

I checked every alternative. **Gemini** had dropped tuning from the API entirely. **Groq**'s is
closed beta. **OpenRouter** has no training stack. **Together** and **Fireworks** still offer it.
Sorting that list by who owns the weights produces the finding I did not go looking for:

> **Every provider still offering fine-tuning tunes OPEN-WEIGHT models.
> Every provider withdrawing it owns closed frontier weights.**

And the labs' own stated reason for the retreat is *"don't fine-tune — use context and
retrieval."* **That is this project's thesis, arrived at independently by the industry.** My Step 2
result — a tuned 0.5B confidently inventing *"$500 to $1,000 per day"* against a truth of 1,000
LKR — reproduces on a laptop the reasoning behind a product withdrawal at frontier scale.

I could not have planned this. The project got a better finding by being *unable* to buy the
thing it budgeted for.

### $4.00 of compute cost $0.04

Together bills LoRA training at $0.48 per 1M tokens. My 300 pairs × 3 epochs came to **77,559
tokens — four cents of compute.** I was charged **$4.00**, the per-job minimum. A **~100× markup**,
and at this dataset size the floor *is* the price; the tokens are rounding error.

**Small jobs are billed as minimums, not rates** — a real property of managed fine-tuning that a
per-token price table hides completely. It was still the right $4: it bought the one producer the
hardware genuinely cannot make (an 8B needs ~15 GB to train; the card has 8), it closed the TRIAD,
and it left me an adapter file I own. **Rent the capability you lack; never rent the one you
already have.**

### The judge costs more than everything it judges — by 17×

Phase B inference: **$0.39**. The judge: **~$6.87 ceiling**. The reason is structural and I did not
see it coming:

> **A producer is billed for the question it reads. The judge is billed for the whole document.**

Each producer reads a 14-token question. The judge reads a **969-token source document**, plus the
rubric, plus the answer — **1,000 times**. The input side, pure rounding error throughout Phase B,
becomes the dominant term the instant you start evaluating.

The consequence is genuinely counterintuitive: **the real price of adding a producer is its judge
bill, not its inference bill.** Six of my ten producers cost $0.00 to run — and every one of them
costs the same ~$0.69 to *evaluate* as `gpt-4.1` does. Evaluation, not inference, is what scales
with the size of an experiment.

### The fine-tuned models are the fastest thing on the board

I expected fine-tuning to lose on facts (it did). I did not expect it to *win on speed by 9×*.
`ft-local-small` answers in **0.94 s** against `prompt-qwen3.5-9b`'s **8.68 s**. Fine-tuning taught
the model the *shape* of the answer, so it stops preambling and just answers. That is a real,
usable property — and it is invisible if you only look at a quality score. **This is what Rule #4
is protecting.**

---

## What I would tell someone starting this project

1. **Build the resume-and-guard machinery before the first paid call, not after.** Every hour spent
   on `assert_complete()`, the model-swap guard and the append-only log came back doubled. The
   bugs that cost me time were all bugs that *let bad data look fine*.
2. **Distrust the answer that looks good.** Every genuinely dangerous bug here — the bf16 adapter,
   the blended Gemini producer, the leaked reasoning traces — produced **fluent, plausible output**
   and no error at all. A crash is a gift. Silence is the enemy.
3. **Estimate before you spend, every time.** `estimate` makes zero API calls and takes seconds. It
   caught the shape of the judge's bill long before the judge ran.
4. **Write down what you expect *before* you look.** Otherwise every result is retro-fitted into a
   story, and the whole point of building a judge was to not do that.

---

## Phase C — after the judge runs

> ⟨To be completed once `results/master_table.csv` exists. Candidate questions to answer honestly:
> Did the judge actually separate the AEC-fee answers correctly, or did I have to fix the rubric?
> Did any result contradict what I predicted here? Where did the judge disagree with my own reading
> of an answer — and, on inspection, **which of us was right**?⟩
