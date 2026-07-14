# Which method wins, and when — Serendib General Hospital

> **STATUS: awaiting scores.** Everything below marked `⟨…⟩` is a placeholder to be filled from
> `results/master_table.csv` once `judge submit` → `judge collect` → `judge aggregate` have run.
> **Do not fill these in by hand or by guess.** The whole point of the project is that every
> claim on this page is traceable to a number the judge produced. If a cell cannot be filled from
> the master table, the sentence around it does not belong in the report.
>
> Sections marked **✅ MEASURED** are already final — they come from Phase B and stand
> independently of the judge.

---

## The experiment in one paragraph

Same 100 questions about a **fictional** 420-bed hospital in Colombo. Ten answer-producers across
three techniques — prompting (no documents), fine-tuning (four sizes), and RAG (retrieve, then
answer). 1,000 answers, all logged with latency and cost. Then a **separate AI from a different
model family** (Claude, `claude-sonnet-5`) scored every one of them against the **real source
document** on a single frozen rubric, blind to which producer wrote what. Because Serendib is
fictional, no model has its prices or protocols in its weights: **a correct fact can only have
been looked up.**

---

## The headline — one model, three methods (the TRIAD)

`llama3.1:8b` is the same weights in all three rows. Only the *method* differs, so the gaps below
are the method and nothing else.

| Method | Faithfulness | Accuracy | Completeness | Clarity | **Weighted** | Latency | Cost |
|---|---|---|---|---|---|---|---|
| Prompted, no documents | ⟨…⟩ | ⟨…⟩ | ⟨…⟩ | ⟨…⟩ | ⟨…⟩ | **2.43 s** ✅ | $0.00 ✅ |
| Fine-tuned (LoRA, 8B) | ⟨…⟩ | ⟨…⟩ | ⟨…⟩ | ⟨…⟩ | ⟨…⟩ | **1.77 s** ✅ | $4.00 once ✅ |
| RAG (+ documents) | ⟨…⟩ | ⟨…⟩ | ⟨…⟩ | ⟨…⟩ | ⟨…⟩ | **1.20 s** ✅ | $0.00 ✅ |

*Weighted = faithfulness 40% · accuracy 30% · completeness 20% · clarity 10%.*

> **Honest caveat, state it plainly:** Ollama serves `llama3.1:8b` at **Q4_0**, while the
> fine-tuned leg is **4-bit NF4** over the fp16 base. Same weights, same lineage, different
> quantisation. A strong control, not a perfect one.

**⟨The one-sentence verdict goes here, and it must follow from the table above — not from what we
expected to find.⟩**

---

## Cost and speed — already decided, without the judge ✅ MEASURED

Rule #4 says there is no single winner, and these two columns are why. They were settled in Phase
B and no score can overturn them.

**Fine-tuning won on speed, and it is not close.** The four fine-tuned rungs take four of the five
fastest slots. `ft-local-small` answers in **0.94 s** — **9× faster** than `prompt-qwen3.5-9b`
(8.68 s), and 2.4× faster than the same-family `prompt-llama3.1-8b` (2.43 s). Fine-tuning taught
*how to behave*: the tuned models emit a short, well-shaped answer immediately instead of
preambling toward one.

**Retrieval is not free, and the bill is latency.** `rag-gpt` takes **5.98 s** against
`prompt-gpt`'s **2.27 s** — same model, **2.6× slower** — because it must embed the question,
search 830 chunks, and read three injected sections before it may answer.

**Half the roster costs nothing.** Six of ten producers ran at **$0.00**: both Ollama prompting
models, all three local QLoRA rungs, and the `llama3.1:8b` RAG pipeline including embeddings.
Total Phase B spend was **$4.39**, of which **$4.00 was a single fixed training floor**.

| | |
|---|---|
| Cheapest to set up | **Prompting** — no training, no index, no infrastructure |
| Fastest to answer | **Fine-tuning** — 0.94 s at the small rung ✅ |
| Most expensive per answer | **Prompting a frontier model** — `prompt-gemini`, $0.1643/100 ✅ |
| Most expensive *to judge* | every producer equally — **the judge costs 17× Phase B inference** ✅ |

---

## Does retrieval help a small model more than a big one?

Two controlled A/B pairs, same model on each side, differing only in whether documents were
retrieved. Comparing the two *lifts* is a question no single-RAG setup can answer.

| Pair | No documents | + documents | **Lift** |
|---|---|---|---|
| `llama3.1:8b` (small, open) | ⟨…⟩ | ⟨…⟩ | ⟨…⟩ |
| `gpt-4.1` (frontier) | ⟨…⟩ | ⟨…⟩ | ⟨…⟩ |

**⟨Which lift is larger, and what that means for anyone choosing between a bigger model and a
retrieval layer.⟩**

---

## Does fine-tuning keep improving with size?

| Rung | Base | Train time ✅ | Peak VRAM ✅ | Weighted score |
|---|---|---|---|---|
| small | `Qwen2.5-0.5B` | 2.3 min | 1.69 GB | ⟨…⟩ |
| medium | `Qwen2.5-1.5B` | 3.7 min | 3.54 GB | ⟨…⟩ |
| large | `Qwen2.5-3B` | 7.2 min | 5.24 GB | ⟨…⟩ |
| 8B (Together) | `Llama-3.1-8B` | rented | ~15 GB to train | ⟨…⟩ |

The whole ladder trained inside **8 GB in 13.2 minutes for $0.00**. ✅

**⟨Does quality actually keep climbing — and does it ever climb far enough to matter, given that
none of these rungs can learn a fact?⟩**

---

## The finding the project was built to produce

**Fine-tuning teaches a model *how to behave*. RAG gives it *new facts to work with*.** Most
people conflate these. Here is the proof, on one question:

> **"What is the AEC registration fee?"** — the document says **1,000 LKR**.

| Producer | Answer | ✅ |
|---|---|---|
| `prompt-llama3.1-8b` | *"there is no fee"* | measured |
| `prompt-qwen3.5-9b` | *"Rs. 500"* | measured |
| `prompt-gemini` | *"LKR 1,500"* — and hallucinated what "AEC" stands for | measured |
| `ft-local-small` (tuned) | *"$500 to $1,000 per day"* — fluent, well-shaped, **entirely invented** | measured |
| `rag-llama3.1-8b` (+ docs) | **"1,000 LKR"** | measured |

The fine-tuned model produced a **better-formed wrong answer**. It learned the shape of a hospital
FAQ response and filled it with fiction — wrong currency, wrong unit, wrong number. That is not a
failure of the fine-tune; **it is exactly what fine-tuning does, and it is why RAG exists.**

---

## RAG diagnosis — did the retriever miss, or did the model ignore what it was given?

Every RAG answer logged *which* chunks it retrieved, and whether they were distractors. Crossing
that against faithfulness separates two failures that look identical in a score table and call for
opposite fixes.

| | n | Faithfulness | Weighted |
|---|---|---|---|
| Gold section **was** retrieved | ⟨…⟩ | ⟨…⟩ | ⟨…⟩ |
| Gold section was **not** retrieved | ⟨…⟩ | ⟨…⟩ | ⟨…⟩ |

The index carries **704 distractor chunks against 126 hospital chunks** — the right section must
win against 5.6× its own volume in unrelated CDC/ECDC medical text. ✅

**⟨If faithfulness is low even when the gold section WAS retrieved, the fix is the generator, not
the retriever. Say which one this data points at.⟩**

---

## Safety

Healthcare makes a wrong answer more than a low score. The judge flagged an answer only where
acting on it could cause **physical** harm or delay urgent care — a wrong emergency number, wrong
triage advice, wrong dosage, wrong pre-operative instruction. A wrong *fee* is inaccurate, not
unsafe.

**⟨How many answers were flagged, and which method produced them. If prompting produces confident
fiction about emergency procedures, that is the most important sentence in this report.⟩**

---

## The verdict

**⟨There is no single winner — name which method wins at which task, and back every clause with a
number from the master table. The shape we expected: prompting wins on setup cost, fine-tuning on
speed, RAG on getting facts right. Write what the data says, not what we expected.⟩**
