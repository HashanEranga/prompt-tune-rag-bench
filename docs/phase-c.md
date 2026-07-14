# Phase C — The Judge & Conclusions (NOT STARTED)

A **separate AI** scores every Phase B answer against the **source document**
(never the drafted answer), then the scores are aggregated into one master table
and a written verdict.

> **Integrity Rule #3:** the judge must be a **different model family** than the
> answerers. A model judging its own family may favour answers that "sound like
> itself."
>
> **This project's assignment:** answerers are GPT · Gemini · Llama · Qwen
> (see the [Phase B roster](phase-b.md#the-answer-producer-roster)) → the judge is
> **Claude** (`ANTHROPIC_API_KEY`). Claude therefore never produces an answer in
> Phase B. Keep it that way.

---

## Step 4 · The Judge  (`src/4_judge.py`)
For each of the ~800 answers, the judge reads **question + real source document +
candidate answer** and returns structured JSON:

| Dimension | Scale | Notes |
|---|---|---|
| Faithfulness to Source | 1–5 | **weighted highest** |
| Accuracy | 1–5 | prices / times / numbers correct? |
| Completeness | 1–5 | everything asked addressed? |
| Clarity | 1–5 | readable for patient/staff? |
| Safety Flag | Yes/No | could a wrong answer cause harm? |
| Reason | text | one-line traceable verdict |

- [ ] Same judge, same rubric, same prompt applied to **every** method's answers.
- [ ] Forced structured JSON output for pandas aggregation.
- **Output:** `results/scores.csv` — one row per answer.

## The final compilation & write-up  (Days 17–18)
- [ ] Aggregate into **one master table** → `results/master_table.csv`
      (weighted score, faithfulness highest, + cost/latency columns).
- [ ] `report/summary.md` — one page: *which method wins in which situation*, backed by numbers.
- [ ] `report/reflection.md` — mistakes watched for + what surprised you.
- [ ] Polish so a stranger can run the whole pipeline from scratch.

---

## Phase C deliverable
- [ ] `results/scores.csv` — judge output, one row per answer.
- [ ] `results/master_table.csv` — the aggregated comparison.
- [ ] `report/summary.md` + `report/reflection.md`.

> **Rule #4:** there is no single winner — name **which method wins at which task**,
> tracking cost and speed alongside quality.
