# Progress tracker — Serendib Hospital · Three Ways, One Judge

Per-phase progress lives here, one file per phase. The root
[`README.md`](../README.md) holds the full project brief and plan; these docs
track **what is actually done**.

| Phase | Tracker | Status | Deliverable |
|---|---|---|---|
| **A** — Build the dataset | [phase-a.md](phase-a.md) | ✅ **Complete** | 100 locked test Qs + 300 train, frozen split |
| **B** — Three contenders | [phase-b.md](phase-b.md) | ✅ **Complete** | **1,000/1,000 answers · 0 errors** (prompting · fine-tuning · RAG) |
| **C** — Judge & conclusions | [phase-c.md](phase-c.md) | ✅ **Judged** | **1,000/1,000 scored** → [`master_table.csv`](../results/master_table.csv) |
| **D** — Retrieval engineering | [phase-d.md](phase-d.md) | ⬜ **Proposed, not built** | a fixed retriever — *no code exists for this yet* |

## Where it landed

The TRIAD — one model (`llama3.1:8b`), three methods, one judge:

**prompted 1.927 → fine-tuned 2.529 → + documents 3.969.**

Retrieval roughly triples faithfulness. Fine-tuning improves format, speed and safety but cannot
install a fact. `rag-gpt` tops the table at **4.244**; `prompt-llama3.1-8b` sits at the bottom on
**1.927**, with **26% of its answers flagged as potentially unsafe** against RAG's 3–5%.

**Phase D exists because of one number.** On the 32 questions where the retriever actually found
the right section, accuracy is **4.66** and completeness **4.84** — near-perfect. On the 30 where it
missed, they collapse to **3.00** and **1.77**. The models are not the bottleneck; the retriever is.
Two concrete root causes are identified in [phase-d.md](phase-d.md), both visible in data already on
disk. **It is a proposal — nothing has been built or run.**

💰 **[cost-analysis.md](cost-analysis.md)** — where the money goes. Phase B came in at
**$4.39 actual** against a $4.83 ceiling; the judge costs more than everything it judges.

**Status legend:** ✅ complete · 🟡 in progress · ⬜ not started

> Update the **Status** cell here whenever a phase moves, and tick the checkboxes
> inside each phase file as individual steps land.
