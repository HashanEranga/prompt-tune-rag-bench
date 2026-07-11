# Serendib Hospital — Three Ways, One Judge

> **BuildrLabs AAC01 · Mini Project 01 · Fundamentals of Agentic AI · Individual Build**
> *"Build it three ways. Let a judge decide."*

📍 **Progress tracker:** [`docs/README.md`](docs/README.md) — per-phase status (A ✅ · B ⬜ · C ⬜).

---

# Part 1 — What this project actually is

This is an **individual (solo) build**, ~2.5–3 weeks of work.

The real goal is **not** "build three chatbots." It's to run a **controlled scientific experiment**: answer the *same* set of hospital questions using **three different AI techniques**, then build a **separate AI judge** that scores every answer on the same rubric — so you can prove, *with your own numbers*, which technique wins for which kind of task.

The sentence you're trying to be able to say at the end, backed by ~800 scored answers, is:

> *"For this task, prompting won on setup cost, fine-tuning won on speed, and RAG won on getting facts right — here's my data."*

The single most important concept the project drills into you:

- **Fine-tuning teaches a model *how to behave*** (response patterns, tone, format) — not new facts.
- **RAG gives a model *new facts to work with*** at answer time (it looks things up).
- Most people confuse these two. This project makes you prove the difference empirically.

**The six numbers that define the scope:**

| Number | Meaning |
|---|---|
| **1** | Solo project — every decision is yours |
| **3** | Approaches compared: Prompting, Fine-tuning, RAG |
| **~3 weeks** | Estimated effort (18-day schedule) |
| **100** | Test questions your models *never see during training* |
| **~800** | Total answers the judge scores (100 questions × ~8 answer-producers) |
| **0** | Single winners — the point is *which method wins at which task*, not one champion |

---

# Part 2 — Your data (and how each folder is used)

You have two folders, and they play **completely different roles**. Getting this straight now saves you from a mistake that would invalidate results.

### `data/raw/FAQs+Websites/` → **THE knowledge base (your ground truth)** — 13 PDFs

This is **Serendib General Hospital**, a *fictional* 420-bed hospital in Colombo. It matches the "~13 documents" the brief describes exactly. It contains:

- Inpatient admission handbook + FAQ
- OPD services directory, surgery pricing + FAQ
- Emergency/casualty protocol reference
- Lab & diagnostic catalog + FAQ
- Insurance/payments/billing policy + FAQ
- Pharmacy info sheet
- A document library index

**Why fictional matters:** No AI on earth has seen this hospital's data. Facts like *cataract surgery = 90,000–140,000 LKR per eye* or *OPD volume ~1,200 patients/day* can **only** come from these documents. If a model gets them right, it actually looked them up — it didn't guess from training memory. **That is the whole test.**

➡️ **Everything derives from these 13 files:** your Q&A pairs come from here, RAG retrieves from here, and the judge treats *these documents* (never your AI-written sample answers) as the authority.

### `data/raw/Medical/` → **external real-world medical corpus** — 15 PDFs

These are **real** documents: US CDC *MMWR* reports (`mm75xx-H.pdf`), ECDC communicable-disease-threat weekly reports, an Ebola tabletop doc, etc. They are **not** about Serendib Hospital.

Their most useful role is as **realistic distractor / noise documents for your RAG retrieval corpus**. A real retrieval system has to find the *right* hospital section inside a big, messy pile of medical text. Mixing these in lets you test whether your retriever finds the needle (hospital fact) in the haystack (general medical PDFs), rather than a toy setup where retrieval can't fail.

> ⚠️ **Key rule:** Q&A **ground truth comes ONLY from the 13 hospital docs.** The `Medical/` folder is for making retrieval realistic — never treat it as the answer authority. (If you'd rather keep RAG simple for v1, index only the 13 hospital docs first and add `Medical/` as distractors later.)

---

# Part 3 — The mental model: 3 contenders + 1 judge

Every test question gets answered **three ways**, then scored by an independent judge.

```
                        ┌─────────────────────────┐
   100 test questions ─▶│  1. PROMPTING (baseline)│──┐
   (from hospital docs) │     no docs given        │  │
                        ├─────────────────────────┤  │   ~800
                        │  2. FINE-TUNING          │  ├─▶  answers ─▶  🧑‍⚖️ JUDGE ─▶ scores
                        │     small / med / large  │  │            (different model family)
                        ├─────────────────────────┤  │
                        │  3. RAG                  │──┘
                        │     retrieve, then answer│
                        └─────────────────────────┘
```

**Where the ~800 comes from:** 100 questions × roughly 8 answer-producers ≈ 800. The 8 producers are typically: a few prompting models (2–3 frontier + 2–3 small) + 3 fine-tuned sizes + 1 RAG pipeline.

### The judge's scorecard — same 5 dimensions on every answer

| Dimension | Question it answers | Scale |
|---|---|---|
| **Faithfulness to Source** | Does the answer match what the *real document* says? | 1–5 **(weighted highest)** |
| **Accuracy** | Are specific facts — prices, times, numbers — correct? | 1–5 |
| **Completeness** | Did it fully address everything asked? | 1–5 |
| **Clarity** | Easy to read for a patient/staff member? | 1–5 |
| **Safety Flag** | Could a wrong answer cause real harm? | **Yes/No** |

The judge also writes a **one-line reason** for each score (a traceable verdict, not a vague opinion). **Faithfulness is weighted highest** — an answer can be beautifully clear and still fail if it contradicts the source. And because this is healthcare, a wrong emergency number or dosage is a **safety error**, not just a low score.

---

# Part 4 — The phases (your "3 phases," reconciled)

The brief lists it as a **dependency chain of 5 steps (Step 0 → Step 4)**. That maps cleanly onto **3 phases** — work in these:

| Your phase | Brief's steps | What it produces |
|---|---|---|
| **Phase A — Foundation** | Step 0: Build the dataset | Clean, verified Q&A pairs, split train/test |
| **Phase B — The three contenders** | Step 1 Prompting · Step 2 Fine-tuning · Step 3 RAG | ~800 logged answers |
| **Phase C — Judgment & conclusions** | Step 4: The Judge + write-up | Master results table + one-page verdict |

---

## 🅰️ PHASE A — Build the Dataset (Days 1–3)

> *"Your entire project is only as good as the data underneath it."* This step deserves **more** time than people give it. Garbage questions → meaningless results.

**Three sub-steps:**

**A1. Clean the documents first.**
Real PDFs have repeated letterheads, signature blocks, page footers, page numbers. Strip all of it. Noisy source text → noisy questions.
- *Tools:* Python + `PyMuPDF` (`fitz`) or `pdfplumber` to extract text; regex/manual passes to remove boilerplate.
- *Output:* one clean `.txt`/`.md` per hospital document in `data/clean/`.

**A2. Segment, then generate Q&A pairs — and verify every pair.**
- Break each cleaned doc into **logical sections** (chunks by heading/topic).
- Use an AI (e.g., Claude) to draft sample **questions + answers** from each section.
- **Manually verify every pair against its source document.** The AI-drafted answers can contain errors — you're the quality gate. This is the tedious-but-critical part.
- Aim for a healthy pool, then reserve **exactly 100 for the test set.**

**A3. Split cleanly — training vs. testing — and never cross the line.**
- Divide verified pairs into a **training group** (`data/qa/train.jsonl`) and a **testing group** (`data/qa/test.jsonl`, the 100 eval questions).
- Keep them **permanently separate.** The 100 test questions must be ones the fine-tuned models **never see during training** — otherwise every downstream number is invalid. Freeze (commit) the split before building anything else.

📦 **Phase A deliverable:** a clean, verified set of Q&A pairs, clearly split into `train/` and `test/` subsets.

---

## 🅱️ PHASE B — Build the three contenders (Days 4–13)

Run all three under **identical conditions** on the same 100 test questions. For each answer, log: question ID, method, model, the answer text, and **latency + cost** (you'll need cost/speed for the "no single winner" conclusion).

### Step 1 · Prompting — the baseline (Days 4–6, *faster than it looks*)

- Ask all 100 test questions **directly** to several models — **no documents given as context.**
- Test **both** well-known frontier models **and** smaller open models.
- Same questions, same conditions, no exceptions.
- **Why:** this is your baseline. Since no model has seen Serendib's real prices/schedules/protocols, raw prompting *should* underperform on hospital-specific facts. If it doesn't — that's a finding worth reporting. Unexpected results are still results.
- *Tools:* Claude / GPT via API for frontier; Llama / Qwen / Mistral via API or local **Ollama** for the small open models.

📦 *Deliverable:* every model's complete answers to all 100 questions, logged and ready for the judge.

### Step 2 · Fine-tuning — small → medium → large (Days 7–12, *the longest stretch*)

- Fine-tune on your **training** Q&A pairs so the model learns *how to behave* on this task type (format/pattern) — **not** new facts.
- Train **three sizes** (small, medium, large) and compare:
  - **Small** — fastest/cheapest; the lower bound of what fine-tuning can do here.
  - **Medium** — often the biggest jump in quality-per-dollar.
  - **Large** — highest capacity; does quality *actually keep climbing*? That's what you're measuring.
- Test each **only on the 100 unseen questions.**
- **Log training time + estimated cost per size** — these numbers go into your final conclusions.
- *Tools:* OpenAI fine-tuning API (easy small/med/large tiers), **or** open models with **LoRA/PEFT** (e.g., `unsloth`, Hugging Face `peft`) using a size ladder like Qwen 0.5B/1.5B/7B or Llama 1B/3B/8B. Cloud options: Together, Fireworks, or a rented GPU.

📦 *Deliverable:* three trained models, their scored answers, and documented **training time + cost per size**.

### Step 3 · RAG — give it the documents (Days 10–13, *overlaps with fine-tuning*)

Build a **retrieve-then-answer** pipeline:
1. **Retriever** — for each incoming question, find the most relevant document section(s).
2. **Injection** — hand those retrieved sections to the model as context *immediately before* it answers.
3. **One fixed configuration** across all 100 questions — don't tune it per question, or results aren't comparable.
- **Why:** RAG is the technique *designed* to inject **fresh, specific facts a model was never trained on** — exactly what Serendib's proprietary pricing/schedules/protocols test. This is where you expect the model to stop guessing and start *looking things up*. How well it actually does that is your finding.
- *Tools:* embeddings (`sentence-transformers`, or Voyage/OpenAI embeddings) → vector store (**FAISS** or **Chroma**) → an LLM for generation. Index the 13 hospital docs; optionally add `data/raw/Medical/` as distractors for a realistic retrieval test.

📦 *Deliverable:* a working retrieval-and-answer pipeline + all scored answers on the full test set.

---

## 🅲️ PHASE C — The Judge & conclusions (Days 14–18)

### Step 4 · The Judge (Days 14–16)

A **separate AI** whose *only* job is scoring. For each of the ~800 answers it reads three things — **the original question, the real source document, and the candidate answer** — and returns a structured verdict: the 5 scores + a one-line reason + the safety flag.

Three properties make it valid:
- **Structured scores** — 1–5 per dimension + written reason (traceable, not vague).
- **Safety flagging** — records whether a wrong answer could cause real harm.
- **Consistent application** — same judge, same rubric, same prompt, applied to *every* method's answers. Consistency is what makes the comparison fair.

> ⚠️ Your entire project is only as trustworthy as your judge. A sloppy or biased judge makes every downstream result meaningless.

- *Tools:* an LLM from a **different family than your answerers** (see the rules below), forced into **structured JSON output** for the 5 dimensions so you can aggregate in pandas.

### The final compilation & write-up (Days 17–18)

- Aggregate all scores into **one master results table** (a pandas DataFrame → `results/master_table.csv`) comparing every method across all answers, with a weighted score (faithfulness weighted highest) plus cost/latency columns.
- Write the **one-page summary** (`report/summary.md`): *which method wins in which situation*, backed by your numbers.
- Write a **short reflection** (`report/reflection.md`): the mistakes you had to watch for, and what surprised you.
- Polish the repo so a stranger can run the whole pipeline **from scratch**.

---

# Part 5 — The four rules you cannot break

These aren't guidelines — they're **integrity constraints**. Violating any one invalidates your results:

1. **Keep training and testing separate.** Mixing lets a model "cheat" by memorizing. Once mixed, there's no valid way to un-mix.
2. **Judge against the real document** — not your AI-written sample answers (those can contain errors). The **source document is always the authority**.
3. **Use a different AI to judge** than the one that answered. Same model family judging itself may unconsciously favor answers that "sound like itself." Keep the judge independent. *(Practical pick: if you answer with GPT/Llama/Qwen, judge with Claude — or vice versa.)*
4. **There is no single winner.** Track **cost and speed** alongside quality. Your job is naming which method wins at which task, not crowning one universal champion.

---

# Part 6 — What you hand in & how you're graded

**You submit:**
- Working code for the full pipeline — **runnable from scratch by anyone**
- One **master results table** comparing every method across all scored answers
- A **one-page summary**: which method wins in which situation, backed by numbers
- A **short reflection** on mistakes to watch for and what surprised you

**Mainly graded on:**
- **Fair, trustworthy comparison** *(weighted highest)* — were conditions controlled and scoring consistent?
- **Thoughtful conclusions** — does the analysis actually follow from the data?
- **Clean, well-checked data** — evidence of careful cleaning, pair verification, proper train/test separation.

---

# Part 7 — Suggested tech stack & repo layout

A structure that satisfies "runnable from scratch by a stranger":

```
serendib-eval/
├── README.md              # this file — how to run the whole thing, top to bottom
├── requirements.txt
├── docs/                  # per-phase progress trackers (see docs/README.md)
│   ├── README.md          # progress index — A/B/C status at a glance
│   ├── phase-a.md         # Phase A — dataset (complete)
│   ├── phase-b.md         # Phase B — three contenders
│   └── phase-c.md         # Phase C — judge & conclusions
├── data/
│   ├── raw/               # the 28 provided PDFs (13 hospital + 15 medical)
│   ├── clean/             # cleaned text per hospital doc
│   └── qa/
│       ├── train.jsonl    # fine-tuning pairs
│       └── test.jsonl     # the LOCKED 100 test questions
├── src/
│   ├── build_dataset/         # Phase A package: extract → clean → gen Q&A → verify → split
│   │   ├── __main__.py        #   CLI: python -m build_dataset <clean|segment|generate|split>
│   │   ├── config.py          #   shared paths + source-filename→slug map
│   │   ├── clean.py           #   stage 1: PDFs → clean Markdown
│   │   ├── segment.py         #   stage 2: Markdown → logical sections
│   │   ├── generate.py        #   stage 3: sections → grounded Q&A pool + verify sheet
│   │   └── split.py           #   stage 4: verified pool → train/test (seeded)
│   ├── 1_prompting.py         # baseline answers, all models
│   ├── 2_finetune.py          # train small/med/large, then answer
│   ├── 3_rag.py               # index → retrieve → answer
│   └── 4_judge.py             # score all answers, structured JSON
├── results/
│   ├── answers.jsonl          # every answer + latency + cost
│   ├── scores.csv             # judge output, one row per answer
│   └── master_table.csv       # aggregated comparison
└── report/
    ├── summary.md             # the one-pager
    └── reflection.md
```

**Language:** Python. **PDF:** PyMuPDF/pdfplumber. **Q&A generation + judge:** Claude (Opus/Sonnet). **Prompting models:** Claude + GPT (frontier) and Llama/Qwen/Mistral via Ollama (open). **Fine-tuning:** OpenAI FT API *or* LoRA (unsloth/PEFT) on an open size-ladder. **RAG:** sentence-transformers + FAISS/Chroma. **Analysis:** pandas + matplotlib.

**Setup:**

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt

export ANTHROPIC_API_KEY=...         # e.g. the judge
export OPENAI_API_KEY=...             # e.g. answerers / fine-tuning
# Ollama (local open models) needs no key — just `ollama serve`
```

**Run, top to bottom** (each step's output feeds the next):

```bash
(cd src && python -m build_dataset clean && python -m build_dataset segment \
       && python -m build_dataset generate && python -m build_dataset split)  # Phase A — foundation
python src/1_prompting.py            # Phase B — baseline
python src/2_finetune.py             # Phase B — small/medium/large
python src/3_rag.py                  # Phase B — retrieve-then-answer
python src/4_judge.py                # Phase C — score everything
```

---

# Part 8 — The 18-day timeline

| Days | Work |
|---|---|
| **1–3** | Documents → clean dataset. Verify every Q&A pair. **Lock the train/test split.** |
| **4–6** | **Prompting baseline.** Run all models, log every answer. (Faster than it looks.) |
| **7–12** | **Fine-tuning — longest stretch.** Train small/med/large; document time + cost. |
| **10–13** | **RAG pipeline** (overlaps with fine-tuning). Build retrieval, run full test set. |
| **14–16** | **Judge + full results table.** Score everything, compile the master comparison. |
| **17–18** | One-page summary, reflection, polish repo so it runs clean for a stranger. |

> Note the **deliberate overlap** between fine-tuning (7–12) and RAG (10–13) — kick off training runs, then build RAG while they train, rather than treating them as strictly sequential.

---

## The one line to keep in mind

> **"There is no single right answer. What matters is the strength of your evidence."**
