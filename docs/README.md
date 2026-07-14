# Progress tracker — Serendib Hospital · Three Ways, One Judge

Per-phase progress lives here, one file per phase. The root
[`README.md`](../README.md) holds the full project brief and plan; these docs
track **what is actually done**.

| Phase | Tracker | Status | Deliverable |
|---|---|---|---|
| **A** — Build the dataset | [phase-a.md](phase-a.md) | ✅ **Complete** | 100 locked test Qs + 300 train, frozen split |
| **B** — Three contenders | [phase-b.md](phase-b.md) | 🟡 **Steps 1–2a done** (700/1,000) · Step 2b + RAG pending | 1,000 logged answers (prompting · fine-tuning · RAG) |
| **C** — Judge & conclusions | [phase-c.md](phase-c.md) | ⬜ Not started | master results table + one-page verdict |

**Phase B next step:** `cd src && uv run python -m contenders estimate` — zero API calls,
prints the projected spend so you approve a number before anything costs money.

💰 **[cost-analysis.md](cost-analysis.md)** — where the money goes: **$8.75** ceiling for the
whole project, half the producers are free, and the judge costs more than everything it judges.

**Status legend:** ✅ complete · 🟡 in progress · ⬜ not started

> Update the **Status** cell here whenever a phase moves, and tick the checkboxes
> inside each phase file as individual steps land.
