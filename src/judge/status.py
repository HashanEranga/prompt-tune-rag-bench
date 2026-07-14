"""`status` — where Phase C stands. Reads only local files, makes no API calls."""
from __future__ import annotations

import json

from contenders.answers import TEST_SIZE, counts_by_producer
from contenders.models import JUDGE_MODEL, PRODUCERS

from .config import BATCH_PATH, MASTER_PATH, SCORES_PATH
from .scores import judges_used, load_all, spend


def cmd_status(_args) -> None:
    rows = load_all()
    answered = counts_by_producer()
    scored: dict[str, int] = {}
    for r in rows:
        scored[r["producer"]] = scored.get(r["producer"], 0) + 1

    w = max(len(p.key) for p in PRODUCERS)
    print(f"\n  {'producer':<{w}}  {'answered':>8}  {'scored':>8}  status")
    print("  " + "-" * (w + 34))
    for p in PRODUCERS:
        got, sc = answered.get(p.key, 0), scored.get(p.key, 0)
        bar = "judged" if sc == TEST_SIZE else ("—" if sc == 0 else "partial")
        print(f"  {p.key:<{w}}  {got:>4}/{TEST_SIZE:<3}  {sc:>4}/{TEST_SIZE:<3}  {bar}")

    total, expected = len(rows), TEST_SIZE * len(PRODUCERS)
    print("  " + "-" * (w + 34))
    print(f"  {'TOTAL':<{w}}  {sum(answered.values()):>4}/{expected:<4} "
          f"{total:>4}/{expected:<4} ${spend():.4f} spent judging")

    judges = judges_used()
    if judges:
        tag = "" if judges == {JUDGE_MODEL} else "   ⚠️  DOES NOT MATCH models.JUDGE_MODEL"
        print(f"\n  judged by: {', '.join(sorted(judges))}{tag}")

    if BATCH_PATH.exists():
        b = json.loads(BATCH_PATH.read_text(encoding="utf-8"))
        print(f"  last batch: {b['batch_id']}  ({b['submitted']} requests, "
              f"thinking={'adaptive' if b.get('thinking') else 'disabled'})")

    if total == 0:
        nxt = "python -m judge estimate     # $0, no API calls"
    elif total < expected:
        nxt = "python -m judge collect      # free, resumes"
    elif not MASTER_PATH.exists():
        nxt = "python -m judge aggregate    # → results/master_table.csv"
    else:
        nxt = "write report/summary.md — the data is in results/master_table.csv"
    print(f"\n  next: {nxt}\n")
