"""`status` — where Phase B stands. `check` — do the model ids and keys actually work?

`status` reads only local files. `check` sends one ~10-token ping per paid provider, the
cheapest way to find out that a model id is wrong or a key is dead before a 1,000-call
run discovers it for you.
"""
from __future__ import annotations

import os

from .answers import TEST_SIZE, counts_by_producer, load_all, spend_by_producer
from .models import PRODUCERS, load_jobs, resolve_model


def cmd_status(_args) -> None:
    counts = counts_by_producer()
    spend = spend_by_producer()
    jobs = load_jobs()
    errors = sum(1 for r in load_all() if r.get("error"))

    w = max(len(p.key) for p in PRODUCERS)
    print(f"\n  {'producer':<{w}}  {'answered':>8}  {'spend $':>9}  status")
    print("  " + "-" * (w + 40))
    for p in PRODUCERS:
        got = counts.get(p.key, 0)
        bar = "complete" if got == TEST_SIZE else ("—" if got == 0 else f"partial")
        if p.method == "finetuned" and not jobs.get(p.key, {}).get("trained_model"):
            bar = "not trained"
        print(f"  {p.key:<{w}}  {got:>4}/{TEST_SIZE:<3}  {spend.get(p.key, 0.0):>9.4f}  {bar}")

    total = sum(counts.values())
    print("  " + "-" * (w + 40))
    print(f"  {'TOTAL':<{w}}  {total:>4}/{TEST_SIZE * len(PRODUCERS):<3}  "
          f"{sum(spend.values()):>9.4f}"
          f"{f'   ({errors} errored rows — re-run to retry)' if errors else ''}")

    if jobs:
        print("\n  Fine-tuning runs")
        for key, j in jobs.items():
            cost = j.get("train_cost_usd", 0.0)
            extra = (f"{j['trained_tokens']:,} tok" if j.get("trained_tokens")
                     else f"peak {j.get('peak_vram_gb', '?')} GB")
            print(f"    {key:<{w}}  {j.get('train_seconds', 0) / 60:5.1f} min  "
                  f"${cost:6.2f}  {extra}")
    print()


def cmd_check(_args) -> None:
    """One tiny call per paid provider — confirms the model id resolves and the key works."""
    from .clients import call

    print("\nPinging each provider with a ~10-token call (costs cents)\n")
    for p in PRODUCERS:
        if p.api_key_env and not os.getenv(p.api_key_env):
            print(f"  ✗ {p.key:<20} {p.api_key_env} is empty in .env")
            continue
        try:
            model = resolve_model(p)
        except SystemExit as exc:
            print(f"  · {p.key:<20} skipped — {exc}")
            continue
        c = call(p, model, "Reply with the single word: ready")
        if c.error:
            print(f"  ✗ {p.key:<20} {model}\n      {c.error[:100]}")
        else:
            print(f"  ✓ {p.key:<20} {model:<28} {c.latency_s:5.2f}s  ${c.cost_usd:.5f}")
    print()
