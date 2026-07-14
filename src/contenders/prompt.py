"""Step 1 — the prompting baseline. No documents, no retrieval, no hints: the control.

Serendib General Hospital is fictional, so its prices and protocols exist in no model's
weights. The baseline is expected to FAIL on hospital-specific facts — a correct answer
here would be a red flag — and that failure is what fine-tuning and RAG are measured
against.
"""
from __future__ import annotations

from .answers import load_test
from .models import PROMPTING, PRODUCERS
from .runner import run_producer


def cmd_prompt(args) -> None:
    questions = load_test()
    if args.limit:
        questions = questions[:args.limit]

    producers = [p for p in PRODUCERS if p.method == PROMPTING]
    if args.producer:
        producers = [p for p in producers if p.key in args.producer]
        if not producers:
            raise SystemExit(f"no prompting producer matches {args.producer}. "
                             f"Choose from: {[p.key for p in PRODUCERS if p.method == PROMPTING]}")
    if args.local_only:
        producers = [p for p in producers if p.is_local]

    print(f"Step 1 · prompting baseline — NO documents in context")
    print(f"{len(producers)} producers x {len(questions)} questions")
    for p in producers:
        run_producer(p, questions)
