"""`reset` — discard answers so a producer can be re-run cleanly.

Needed above all when a producer's MODEL changes: resuming on top of the old model's rows
would leave one producer key holding two models, which the judge would score as a single
contender and which cannot be un-mixed afterwards. Also clears errored rows, though those
are already retried automatically — `load_done()` never counts a failed row as done.
"""
from __future__ import annotations

import sys

from .answers import counts_by_producer, drop_rows, models_used
from .models import BY_KEY


def cmd_reset(args) -> None:
    if not args.producer and not args.errors:
        sys.exit("Nothing selected. Use --producer KEY (discard a producer's answers, "
                 "e.g. after changing its model) or --errors (discard failed rows only).")

    if args.producer and args.producer not in BY_KEY:
        sys.exit(f"unknown producer {args.producer!r}. Choose from:\n  " +
                 "\n  ".join(BY_KEY))

    if args.producer:
        got = counts_by_producer().get(args.producer, 0)
        models = models_used(args.producer)
        scope = f"{args.producer}"
        if args.errors:
            scope += " (errored rows only)"
        print(f"\nAbout to discard: {scope}")
        print(f"  {got} good answers on disk"
              f"{' from ' + ', '.join(sorted(models)) if models else ''}")
    else:
        print("\nAbout to discard: ALL errored rows, every producer")

    if not args.yes:
        reply = input("\nThis rewrites results/answers.jsonl. Proceed? [y/N] ").strip().lower()
        if reply != "y":
            sys.exit("aborted — nothing changed")

    dropped = drop_rows(producer=args.producer, errors_only=args.errors)
    print(f"\n  ✓ dropped {dropped} rows from results/answers.jsonl")
    if args.producer and not args.errors:
        print(f"  Re-run it with:  python -m contenders prompt --producer {args.producer}"
              if BY_KEY[args.producer].method == "prompting" else
              f"  Re-run the step that produces {args.producer}.")
