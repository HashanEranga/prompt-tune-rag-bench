#!/usr/bin/env python3
"""Phase C — score all 1,000 Phase B answers, then aggregate the verdict.

Run as subcommands, from the ``src/`` directory. Cheapest and safest first:

    python -m judge estimate              # price the whole run — ZERO API calls
    python -m judge submit --limit 5      # cents — prove the rubric round-trips
    python -m judge collect               # free — poll, validate, write scores.csv
    python -m judge submit                # the real 1,000-answer batch
    python -m judge collect               # free, resumes, re-runnable
    python -m judge status                # scored + spend so far
    python -m judge aggregate             # → results/master_table.csv

`submit` is the one irreversible spend; `collect` is free and re-runnable. They are separate
commands (and separate PyCharm buttons) for the same reason Phase B split the $4 Together
training from the free local answering: the paid step must not be reachable by reflex while
you iterate on the free one.

The judge is Claude — a different model family than every Phase B answerer (GPT, Gemini,
Llama, Qwen). That is Integrity Rule #3, and it is why Claude never produced an answer.
"""
from __future__ import annotations

import argparse

from .aggregate import cmd_aggregate
from .batch import cmd_collect, cmd_submit
from .estimate import cmd_estimate
from .status import cmd_status


def main() -> None:
    p = argparse.ArgumentParser(prog="judge", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("estimate", help="price the judge run (no API calls)"
                   ).set_defaults(func=cmd_estimate)

    sp = sub.add_parser("submit", help="submit the answers to the judge (THE SPEND)")
    sp.add_argument("--limit", type=int, metavar="N",
                    help="judge only the first N answers (smoke test before the full run)")
    sp.add_argument("--thinking", action="store_true",
                    help="adaptive thinking — a more deliberative judge, ~2x the output "
                         "tokens. `estimate` prices both.")
    sp.set_defaults(func=cmd_submit)

    sub.add_parser("collect", help="poll the batch, validate verdicts, write scores.csv"
                   ).set_defaults(func=cmd_collect)

    sub.add_parser("status", help="answers scored + spend, per producer"
                   ).set_defaults(func=cmd_status)

    sub.add_parser("aggregate", help="scores + answers → results/master_table.csv"
                   ).set_defaults(func=cmd_aggregate)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
