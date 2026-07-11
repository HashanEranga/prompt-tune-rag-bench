#!/usr/bin/env python3
"""Phase A — Build the Serendib Hospital Q&A dataset.

Pipeline (run as subcommands, in order, from the ``src/`` directory):

    python -m build_dataset clean       # PDFs -> data/clean/*.md  (deterministic)
    python -m build_dataset segment     # clean -> data/interim/sections.jsonl
    python -m build_dataset generate    # (documented) sections -> data/qa/pool.jsonl
    python -m build_dataset split       # verified pool -> train.jsonl / test.jsonl (seeded)

Only the 13 hospital docs in data/raw/hospital/ are in scope for Phase A.
The `clean` and `split` stages are fully deterministic and reproducible from
scratch. `generate` documents the Anthropic-API method; the committed
data/qa/pool.jsonl is produced in-session (see README / plan) so no API key is
needed to reproduce clean+split.
"""
from __future__ import annotations

import argparse

from .clean import cmd_clean
from .generate import cmd_generate
from .segment import cmd_segment
from .split import cmd_split


def main() -> None:
    p = argparse.ArgumentParser(prog="build_dataset", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("clean", help="extract + de-boilerplate the 13 hospital PDFs").set_defaults(func=cmd_clean)
    sub.add_parser("segment", help="split clean docs into logical sections").set_defaults(func=cmd_segment)
    sub.add_parser("generate", help="draft grounded Q&A pairs + verify sheet").set_defaults(func=cmd_generate)
    sp = sub.add_parser("split", help="split verified pool into train/test")
    sp.add_argument("--all", action="store_true",
                    help="PREVIEW: split the full pool even if pairs aren't verified yet")
    sp.set_defaults(func=cmd_split)
    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
