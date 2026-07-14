#!/usr/bin/env python3
"""Phase B — answer the 100 locked test questions three ways.

Run as subcommands, from the ``src/`` directory. Cheapest and safest first:

    python -m contenders estimate              # price the whole run — ZERO API calls
    python -m contenders check                 # do the model ids + keys work? (cents)
    python -m contenders rag --build-index     # free, fully local (Ollama + FAISS)
    python -m contenders prompt --local-only --limit 3   # free end-to-end smoke test
    python -m contenders prompt                # Step 1 — baseline, no documents
    python -m contenders finetune --backend local        # Step 2a — QLoRA on the 4060 (free)
    python -m contenders finetune --backend together     # Step 2b — the 8B rung ($4, once)
    python -m contenders rag                   # Step 3 — retrieve, then answer
    python -m contenders status                # progress + spend so far

Every command resumes: answers already in results/answers.jsonl are skipped, so a
crash or a Ctrl-C never costs a paid call twice. Add --limit N to any answering
command to try it on N questions first.
"""
from __future__ import annotations

import argparse

from .estimate import DEFAULT_TOP_K, cmd_estimate
from .finetune import DEFAULT_EPOCHS, cmd_finetune
from .prompt import cmd_prompt
from .rag import cmd_rag
from .reset import cmd_reset
from .status import cmd_check, cmd_status


def _answering_flags(sp) -> None:
    sp.add_argument("--limit", type=int, metavar="N",
                    help="answer only the first N questions (smoke test before the full run)")
    sp.add_argument("--local-only", action="store_true",
                    help="skip paid producers — run the free local ones only")


def main() -> None:
    p = argparse.ArgumentParser(prog="contenders", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("estimate", help="project the cost of the whole run (no API calls)")
    sp.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    sp.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    sp.add_argument("--max-output", type=int, default=300,
                    help="output tokens charged per answer (the worst-case cap)")
    sp.set_defaults(func=cmd_estimate)

    sub.add_parser("check", help="ping each provider: do the model ids and keys work?"
                   ).set_defaults(func=cmd_check)

    sp = sub.add_parser("prompt", help="Step 1 — baseline answers, NO documents")
    _answering_flags(sp)
    sp.add_argument("--producer", nargs="+", metavar="KEY", help="run only these producers")
    sp.set_defaults(func=cmd_prompt)

    sp = sub.add_parser("finetune", help="Step 2 — train the small/medium/large ladder, then answer")
    _answering_flags(sp)
    sp.add_argument("--backend", choices=["openai", "local", "together"], required=True,
                    help="local = QLoRA on your GPU (free); together = the 8B rung your GPU "
                         "can't train ($4 once, then served locally); openai = WITHDRAWN")
    sp.add_argument("--size", nargs="+", choices=["small", "medium", "large"],
                    help="train/answer only these rungs of the ladder")
    sp.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    sp.add_argument("--retrain", action="store_true", help="retrain even if already trained")
    sp.add_argument("--train-only", action="store_true", help="train, but don't answer yet")
    sp.add_argument("--answer-only", action="store_true", help="skip training, answer with existing")
    sp.set_defaults(func=cmd_finetune)

    sp = sub.add_parser("rag", help="Step 3 — build the index, then retrieve-and-answer")
    _answering_flags(sp)
    sp.add_argument("--build-index", action="store_true",
                    help="embed hospital sections + medical distractors into FAISS (free, local)")
    sp.set_defaults(func=cmd_rag)

    sub.add_parser("status", help="answers logged + spend, per producer"
                   ).set_defaults(func=cmd_status)

    sp = sub.add_parser("reset", help="discard a producer's answers (e.g. after changing its model)")
    sp.add_argument("--producer", metavar="KEY",
                    help="discard every answer from this producer — required if you change its model")
    sp.add_argument("--errors", action="store_true", help="discard errored rows only")
    sp.add_argument("--yes", action="store_true", help="skip the confirmation prompt")
    sp.set_defaults(func=cmd_reset)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
