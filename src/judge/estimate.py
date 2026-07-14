"""Step 0 — price the judge BEFORE running it. Zero API calls.

Same discipline as `contenders estimate`: count the real tokens in the real prompts the
judge will actually send, multiply by the PRICING table, and print a ceiling. You approve a
number before any money moves.

The one figure that cannot be measured offline is the output length, so it is charged at
the full MAX_OUTPUT_TOKENS cap on every one of the 1,000 calls — the verdict is ~120 tokens
in practice, so the real bill lands well under this.
"""
from __future__ import annotations

from contenders.answers import load_test
from contenders.models import JUDGE_MODEL, PRICES_VERIFIED, PRICING, PRODUCERS

from .batch import BATCH_DISCOUNT, MAX_OUTPUT_TOKENS
from .rubric import SYSTEM_PROMPT, build_user
from .scores import load_done
from .sources import assert_all_resolve, document_for

# A judged answer is charged the Phase B output cap: the longest answer any producer could
# have written is the longest string the judge could have to read.
ANSWER_TOKENS_CAP = 300

# Adaptive thinking (`--thinking`) is unbounded by nature. This allowance is what the
# estimate charges for it, so the two options can be compared as numbers.
THINKING_ALLOWANCE = 400


def _encoder():
    """o200k_base, as in `contenders estimate`. It is not Claude's tokenizer — treat the
    figure as ±20%, which is plenty to decide whether to spend $3."""
    import tiktoken
    return tiktoken.get_encoding("o200k_base")


def cmd_estimate(_args) -> None:
    enc = _encoder()
    test = load_test()
    assert_all_resolve(test)

    n_producers = len(PRODUCERS)
    rubric_tokens = len(enc.encode(SYSTEM_PROMPT))

    # The real prompt, measured — full source document included, exactly as submit builds it.
    scaffold = len(enc.encode(build_user("", "", "")))
    per_question = []
    for q in test:
        doc = len(enc.encode(document_for(q)))
        qt = len(enc.encode(q["question"]))
        per_question.append(scaffold + doc + qt + ANSWER_TOKENS_CAP)

    doc_tokens = [len(enc.encode(document_for(q))) for q in test]
    in_per_call = [rubric_tokens + t for t in per_question]
    total_in = sum(in_per_call) * n_producers

    already = len(load_done())
    remaining = len(test) * n_producers - already

    print(f"\nMeasured from the frozen dataset and the real rubric (no API calls):")
    print(f"  answers to judge    : {len(test)} questions x {n_producers} producers = "
          f"{len(test) * n_producers:,}"
          f"{f'  ({already:,} already scored, {remaining:,} left)' if already else ''}")
    print(f"  rubric (system)     : {rubric_tokens} tok  (identical on every call)")
    print(f"  source document     : mean {sum(doc_tokens) // len(doc_tokens)} tok, "
          f"max {max(doc_tokens)} tok  (the FULL cleaned doc — Rule #2)")
    print(f"  candidate answer    : charged at the {ANSWER_TOKENS_CAP}-tok Phase B cap")
    print(f"  mean input per call : {sum(in_per_call) // len(in_per_call):,} tok\n")

    price = PRICING[JUDGE_MODEL]

    def bill(out_per_call: int) -> tuple[float, float]:
        out = out_per_call * len(test) * n_producers
        full = (total_in * price.inp + out * price.out) / 1_000_000
        return full, full * BATCH_DISCOUNT

    off_full, off_batch = bill(MAX_OUTPUT_TOKENS)
    on_full, on_batch = bill(MAX_OUTPUT_TOKENS + THINKING_ALLOWANCE)

    print(f"  Judge: {JUDGE_MODEL}   (${price.inp}/${price.out} per 1M in/out)")
    print(f"  {'':<26}{'sync $':>10}{'BATCHED $':>12}")
    print("  " + "-" * 48)
    print(f"  {'thinking disabled':<26}{off_full:>10.2f}{off_batch:>12.2f}   <- default")
    print(f"  {'--thinking (adaptive)':<26}{on_full:>10.2f}{on_batch:>12.2f}")
    print("  " + "-" * 48)
    print(f"\n  Phase C TOTAL (as configured) $ {off_batch:7.2f}   "
          f"— {len(test) * n_producers:,} verdicts, Batches API at {BATCH_DISCOUNT:.0%}\n")

    print("  Output is charged at the full cap on every call, so this is a CEILING:")
    print(f"  a real verdict is ~120 tok against the {MAX_OUTPUT_TOKENS}-tok cap charged here.\n")

    if not PRICES_VERIFIED:
        print("  " + "=" * 68)
        print("  ⚠️  PRICES ARE UNVERIFIED — confirm before trusting these figures.")
        print(f"    {price.source}   (judge)")
        print("  Note: Claude Sonnet 5 carries introductory pricing ($2/$10 per 1M)")
        print("  through 2026-08-31, so the real bill should land BELOW the number")
        print("  above, which is computed at the standard $3/$15 list rate.")
        print("  " + "=" * 68 + "\n")
