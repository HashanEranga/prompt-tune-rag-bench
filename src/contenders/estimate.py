"""Step 0 — price the run BEFORE running it. Makes zero API calls.

Pure local arithmetic: tiktoken counts the real tokens in the real 100 questions, then
the ``PRICING`` table in models.py multiplies them out, so the number you approve is
grounded in the actual data. The output is deliberately a CEILING — every producer is
charged the full ``MAX_OUTPUT_TOKENS`` cap on every answer, so real spend comes in under.
"""
from __future__ import annotations

import json

from .clients import MAX_OUTPUT_TOKENS
from .config import SECTIONS_PATH, TRAIN_PATH
from .answers import TEST_SIZE, load_test
from .models import (FINETUNED, FT_INFERENCE_MULTIPLIER, JUDGE_MODEL, OPENAI, PRICES_VERIFIED,
                     PRICING, PRODUCERS, RAG, TOGETHER_JOB_MINIMUM_USD, Producer, price_for)

DEFAULT_EPOCHS = 3          # OpenAI FT default; --epochs to test sensitivity
DEFAULT_TOP_K = 3           # RAG sections injected per question (the fixed config)


def _encoder():
    """One tokenizer for every model. Tokenizers differ by family, so treat the non-OpenAI
    figures as ±20% — plenty accurate for a go/no-go on spend."""
    import tiktoken
    return tiktoken.get_encoding("o200k_base")


def _mean_section_tokens(enc) -> int:
    """Mean tokens per retrievable section — the unit RAG injects and the judge reads,
    measured from the real Phase A sections rather than assumed."""
    if not SECTIONS_PATH.exists():
        return 400
    totals = []
    for line in SECTIONS_PATH.open(encoding="utf-8"):
        sec = json.loads(line)
        parts = [sec["heading"]]
        for b in sec["content"]:
            if b["kind"] == "para":
                parts.append(b["text"])
            elif b["kind"] == "bullets":
                parts += b["items"]
            elif b["kind"] == "table":
                parts.append(" ".join(b["header"]))
                parts += [" ".join(r) for r in b["rows"]]
        totals.append(len(enc.encode("\n".join(parts))))
    return round(sum(totals) / max(1, len(totals)))


def _train_tokens(enc) -> int:
    if not TRAIN_PATH.exists():
        return 0
    total = 0
    for line in TRAIN_PATH.open(encoding="utf-8"):
        for m in json.loads(line)["messages"]:
            total += len(enc.encode(m["content"]))
    return total


def _input_tokens(p: Producer, questions: list[int], sys_tokens: int, ctx_tokens: int) -> int:
    """Total input tokens this producer sends across all 100 questions."""
    per_q = sys_tokens + (ctx_tokens if p.method == RAG else 0)
    return sum(q + per_q for q in questions)


def _ft_rate(p: Producer):
    """A fine-tuned model bills above its base — see FT_INFERENCE_MULTIPLIER."""
    base = price_for(p.model)
    if p.is_local:
        return base
    return type(base)(base.inp * FT_INFERENCE_MULTIPLIER, base.out * FT_INFERENCE_MULTIPLIER,
                      base.train, base.source)


def cmd_estimate(args) -> None:
    from build_dataset.split import SYSTEM_PROMPT

    enc = _encoder()
    epochs = args.epochs
    top_k = args.top_k
    max_out = args.max_output

    test = load_test()
    q_tokens = [len(enc.encode(r["question"])) for r in test]
    sys_tokens = len(enc.encode(SYSTEM_PROMPT))
    sec_tokens = _mean_section_tokens(enc)
    ctx_tokens = sec_tokens * top_k
    train_tokens = _train_tokens(enc)

    print(f"\nMeasured from the frozen dataset (no API calls):")
    print(f"  test questions      : {len(test)}  (mean {sum(q_tokens)//max(1,len(q_tokens))} tok)")
    print(f"  system prompt       : {sys_tokens} tok  (identical for every producer)")
    print(f"  mean section        : {sec_tokens} tok  -> RAG context = {top_k} x {sec_tokens} "
          f"= {ctx_tokens} tok")
    print(f"  train.jsonl         : {train_tokens:,} tok  x {epochs} epochs = "
          f"{train_tokens * epochs:,} trained tok")
    print(f"  output charged at   : {max_out} tok/answer (the cap — this is a CEILING)\n")

    rows, paid_total, train_total = [], 0.0, 0.0
    for p in PRODUCERS:
        in_tok = _input_tokens(p, q_tokens, sys_tokens, ctx_tokens)
        out_tok = max_out * TEST_SIZE
        rate = _ft_rate(p) if p.method == FINETUNED else price_for(p.model)
        infer = (in_tok * rate.inp + out_tok * rate.out) / 1_000_000

        train_cost = 0.0
        if p.method == FINETUNED and p.provider == OPENAI:
            base = price_for(p.model)
            train_cost = (train_tokens * epochs * (base.train or 0.0)) / 1_000_000
            train_total += train_cost
        elif p.trains_remotely:
            # Charge Together's flat per-job floor, not the tokens: our tokens are worth
            # ~$0.04 against a $4.00 minimum, so billing them under-reports by ~100x.
            train_cost = TOGETHER_JOB_MINIMUM_USD
            train_total += train_cost

        paid_total += infer + train_cost
        rows.append((p, in_tok, out_tok, infer, train_cost))

    w = max(len(p.key) for p in PRODUCERS)
    print(f"  {'producer':<{w}}  {'calls':>5}  {'in tok':>9}  {'out tok':>8}  "
          f"{'infer $':>9}  {'train $':>9}")
    print("  " + "-" * (w + 48))
    for p, in_tok, out_tok, infer, train_cost in rows:
        if p.trains_remotely:
            tag = "  (trained on Together, ANSWERS free & local)"
        elif p.is_local:
            tag = "  (free)"
        else:
            tag = ""
        print(f"  {p.key:<{w}}  {TEST_SIZE:>5}  {in_tok:>9,}  {out_tok:>8,}  "
              f"{infer:>9.2f}  {train_cost:>9.2f}{tag}")

    n_answers = TEST_SIZE * len(PRODUCERS)
    print("  " + "-" * (w + 48))
    print(f"\n  Phase B — inference          $ {paid_total - train_total:7.2f}")
    print(f"  Phase B — FT training (1x)   $ {train_total:7.2f}")
    print(f"  Phase B TOTAL                $ {paid_total:7.2f}   ({n_answers:,} answers)")
    # Phase C is NOT projected here. The judge's bill is dominated by the source document it
    # reads on every call, and only `judge estimate` knows the real rubric and the real
    # documents — so it owns that number. Two estimates of one figure is one too many.
    print(f"\n  Phase C — judge ({JUDGE_MODEL}):  python -m judge estimate")
    print("    (it reads the real rubric + the real source documents — also $0, no API calls)\n")

    if not PRICES_VERIFIED:
        print("  " + "=" * 68)
        print("  ⚠️  PRICES ARE UNVERIFIED — do not trust these figures yet.")
        print("  The PRICING table in models.py ships with best-known values that")
        print("  have NOT been checked against the live pricing pages. Confirm each")
        print("  row below, correct it, then set PRICES_VERIFIED = True.")
        print("  " + "=" * 68)
        seen = set()
        for p in PRODUCERS:
            pr = price_for(p.model)
            if pr.source and pr.source not in seen and not p.is_local:
                seen.add(pr.source)
                print(f"    {pr.source}")
        print(f"    {PRICING[JUDGE_MODEL].source}   (judge)")
        print()
