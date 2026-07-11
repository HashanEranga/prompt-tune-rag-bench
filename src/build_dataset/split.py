"""Stage 4 — stratified, seeded train/test split with a leakage guard.

Fully deterministic: verified ``data/qa/pool.jsonl`` -> ``train.jsonl`` +
``test.jsonl`` + a frozen ``MANIFEST.md``.
"""
from __future__ import annotations

import hashlib
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

from .config import QA_DIR, ROOT

SEED = 42
TEST_SIZE = 100
DUP_THRESHOLD = 0.80  # Jaccard token overlap above which two questions are "duplicates"
SYSTEM_PROMPT = (
    "You are a helpful assistant for Serendib General Hospital in Colombo, Sri Lanka. "
    "Answer patient and staff questions using the hospital's policies, procedures, and "
    "published figures. Be concise, specific, and accurate."
)


def _norm_tokens(q: str) -> frozenset:
    return frozenset(re.sub(r"[^a-z0-9 ]", " ", q.lower()).split())


def _jaccard(a: frozenset, b: frozenset) -> float:
    return len(a & b) / max(1, len(a | b))


def _same_fact(r1: dict, r2: dict) -> bool:
    """Two pairs are the same fact only if BOTH the question and the answer are
    near-identical — so template siblings ('cost of appendectomy' vs 'cost of
    cholecystectomy', different answers) are kept as distinct facts."""
    return (_jaccard(_norm_tokens(r1["question"]), _norm_tokens(r2["question"])) >= DUP_THRESHOLD
            and _jaccard(_norm_tokens(r1["answer"]), _norm_tokens(r2["answer"])) >= DUP_THRESHOLD)


def _dedup(pool: list[dict]) -> tuple[list[dict], list[dict]]:
    """Drop same-fact duplicate pairs (keep the first) so no fact straddles the
    train/test line. Returns (unique, dropped)."""
    kept, dropped = [], []
    for r in pool:
        if any(_same_fact(r, k) for k in kept):
            dropped.append(r)
        else:
            kept.append(r)
    return kept, dropped


def _allocate(sizes: dict, k: int) -> dict:
    """Largest-remainder proportional allocation of k test slots across docs."""
    total = sum(sizes.values())
    raw = {d: sizes[d] * k / total for d in sizes}
    quota = {d: int(raw[d]) for d in sizes}
    for d in sorted(sizes, key=lambda d: raw[d] - quota[d], reverse=True)[:k - sum(quota.values())]:
        quota[d] += 1
    return {d: min(quota[d], sizes[d]) for d in sizes}


def cmd_split(args) -> None:
    pool = [json.loads(l) for l in (QA_DIR / "pool.jsonl").open(encoding="utf-8")]
    verified = [r for r in pool if r.get("verified")]
    if not args.all and len(verified) < TEST_SIZE:
        sys.exit(
            f"Only {len(verified)} of {len(pool)} pairs are marked verified — need at "
            f"least {TEST_SIZE} for the test set. Verify pairs (set \"verified\": true "
            f"in pool.jsonl after checking data/qa/verify_sheet.md), or re-run with "
            f"--all to build a PREVIEW split from the full pool.")
    working = verified if verified else pool
    preview = not verified
    working, dropped = _dedup(working)

    by_doc: dict = {}
    for r in working:
        by_doc.setdefault(r["doc_id"], []).append(r)
    quota = _allocate({d: len(v) for d, v in by_doc.items()}, TEST_SIZE)

    rng = random.Random(SEED)
    test = []
    for d in sorted(by_doc):
        items = by_doc[d][:]
        rng.shuffle(items)
        # round-robin across answer_types so the quota gets a balanced mix
        buckets: dict = {}
        for it in items:
            buckets.setdefault(it["answer_type"], []).append(it)
        order, queues = [], [buckets[t] for t in sorted(buckets)]
        while any(queues):
            for qd in queues:
                if qd:
                    order.append(qd.pop(0))
        test += order[:quota[d]]
    # top up / trim to exactly TEST_SIZE
    test_ids = {r["id"] for r in test}
    if len(test) < TEST_SIZE:
        extra = [r for r in working if r["id"] not in test_ids]
        rng.shuffle(extra)
        test += extra[:TEST_SIZE - len(test)]
        test_ids = {r["id"] for r in test}
    test = test[:TEST_SIZE]
    test_ids = {r["id"] for r in test}
    train = [r for r in working if r["id"] not in test_ids]

    # ---- integrity assertions ----
    assert len(test) == TEST_SIZE, f"test set is {len(test)}, expected {TEST_SIZE}"
    assert not (test_ids & {r["id"] for r in train}), "train/test id overlap"
    for tr in train:
        assert not any(_same_fact(tr, te) for te in test), \
            f"same-fact question leaks across split: {tr['id']}"

    # ---- write test.jsonl (full records for the judge) ----
    test_path = QA_DIR / "test.jsonl"
    keep = ["id", "question", "answer", "source_doc", "clean_path", "doc_id",
            "section", "evidence", "topic", "answer_type"]
    with test_path.open("w", encoding="utf-8") as fh:
        for r in test:
            fh.write(json.dumps({k: r[k] for k in keep}, ensure_ascii=False) + "\n")

    # ---- write train.jsonl (OpenAI-FT chat format + passthrough metadata) ----
    train_path = QA_DIR / "train.jsonl"
    with train_path.open("w", encoding="utf-8") as fh:
        for r in train:
            rec = {
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": r["question"]},
                    {"role": "assistant", "content": r["answer"]},
                ],
                "id": r["id"], "source_doc": r["source_doc"],
                "topic": r["topic"], "answer_type": r["answer_type"],
            }
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def sha(p: Path) -> str:
        return hashlib.sha256(p.read_bytes()).hexdigest()

    # ---- MANIFEST.md (frozen split record) ----
    tdoc, tetype = Counter(r["doc_id"] for r in train), Counter(r["answer_type"] for r in test)
    edoc = Counter(r["doc_id"] for r in test)
    man = [
        "# Phase A dataset manifest",
        "",
        ("> ⚠️ **PREVIEW** — built from the full pool with `--all`; not yet human-verified. "
         "Re-run `split` after verification to freeze." if preview else
         "Frozen split of the human-verified pool."),
        "",
        f"- Seed: `{SEED}`",
        f"- Pool: {len(pool)} pairs; near-duplicates dropped: {len(dropped)}; "
        f"working set: {len(working)}",
        f"- **Train: {len(train)}** → `train.jsonl`  ·  **Test: {len(test)}** → `test.jsonl`",
        f"- sha256(train.jsonl): `{sha(train_path)}`",
        f"- sha256(test.jsonl):  `{sha(test_path)}`",
        "",
        "## Test-set coverage by document",
        "", "| doc_id | test | train |", "|---|---|---|",
    ]
    for d in sorted(by_doc):
        man.append(f"| {d} | {edoc.get(d, 0)} | {tdoc.get(d, 0)} |")
    man += ["", "## Test-set answer-type mix", "", "| answer_type | count |", "|---|---|"]
    for t, c in tetype.most_common():
        man.append(f"| {t} | {c} |")
    man.append("")
    (QA_DIR / "MANIFEST.md").write_text("\n".join(man), encoding="utf-8")

    tag = " (PREVIEW — pool not yet verified)" if preview else ""
    print(f"Split{tag}: {len(train)} train / {len(test)} test "
          f"(dropped {len(dropped)} near-dupes) -> {QA_DIR.relative_to(ROOT)}/")
    print(f"  all 13 docs in test: {len(edoc) == len(by_doc)}; "
          f"test answer_types: {dict(tetype)}")
