"""``results/answers.jsonl`` — the Phase B deliverable and the Phase C input.

One record per (question_id, producer): 10 producers x 100 questions = 1,000 rows.
``question_id`` joins back to ``test.jsonl.id``, which carries the gold answer, the
verbatim evidence and the source path the judge scores against.

The file is append-only and resume-safe: a rerun skips what is already on disk, so a
crash at answer 900 costs nothing and no paid call is ever paid for twice.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone

from .config import ANSWERS_PATH, RESULTS_DIR, TEST_PATH
from .models import BY_KEY, Producer

TEST_SIZE = 100


@dataclass(frozen=True)
class AnswerRecord:
    question_id: str
    producer: str
    method: str
    model: str
    answer: str
    latency_s: float
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float
    size: str | None = None
    retrieved: list[dict] = field(default_factory=list)  # RAG only: what was pulled, and did it hit
    error: str | None = None
    ts: str = ""

    @staticmethod
    def now() -> str:
        return datetime.now(timezone.utc).isoformat(timespec="seconds")


def load_test() -> list[dict]:
    """The 100 locked test questions. Frozen in Phase A — never regenerate."""
    return [json.loads(l) for l in TEST_PATH.open(encoding="utf-8")]


def load_all() -> list[dict]:
    if not ANSWERS_PATH.exists():
        return []
    return [json.loads(l) for l in ANSWERS_PATH.open(encoding="utf-8")]


def load_done() -> set[tuple[str, str]]:
    """(question_id, producer) pairs already answered. Errored rows are NOT
    counted as done, so a rerun retries them."""
    return {(r["question_id"], r["producer"]) for r in load_all() if not r.get("error")}


def append(rec: AnswerRecord) -> None:
    """Append one answer, flushed immediately — a crash must not lose paid work."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with ANSWERS_PATH.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(asdict(rec), ensure_ascii=False) + "\n")


def models_used(producer: str) -> set[str]:
    """Which model ids have answered under this producer key. More than one means a
    swapped model has contaminated the producer — see the guard in runner.py."""
    return {r["model"] for r in load_all()
            if r["producer"] == producer and not r.get("error")}


def drop_rows(producer: str | None = None, errors_only: bool = False) -> int:
    """Rewrite answers.jsonl without the matching rows, returning how many were dropped.
    Needed when a producer's model changes: its old answers are no longer comparable and
    must not be mixed in under the same key.
    """
    rows = load_all()
    if not rows:
        return 0

    def doomed(r: dict) -> bool:
        if producer and r["producer"] != producer:
            return False
        if errors_only and not r.get("error"):
            return False
        return True

    keep = [r for r in rows if not doomed(r)]
    dropped = len(rows) - len(keep)
    if not dropped:
        return 0
    with ANSWERS_PATH.open("w", encoding="utf-8") as fh:
        for r in keep:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    return dropped


def counts_by_producer() -> dict[str, int]:
    out: dict[str, int] = {}
    for r in load_all():
        if not r.get("error"):
            out[r["producer"]] = out.get(r["producer"], 0) + 1
    return out


def spend_by_producer() -> dict[str, float]:
    out: dict[str, float] = {}
    for r in load_all():
        out[r["producer"]] = out.get(r["producer"], 0.0) + r.get("cost_usd", 0.0)
    return out


def assert_complete(producers: tuple[Producer, ...] = ()) -> None:
    """Phase B is done only when every producer answered every question — a missing row
    would silently shrink a producer's denominator and skew its mean score."""
    wanted = producers or tuple(BY_KEY.values())
    rows = load_all()
    test_ids = {r["id"] for r in load_test()}
    errors = [r for r in rows if r.get("error")]
    counts = counts_by_producer()

    for p in wanted:
        got = counts.get(p.key, 0)
        assert got == TEST_SIZE, f"{p.key}: {got} answers, expected {TEST_SIZE}"
        answered = {r["question_id"] for r in rows
                    if r["producer"] == p.key and not r.get("error")}
        missing = test_ids - answered
        assert not missing, f"{p.key} missing {len(missing)} questions, e.g. {sorted(missing)[:3]}"

    assert not errors, f"{len(errors)} errored rows remain, e.g. {errors[0]['error'][:80]}"
    print(f"✓ complete: {len(wanted)} producers x {TEST_SIZE} questions = "
          f"{len(wanted) * TEST_SIZE} answers, 0 errors")
