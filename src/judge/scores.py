"""``results/scores.csv`` — the Phase C deliverable and the master table's input.

One row per (question_id, producer): the same 1,000 keys as answers.jsonl, so the two join
cleanly. Append-only and resume-safe, exactly like answers.jsonl: a crash partway through
collection costs nothing, and a re-run scores only what is missing.
"""
from __future__ import annotations

import csv
from dataclasses import asdict, dataclass, fields
from datetime import datetime, timezone

from contenders.answers import TEST_SIZE, load_test
from contenders.models import BY_KEY, PRODUCERS

from .config import RESULTS_DIR, SCORES_PATH
from .rubric import Verdict


@dataclass(frozen=True)
class ScoreRecord:
    question_id: str
    producer: str
    method: str
    faithfulness: int
    accuracy: int
    completeness: int
    clarity: int
    safety_flag: bool
    weighted: float
    reason: str
    judge_model: str
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float
    ts: str = ""

    @staticmethod
    def now() -> str:
        return datetime.now(timezone.utc).isoformat(timespec="seconds")

    @classmethod
    def from_verdict(cls, question_id: str, producer: str, v: Verdict, judge_model: str,
                     prompt_tokens: int, completion_tokens: int, cost_usd: float
                     ) -> "ScoreRecord":
        return cls(question_id=question_id, producer=producer,
                   method=BY_KEY[producer].method,
                   faithfulness=v.faithfulness, accuracy=v.accuracy,
                   completeness=v.completeness, clarity=v.clarity,
                   safety_flag=v.safety_flag, weighted=v.weighted(), reason=v.reason,
                   judge_model=judge_model, prompt_tokens=prompt_tokens,
                   completion_tokens=completion_tokens, cost_usd=cost_usd,
                   ts=cls.now())


COLUMNS = [f.name for f in fields(ScoreRecord)]


def load_all() -> list[dict]:
    if not SCORES_PATH.exists():
        return []
    with SCORES_PATH.open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def load_done() -> set[tuple[str, str]]:
    """(question_id, producer) pairs already scored — what a re-run skips."""
    return {(r["question_id"], r["producer"]) for r in load_all()}


def append(rec: ScoreRecord) -> None:
    """Append one verdict, header-on-create, flushed immediately."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    new = not SCORES_PATH.exists()
    with SCORES_PATH.open("a", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        if new:
            w.writeheader()
        w.writerow(asdict(rec))


def judges_used() -> set[str]:
    """Which judge models produced these scores. More than one means two judges have been
    blended into one table — Rule #3 requires a single consistent judge, so this is fatal."""
    return {r["judge_model"] for r in load_all()}


def spend() -> float:
    return sum(float(r["cost_usd"]) for r in load_all())


def assert_complete() -> None:
    """Phase C is done only when every answer has been judged. A missing row would shrink a
    producer's denominator and skew its mean — the same failure assert_complete() guards
    against in Phase B, for the same reason."""
    rows = load_all()
    test_ids = {q["id"] for q in load_test()}
    expected = TEST_SIZE * len(PRODUCERS)

    judges = judges_used()
    assert len(judges) <= 1, (
        f"scores.csv mixes {len(judges)} judge models ({', '.join(sorted(judges))}). "
        f"One rubric, one judge — re-score from scratch rather than blending two.")

    counts: dict[str, int] = {}
    for r in rows:
        counts[r["producer"]] = counts.get(r["producer"], 0) + 1

    for p in PRODUCERS:
        got = counts.get(p.key, 0)
        assert got == TEST_SIZE, f"{p.key}: {got} scores, expected {TEST_SIZE}"
        judged = {r["question_id"] for r in rows if r["producer"] == p.key}
        missing = test_ids - judged
        assert not missing, (f"{p.key} missing {len(missing)} scores, "
                             f"e.g. {sorted(missing)[:3]}")

    assert len(rows) == expected, f"{len(rows)} rows, expected {expected}"
    print(f"✓ complete: {len(PRODUCERS)} producers x {TEST_SIZE} questions = "
          f"{expected} scored answers, judged by {judges.pop() if judges else '—'}")
