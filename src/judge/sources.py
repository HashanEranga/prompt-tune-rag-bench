"""question_id → the full cleaned source document the answer must be judged against.

Rule #2 in one function. Each test question carries the ``clean_path`` of the document its
Q&A pair was derived from, and all 100 resolve against ``data/clean/``.

Why the whole document rather than the one tagged section: the section is ~100 tokens, and
an answer that correctly draws on a neighbouring section of the same document would be
marked unfaithful for it — punishing the model for the dataset's own segmentation. The 13
cleaned documents total 60 KB, so handing over the whole file costs little and removes that
artefact entirely.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from .config import CLEAN_DIR, ROOT


@lru_cache(maxsize=None)
def _read(path: str) -> str:
    p = Path(path)
    if not p.is_absolute():
        p = ROOT / p
    if not p.exists():
        raise SystemExit(
            f"source document not found: {p}\nPhase A's data/clean/ must be present — the "
            f"judge scores against the real documents, never against the drafted answers.")
    return p.read_text(encoding="utf-8").strip()


def document_for(question: dict) -> str:
    """The cleaned document behind one test question."""
    return _read(question["clean_path"])


def assert_all_resolve(questions: list[dict]) -> None:
    """Fail before spending, not at answer 700. A question whose document is missing cannot
    be judged against a document at all."""
    missing = sorted({q["clean_path"] for q in questions
                      if not (ROOT / q["clean_path"]).exists()})
    if missing:
        raise SystemExit(f"{len(missing)} source document(s) missing from {CLEAN_DIR}:\n  "
                         + "\n  ".join(missing))
