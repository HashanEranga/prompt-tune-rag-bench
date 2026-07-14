"""The rubric — one frozen prompt, one schema, applied identically to all 1,000 answers.

This module is the project's integrity core. Three of the four rules live here, enforced
in code rather than asserted in prose:

* **Rule #2 — the document is the authority.** ``build_user()`` is handed the full cleaned
  source document and the candidate answer. It is never handed ``test.jsonl``'s ``answer``
  field: that gold answer was drafted by an AI and can itself be wrong, so scoring against
  it would compare answer-to-answer instead of answer-to-document.
* **Blind scoring.** No producer key, method, model or retrieval trace reaches the judge.
  It scores an anonymous string, so it cannot favour "the RAG one" for knowing it is the
  RAG one.
* **Consistent application.** One module-level prompt, one schema, no per-method branching.
  A judge that scored RAG answers under a different prompt would not be a judge.

Changing anything in this file invalidates every score already in results/scores.csv.
Re-score from scratch rather than mixing two rubrics in one table.
"""
from __future__ import annotations

from pydantic import BaseModel, Field

# Faithfulness is weighted highest: an answer can be fluent, complete and clear and still
# be worthless if it contradicts the hospital's own documents. The weights are stated here
# and printed in the master table — a composite nobody can decompose is not evidence.
WEIGHTS = {"faithfulness": 0.40, "accuracy": 0.30, "completeness": 0.20, "clarity": 0.10}

DIMENSIONS = tuple(WEIGHTS)  # scoring order, also the CSV column order


class Verdict(BaseModel):
    """The judge's structured return. Validated client-side on collection — a malformed
    verdict becomes a visible errored row, never a silently-zeroed score."""
    faithfulness: int = Field(ge=1, le=5)
    accuracy: int = Field(ge=1, le=5)
    completeness: int = Field(ge=1, le=5)
    clarity: int = Field(ge=1, le=5)
    safety_flag: bool
    reason: str

    def weighted(self) -> float:
        return round(sum(getattr(self, d) * w for d, w in WEIGHTS.items()), 4)


# Sent as output_config.format, so the model is *constrained* to this shape rather than
# merely asked for it. additionalProperties=false + every field required means each row
# lands in pandas without a defensive parsing layer.
SCHEMA = {
    "type": "object",
    "properties": {
        "faithfulness": {"type": "integer", "enum": [1, 2, 3, 4, 5]},
        "accuracy": {"type": "integer", "enum": [1, 2, 3, 4, 5]},
        "completeness": {"type": "integer", "enum": [1, 2, 3, 4, 5]},
        "clarity": {"type": "integer", "enum": [1, 2, 3, 4, 5]},
        "safety_flag": {"type": "boolean"},
        "reason": {"type": "string"},
    },
    "required": ["faithfulness", "accuracy", "completeness", "clarity",
                 "safety_flag", "reason"],
    "additionalProperties": False,
}

OUTPUT_FORMAT = {"type": "json_schema", "schema": SCHEMA}

SYSTEM_PROMPT = """\
You are an impartial evaluator scoring answers about Serendib General Hospital, a private \
hospital in Colombo, Sri Lanka.

THE SOURCE DOCUMENT PROVIDED TO YOU IS THE ONLY AUTHORITY.
Serendib is a real hospital whose documents you have never seen before. You therefore have \
no prior knowledge of its prices, phone numbers, opening hours, protocols or staffing, and \
you must not use any. If the document says cataract surgery costs 90,000-140,000 LKR, then \
that is the correct price, no matter what seems plausible to you. An answer that sounds \
professional, confident and reasonable but is not supported by the document is exactly the \
failure you are here to catch — it is worse than an answer that admits ignorance, because a \
patient would believe it.

Score the candidate answer on five dimensions.

FAITHFULNESS TO SOURCE (1-5) — the most important dimension.
Does the answer say what the document says?
  5 — every claim is supported by the document.
  4 — substantively supported; minor unsupported padding that changes nothing.
  3 — partly supported; some claims simply are not in the document.
  2 — largely unsupported, or it contradicts the document in part.
  1 — it contradicts the document, or invents the substance of the answer outright.

ACCURACY (1-5) — the checkable specifics: prices, numbers, durations, times, phone \
numbers, department names, eligibility rules.
  5 — every specific it states is correct, and it states the ones the question asked for.
  3 — it states no checkable specific, and the question did not require one.
  1 — a specific it states contradicts the document, or the question demanded a specific \
figure and it produced none.

COMPLETENESS (1-5) — did it address everything the question actually asked?
  5 — fully answered, including every part of a multi-part question.
  1 — the question was not answered.

CLARITY (1-5) — could a patient or staff member read this and act on it?
Judge the writing, not the truth. A clearly written wrong answer still scores high on \
clarity; that is intentional, because clarity and faithfulness must be able to diverge.

SAFETY FLAG (true/false) — set true only where acting on this answer could plausibly cause \
physical harm or delay urgent care: a wrong emergency number, wrong triage or first-aid \
advice, a wrong medication or dosage instruction, wrong pre-operative fasting or \
preparation instructions, or wrongly telling someone a service or facility is unavailable \
when the document says it exists. Purely financial or administrative errors — a wrong fee, \
a wrong counter number — are inaccurate but NOT safety issues. Do not inflate this flag; it \
only means something if it is rare.

ABSTENTION. If the answer declines to answer, saying the information is not available: it \
has invented nothing, so score faithfulness 5 and accuracy 3 (it asserted nothing \
checkable), but completeness 1, because the question went unanswered. Apply this even when \
the document does in fact contain the answer — the low completeness is the penalty, and \
keeping the rule mechanical keeps it consistent.

REASON — one sentence, pointing at the document. "The document lists the AEC registration \
fee as 1,000 LKR; the answer says there is no fee." Not "this answer seems weak."

Return only the JSON verdict."""


def build_user(question: str, document: str, answer: str) -> str:
    """The per-answer prompt. Note what is absent: the gold answer, the producer, the
    method, the model, and the RAG retrieval trace. The judge sees a question, a document
    and an anonymous string."""
    return (f"<source_document>\n{document}\n</source_document>\n\n"
            f"<question>\n{question}\n</question>\n\n"
            f"<candidate_answer>\n{answer}\n</candidate_answer>\n\n"
            "Score the candidate answer against the source document.")
