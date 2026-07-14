"""The shared answer loop — used by prompting, fine-tuning and RAG alike.

One loop for all three methods is what makes the comparison controlled: every producer is
resumed, retried, timed, priced and logged by identical code, so a score difference can
only come from the method, never from the harness.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

from .answers import AnswerRecord, append, drop_rows, load_done, models_used
from .clients import call
from .models import GOOGLE, OPENAI, Producer, resolve_model

# Gemini stays below OpenAI: 4 is comfortable on a Tier-1 key, but drop it to 2 on a free
# one, where 8 at once triggers 429 RESOURCE_EXHAUSTED faster than backoff can absorb.
WORKERS = {OPENAI: 8, GOOGLE: 4}
DEFAULT_WORKERS = 4


def run_producer(p: Producer, questions: list[dict], build_prompt=None) -> None:
    """Answer every question with one producer, skipping what's already on disk.

    ``build_prompt(question_record) -> (user_text, retrieved)`` lets RAG inject retrieved
    context and record what it pulled. Prompting and fine-tuned producers leave it None
    and send the bare question — no documents, which is the point of the baseline.
    """
    model = resolve_model(p)

    # Resuming a producer that already has answers from a DIFFERENT model would blend the
    # two under one key, and the judge would score the mix as a single contender.
    stale = models_used(p.key) - {model}
    if stale:
        raise SystemExit(
            f"{p.key} already has answers from {', '.join(sorted(stale))}, but is now "
            f"configured as {model}.\nResuming would mix two models under one producer "
            f"key and silently invalidate the comparison.\n\nDiscard the old answers "
            f"first:\n    python -m contenders reset --producer {p.key}")

    # Clear this producer's failed rows before retrying them, or every re-run appends a
    # fresh error row while the old ones linger.
    stale_errors = drop_rows(producer=p.key, errors_only=True)
    if stale_errors:
        print(f"  {p.key}: clearing {stale_errors} errored row(s) to retry them")

    done = load_done()
    todo = [q for q in questions if (q["id"], p.key) not in done]
    if not todo:
        print(f"  {p.key}: already complete ({len(questions)} answers) — skipping")
        return
    skipped = len(questions) - len(todo)
    note = f", resuming ({skipped} already done)" if skipped else ""
    print(f"\n{p.key}  [{p.method} · {model}]  {len(todo)} to answer{note}")

    def one(q: dict) -> AnswerRecord:
        user, retrieved = (build_prompt(q) if build_prompt else (q["question"], []))
        c = call(p, model, user)
        return AnswerRecord(
            question_id=q["id"], producer=p.key, method=p.method, model=model,
            answer=c.text, latency_s=round(c.latency_s, 3),
            prompt_tokens=c.prompt_tokens, completion_tokens=c.completion_tokens,
            cost_usd=c.cost_usd, size=p.size, retrieved=retrieved,
            error=c.error, ts=AnswerRecord.now())

    # Local producers run serially: parallelising them thrashes the one 8 GB GPU and
    # corrupts the latency numbers, which are a deliverable.
    if p.is_local:
        records = (one(q) for q in todo)
    else:
        pool = ThreadPoolExecutor(max_workers=WORKERS.get(p.provider, DEFAULT_WORKERS))
        records = pool.map(one, todo)

    spent, failed = 0.0, 0
    for rec in records:
        append(rec)
        spent += rec.cost_usd
        if rec.error:
            failed += 1
            print(f"  ! {rec.question_id}  {rec.error[:70]}")
        else:
            print(f"  ✓ {rec.question_id}  {rec.latency_s:5.2f}s  "
                  f"{rec.completion_tokens:>3} tok  ${rec.cost_usd:.5f}")
    tag = f", {failed} FAILED (re-run to retry)" if failed else ""
    print(f"  {p.key}: {len(todo) - failed}/{len(todo)} answered, ${spent:.4f}{tag}")
