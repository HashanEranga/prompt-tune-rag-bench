"""Step 4 — score all 1,000 answers through the Message Batches API.

Split deliberately into two commands, for the same reason Phase B split the Together
fine-tune into ``--train-only`` and ``--answer-only``:

    submit   the irreversible spend. One button, run once.
    collect  free, re-runnable, resume-safe. Poll and write.

Batching halves the price of every token, and scoring 1,000 answers is the least
latency-sensitive job in the project — nothing is waiting on the verdicts but us.
"""
from __future__ import annotations

import json
import re
import time

from anthropic import Anthropic
from dotenv import load_dotenv
from pydantic import ValidationError

from contenders.answers import load_all as load_answers, load_test
from contenders.models import BY_KEY, JUDGE_MODEL, PRODUCERS, cost_usd

from .config import BATCH_PATH, SCORES_PATH
from .rubric import OUTPUT_FORMAT, SYSTEM_PROMPT, Verdict, build_user
from .scores import ScoreRecord, append, load_done
from .sources import assert_all_resolve, document_for

load_dotenv()

# The Batches API bills every token at half price. Applied to the usage the API reports
# back, so scores.csv carries what the run actually cost, not a projection.
BATCH_DISCOUNT = 0.5

# Room for the verdict and nothing else. The rubric is a bounded extraction against a
# document that is right there in the prompt, not a puzzle — see THINKING below.
MAX_OUTPUT_TOKENS = 512

# ⚠️ NO `temperature` HERE, deliberately. Phase B pinned every answerer at temperature=0.0
# for reproducibility, and the reflex is to do the same for the judge. Claude Sonnet 5
# REJECTS a non-default sampling parameter with a 400 — `temperature`, `top_p` and `top_k`
# are gone. Decoding is steered by the prompt now, so the rubric carries that weight.
#
# Thinking is disabled by default. Adaptive thinking would roughly double the run's output
# tokens for a task that is a bounded 5-field judgement against a document already in the
# prompt, and fewer moving parts means the rubric is applied more consistently — which is
# the thing Phase C is actually being graded on. `--thinking` opts into adaptive reasoning
# at low effort if you want to pay for a more deliberative judge; `judge estimate` prices
# both so the choice is made against numbers.
THINKING_OFF = {"type": "disabled"}
THINKING_ON = {"type": "adaptive"}

POLL_SECONDS = 30

# Anthropic constrains custom_id to ^[a-zA-Z0-9_-]{1,64}$ — and producer keys contain a
# dot (`prompt-llama3.1-8b`), which is not in that set. So the id is built from the
# producer's INDEX, and results/judge_batch.json carries the manifest that maps it back.
# Never re-derive the mapping from PRODUCERS order at collect time: reordering the registry
# between submit and collect would silently reassign every score to the wrong producer.
CUSTOM_ID_RE = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")


def _params(question: dict, answer_text: str, thinking: bool) -> dict:
    return {
        "model": JUDGE_MODEL,
        "max_tokens": MAX_OUTPUT_TOKENS,
        "system": SYSTEM_PROMPT,
        "messages": [{"role": "user", "content": build_user(
            question=question["question"],
            document=document_for(question),
            answer=answer_text)}],
        "thinking": THINKING_ON if thinking else THINKING_OFF,
        # Constrained decoding: the model cannot return a shape other than the rubric's.
        "output_config": {"format": OUTPUT_FORMAT},
    }


def _pending(limit: int | None) -> list[tuple[str, dict, dict]]:
    """(custom_id, question, answer_row) for every answer not yet scored.

    Ordered question-major so `--limit 5` smoke-tests one question across five different
    producers — a spread that shows the rubric discriminating, rather than five near-identical
    answers from one model.
    """
    test = {q["id"]: q for q in load_test()}
    assert_all_resolve(list(test.values()))
    index = {p.key: i for i, p in enumerate(PRODUCERS)}
    done = load_done()

    answers = {(r["question_id"], r["producer"]): r for r in load_answers()
               if not r.get("error")}

    out = []
    for qid in test:
        for p in PRODUCERS:
            key = (qid, p.key)
            if key in done or key not in answers:
                continue
            cid = f"{qid}--p{index[p.key]:02d}"
            assert CUSTOM_ID_RE.match(cid), f"custom_id {cid!r} is not API-safe"
            out.append((cid, test[qid], answers[key]))
            if limit and len(out) >= limit:
                return out
    return out


def cmd_submit(args) -> None:
    pending = _pending(args.limit)
    if not pending:
        print("\n  Every answer is already scored — nothing to submit. "
              "Run `judge status` or `judge aggregate`.\n")
        return

    thinking = bool(args.thinking)
    print(f"\n  judge      : {JUDGE_MODEL}")
    print(f"  thinking   : {'adaptive' if thinking else 'disabled'}")
    print(f"  to score   : {len(pending)} answers")
    print(f"  discount   : Batches API — every token at {BATCH_DISCOUNT:.0%}\n")

    requests, manifest = [], {}
    for cid, question, answer in pending:
        requests.append({"custom_id": cid,
                         "params": _params(question, answer["answer"], thinking)})
        manifest[cid] = {"question_id": question["id"], "producer": answer["producer"]}

    client = Anthropic()
    batch = client.messages.batches.create(requests=requests)

    # Persist the id BEFORE anything else can fail. A submitted batch whose id we lost is
    # money spent on verdicts we cannot fetch.
    BATCH_PATH.write_text(json.dumps({
        "batch_id": batch.id,
        "judge_model": JUDGE_MODEL,
        "thinking": thinking,
        "submitted": len(requests),
        "manifest": manifest,
    }, indent=2), encoding="utf-8")

    print(f"  ✓ submitted  {batch.id}   status: {batch.processing_status}")
    print(f"    manifest written to {BATCH_PATH.name}\n")
    print("  Next:  python -m judge collect     (free, re-runnable, resumes)\n")


def _text_of(message) -> str:
    """The JSON verdict. output_config guarantees a text block holding valid JSON; with
    --thinking, thinking blocks precede it, so select by type rather than by position."""
    for block in message.content:
        if block.type == "text":
            return block.text
    raise ValueError(f"no text block in response (stop_reason={message.stop_reason})")


def cmd_collect(args) -> None:
    if not BATCH_PATH.exists():
        raise SystemExit("no results/judge_batch.json — submit a batch first:\n"
                         "    python -m judge submit --limit 5")
    state = json.loads(BATCH_PATH.read_text(encoding="utf-8"))
    batch_id, manifest = state["batch_id"], state["manifest"]
    judge_model = state["judge_model"]

    client = Anthropic()
    print(f"\n  batch  {batch_id}   ({state['submitted']} requests)")

    while True:
        batch = client.messages.batches.retrieve(batch_id)
        if batch.processing_status == "ended":
            break
        c = batch.request_counts
        print(f"    {batch.processing_status}  processing={c.processing} "
              f"succeeded={c.succeeded} errored={c.errored}  "
              f"(polling every {POLL_SECONDS}s — Ctrl-C is safe, re-run collect)")
        time.sleep(POLL_SECONDS)

    done = load_done()
    written = failed = skipped = 0
    spent = 0.0

    # Results come back in ARBITRARY order — key by custom_id, never by position.
    for result in client.messages.batches.results(batch_id):
        entry = manifest.get(result.custom_id)
        if entry is None:
            print(f"  ! {result.custom_id}: not in the manifest — ignoring")
            failed += 1
            continue
        qid, producer = entry["question_id"], entry["producer"]

        if (qid, producer) in done:
            skipped += 1
            continue

        if result.result.type != "succeeded":
            # Left unwritten on purpose: an unscored answer is "not done", so a re-run
            # retries it. A zeroed row would look like a real verdict of zero.
            print(f"  ! {qid} / {producer}: {result.result.type}")
            failed += 1
            continue

        message = result.result.message
        try:
            verdict = Verdict.model_validate_json(_text_of(message))
        except (ValidationError, ValueError, json.JSONDecodeError) as exc:
            print(f"  ! {qid} / {producer}: malformed verdict — {str(exc)[:90]}")
            failed += 1
            continue

        u = message.usage
        cost = cost_usd(judge_model, u.input_tokens, u.output_tokens) * BATCH_DISCOUNT
        append(ScoreRecord.from_verdict(qid, producer, verdict, judge_model,
                                        u.input_tokens, u.output_tokens, cost))
        written += 1
        spent += cost

    print(f"\n  wrote {written} verdicts to {SCORES_PATH.name}"
          f"{f', skipped {skipped} already scored' if skipped else ''}"
          f"{f', {failed} FAILED (re-submit to retry)' if failed else ''}")
    print(f"  spend: ${spent:.4f}\n")
