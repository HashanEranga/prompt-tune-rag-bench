"""The producer registry and the price table — Phase B's single source of truth.

Ten answer-producers x 100 locked test questions = 1,000 answers. Everything
downstream (estimation, dispatch, cost accounting, the Phase C join) keys off the
``Producer.key`` slugs declared here.

The three ``ft-openai-*`` producers were removed when OpenAI withdrew fine-tuning at
the organisation level; Together replaced them with the single ``ft-llama3.1-8b`` rung.
Claude is the Phase C judge and so appears nowhere here — answerers are GPT, Gemini,
Llama and Qwen only.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

from .config import RESULTS_DIR

PROMPTING = "prompting"
FINETUNED = "finetuned"
RAG = "rag"

OPENAI = "openai"     # also every OpenAI-wire-compatible host, via base_url
GOOGLE = "google"
OLLAMA = "ollama"
HF_LOCAL = "hf-local"  # local QLoRA adapters, served through transformers


@dataclass(frozen=True)
class Producer:
    """One answer-producer. ``key`` is what lands in answers.jsonl."""
    key: str
    method: str
    provider: str           # where it ANSWERS from
    model: str              # for FINETUNED this is the BASE model; see resolve_model()
    size: str | None = None       # small | medium | large  (fine-tuned ladder only)
    api_key_env: str | None = None
    base_url: str | None = None   # set for Groq / OpenRouter / DashScope
    train_id: str | None = None   # where it TRAINS, when that isn't where it answers

    @property
    def is_local(self) -> bool:
        return self.provider in (OLLAMA, HF_LOCAL)

    @property
    def trains_remotely(self) -> bool:
        return self.train_id is not None


# Order here is the order they run and the order they print.
PRODUCERS: tuple[Producer, ...] = (
    # Step 1 — prompting, no documents in context (the control).
    Producer("prompt-gpt", PROMPTING, OPENAI, "gpt-4.1", api_key_env="OPENAI_API_KEY"),
    Producer("prompt-gemini", PROMPTING, GOOGLE, "gemini-3.5-flash", api_key_env="GOOGLE_API_KEY"),
    Producer("prompt-llama3.1-8b", PROMPTING, OLLAMA, "llama3.1:8b"),
    Producer("prompt-qwen3.5-9b", PROMPTING, OLLAMA, "qwen3.5:9b"),

    # Step 2 — fine-tuned ladder, self-hosted QLoRA on the RTX 4060.
    Producer("ft-local-small", FINETUNED, HF_LOCAL, "Qwen/Qwen2.5-0.5B-Instruct", "small"),
    Producer("ft-local-medium", FINETUNED, HF_LOCAL, "Qwen/Qwen2.5-1.5B-Instruct", "medium"),
    Producer("ft-local-large", FINETUNED, HF_LOCAL, "Qwen/Qwen2.5-3B-Instruct", "large"),

    # Step 2b — trained on Together, served here. NousResearch mirrors Meta's weights
    # bit-for-bit and is ungated; the official meta-llama repo 403s without a licence.
    Producer("ft-llama3.1-8b", FINETUNED, HF_LOCAL,
             "NousResearch/Meta-Llama-3.1-8B-Instruct",
             train_id="meta-llama/Meta-Llama-3.1-8B-Instruct-Reference"),

    # Step 3 — RAG. Same two models as prompting, but WITH documents.
    Producer("rag-llama3.1-8b", RAG, OLLAMA, "llama3.1:8b"),
    Producer("rag-gpt", RAG, OPENAI, "gpt-4.1", api_key_env="OPENAI_API_KEY"),
)

BY_KEY = {p.key: p for p in PRODUCERS}

# The two controlled A/B pairs: same model, docs vs no docs. Any lift is purely retrieval.
AB_PAIRS = (("prompt-llama3.1-8b", "rag-llama3.1-8b"), ("prompt-gpt", "rag-gpt"))

# All three methods on ONE model, so a score difference can only be the method. Caveat to
# report: Ollama serves llama3.1:8b at Q4_0 while the fine-tuned leg is 4-bit NF4 over the
# fp16 base — same weights and lineage, different quantisation.
TRIAD = ("prompt-llama3.1-8b", "ft-llama3.1-8b", "rag-llama3.1-8b")

# The embedder is a different family than the Qwen answerer on purpose, so retrieval
# quality isn't entangled with one lineage.
EMBED_MODEL = "nomic-embed-text"
JUDGE_MODEL = "claude-sonnet-4-5"  # Phase C only; priced here so `estimate` sees the whole project


@dataclass(frozen=True)
class Price:
    """USD per 1M tokens. ``train`` is the one-off fine-tuning training rate."""
    inp: float
    out: float
    train: float | None = None
    source: str = ""


# ⚠️ Best-known figures, NOT confirmed against the live pricing pages. `contenders
# estimate` prints a banner until each row is checked against its `source` URL and this
# flag is flipped.
PRICES_VERIFIED = False

FREE = Price(0.0, 0.0, 0.0, "local — hardware only, no API cost")

PRICING: dict[str, Price] = {
    # The gpt-4.1 `train` rate is dead weight since OpenAI withdrew fine-tuning; it is kept
    # only so the estimate's historical FT projection can still be reproduced.
    "gpt-4.1": Price(2.00, 8.00, 25.00, "https://openai.com/api/pricing/"),
    # Gemini's output price includes thinking tokens — hence thinking_budget=0 in clients.py.
    "gemini-3.5-flash": Price(1.50, 9.00, None, "https://ai.google.dev/gemini-api/docs/pricing"),
    # Retired: still in models.list(), but calling it 404s. Kept to price the 21 answers it
    # produced before the switch.
    "gemini-2.5-flash": Price(0.30, 2.50, None, "retired 2026 — new users get 404"),
    "claude-sonnet-4-5": Price(3.00, 15.00, None, "https://www.anthropic.com/pricing#api"),
    "llama3.1:8b": FREE,
    "qwen3.5:9b": FREE,
    "nomic-embed-text": FREE,
    "Qwen/Qwen2.5-0.5B-Instruct": FREE,
    "Qwen/Qwen2.5-1.5B-Instruct": FREE,
    "Qwen/Qwen2.5-3B-Instruct": FREE,
    # Trained on Together, but ANSWERS here — inference is free like any local model.
    "NousResearch/Meta-Llama-3.1-8B-Instruct": FREE,
}

# Together charges $0.48/1M training tokens but enforces a $4.00 minimum per job, and our
# 300 pairs are only ~$0.04 of compute — so the floor is the price. A flat per-job fee
# cannot live in `Price.train`; `estimate` adds it separately.
TOGETHER_JOB_MINIMUM_USD = 4.00

# A fine-tuned model bills at a premium over its base. Applied to the base rate when we
# can't yet know the ft:… model id (i.e. during estimation).
FT_INFERENCE_MULTIPLIER = 2.0


def price_for(model: str) -> Price:
    """Price of a model id. Fine-tuned ids (``ft:gpt-4.1-nano:org::abc``) are
    resolved back to their base model, then charged at the fine-tuned rate."""
    if model in PRICING:
        return PRICING[model]
    if model.startswith("ft:"):
        base = model.split(":")[1]
        if base in PRICING:
            p = PRICING[base]
            return Price(p.inp * FT_INFERENCE_MULTIPLIER, p.out * FT_INFERENCE_MULTIPLIER,
                         p.train, p.source)
    raise KeyError(f"no price for model {model!r} — add it to PRICING in models.py")


def cost_usd(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    p = price_for(model)
    return (prompt_tokens * p.inp + completion_tokens * p.out) / 1_000_000


JOBS_PATH = RESULTS_DIR / "finetune_jobs.json"


def load_jobs() -> dict:
    """Trained-model ids + training time/cost, written by `contenders finetune`."""
    if not JOBS_PATH.exists():
        return {}
    return json.loads(JOBS_PATH.read_text(encoding="utf-8"))


def resolve_model(p: Producer) -> str:
    """The model id to actually call — for fine-tuned producers the trained artifact
    recorded at training time, since ``Producer.model`` only names the base."""
    if p.method != FINETUNED:
        return p.model
    job = load_jobs().get(p.key)
    if not job or not job.get("trained_model"):
        backend = "together" if p.trains_remotely else "local"
        raise SystemExit(
            f"{p.key} has not been fine-tuned yet — no trained model in "
            f"{JOBS_PATH.name}. Run:  python -m contenders finetune "
            f"--backend {backend}")
    return job["trained_model"]
