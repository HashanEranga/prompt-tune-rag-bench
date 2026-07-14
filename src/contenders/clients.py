"""One ``call()`` for every provider — the only place Phase B talks to a model.

Four adapters cover all ten producers: OpenAI-wire (and by base_url Groq / OpenRouter /
DashScope), Google GenAI, Ollama, and local QLoRA adapters via transformers.

It is a single function so that two invariants hold by construction: every producer gets
the identical SYSTEM_PROMPT (the string the FT models were trained with), and the
identical decoding settings. A comparison in which one contender sampled differently, or
saw a different system prompt, is not a comparison.
"""
from __future__ import annotations

import os
import random
import time
from dataclasses import dataclass

from dotenv import load_dotenv

from build_dataset.split import SEED, SYSTEM_PROMPT

from .models import GOOGLE, HF_LOCAL, OLLAMA, OPENAI, Producer, cost_usd

load_dotenv()

# Fixed decoding settings — identical for all 10 producers.
TEMPERATURE = 0.0
MAX_OUTPUT_TOKENS = 300   # also the worst-case output used by `contenders estimate`
NUM_CTX = 8192            # RAG-sized; llama3.1:8b still fits 100% on an 8 GB GPU (6.3 GB)

RETRY_ATTEMPTS = 4
RETRY_BASE_DELAY = 2.0        # seconds; doubles each attempt, plus jitter
RATE_LIMIT_DELAY = 65.0       # a 429 is a per-MINUTE quota — back off past the window


@dataclass(frozen=True)
class Completion:
    """One model response. ``error`` is set instead of raising, so a provider failure
    lands in answers.jsonl as a visible row rather than silently shrinking a
    producer's denominator."""
    text: str
    latency_s: float
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float
    error: str | None = None


PERMANENT = ("authentication", "invalid_api_key", "permission", "not_found",
             "does not exist", "model_not_found", "invalid_request")

# A bug in this repo is not a provider blip: retrying one burns ~14s of backoff per
# question to hide the traceback that would have named the line.
BUGS = (AttributeError, TypeError, NameError, ImportError)


def _is_transient(exc: Exception) -> bool:
    if isinstance(exc, BUGS):
        return False
    msg = str(exc).lower()
    if any(p in msg for p in PERMANENT):
        return False
    return True


def _is_rate_limit(exc: Exception) -> bool:
    msg = str(exc).lower()
    return "429" in msg or "resource_exhausted" in msg or "rate limit" in msg


def _with_retry(fn):
    """Retry transient failures only; a wrong key or a nonexistent model id must fail
    loudly. Rate limits wait the quota window out — 2s/4s/8s backoff would just fail
    three more times inside the same minute.
    """
    for attempt in range(RETRY_ATTEMPTS):
        try:
            return fn()
        except Exception as exc:
            if attempt == RETRY_ATTEMPTS - 1 or not _is_transient(exc):
                raise
            if _is_rate_limit(exc):
                print(f"    rate-limited, waiting {RATE_LIMIT_DELAY:.0f}s for the quota "
                      f"window (attempt {attempt + 1}/{RETRY_ATTEMPTS})...")
                time.sleep(RATE_LIMIT_DELAY + random.random() * 5)
            else:
                time.sleep(RETRY_BASE_DELAY * 2 ** attempt + random.random())


# Built once, reused across all 100 questions.
_clients: dict = {}


def _require_key(p: Producer) -> str:
    key = os.getenv(p.api_key_env or "")
    if not key:
        raise SystemExit(
            f"{p.key} needs {p.api_key_env}, which is empty. Add it to .env "
            f"(see .env.example), or drop this producer from PRODUCERS in models.py.")
    return key


def _openai_client(p: Producer):
    if p.key not in _clients:
        from openai import OpenAI
        _clients[p.key] = OpenAI(api_key=_require_key(p), base_url=p.base_url)
    return _clients[p.key]


def _google_client(p: Producer):
    if p.key not in _clients:
        from google import genai
        _clients[p.key] = genai.Client(api_key=_require_key(p))
    return _clients[p.key]


def _hf_pipeline(p: Producer, model_id: str):
    """Local QLoRA adapter served through transformers, on a 4-bit base loaded with the
    identical BitsAndBytesConfig it was trained under (``finetune._train_local``).
    Serving the adapter on a bf16 base would be both a different model than the one
    tuned and too big for the 8 GB card. torch is imported lazily — it is an optional
    extra, and the API-only producers must run without it.
    """
    if p.key not in _clients:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        tok = AutoTokenizer.from_pretrained(p.model)
        base = AutoModelForCausalLM.from_pretrained(
            p.model, device_map="auto",
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True))
        _clients[p.key] = (PeftModel.from_pretrained(base, model_id).eval(), tok)
    return _clients[p.key]


def _call_openai(p: Producer, model: str, user: str) -> tuple[str, int, int]:
    r = _openai_client(p).chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": SYSTEM_PROMPT},
                  {"role": "user", "content": user}],
        temperature=TEMPERATURE, seed=SEED, max_tokens=MAX_OUTPUT_TOKENS)
    u = r.usage
    return r.choices[0].message.content or "", u.prompt_tokens, u.completion_tokens


def _call_google(p: Producer, model: str, user: str) -> tuple[str, int, int]:
    from google.genai import types
    # thinking_budget=0 for the same reason Ollama gets think=False: reasoning is on by
    # default, billed at the output rate, and its trace leaks into the answer the judge
    # scores. Every producer answers directly; none get to reason first.
    r = _google_client(p).models.generate_content(
        model=model, contents=user,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT, temperature=TEMPERATURE,
            max_output_tokens=MAX_OUTPUT_TOKENS,
            thinking_config=types.ThinkingConfig(thinking_budget=0)))
    u = r.usage_metadata
    return (r.text or "", u.prompt_token_count or 0, u.candidates_token_count or 0)


def _call_ollama(p: Producer, model: str, user: str) -> tuple[str, int, int]:
    import ollama
    # think=False is not optional: with Qwen3.x reasoning on, one question ran >9 min
    # without completing here, vs 3.6 s with it off.
    r = ollama.chat(
        model=model,
        messages=[{"role": "system", "content": SYSTEM_PROMPT},
                  {"role": "user", "content": user}],
        think=False,
        options={"temperature": TEMPERATURE, "seed": SEED,
                 "num_ctx": NUM_CTX, "num_predict": MAX_OUTPUT_TOKENS})
    return (r["message"]["content"], r.get("prompt_eval_count", 0), r.get("eval_count", 0))


def _call_hf_local(p: Producer, model: str, user: str) -> tuple[str, int, int]:
    import torch
    model_obj, tok = _hf_pipeline(p, model)
    chat = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]
    # transformers v5 returns a BatchEncoding, not a bare tensor — ask for the dict
    # explicitly and unpack it, so attention_mask reaches generate() too.
    enc = tok.apply_chat_template(chat, add_generation_prompt=True, return_tensors="pt",
                                  return_dict=True).to(model_obj.device)
    n_in = enc["input_ids"].shape[-1]
    with torch.no_grad():
        out = model_obj.generate(**enc, max_new_tokens=MAX_OUTPUT_TOKENS, do_sample=False,
                                 pad_token_id=tok.eos_token_id)
    text = tok.decode(out[0][n_in:], skip_special_tokens=True)
    return text, int(n_in), int(out.shape[-1] - n_in)


ADAPTERS = {OPENAI: _call_openai, GOOGLE: _call_google,
            OLLAMA: _call_ollama, HF_LOCAL: _call_hf_local}


def call(p: Producer, model: str, user: str) -> Completion:
    """Ask one producer one question. ``user`` is the bare question for prompting
    and fine-tuned producers, or the retrieval-augmented prompt for RAG — the
    system prompt and decoding settings are applied here, identically, for all."""
    adapter = ADAPTERS[p.provider]
    # Load the weights before the clock starts, or the first question of every local
    # producer is charged the ~5-7 s model load on top of its own generation.
    if p.provider == HF_LOCAL:
        _hf_pipeline(p, model)
    started = time.perf_counter()
    try:
        text, in_tok, out_tok = _with_retry(lambda: adapter(p, model, user))
    except SystemExit:
        raise
    except Exception as exc:
        return Completion("", time.perf_counter() - started, 0, 0, 0.0,
                          error=f"{type(exc).__name__}: {exc}")
    latency = time.perf_counter() - started
    # Local producers are free, and a QLoRA adapter path is not a priced model id.
    cost = 0.0 if p.is_local else cost_usd(model, in_tok, out_tok)
    return Completion(text.strip(), latency, in_tok, out_tok, cost)
