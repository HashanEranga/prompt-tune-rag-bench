"""Step 2 — the fine-tuned ladder, small -> medium -> large, two ways.

Two live backends, same 300 pairs, identical hyperparameters:

  --backend local     QLoRA on the RTX 4060 (Qwen2.5 0.5B/1.5B/3B). Costs electricity.
  --backend together  The 8B rung the 4060 cannot train. Costs $4.00, once. Together
                      only TRAINS: the adapter is downloaded and served locally for $0.

``--backend openai`` is dead — the API returns 403 training_not_available at the
organisation level. ``_train_openai`` is kept only as the record of what was attempted;
it is unreachable from ``PRODUCERS`` and ``cmd_finetune`` refuses the flag.
"""
from __future__ import annotations

import json
import os
import shutil
import tarfile
import time

from .answers import load_test
from .clients import _require_key
from .config import ADAPTERS_DIR, RESULTS_DIR, TRAIN_PATH
from .models import (FINETUNED, HF_LOCAL, JOBS_PATH, PRODUCERS, TOGETHER_JOB_MINIMUM_USD,
                     Producer, load_jobs, price_for)
from .runner import run_producer

DEFAULT_EPOCHS = 3
POLL_SECONDS = 30

# QLoRA — sized for 8 GB VRAM. seq 512 covers a Q&A pair with headroom.
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
MAX_SEQ_LEN = 512
BATCH_SIZE = 1
GRAD_ACCUM = 8
LEARNING_RATE = 2e-4


def _load_train() -> list[dict]:
    """Load the training pairs, refusing to train if a test question is among them —
    a memorised eval question invalidates every score built on top of it."""
    train = [json.loads(l) for l in TRAIN_PATH.open(encoding="utf-8")]
    train_ids = {r["id"] for r in train}
    test_ids = {r["id"] for r in load_test()}
    leaked = train_ids & test_ids
    assert not leaked, (
        f"INTEGRITY RULE #1 VIOLATED: {len(leaked)} test question(s) are in the "
        f"training set, e.g. {sorted(leaked)[:3]}. Training would invalidate every "
        f"downstream result. Re-run `python -m build_dataset split`.")
    print(f"✓ Rule #1: {len(train)} train / {len(test_ids)} test, zero overlap")
    return train


def _record_job(key: str, **fields) -> None:
    jobs = load_jobs()
    jobs[key] = {**jobs.get(key, {}), **fields}
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    JOBS_PATH.write_text(json.dumps(jobs, indent=2), encoding="utf-8")


def _write_messages_jsonl(train: list[dict], name: str):
    """Both managed backends validate strictly and want bare ``messages`` rows, so
    Phase A's passthrough metadata (id/topic/answer_type/source_doc) is stripped here."""
    path = RESULTS_DIR / name
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for r in train:
            fh.write(json.dumps({"messages": r["messages"]}, ensure_ascii=False) + "\n")
    return path


def _train_openai(p: Producer, train: list[dict], epochs: int) -> None:
    from openai import OpenAI
    client = OpenAI(api_key=_require_key(p))

    upload = _write_messages_jsonl(train, "train_openai.jsonl")

    print(f"\n{p.key}  [{p.size} · {p.model}]")
    fobj = client.files.create(file=upload.open("rb"), purpose="fine-tune")
    job = client.fine_tuning.jobs.create(
        training_file=fobj.id, model=p.model,
        hyperparameters={"n_epochs": epochs})
    print(f"  job {job.id} submitted, polling every {POLL_SECONDS}s...")

    started = time.perf_counter()
    while True:
        job = client.fine_tuning.jobs.retrieve(job.id)
        if job.status in ("succeeded", "failed", "cancelled"):
            break
        print(f"  {job.status}...", end="\r")
        time.sleep(POLL_SECONDS)
    elapsed = time.perf_counter() - started

    if job.status != "succeeded":
        raise SystemExit(f"{p.key}: fine-tuning {job.status} — {getattr(job, 'error', '')}")

    trained = job.trained_tokens or 0
    cost = trained * (price_for(p.model).train or 0.0) / 1_000_000
    _record_job(p.key, backend="openai", base_model=p.model, size=p.size,
                trained_model=job.fine_tuned_model, trained_tokens=trained,
                epochs=epochs, train_seconds=round(elapsed, 1), train_cost_usd=round(cost, 4))
    print(f"  ✓ {job.fine_tuned_model}")
    print(f"    {trained:,} trained tokens · {elapsed / 60:.1f} min · ${cost:.2f}")


# Stems, not exact states: Together reports "cancel_requested" before "cancelled", so
# matching only the past tense makes the poll loop wait forever on an already-dead job.
DEAD_STATES = ("error", "cancel", "failed")


def _find_remote_job(client, p: Producer):
    """The newest live-or-finished job for this producer (matched on ``suffix``), meaning
    'do not pay again'. Dead jobs are ignored — those do need a fresh submission."""
    try:
        jobs = client.fine_tuning.list().data
    except Exception as exc:
        print(f"  (couldn't list existing Together jobs: {exc} — proceeding)")
        return None
    mine = [j for j in jobs
            if getattr(j, "suffix", None) == p.key
            and not any(s in str(getattr(j, "status", "")).lower() for s in DEAD_STATES)]
    if not mine:
        return None
    return sorted(mine, key=lambda j: str(getattr(j, "created_at", "")))[-1]


def _download_adapter(client, job, out_dir) -> None:
    """Download and unpack the trained LoRA into a PEFT directory ``_hf_pipeline`` can serve.

    Together hands back one ``.tar.zst`` archive, not a directory. ``output=`` must be
    passed — left to the SDK the filename comes from the org-prefixed
    ``x_model_output_name``, a nested path whose parent does not exist here. The zstd
    archive needs no ``zstandard`` import; stdlib tarfile reads ``.tar.zst`` natively.
    """
    from together.lib import DownloadManager

    # Not out_dir.with_suffix(): pathlib reads `ft-llama3.1-8b` as stem `ft-llama3` +
    # suffix `.1-8b` and would silently rename the archive to ft-llama3.tar.zst.
    archive = out_dir.parent / f"{out_dir.name}.tar.zst"
    DownloadManager(client).download(
        url=f"/finetune/download?ft_id={job.id}&checkpoint=adapter",
        output=archive, remote_name=job.x_model_output_name, fetch_metadata=True)

    with tarfile.open(archive, "r:*") as tar:
        tar.extractall(out_dir, filter="data")   # filter="data" refuses paths outside out_dir

    # Some checkpoints unpack under a single top-level folder, some at the root.
    # PEFT only loads the latter, so flatten the former.
    entries = list(out_dir.iterdir())
    if len(entries) == 1 and entries[0].is_dir():
        for item in entries[0].iterdir():
            shutil.move(str(item), out_dir)
        entries[0].rmdir()

    # Fail loudly here rather than record a trained_model pointing at an empty directory —
    # that error would resurface hours later inside an answer run, far from its cause.
    missing = [f for f in ("adapter_config.json", "adapter_model.safetensors")
               if not (out_dir / f).exists()]
    if missing:
        raise SystemExit(
            f"Adapter extracted to {out_dir} but {', '.join(missing)} is missing.\n"
            f"The archive is kept at {archive} — inspect it with `tar -tf` before re-running.")


def _await_and_download(client, p: Producer, job, out_dir, epochs: int) -> None:
    """Poll a Together job to completion, then pull the adapter down and record it.

    Shared by the fresh-submit and reattach paths, so a crashed poll is resumed by simply
    re-running the command — without paying the $4 floor twice.
    """
    started = time.perf_counter()
    status = str(getattr(job, "status", "")).lower()
    while not any(s in status for s in ("completed", *DEAD_STATES)):
        print(f"  {status}... (Together queues jobs; pending for a while is normal)   ",
              end="\r")
        time.sleep(POLL_SECONDS)
        job = client.fine_tuning.retrieve(job.id)
        status = str(getattr(job, "status", "")).lower()
    elapsed = time.perf_counter() - started

    if "completed" not in status:
        raise SystemExit(
            f"\n{p.key}: Together fine-tuning ended as {status!r}. Nothing was downloaded.\n"
            f"Check the job at https://api.together.ai/fine-tuning/{job.id}")

    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n  ✓ job {job.id} completed — downloading adapter -> {out_dir.name}/")
    _download_adapter(client, job, out_dir)

    _record_job(p.key, backend="together", base_model=p.model, train_id=p.train_id,
                size=p.size, trained_model=str(out_dir), remote_job_id=job.id,
                epochs=epochs, train_seconds=round(elapsed, 1),
                train_cost_usd=TOGETHER_JOB_MINIMUM_USD, served="local (4-bit NF4)")
    print(f"  ✓ adapter -> {out_dir}")
    print(f"    waited {elapsed / 60:.1f} min · ${TOGETHER_JOB_MINIMUM_USD:.2f} (one-off) · "
          f"every answer from here on is LOCAL and free")


def _train_together(p: Producer, train: list[dict], epochs: int) -> None:
    """Train an 8B LoRA on Together's GPU, then download the adapter and serve it here.

    Fine-tuning an 8B needs ~15 GB but serving one at 4-bit needs only ~5.5 GB, so the big
    GPU is rented for the ten minutes we cannot do ourselves and the adapter kept on disk.

    Hyperparameters match ``_train_local`` where the platforms allow, but not fully:
    Together defaults to all-linear LoRA modules, seq 131072 and grad-accum 1, against our
    q_proj/v_proj, 512 and 8. So ft-llama3.1-8b vs ft-local-* does NOT isolate managed vs
    self-hosted — the recipe differs too. The TRIAD comparison is unaffected.
    """
    from together import Together

    key = os.getenv("TOGETHER_API_KEY")
    if not key:
        raise SystemExit(
            "ft-llama3.1-8b needs TOGETHER_API_KEY, which is empty. Add it to .env "
            "(see .env.example).\nOpenAI's fine-tuning API is withdrawn at the org level, "
            "so Together is the managed path — see docs/cost-analysis.md.")
    client = Together(api_key=key)

    out_dir = ADAPTERS_DIR / p.key

    # A submitted job lives on Together, not here: if this process dies while polling,
    # finetune_jobs.json is never written and a naive re-run pays the $4 floor a second
    # time for a model we already own. Reattach instead of creating.
    existing = _find_remote_job(client, p)
    if existing is not None:
        print(f"\n{p.key}: reattaching to existing Together job {existing.id} "
              f"(status: {existing.status})")
        print(f"  Not creating a new one — that would pay the ${TOGETHER_JOB_MINIMUM_USD:.2f} "
              f"floor a second time for training we have already bought.")
        return _await_and_download(client, p, existing, out_dir, epochs)

    upload = _write_messages_jsonl(train, "train_together.jsonl")

    print(f"\n{p.key}  [{p.train_id}]  LoRA SFT on Together")
    print(f"  ⚠️  Together bills a ${TOGETHER_JOB_MINIMUM_USD:.2f} MINIMUM per job. Our "
          f"{len(train)} pairs are worth ~$0.04 of tokens,\n      so this costs the floor — "
          f"a ~100x markup on what we consume. It is charged ONCE.")

    fobj = client.files.upload(file=str(upload), check=True)
    job = client.fine_tuning.create(
        training_file=fobj.id,
        model=p.train_id,
        lora=True,
        lora_r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
        n_epochs=epochs, learning_rate=LEARNING_RATE,
        suffix=p.key,
    )
    print(f"  job {job.id} submitted, polling every {POLL_SECONDS}s...")
    return _await_and_download(client, p, job, out_dir, epochs)


def _train_local(p: Producer, train: list[dict], epochs: int) -> None:
    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoTokenizer, BitsAndBytesConfig
    from trl import SFTConfig, SFTTrainer

    out_dir = ADAPTERS_DIR / p.key
    print(f"\n{p.key}  [{p.size} · {p.model}]  QLoRA 4-bit -> {out_dir.name}/")

    ds = Dataset.from_list([{"messages": r["messages"]} for r in train])
    quant = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    tok = AutoTokenizer.from_pretrained(p.model)

    trainer = SFTTrainer(
        model=p.model,
        train_dataset=ds,
        processing_class=tok,
        peft_config=LoraConfig(r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
                               task_type="CAUSAL_LM"),
        args=SFTConfig(
            output_dir=str(out_dir), num_train_epochs=epochs,
            per_device_train_batch_size=BATCH_SIZE, gradient_accumulation_steps=GRAD_ACCUM,
            learning_rate=LEARNING_RATE, max_length=MAX_SEQ_LEN,
            gradient_checkpointing=True, bf16=True, logging_steps=10,
            report_to=[], save_strategy="no",
            model_init_kwargs={"quantization_config": quant, "device_map": "auto"}),
    )

    started = time.perf_counter()
    trainer.train()
    elapsed = time.perf_counter() - started
    trainer.save_model(str(out_dir))

    peak_gb = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0
    _record_job(p.key, backend="local", base_model=p.model, size=p.size,
                trained_model=str(out_dir), epochs=epochs,
                train_seconds=round(elapsed, 1), train_cost_usd=0.0,
                gpu_hours=round(elapsed / 3600, 4), peak_vram_gb=round(peak_gb, 2))
    print(f"  ✓ adapter -> {out_dir}")
    print(f"    {elapsed / 60:.1f} min · {elapsed / 3600:.3f} GPU-hours · "
          f"peak {peak_gb:.1f} GB VRAM · $0.00")


def _warn_if_ollama_loaded() -> None:
    """Ollama and QLoRA share the same 8 GB. A resident model will OOM training."""
    try:
        import ollama
        loaded = [m.model for m in ollama.ps().models]
    except Exception:
        return
    if loaded:
        print(f"\n  ⚠️  Ollama is holding {', '.join(loaded)} in VRAM. Training shares the "
              f"same 8 GB and may OOM.\n     Free it first:  " +
              "  ".join(f"ollama stop {m}" for m in loaded) + "\n")


def cmd_finetune(args) -> None:
    if args.backend == "openai":
        raise SystemExit(
            "--backend openai is withdrawn. OpenAI's fine-tuning API returns\n"
            "  403 training_not_available — 'OpenAI is winding down the fine-tuning platform\n"
            "  and your organization is no longer able to create new fine-tuning training jobs.'\n\n"
            "That is revoked at the organisation level: no base model, retry or key gets around\n"
            "it, so the three ft-openai-* producers were removed from models.py.\n\n"
            "Step 2 now has two live backends:\n"
            "    python -m contenders finetune --backend local      # QLoRA ladder, free\n"
            "    python -m contenders finetune --backend together   # the 8B triad rung, $4 once")

    # Both backends serve locally; they differ only in where they TRAIN.
    remote = args.backend == "together"
    producers = [p for p in PRODUCERS
                 if p.method == FINETUNED and p.provider == HF_LOCAL
                 and p.trains_remotely == remote]
    if args.size:
        producers = [p for p in producers if p.size in args.size]
    if not producers:
        raise SystemExit(f"--backend {args.backend} matches no producers in models.py")

    train = _load_train()
    _warn_if_ollama_loaded()

    if not args.answer_only:
        trainer = _train_together if remote else _train_local
        for p in producers:
            if load_jobs().get(p.key, {}).get("trained_model") and not args.retrain:
                print(f"\n{p.key}: already trained — skipping (use --retrain to force)")
                continue
            trainer(p, train, args.epochs)

    if args.train_only:
        return

    questions = load_test()
    if args.limit:
        questions = questions[:args.limit]
    print(f"\nAnswering {len(questions)} test questions with the {args.backend} ladder")
    for p in producers:
        run_producer(p, questions)
