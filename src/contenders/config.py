"""Shared paths for Phase B.

Phase A's ``build_dataset.config`` owns the dataset layout and stays frozen; this module
re-exports what Phase B reads from it and adds the directories Phase B writes, so the
on-disk layout stays defined in exactly one place per package.
"""
from __future__ import annotations

from build_dataset.config import CLEAN_DIR, INTERIM_DIR, QA_DIR, ROOT

RAW_MEDICAL = ROOT / "data" / "raw" / "medical"   # RAG distractors (gitignored)
INDEX_DIR = ROOT / "data" / "index"               # FAISS index + chunk metadata
RESULTS_DIR = ROOT / "results"                    # answers.jsonl, rag_config.json, ...
ADAPTERS_DIR = ROOT / "models"                    # local QLoRA adapters

ANSWERS_PATH = RESULTS_DIR / "answers.jsonl"
TEST_PATH = QA_DIR / "test.jsonl"
TRAIN_PATH = QA_DIR / "train.jsonl"
SECTIONS_PATH = INTERIM_DIR / "sections.jsonl"

__all__ = ["ADAPTERS_DIR", "ANSWERS_PATH", "CLEAN_DIR", "INDEX_DIR", "INTERIM_DIR",
           "QA_DIR", "RAW_MEDICAL", "RESULTS_DIR", "ROOT", "SECTIONS_PATH",
           "TEST_PATH", "TRAIN_PATH"]
