"""Shared paths for Phase C.

Phase B's ``contenders.config`` already owns the layout Phase C reads from (the answers,
the test set, the cleaned documents); this module re-exports what the judge needs and adds
the three files it writes, so the on-disk layout stays defined in one place per package.
"""
from __future__ import annotations

from contenders.config import (ANSWERS_PATH, CLEAN_DIR, RESULTS_DIR, ROOT, SECTIONS_PATH,
                               TEST_PATH)

SCORES_PATH = RESULTS_DIR / "scores.csv"        # one row per judged answer
MASTER_PATH = RESULTS_DIR / "master_table.csv"  # the aggregated comparison
BATCH_PATH = RESULTS_DIR / "judge_batch.json"   # the submitted batch id — losing it loses the run
REPORT_DIR = ROOT / "report"                    # summary.md + reflection.md

__all__ = ["ANSWERS_PATH", "BATCH_PATH", "CLEAN_DIR", "MASTER_PATH", "REPORT_DIR",
           "RESULTS_DIR", "ROOT", "SCORES_PATH", "SECTIONS_PATH", "TEST_PATH"]
