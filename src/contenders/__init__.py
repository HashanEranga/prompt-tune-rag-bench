"""Phase B — the three contenders: prompting, fine-tuning, RAG.

Answers the 100 locked Phase A test questions with 10 producers under identical
conditions, logging latency and cost for every answer into results/answers.jsonl
— the file the Phase C judge scores.
"""
from __future__ import annotations

from .estimate import cmd_estimate
from .finetune import cmd_finetune
from .prompt import cmd_prompt
from .rag import cmd_rag
from .reset import cmd_reset
from .status import cmd_check, cmd_status

__all__ = ["cmd_check", "cmd_estimate", "cmd_finetune", "cmd_prompt", "cmd_rag",
           "cmd_reset", "cmd_status"]
