"""Phase A dataset builder for the Serendib Hospital Q&A project.

Four deterministic-where-possible stages, each exposed as a ``cmd_*`` function
and wired to a subcommand in :mod:`build_dataset.__main__`:

    clean    -> :func:`build_dataset.clean.cmd_clean`
    segment  -> :func:`build_dataset.segment.cmd_segment`
    generate -> :func:`build_dataset.generate.cmd_generate`
    split    -> :func:`build_dataset.split.cmd_split`

Run: ``python -m build_dataset <stage>`` from the ``src/`` directory.
"""
from __future__ import annotations

from .clean import cmd_clean
from .generate import cmd_generate
from .segment import cmd_segment
from .split import cmd_split

__all__ = ["cmd_clean", "cmd_segment", "cmd_generate", "cmd_split"]
