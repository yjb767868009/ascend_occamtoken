"""Small logging helpers for patch experiments."""

from __future__ import annotations

import os
import sys
from collections.abc import Iterable

from .pruning import PruneStats


def log_stats(stats: Iterable[PruneStats]) -> None:
    if os.getenv("VLLM_ASCEND_OCCAMTOKEN_LOG_STATS", "0").lower() not in {
        "1",
        "true",
        "yes",
        "on",
    }:
        return
    for item in stats:
        print(
            "[occamtoken] "
            f"stage={item.stage} original={item.original_tokens} "
            f"kept={item.kept_tokens} retention={item.retention:.4f}",
            file=sys.stderr,
            flush=True,
        )

