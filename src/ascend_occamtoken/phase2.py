"""Helpers for direct multimodal encoder paths outside stock vLLM.

Some deployments run the vision encoder in an optimized worker path and validate
encoder output rows before the model-level ``_process_image_input`` patch can
run. These helpers let that path apply the same Stage-I true pruning immediately
after image encoding and before row-count validation.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

from .config import OccamTokenConfig
from .logging import log_stats
from .pruning import prune_true_image_tokens


def prune_phase2_local_image_outputs(
    local_outputs: Sequence[torch.Tensor],
    *,
    my_image_indices: Sequence[int],
    output_sizes: Sequence[int],
    config: OccamTokenConfig | None = None,
) -> list[torch.Tensor]:
    """Apply true Stage-I pruning to phase2 direct-encoder image outputs.

    Args:
        local_outputs: Per-image encoder outputs, already split by original
            image grid sizes.
        my_image_indices: Global image indices corresponding to ``local_outputs``.
        output_sizes: Expected final placeholder lengths for all request images.
        config: Optional OccamToken config. Defaults to environment config.

    Returns:
        The original outputs when true Stage-I is disabled; otherwise a list of
        pruned per-image outputs whose row counts match ``output_sizes``.
    """
    config = config or OccamTokenConfig.from_env()
    if not config.true_stage1_active():
        return list(local_outputs)

    if len(local_outputs) != len(my_image_indices):
        raise RuntimeError(
            "OccamToken phase2 local output count mismatch: "
            f"local_outputs={len(local_outputs)} "
            f"my_image_indices={len(my_image_indices)}"
        )

    pruned_outputs: list[torch.Tensor] = []
    stats = []
    for local_pos, (global_image_idx, image_embeds) in enumerate(
        zip(my_image_indices, local_outputs, strict=True)
    ):
        pruned, item_stats = prune_true_image_tokens(image_embeds, config)
        expected = int(output_sizes[int(global_image_idx)])
        actual = int(pruned.shape[0])
        if actual != expected:
            raise RuntimeError(
                "OccamToken phase2 prune size mismatch: "
                f"local_pos={local_pos} "
                f"global_image_idx={int(global_image_idx)} "
                f"original={int(image_embeds.shape[0])} "
                f"expected={expected} actual={actual} "
                f"stage={config.stage} "
                f"target_ratio={config.target_ratio} "
                f"stage1_ratio={config.stage1_ratio} "
                f"target_tokens={config.target_tokens} "
                f"stage1_tokens={config.stage1_tokens}"
            )
        pruned_outputs.append(pruned)
        stats.append(item_stats)

    log_stats(stats)
    return pruned_outputs
