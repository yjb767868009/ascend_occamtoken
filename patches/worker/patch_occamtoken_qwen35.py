"""OccamToken-style masked pruning patch for Qwen3.5.

This is phase 1 of the experiment. It intentionally keeps the visual token
count unchanged by replacing pruned embeddings instead of removing tokens.
That avoids vLLM multimodal placeholder and M-RoPE accounting changes while we
measure quality effects for fixed, Stage-I, Stage-II-lite, and full modes.
"""

from __future__ import annotations

import torch

from vllm_ascend.occamtoken.config import OccamTokenConfig
from vllm_ascend.occamtoken.logging import log_stats
from vllm_ascend.occamtoken.pruning import (
    prune_stage1_masked,
    prune_stage2_masked,
    select_text_window,
)
from vllm.model_executor.models.qwen3_5 import (
    Qwen3_5ForConditionalGeneration,
    _merge_multimodal_embeddings,
    _require_is_multimodal,
)


_ORIG_PROCESS_IMAGE_INPUT = Qwen3_5ForConditionalGeneration._process_image_input


def _patched_process_image_input(self, image_input):
    config = OccamTokenConfig.from_env()
    image_embeds_split = _ORIG_PROCESS_IMAGE_INPUT(self, image_input)
    if not config.stage1_active():
        return image_embeds_split

    output = []
    stats = []
    for image_embeds in image_embeds_split:
        pruned, item_stats = prune_stage1_masked(image_embeds, config)
        output.append(pruned)
        stats.append(item_stats)
    log_stats(stats)
    return tuple(output)


def _patched_embed_input_ids(
    self,
    input_ids: torch.Tensor,
    multimodal_embeddings=None,
    *,
    is_multimodal: torch.Tensor | None = None,
) -> torch.Tensor:
    config = OccamTokenConfig.from_env()
    inputs_embeds = self._embed_text_input_ids(
        input_ids,
        self.language_model.embed_input_ids,
        is_multimodal=is_multimodal,
    )

    if multimodal_embeddings is None or len(multimodal_embeddings) == 0:
        return inputs_embeds

    is_multimodal = _require_is_multimodal(is_multimodal)

    if config.stage2_active():
        text_embeddings = inputs_embeds[~is_multimodal]
        text_embeddings = select_text_window(
            text_embeddings,
            max_text_tokens=config.max_text_tokens,
            question_tail_tokens=config.question_tail_tokens,
        )
        output = []
        stats = []
        for visual_embeddings in multimodal_embeddings:
            pruned, item_stats = prune_stage2_masked(
                visual_embeddings,
                text_embeddings,
                config,
            )
            output.append(pruned)
            stats.append(item_stats)
        multimodal_embeddings = tuple(output)
        log_stats(stats)

    return _merge_multimodal_embeddings(
        inputs_embeds=inputs_embeds,
        multimodal_embeddings=multimodal_embeddings,
        is_multimodal=is_multimodal,
    )


Qwen3_5ForConditionalGeneration._process_image_input = _patched_process_image_input
Qwen3_5ForConditionalGeneration.embed_input_ids = _patched_embed_input_ids
