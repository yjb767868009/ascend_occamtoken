"""OccamToken-style pruning patch for Qwen3.5.

Masked mode keeps the visual token count unchanged by replacing pruned
embeddings. True mode reduces image placeholders and image embeddings together
before vLLM schedules the prompt, so the language model sees fewer image tokens.
"""

from __future__ import annotations

import torch

from vllm_ascend.occamtoken.config import OccamTokenConfig
from vllm_ascend.occamtoken.logging import log_stats
from vllm_ascend.occamtoken.pruning import (
    prune_stage1_masked,
    prune_stage1_true,
    prune_stage2_masked,
    select_text_window,
)
from vllm.multimodal.processing import PromptReplacement
from vllm.model_executor.models.qwen3_5 import (
    Qwen3_5ForConditionalGeneration,
    _merge_multimodal_embeddings,
    _require_is_multimodal,
)
from vllm.model_executor.models.qwen3_vl import Qwen3VLMultiModalProcessor


_ORIG_PROCESS_IMAGE_INPUT = Qwen3_5ForConditionalGeneration._process_image_input
_ORIG_GET_PROMPT_UPDATES = Qwen3VLMultiModalProcessor._get_prompt_updates


def _patched_get_prompt_updates(
    self,
    mm_items,
    processor_mm_kwargs,
    out_mm_kwargs,
):
    updates = list(
        _ORIG_GET_PROMPT_UPDATES(
            self,
            mm_items,
            processor_mm_kwargs,
            out_mm_kwargs,
        )
    )
    config = OccamTokenConfig.from_env()
    if not config.true_stage1_active():
        return updates

    get_processor = getattr(self.info, "get_" + "hf" + "_processor")
    processor = get_processor(**processor_mm_kwargs)
    image_processor = self.info.get_image_processor(**processor_mm_kwargs)
    merge_length = image_processor.merge_size**2

    def get_image_replacement_qwen35_occamtoken(item_idx: int):
        out_item = out_mm_kwargs["image"][item_idx]
        grid_thw = out_item["image_grid_thw"].data
        num_tokens = int(grid_thw.prod()) // merge_length
        budget = config.stage1_budget(num_tokens)
        return [processor.image_token_id] * budget

    image_update = PromptReplacement(
        modality="image",
        target=processor.image_token,
        replacement=get_image_replacement_qwen35_occamtoken,
    )

    return [image_update, *(u for u in updates if u.modality != "image")]


def _patched_process_image_input(self, image_input):
    config = OccamTokenConfig.from_env()
    image_embeds_split = _ORIG_PROCESS_IMAGE_INPUT(self, image_input)
    if not config.stage1_active():
        return image_embeds_split

    output = []
    stats = []
    for image_embeds in image_embeds_split:
        if config.true_stage1_active():
            pruned, item_stats = prune_stage1_true(image_embeds, config)
        else:
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

    if config.stage2_active() and not config.true_sparse_active():
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
Qwen3VLMultiModalProcessor._get_prompt_updates = _patched_get_prompt_updates
