"""OccamToken-style pruning patch for Qwen3.5.

Masked mode keeps the visual token count unchanged by replacing pruned
embeddings. True mode reduces image placeholders and image embeddings together
before vLLM schedules the prompt, so the language model sees fewer image tokens.
"""

from __future__ import annotations

import sys

import torch

from vllm_ascend.occamtoken.config import OccamTokenConfig
from vllm_ascend.occamtoken.logging import log_stats
from vllm_ascend.occamtoken.mrope import install_mrope_patch
from vllm_ascend.occamtoken.pruning import (
    prune_stage1_masked,
    prune_stage2_masked,
    prune_true_image_tokens,
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
_FALLBACK_LOGGED = False
_BUDGET_MISMATCH_LOGGED = False
_DISABLE_NEXT_TRUE_IMAGE_PRUNE = False
_PENDING_IMAGE_BUDGETS: list[tuple[int, int]] = []


def _describe_out_mm_kwargs(out_mm_kwargs, item_idx: int) -> str:
    try:
        top_keys = list(out_mm_kwargs.keys())
    except AttributeError:
        return f"out_mm_kwargs_type={type(out_mm_kwargs).__name__}"

    image_items = out_mm_kwargs.get("image")
    image_len = len(image_items) if hasattr(image_items, "__len__") else "unknown"
    parts = [
        f"out_mm_kwargs_keys={top_keys}",
        f"image_items_type={type(image_items).__name__}",
        f"image_items_len={image_len}",
        f"item_idx={item_idx}",
    ]

    try:
        out_item = image_items[item_idx]
        parts.append(f"image_item_type={type(out_item).__name__}")
        if hasattr(out_item, "keys"):
            item_keys = list(out_item.keys())
            parts.append(f"image_item_keys={item_keys}")
            grid_field = out_item.get("image_grid_thw")
            parts.append(f"image_grid_thw_type={type(grid_field).__name__}")
            grid_data = getattr(grid_field, "data", None)
            if grid_data is not None:
                parts.append(f"image_grid_thw_data_type={type(grid_data).__name__}")
    except Exception as exc:  # pragma: no cover - diagnostics only.
        parts.append(f"describe_error={type(exc).__name__}: {exc}")

    return " ".join(parts)


def _log_prompt_update_fallback(reason: Exception, out_mm_kwargs, item_idx: int) -> None:
    global _DISABLE_NEXT_TRUE_IMAGE_PRUNE
    _DISABLE_NEXT_TRUE_IMAGE_PRUNE = True
    global _FALLBACK_LOGGED
    if _FALLBACK_LOGGED:
        return
    _FALLBACK_LOGGED = True
    print(
        "[occamtoken] true Stage-I prompt replacement fallback: "
        f"reason={type(reason).__name__}: {reason}; "
        f"{_describe_out_mm_kwargs(out_mm_kwargs, item_idx)}",
        file=sys.stderr,
    )


def _log_processor_fallback(reason: str, processor) -> None:
    global _DISABLE_NEXT_TRUE_IMAGE_PRUNE
    _DISABLE_NEXT_TRUE_IMAGE_PRUNE = True
    global _FALLBACK_LOGGED
    if _FALLBACK_LOGGED:
        return
    _FALLBACK_LOGGED = True
    attrs = sorted(
        name
        for name in dir(processor)
        if "image" in name and not name.startswith("__")
    )
    print(
        "[occamtoken] true Stage-I prompt replacement fallback: "
        f"reason={reason}; processor_type={type(processor).__name__}; "
        f"image_attrs={attrs}",
        file=sys.stderr,
    )


def _log_budget_mismatch(message: str) -> None:
    global _BUDGET_MISMATCH_LOGGED
    if _BUDGET_MISMATCH_LOGGED:
        return
    _BUDGET_MISMATCH_LOGGED = True
    print(
        f"[occamtoken] true Stage-I multi-image budget warning: {message}",
        file=sys.stderr,
    )


def _patched_get_prompt_updates(
    self,
    mm_items,
    hf_processor_mm_kwargs,
    out_mm_kwargs,
):
    updates = list(
        _ORIG_GET_PROMPT_UPDATES(
            self,
            mm_items,
            hf_processor_mm_kwargs,
            out_mm_kwargs,
        )
    )
    config = OccamTokenConfig.from_env()
    if not config.true_stage1_active():
        return updates

    original_image_update = next((u for u in updates if u.modality == "image"), None)
    if original_image_update is None:
        return updates

    get_processor = getattr(self.info, "get_" + "hf" + "_processor")
    processor = get_processor(**hf_processor_mm_kwargs)
    image_token_id = getattr(processor, "image_token_id", None)
    image_token = getattr(processor, "image_token", None)
    if image_token_id is None or image_token is None:
        if config.strict:
            raise RuntimeError(
                "OccamToken true Stage-I requires processor.image_token and "
                "processor.image_token_id, but at least one is missing. "
                f"processor_type={type(processor).__name__}"
            )
        _log_processor_fallback(
            "processor_missing_image_token_or_id",
            processor,
        )
        return updates

    image_processor = self.info.get_image_processor(**hf_processor_mm_kwargs)
    merge_length = image_processor.merge_size**2

    def fallback_image_replacement(item_idx: int):
        replacement = original_image_update.content
        if callable(replacement):
            return replacement(item_idx)
        return replacement

    def get_image_replacement_qwen35_occamtoken(item_idx: int):
        try:
            image_items = out_mm_kwargs["image"]
            out_item = image_items[item_idx]
            grid_field = out_item["image_grid_thw"]
            grid_thw = getattr(grid_field, "data", grid_field)
            if not isinstance(grid_thw, torch.Tensor):
                grid_thw = torch.as_tensor(grid_thw)
            num_tokens = int(grid_thw.prod().item()) // merge_length
        except (KeyError, IndexError, TypeError, AttributeError, RuntimeError) as exc:
            if config.strict:
                raise RuntimeError(
                    "OccamToken true Stage-I cannot read image_grid_thw from "
                    "out_mm_kwargs. "
                    f"{_describe_out_mm_kwargs(out_mm_kwargs, item_idx)}"
                ) from exc
            _log_prompt_update_fallback(exc, out_mm_kwargs, item_idx)
            return fallback_image_replacement(item_idx)

        budget = config.true_image_budget(num_tokens)
        _PENDING_IMAGE_BUDGETS.append((num_tokens, budget))
        return [image_token_id] * budget

    image_update = PromptReplacement(
        modality="image",
        target=image_token,
        replacement=get_image_replacement_qwen35_occamtoken,
    )

    return [image_update, *(u for u in updates if u.modality != "image")]


def _patched_process_image_input(self, image_input):
    global _DISABLE_NEXT_TRUE_IMAGE_PRUNE
    config = OccamTokenConfig.from_env()
    image_embeds_split = _ORIG_PROCESS_IMAGE_INPUT(self, image_input)
    if not config.stage1_active():
        return image_embeds_split
    if config.true_stage1_active() and _DISABLE_NEXT_TRUE_IMAGE_PRUNE:
        _DISABLE_NEXT_TRUE_IMAGE_PRUNE = False
        _PENDING_IMAGE_BUDGETS.clear()
        return image_embeds_split

    output = []
    stats = []
    for item_idx, image_embeds in enumerate(image_embeds_split):
        if config.true_stage1_active():
            pruned, item_stats = prune_true_image_tokens(image_embeds, config)
            if _PENDING_IMAGE_BUDGETS:
                expected_original, expected_budget = _PENDING_IMAGE_BUDGETS.pop(0)
                actual_original = int(image_embeds.shape[0])
                actual_budget = int(pruned.shape[0])
                if (
                    expected_original != actual_original
                    or expected_budget != actual_budget
                ):
                    _log_budget_mismatch(
                        "placeholder budget and image embedding split differ: "
                        f"item_idx={item_idx} "
                        f"expected_original={expected_original} "
                        f"actual_original={actual_original} "
                        f"expected_budget={expected_budget} "
                        f"actual_budget={actual_budget}"
                    )
            else:
                _log_budget_mismatch(
                    "missing pending placeholder budget for image embedding split: "
                    f"item_idx={item_idx} actual_original={int(image_embeds.shape[0])} "
                    f"actual_budget={int(pruned.shape[0])}"
                )
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
install_mrope_patch()
