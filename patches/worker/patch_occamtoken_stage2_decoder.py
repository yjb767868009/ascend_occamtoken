"""Decoder-layer hooks for true OccamToken Stage-II experiments."""

from __future__ import annotations

from itertools import islice

import torch

from vllm_ascend.occamtoken.config import OccamTokenConfig
from vllm_ascend.occamtoken.logging import log_stats
from vllm_ascend.occamtoken.pruning import stage2_true_keep_mask
from vllm.distributed import get_pp_group
from vllm.sequence import IntermediateTensors
from vllm.model_executor.models.qwen3_5 import (
    Qwen3_5ForConditionalGeneration,
    Qwen3_5Model,
)


_ORIG_QWEN35_FORWARD = Qwen3_5ForConditionalGeneration.forward


def _forward_until_layer(
    self: Qwen3_5Model,
    input_ids: torch.Tensor | None,
    positions: torch.Tensor,
    *,
    stop_layer: int,
    intermediate_tensors: IntermediateTensors | None = None,
    inputs_embeds: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Run Qwen3.5 text layers through ``stop_layer`` inclusive."""
    if stop_layer < self.start_layer or stop_layer >= self.end_layer:
        raise RuntimeError(
            "OccamToken Stage-II stop_layer is outside local pipeline range: "
            f"stop_layer={stop_layer} start={self.start_layer} end={self.end_layer}"
        )

    if get_pp_group().is_first_rank:
        if inputs_embeds is not None:
            hidden_states = inputs_embeds
        else:
            hidden_states = self.embed_input_ids(input_ids)
        residual = None
    else:
        if intermediate_tensors is None:
            raise RuntimeError(
                "OccamToken Stage-II prefix forward requires intermediate_tensors "
                "on non-first pipeline ranks."
            )
        hidden_states = intermediate_tensors["hidden_states"]
        residual = intermediate_tensors["residual"]

    for layer_idx, layer in enumerate(
        islice(self.layers, self.start_layer, stop_layer + 1),
        start=self.start_layer,
    ):
        if layer_idx in self.aux_hidden_state_layers:
            raise RuntimeError(
                "OccamToken Stage-II prototype does not yet support auxiliary "
                f"hidden state capture at layer {layer_idx}."
            )
        hidden_states, residual = layer(
            positions=positions,
            hidden_states=hidden_states,
            residual=residual,
        )

    return hidden_states, residual


def _forward_from_layer(
    self: Qwen3_5Model,
    hidden_states: torch.Tensor,
    residual: torch.Tensor | None,
    positions: torch.Tensor,
    *,
    start_layer: int,
) -> torch.Tensor | IntermediateTensors:
    """Resume Qwen3.5 text layers from ``start_layer``."""
    if start_layer < self.start_layer or start_layer > self.end_layer:
        raise RuntimeError(
            "OccamToken Stage-II start_layer is outside local pipeline range: "
            f"start_layer={start_layer} start={self.start_layer} end={self.end_layer}"
        )

    for layer_idx, layer in enumerate(
        islice(self.layers, start_layer, self.end_layer),
        start=start_layer,
    ):
        if layer_idx in self.aux_hidden_state_layers:
            raise RuntimeError(
                "OccamToken Stage-II prototype does not yet support auxiliary "
                f"hidden state capture at layer {layer_idx}."
            )
        hidden_states, residual = layer(
            positions=positions,
            hidden_states=hidden_states,
            residual=residual,
        )

    if not get_pp_group().is_last_rank:
        return IntermediateTensors(
            {"hidden_states": hidden_states, "residual": residual}
        )

    hidden_states, _ = self.norm(hidden_states, residual)
    return hidden_states


Qwen3_5Model.forward_until_layer = _forward_until_layer  # type: ignore[attr-defined]
Qwen3_5Model.forward_from_layer = _forward_from_layer  # type: ignore[attr-defined]


def _positions_index_select(positions: torch.Tensor, keep_mask: torch.Tensor):
    if positions.ndim == 2:
        return positions[:, keep_mask]
    return positions[keep_mask]


def _patched_qwen35_forward_stage2_true(
    self: Qwen3_5ForConditionalGeneration,
    input_ids: torch.Tensor,
    positions: torch.Tensor,
    intermediate_tensors: IntermediateTensors | None = None,
    inputs_embeds: torch.Tensor | None = None,
    **kwargs: object,
) -> torch.Tensor | IntermediateTensors:
    config = OccamTokenConfig.from_env()
    if not (
        config.true_sparse_active()
        and config.stage == "full"
        and config.stage2_active()
    ):
        return _ORIG_QWEN35_FORWARD(
            self,
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )

    if intermediate_tensors is not None:
        raise RuntimeError(
            "OccamToken true Stage-II prototype does not support pipeline "
            "intermediate_tensors yet."
        )
    if inputs_embeds is None:
        raise RuntimeError(
            "OccamToken true Stage-II requires merged inputs_embeds."
        )

    is_multimodal = getattr(self, "_occamtoken_last_is_multimodal", None)
    if is_multimodal is None:
        raise RuntimeError(
            "OccamToken true Stage-II cannot find the multimodal token mask. "
            "The Qwen3.5 embed_input_ids patch must run before forward()."
        )
    if int(is_multimodal.shape[0]) != int(inputs_embeds.shape[0]):
        raise RuntimeError(
            "OccamToken true Stage-II multimodal mask length mismatch: "
            f"is_multimodal={tuple(is_multimodal.shape)} "
            f"inputs_embeds={tuple(inputs_embeds.shape)}"
        )

    model = self.language_model.model
    stage2_layer = int(config.stage2_layer)
    hidden_states, residual = model.forward_until_layer(
        input_ids=input_ids,
        positions=positions,
        inputs_embeds=inputs_embeds,
        intermediate_tensors=None,
        stop_layer=stage2_layer,
    )

    image_mask = is_multimodal.to(device=hidden_states.device, dtype=torch.bool)
    text_mask = ~image_mask
    target_image_tokens = config.final_budget(int(image_mask.sum().item()))
    keep_mask, stats = stage2_true_keep_mask(
        hidden_states,
        image_mask=image_mask,
        text_mask=text_mask,
        target_image_tokens=target_image_tokens,
        config=config,
    )
    log_stats([stats])

    hidden_states = hidden_states[keep_mask]
    if residual is not None:
        residual = residual[keep_mask]
    positions = _positions_index_select(positions, keep_mask.to(device=positions.device))

    return model.forward_from_layer(
        hidden_states=hidden_states,
        residual=residual,
        positions=positions,
        start_layer=stage2_layer + 1,
    )


Qwen3_5ForConditionalGeneration.forward = _patched_qwen35_forward_stage2_true
