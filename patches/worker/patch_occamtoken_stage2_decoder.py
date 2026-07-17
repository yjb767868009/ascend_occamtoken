"""Decoder-layer helpers for true OccamToken Stage-II experiments.

The real Stage-II execution path is installed by ``patch_occamtoken_runner``.
This module only adds Qwen3.5 layer-splitting helpers so the runner can execute
layers 0..K and K+1..end under different attention metadata contexts.
"""

from __future__ import annotations

from itertools import islice

import torch

from vllm.distributed import get_pp_group
from vllm.sequence import IntermediateTensors
from vllm.model_executor.models.qwen3_5 import (
    Qwen3_5Model,
)


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
