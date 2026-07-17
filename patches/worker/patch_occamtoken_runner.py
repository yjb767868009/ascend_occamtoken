"""Runner-controlled true Stage-II pruning for Qwen3.5.

Stage-II cannot be a model-only ``hidden_states`` slice. Attention metadata,
GDN query starts, KV slot mapping, M-RoPE positions, and sampling indices all
describe the scheduled token layout. This patch splits the model forward inside
``NPUModelRunner._model_forward`` so the suffix layers run under metadata rebuilt
from the pruned sequence.
"""

from __future__ import annotations

import numpy as np
import torch

from vllm.config import CUDAGraphMode
from vllm.forward_context import BatchDescriptor, get_forward_context
from vllm.sequence import IntermediateTensors
from vllm_ascend.ascend_forward_context import set_ascend_forward_context
from vllm_ascend.occamtoken.config import OccamTokenConfig
from vllm_ascend.occamtoken.logging import log_stats
from vllm_ascend.occamtoken.pruning import stage2_true_keep_mask
from vllm_ascend.worker.model_runner_v1 import NPUModelRunner


_ORIG_PREPARE_INPUTS = NPUModelRunner._prepare_inputs
_ORIG_MODEL_FORWARD = NPUModelRunner._model_forward
_ORIG_PROPOSE_DRAFT_TOKEN_IDS = NPUModelRunner.propose_draft_token_ids
_ORIG_BOOKKEEPING_SYNC = NPUModelRunner._bookkeeping_sync

_kept_indices: torch.Tensor | None = None
_new_cu_seqlens: torch.Tensor | None = None
_new_seq_lens: torch.Tensor | None = None
_new_max_seq_len: int | None = None


def _patched_prepare_inputs(self, scheduler_output, num_scheduled_tokens):
    _clear_compact_metadata_plan()
    self._occamtoken_last_suffix_attn_metadata = None
    self._occamtoken_last_spec_decode_common_attn_metadata = None
    self._occamtoken_last_compact_total_tokens = None
    result = _ORIG_PREPARE_INPUTS(self, scheduler_output, num_scheduled_tokens)
    logits_indices, spec_decode_metadata, total_num_scheduled_tokens = result
    self._occamtoken_last_logits_indices = logits_indices
    self._occamtoken_last_spec_decode_metadata = spec_decode_metadata
    self._occamtoken_last_num_scheduled_tokens_np = np.array(
        num_scheduled_tokens, dtype=np.int32, copy=True
    )
    self._occamtoken_last_total_num_scheduled_tokens = int(total_num_scheduled_tokens)
    return result


def _clear_compact_metadata_plan() -> None:
    global _kept_indices, _new_cu_seqlens, _new_seq_lens, _new_max_seq_len
    _kept_indices = None
    _new_cu_seqlens = None
    _new_seq_lens = None
    _new_max_seq_len = None


def _publish_compact_metadata_plan(
    *,
    keep_mask: torch.Tensor,
    new_cu_seqlens: torch.Tensor,
    new_seq_lens: torch.Tensor,
    new_max_seq_len: int,
) -> None:
    global _kept_indices, _new_cu_seqlens, _new_seq_lens, _new_max_seq_len
    _kept_indices = keep_mask.nonzero(as_tuple=False).flatten().detach()
    _new_cu_seqlens = new_cu_seqlens.detach()
    _new_seq_lens = new_seq_lens.detach()
    _new_max_seq_len = int(new_max_seq_len)


def _apply_compact_metadata_plan(common_attn_metadata):
    if common_attn_metadata is None:
        return None
    if _kept_indices is None or _new_cu_seqlens is None or _new_seq_lens is None:
        return common_attn_metadata

    device = common_attn_metadata.query_start_loc.device
    cu_cpu = _new_cu_seqlens.to(device="cpu", dtype=torch.int32)
    cu_gpu = cu_cpu.to(device=device, non_blocking=True)
    seq_cpu = _new_seq_lens.to(device="cpu", dtype=torch.int32)
    seq_device = (
        common_attn_metadata.seq_lens.device
        if getattr(common_attn_metadata, "seq_lens", None) is not None
        else device
    )
    seq_gpu = seq_cpu.to(device=seq_device, non_blocking=True)
    max_query_len = int((cu_cpu[1:] - cu_cpu[:-1]).max().item())

    common_attn_metadata.query_start_loc = cu_gpu
    common_attn_metadata.query_start_loc_cpu = cu_cpu
    common_attn_metadata.seq_lens = seq_gpu
    common_attn_metadata.seq_lens_cpu = seq_cpu
    if getattr(common_attn_metadata, "_seq_lens_cpu", None) is not None:
        common_attn_metadata._seq_lens_cpu = seq_cpu
    common_attn_metadata.num_actual_tokens = int(cu_cpu[-1].item())
    common_attn_metadata.max_query_len = max_query_len
    common_attn_metadata.max_seq_len = int(_new_max_seq_len or seq_cpu.max().item())
    common_attn_metadata.actual_seq_lengths_q = cu_cpu[1:].tolist()
    common_attn_metadata.num_input_tokens = common_attn_metadata.num_actual_tokens
    _assert_compact_metadata("mtp_common_attn_metadata", common_attn_metadata)
    return common_attn_metadata


def _assert_compact_metadata(name: str, common_attn_metadata) -> None:
    if _new_cu_seqlens is None or _new_seq_lens is None:
        return

    qsl_cpu = common_attn_metadata.query_start_loc_cpu.to(
        device="cpu", dtype=torch.int32
    )
    seq_cpu = common_attn_metadata.seq_lens_cpu.to(device="cpu", dtype=torch.int32)
    expected_qsl = _new_cu_seqlens.to(device="cpu", dtype=torch.int32)
    expected_seq = _new_seq_lens.to(device="cpu", dtype=torch.int32)

    if qsl_cpu.shape != expected_qsl.shape or not torch.equal(qsl_cpu, expected_qsl):
        raise RuntimeError(
            f"OccamToken compact metadata mismatch in {name}: "
            f"query_start_loc_cpu={qsl_cpu.tolist()} "
            f"expected={expected_qsl.tolist()}"
        )
    if seq_cpu.shape != expected_seq.shape or not torch.equal(seq_cpu, expected_seq):
        raise RuntimeError(
            f"OccamToken compact metadata mismatch in {name}: "
            f"seq_lens_cpu={seq_cpu.tolist()} expected={expected_seq.tolist()}"
        )

    expected_total = int(expected_qsl[-1].item())
    if int(common_attn_metadata.num_actual_tokens) != expected_total:
        raise RuntimeError(
            f"OccamToken compact metadata mismatch in {name}: "
            f"num_actual_tokens={common_attn_metadata.num_actual_tokens} "
            f"expected={expected_total}"
        )

    expected_max_query = int((expected_qsl[1:] - expected_qsl[:-1]).max().item())
    if int(common_attn_metadata.max_query_len) != expected_max_query:
        raise RuntimeError(
            f"OccamToken compact metadata mismatch in {name}: "
            f"max_query_len={common_attn_metadata.max_query_len} "
            f"expected={expected_max_query}"
        )


def _index_positions(positions: torch.Tensor, keep_mask: torch.Tensor) -> torch.Tensor:
    if positions.ndim == 2:
        return positions[:, keep_mask]
    return positions[keep_mask]


def _stage2_enabled(inputs_embeds: torch.Tensor | None) -> bool:
    config = OccamTokenConfig.from_env()
    return bool(
        inputs_embeds is not None
        and config.true_sparse_active()
        and config.stage == "full"
        and config.stage2_active()
    )


def _spec_method(self) -> str | None:
    speculative_config = getattr(self, "speculative_config", None)
    method = getattr(speculative_config, "method", None)
    return None if method is None else str(method)


def _is_mtp_method(method: str | None) -> bool:
    return method in {"mtp", "qwen3_5_mtp"}


def _get_qwen35_text_model(model):
    language_model = getattr(model, "language_model", None)
    if language_model is None:
        return None
    text_model = getattr(language_model, "model", None)
    if text_model is None:
        return None
    if not (
        hasattr(text_model, "forward_until_layer")
        and hasattr(text_model, "forward_from_layer")
    ):
        return None
    return text_model


def _is_decode_query_len(query_len: int, decode_token_per_req: int) -> bool:
    return int(query_len) <= int(max(1, decode_token_per_req))


def _build_new_query_lens(self, keep_mask_cpu: np.ndarray) -> np.ndarray:
    num_reqs = int(self.input_batch.num_reqs)
    old_qsl = self.query_start_loc.np[: num_reqs + 1].astype(np.int64, copy=True)
    new_lens = np.empty(num_reqs, dtype=np.int32)
    for req_idx in range(num_reqs):
        start = int(old_qsl[req_idx])
        end = int(old_qsl[req_idx + 1])
        new_lens[req_idx] = int(keep_mask_cpu[start:end].sum())
    return new_lens


def _remap_indices(name: str, indices: torch.Tensor, old_to_new: torch.Tensor) -> torch.Tensor:
    mapped = old_to_new.index_select(0, indices.to(device=old_to_new.device, dtype=torch.long))
    if bool((mapped < 0).any().item()):
        bad = indices[mapped < 0][:8].detach().cpu().tolist()
        raise RuntimeError(
            "OccamToken Stage-II attempted to remap an index that was pruned: "
            f"{name} bad_old_indices={bad}"
        )
    return mapped.to(device=indices.device, dtype=indices.dtype)


def _update_metadata_after_drop(
    self,
    *,
    current_layer_idx: int,
    suffix_attn_metadata,
    keep_mask: torch.Tensor,
    new_query_lens: np.ndarray,
    new_total_tokens: int,
    slot_mapping_out: torch.Tensor,
):
    """Apply compact token metadata to the suffix forward context.

    ``new_query_lens`` describes the compact query plan. KV ``seq_lens`` are
    updated separately because mixed prefill+decode batches need decode rows to
    keep their original cache lengths.
    """
    fwd_ctx = get_forward_context()
    if fwd_ctx is None:
        return suffix_attn_metadata

    num_reqs = int(self.input_batch.num_reqs)
    new_cu = np.zeros(num_reqs + 1, dtype=np.int32)
    np.cumsum(new_query_lens, out=new_cu[1:])

    new_cu_cpu = torch.from_numpy(new_cu)
    new_seq_cpu = self.optimistic_seq_lens_cpu[:num_reqs].clone()
    new_max_seq_len = int(new_seq_cpu.max().item()) if num_reqs else 0
    _publish_compact_metadata_plan(
        keep_mask=keep_mask,
        new_cu_seqlens=new_cu_cpu,
        new_seq_lens=new_seq_cpu,
        new_max_seq_len=new_max_seq_len,
    )
    fwd_ctx.token_drop_applied = True
    fwd_ctx.token_drop_layer_idx = int(current_layer_idx)
    fwd_ctx.new_cu_seqlens = new_cu_cpu.to(device=self.device, non_blocking=True)
    fwd_ctx.new_cu_seqlens_cpu = new_cu_cpu
    fwd_ctx.new_seq_lens = new_seq_cpu.to(device=self.device, non_blocking=True)
    fwd_ctx.new_seq_lens_cpu = new_seq_cpu
    fwd_ctx.new_total_tokens = int(new_total_tokens)
    fwd_ctx.new_max_seq_len = new_max_seq_len
    fwd_ctx.keep_mask_after_token_drop = keep_mask
    fwd_ctx.slot_mapping_out = slot_mapping_out

    def patch_one(meta):
        if hasattr(meta, "slot_mapping"):
            meta.slot_mapping = slot_mapping_out[:new_total_tokens]
        if hasattr(meta, "seq_lens"):
            meta.seq_lens = self.seq_lens[:num_reqs]
        if hasattr(meta, "seq_lens_cpu"):
            meta.seq_lens_cpu = self.optimistic_seq_lens_cpu[:num_reqs]
        if hasattr(meta, "seq_lens_list"):
            meta.seq_lens_list = self.optimistic_seq_lens_cpu[:num_reqs].tolist()
        if hasattr(meta, "query_start_loc"):
            meta.query_start_loc = self.query_start_loc.gpu[: num_reqs + 1]
        if hasattr(meta, "actual_seq_lengths_q"):
            meta.actual_seq_lengths_q = new_cu[1:].tolist()
        if hasattr(meta, "max_query_len"):
            meta.max_query_len = int(new_query_lens.max()) if num_reqs else 0
        if hasattr(meta, "max_seq_len"):
            meta.max_seq_len = int(self.optimistic_seq_lens_cpu[:num_reqs].max().item())
        if hasattr(meta, "num_actual_tokens"):
            meta.num_actual_tokens = int(new_total_tokens)
        if hasattr(meta, "num_actual_tokens_pcp_padded"):
            meta.num_actual_tokens_pcp_padded = int(new_total_tokens)
        if hasattr(meta, "nums_dict") or hasattr(meta, "batch_ptr"):
            # GDN causal_conv1d metadata is rebuilt by _build_attention_metadata.
            # The assignments here make the ownership explicit for downstream
            # debugging; they intentionally do not synthesize conv metadata.
            pass
        return meta

    if isinstance(suffix_attn_metadata, list):
        for ub_meta in suffix_attn_metadata:
            for meta in ub_meta.values():
                patch_one(meta)
    elif isinstance(suffix_attn_metadata, dict):
        for meta in suffix_attn_metadata.values():
            patch_one(meta)

    fwd_slot_mapping = getattr(fwd_ctx, "slot_mapping", None)
    if isinstance(fwd_slot_mapping, dict):
        for layer_idx, _old in list(fwd_slot_mapping.items()):
            try:
                should_update = int(layer_idx) > int(current_layer_idx)
            except (TypeError, ValueError):
                should_update = True
            if should_update:
                fwd_slot_mapping[layer_idx] = slot_mapping_out
    elif fwd_slot_mapping is not None:
        fwd_ctx.slot_mapping = slot_mapping_out

    return suffix_attn_metadata


def _apply_pruned_runner_layout(
    self,
    *,
    keep_mask: torch.Tensor,
    positions: torch.Tensor,
):
    num_reqs = int(self.input_batch.num_reqs)
    old_num_tokens = int(keep_mask.shape[0])
    keep_mask_cpu = keep_mask.detach().to(device="cpu", dtype=torch.bool).numpy()
    old_qsl = self.query_start_loc.np[: num_reqs + 1].astype(np.int64, copy=True)
    old_lens = old_qsl[1:] - old_qsl[:-1]
    new_lens = _build_new_query_lens(self, keep_mask_cpu)
    new_num_tokens = int(new_lens.sum())
    new_qsl = np.zeros(num_reqs + 1, dtype=np.int32)
    np.cumsum(new_lens, out=new_qsl[1:])

    if new_num_tokens <= 0:
        raise RuntimeError("OccamToken Stage-II pruned the entire scheduled batch.")

    self.query_lens = torch.from_numpy(new_lens)
    self.query_start_loc.np[0] = 0
    self.query_start_loc.np[1 : num_reqs + 1] = new_qsl[1:]
    self.query_start_loc.np[num_reqs + 1 :].fill(-1)
    self.query_start_loc.copy_to_gpu()

    if getattr(self, "_has_gdn", False):
        self.gdn_query_start_loc.np[0] = 0
        self.gdn_query_start_loc.np[1 : num_reqs + 1] = new_qsl[1:]
        self.gdn_query_start_loc.np[num_reqs + 1 :].fill(new_qsl[-1])
        self.gdn_query_start_loc.copy_to_gpu()

    self.num_scheduled_tokens.np[:num_reqs] = new_lens
    self.num_scheduled_tokens.copy_to_gpu(num_reqs)

    # Keep the runner's 1D absolute positions in sync for slot mapping and
    # drafter helpers; pass the 2D M-RoPE positions separately to the model.
    keep_on_positions = keep_mask.to(device=self.positions.device, dtype=torch.bool)
    pruned_1d_positions = self.positions[:old_num_tokens][keep_on_positions]
    self.positions[:new_num_tokens].copy_(pruned_1d_positions, non_blocking=True)

    if positions.ndim == 2 and hasattr(self, "mrope_positions"):
        self.mrope_positions.gpu[:, :new_num_tokens].copy_(
            positions[:, :new_num_tokens], non_blocking=True
        )

    old_seq_lens = self.seq_lens[:num_reqs].clone()
    new_seq_lens = self.num_computed_tokens[:num_reqs] + self.num_scheduled_tokens.gpu[:num_reqs]
    decode_token_per_req = int(getattr(self, "decode_token_per_req", 1))
    decode_req_mask = torch.tensor(
        [
            _is_decode_query_len(int(old_lens[i]), decode_token_per_req)
            for i in range(num_reqs)
        ],
        dtype=torch.bool,
        device=self.seq_lens.device,
    )
    new_seq_lens = torch.where(decode_req_mask, old_seq_lens, new_seq_lens)
    self.seq_lens[:num_reqs] = new_seq_lens
    self.seq_lens[num_reqs:].fill_(0)
    self.optimistic_seq_lens_cpu[:num_reqs].copy_(
        self.seq_lens[:num_reqs], non_blocking=False
    )
    self.optimistic_seq_lens_cpu[num_reqs:].fill_(0)
    self._seq_lens_cpu_event_pending = False

    self.input_batch.block_table.compute_slot_mapping(
        num_reqs,
        self.query_start_loc.gpu[: num_reqs + 1],
        self.positions[:new_num_tokens],
    )

    old_to_new = torch.full(
        (old_num_tokens,),
        -1,
        dtype=torch.long,
        device=keep_mask.device,
    )
    old_to_new[keep_mask] = torch.arange(
        new_num_tokens,
        dtype=torch.long,
        device=keep_mask.device,
    )

    logits_indices = getattr(self, "_occamtoken_last_logits_indices", None)
    if logits_indices is not None:
        mapped_logits_indices = _remap_indices(
            "logits_indices", logits_indices, old_to_new
        )
        if mapped_logits_indices.shape == logits_indices.shape:
            logits_indices.copy_(mapped_logits_indices)
            mapped_logits_indices = logits_indices
        self.logits_indices = mapped_logits_indices
        self._occamtoken_last_logits_indices = mapped_logits_indices
        spec_decode_metadata = getattr(
            self, "_occamtoken_last_spec_decode_metadata", None
        )
        if spec_decode_metadata is not None:
            spec_decode_metadata.logits_indices = mapped_logits_indices

    slot_mapping_out = self.input_batch.block_table[0].slot_mapping.gpu[:new_num_tokens]
    return new_lens, new_num_tokens, slot_mapping_out


def _patched_model_forward(
    self,
    num_tokens_padded: int,
    input_ids: torch.Tensor | None = None,
    positions: torch.Tensor | None = None,
    intermediate_tensors: IntermediateTensors | None = None,
    inputs_embeds: torch.Tensor | None = None,
    **model_kwargs,
):
    if not _stage2_enabled(inputs_embeds):
        return _ORIG_MODEL_FORWARD(
            self,
            num_tokens_padded,
            input_ids,
            positions,
            intermediate_tensors,
            inputs_embeds,
            **model_kwargs,
        )

    config = OccamTokenConfig.from_env()
    text_model = _get_qwen35_text_model(self.model)
    if text_model is None or positions is None:
        return _ORIG_MODEL_FORWARD(
            self,
            num_tokens_padded,
            input_ids,
            positions,
            intermediate_tensors,
            inputs_embeds,
            **model_kwargs,
        )
    if intermediate_tensors is not None:
        raise RuntimeError(
            "OccamToken Stage-II true runner path requires the pruning layer "
            "to be on the first pipeline rank for now."
        )

    is_multimodal = getattr(self.model, "_occamtoken_last_is_multimodal", None)
    if is_multimodal is None:
        raise RuntimeError(
            "OccamToken Stage-II true runner path cannot find the multimodal "
            "token mask recorded by the Qwen3.5 embed_input_ids patch."
        )

    old_num_tokens = int(inputs_embeds.shape[0])
    if int(is_multimodal.shape[0]) != old_num_tokens:
        raise RuntimeError(
            "OccamToken Stage-II multimodal mask length mismatch: "
            f"mask={tuple(is_multimodal.shape)} embeds={tuple(inputs_embeds.shape)}"
        )
    stage2_layer = int(config.stage2_layer)

    hidden_states, residual = text_model.forward_until_layer(
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

    if int(keep_mask.sum().item()) == old_num_tokens:
        return text_model.forward_from_layer(
            hidden_states=hidden_states,
            residual=residual,
            positions=positions,
            start_layer=stage2_layer + 1,
        )

    hidden_states = hidden_states[keep_mask]
    if residual is not None:
        residual = residual[keep_mask]
    suffix_positions = _index_positions(positions, keep_mask.to(device=positions.device))
    new_lens, new_num_tokens, slot_mapping_out = _apply_pruned_runner_layout(
        self,
        keep_mask=keep_mask,
        positions=suffix_positions,
    )

    num_reqs = int(self.input_batch.num_reqs)
    spec_decode_metadata = getattr(self, "_occamtoken_last_spec_decode_metadata", None)
    use_spec_decode = spec_decode_metadata is not None
    logits_indices = getattr(self, "_occamtoken_last_logits_indices", None)
    suffix_attn_metadata, spec_decode_common_attn_metadata = self._build_attention_metadata(
        num_tokens=new_num_tokens,
        num_tokens_padded=new_num_tokens,
        num_reqs=num_reqs,
        num_reqs_padded=num_reqs,
        max_query_len=int(new_lens.max()),
        ubatch_slices=None,
        logits_indices=logits_indices,
        use_spec_decode=use_spec_decode,
        num_scheduled_tokens=None,
        num_scheduled_tokens_np=new_lens,
        cascade_attn_prefix_lens=None,
    )
    self._occamtoken_last_suffix_attn_metadata = suffix_attn_metadata
    self._occamtoken_last_spec_decode_common_attn_metadata = (
        spec_decode_common_attn_metadata
    )
    self._occamtoken_last_compact_total_tokens = int(new_num_tokens)

    with set_ascend_forward_context(
        suffix_attn_metadata,
        self.vllm_config,
        num_tokens=new_num_tokens,
        num_tokens_across_dp=None,
        aclgraph_runtime_mode=CUDAGraphMode.NONE,
        batch_descriptor=BatchDescriptor(new_num_tokens),
        num_actual_tokens=new_num_tokens,
        model_instance=self.model,
        max_tokens_across_pcp=0,
        skip_compiled=True,
    ):
        suffix_attn_metadata = _update_metadata_after_drop(
            self,
            current_layer_idx=stage2_layer,
            suffix_attn_metadata=suffix_attn_metadata,
            keep_mask=keep_mask,
            new_query_lens=new_lens,
            new_total_tokens=new_num_tokens,
            slot_mapping_out=slot_mapping_out,
        )
        suffix_ctx = get_forward_context()
        if suffix_ctx is not None:
            suffix_ctx.attn_metadata = suffix_attn_metadata
        return text_model.forward_from_layer(
            hidden_states=hidden_states,
            residual=residual,
            positions=suffix_positions,
            start_layer=stage2_layer + 1,
        )


NPUModelRunner._prepare_inputs = _patched_prepare_inputs
NPUModelRunner._model_forward = _patched_model_forward


def _patched_propose_draft_token_ids(
    self,
    valid_sampled_token_ids,
    sampling_metadata,
    scheduler_output,
    spec_decode_metadata,
    spec_decode_common_attn_metadata,
    positions,
    num_scheduled_tokens,
    hidden_states,
    aux_hidden_states=None,
    sample_hidden_states=None,
    target_model_batch_desc=None,
):
    compact_common = getattr(
        self, "_occamtoken_last_spec_decode_common_attn_metadata", None
    )
    if _kept_indices is not None and not _is_mtp_method(_spec_method(self)):
        raise RuntimeError(
            "OccamToken compact speculative metadata is currently wired for "
            f"qwen3_5_mtp/MTP, but got method={_spec_method(self)!r}."
        )
    if compact_common is not None:
        spec_decode_common_attn_metadata = compact_common
    spec_decode_common_attn_metadata = _apply_compact_metadata_plan(
        spec_decode_common_attn_metadata
    )
    compact_total = getattr(self, "_occamtoken_last_compact_total_tokens", None)
    if compact_total is not None:
        num_scheduled_tokens = int(compact_total)
    try:
        return _ORIG_PROPOSE_DRAFT_TOKEN_IDS(
            self,
            valid_sampled_token_ids,
            sampling_metadata,
            scheduler_output,
            spec_decode_metadata,
            spec_decode_common_attn_metadata,
            positions,
            num_scheduled_tokens,
            hidden_states,
            aux_hidden_states,
            sample_hidden_states,
            target_model_batch_desc,
        )
    finally:
        _clear_compact_metadata_plan()


def _patched_bookkeeping_sync(
    self,
    scheduler_output,
    sampler_output,
    logits,
    hidden_states,
    num_scheduled_tokens,
    spec_decode_metadata,
):
    compact_total = getattr(self, "_occamtoken_last_compact_total_tokens", None)
    if compact_total is not None:
        num_scheduled_tokens = min(int(num_scheduled_tokens), int(compact_total))
    return _ORIG_BOOKKEEPING_SYNC(
        self,
        scheduler_output,
        sampler_output,
        logits,
        hidden_states,
        num_scheduled_tokens,
        spec_decode_metadata,
    )


NPUModelRunner.propose_draft_token_ids = _patched_propose_draft_token_ids
NPUModelRunner._bookkeeping_sync = _patched_bookkeeping_sync
