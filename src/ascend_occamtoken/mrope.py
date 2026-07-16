"""M-RoPE compatibility patch for true image-token removal."""

from __future__ import annotations

import numpy as np
import torch


_INSTALLED = False


def _count_contiguous_tokens(input_tokens: list[int], offset: int, token_id: int) -> int:
    count = 0
    for token in input_tokens[offset:]:
        if token != token_id:
            break
        count += 1
    return count


def install_mrope_patch() -> None:
    """Patch Qwen3VL M-RoPE position init for shortened image placeholders.

    The stock Qwen3VL implementation derives image token counts from
    ``image_grid_thw``. OccamToken true Stage-I shortens image placeholders, so
    M-RoPE must use the actual placeholder span in ``input_tokens`` instead.
    """
    global _INSTALLED
    if _INSTALLED:
        return

    from vllm.model_executor.models.qwen3_vl import Qwen3VLForConditionalGeneration

    original_iter_mm_grid_hw = Qwen3VLForConditionalGeneration._iter_mm_grid_hw

    def iter_mm_grid_hw_occamtoken(
        input_tokens,
        mm_features,
        *,
        video_token_id,
        vision_start_token_id,
        vision_end_token_id,
        spatial_merge_size,
    ):
        sorted_features = sorted(mm_features, key=lambda f: f.mm_position.offset)
        for feature_idx, mm_feature in enumerate(sorted_features):
            offset = mm_feature.mm_position.offset
            if mm_feature.modality == "image":
                image_token_id = input_tokens[offset]
                t, h, w = mm_feature.data["image_grid_thw"].data.tolist()
                assert t == 1, f"Image must have 1 frame, got {t}"
                llm_grid_h = h // spatial_merge_size
                llm_grid_w = w // spatial_merge_size
                actual_num_tokens = _count_contiguous_tokens(
                    input_tokens, offset, image_token_id
                )
                if feature_idx + 1 < len(sorted_features):
                    next_offset = sorted_features[feature_idx + 1].mm_position.offset
                    actual_num_tokens = min(actual_num_tokens, next_offset - offset)
                yield offset, llm_grid_h, llm_grid_w, actual_num_tokens
            elif mm_feature.modality == "video":
                yield from original_iter_mm_grid_hw(
                    input_tokens,
                    [mm_feature],
                    video_token_id=video_token_id,
                    vision_start_token_id=vision_start_token_id,
                    vision_end_token_id=vision_end_token_id,
                    spatial_merge_size=spatial_merge_size,
                )
            else:
                raise ValueError(f"Unsupported modality: {mm_feature.modality}")

    def get_mrope_input_positions_occamtoken(input_tokens, mm_features, config):
        llm_pos_ids_list = []
        st = 0
        for (
            offset,
            llm_grid_h,
            llm_grid_w,
            actual_num_tokens,
        ) in iter_mm_grid_hw_occamtoken(
            input_tokens,
            mm_features,
            video_token_id=config.video_token_id,
            vision_start_token_id=config.vision_start_token_id,
            vision_end_token_id=config.vision_end_token_id,
            spatial_merge_size=config.vision_config.spatial_merge_size,
        ):
            if actual_num_tokens == 0:
                continue

            text_len = offset - st
            if text_len < 0:
                raise ValueError(
                    "OccamToken true Stage-I M-RoPE alignment failed: "
                    f"offset={offset} previous_end={st} "
                    f"actual_num_tokens={actual_num_tokens}"
                )
            st_idx = llm_pos_ids_list[-1].max() + 1 if llm_pos_ids_list else 0
            llm_pos_ids_list.append(
                np.broadcast_to(np.arange(text_len), (3, text_len)) + st_idx
            )

            expected_tokens_per_frame = llm_grid_h * llm_grid_w
            if actual_num_tokens > expected_tokens_per_frame:
                num_logical_frames = actual_num_tokens // expected_tokens_per_frame
                remainder = actual_num_tokens % expected_tokens_per_frame
                for _ in range(num_logical_frames):
                    grid_indices = np.indices((1, llm_grid_h, llm_grid_w)).reshape(
                        3, -1
                    )
                    llm_pos_ids_list.append(grid_indices + text_len + st_idx)
                    st_idx = llm_pos_ids_list[-1].max() + 1
                    text_len = 0
                if remainder > 0:
                    full_grid = np.indices((1, llm_grid_h, llm_grid_w)).reshape(3, -1)
                    llm_pos_ids_list.append(
                        full_grid[:, :remainder] + text_len + st_idx
                    )
            else:
                full_grid = np.indices((1, llm_grid_h, llm_grid_w)).reshape(3, -1)
                llm_pos_ids_list.append(
                    full_grid[:, :actual_num_tokens] + text_len + st_idx
                )

            st = offset + actual_num_tokens

        if st < len(input_tokens):
            st_idx = llm_pos_ids_list[-1].max() + 1 if llm_pos_ids_list else 0
            text_len = len(input_tokens) - st
            llm_pos_ids_list.append(
                np.broadcast_to(np.arange(text_len), (3, text_len)) + st_idx
            )

        llm_positions = np.concatenate(llm_pos_ids_list, axis=1).reshape(3, -1)
        mrope_position_delta = (llm_positions.max() + 1 - len(input_tokens)).item()
        return torch.from_numpy(llm_positions), mrope_position_delta

    Qwen3VLForConditionalGeneration._get_mrope_input_positions = staticmethod(
        get_mrope_input_positions_occamtoken
    )
    _INSTALLED = True
