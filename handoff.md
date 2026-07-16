# Handoff: Phase2 Direct Encoder Compatibility

This file only records the changes and cautions added after the previous working Stage-I true sparse implementation.

## What Changed

Added an early M-RoPE patch helper:

```text
src/ascend_occamtoken/mrope.py
```

It exposes:

```python
from vllm_ascend.occamtoken.mrope import install_mrope_patch
```

Added a phase2 direct-encoder pruning helper:

```text
src/ascend_occamtoken/phase2.py
```

It exposes:

```python
from vllm_ascend.occamtoken.phase2 import prune_phase2_local_image_outputs
```

Updated:

```text
patches/worker/patch_occamtoken_qwen35.py
```

The worker patch now calls `install_mrope_patch()` instead of carrying a separate inline M-RoPE monkey patch.

Added detailed internal patch note:

```text
docs/phase2_direct_encoder_patch.md
```

## Why This Was Needed

Two failures appeared in optimized internal phase2/direct-encoder mode.

First failure:

```text
qwen3_vl.py::_get_mrope_input_positions
ValueError: all elements of broadcast shape must be non-negative
```

Cause:

```text
OccamToken shortened image placeholders, for example 88 -> 79.
Stock Qwen3VL M-RoPE still advanced by original image_grid_thw count, for example 88.
For multi-image prompts, the next image offset became smaller than st.
text_len became negative.
```

Second failure:

```text
local encoder rows mismatch: got 2916, expected rows=2624
```

Cause:

```text
Phase2 direct encoder validates encoder row count before model-level _patched_process_image_input can prune.
The encoder returns original rows.
The expected row count already comes from shortened placeholders.
```

## Internal Phase2 Code Changes Needed

Only the internal phase2/direct-encoder code needs to be changed, assuming the two helper modules above are already importable from `vllm_ascend.occamtoken`.

At the top of internal `patch_mm_opt/phase2.py` or equivalent plugin module:

```python
from vllm_ascend.occamtoken.mrope import install_mrope_patch
from vllm_ascend.occamtoken.phase2 import prune_phase2_local_image_outputs

install_mrope_patch()
```

Important: `install_mrope_patch()` must run before `_init_mrope_positions()`. If the stack trace still points to stock `site-packages/vllm/model_executor/models/qwen3_vl.py::_get_mrope_input_positions`, the patch was loaded too late or not loaded in that worker process.

In internal `_run_local_encode()`, after `_encode_local_images()` returns and before `torch.cat` / row-count validation:

```python
local_outputs = _encode_local_images(
    self, model, processor, my_pil_images, my_image_indices,
)

local_outputs = prune_phase2_local_image_outputs(
    local_outputs,
    my_image_indices=my_image_indices,
    output_sizes=output_sizes,
)

local_cat = torch.cat([out.contiguous() for out in local_outputs], dim=0)
```

Do not wait for `_patched_process_image_input`; in the phase2 direct path, row-count validation happens before that hook can run.

## Cautions

`prune_phase2_local_image_outputs()` assumes `local_outputs` is already split per image. This matches the internal `_encode_local_images()` snippet:

```python
sizes = (grid_thw.prod(-1) // merge_size // merge_size).tolist()
batch_outputs = list(image_embeds.split(sizes))
local_outputs.extend(batch_outputs)
```

The helper intentionally raises if the pruned length does not match `output_sizes[global_image_idx]`. Do not silently fallback in performance tests; a mismatch means placeholder pruning and encoder pruning are inconsistent.

The M-RoPE patch is length-correct but not yet top-k-index-correct. It assigns positions to the first N grid positions for the shortened placeholder span. This should stop the multi-image crash, but a future quality improvement should propagate actual Stage-I keep indices into M-RoPE position generation.

True Stage-II is still not implemented. In true mode, keep testing Stage-I only.

MoE internals still do not need changes. Pruning happens before the language model; MoE sees a shorter sequence.

## Next Smoke Test

Recommended order:

```text
8k text + 2 images, ratio=0.9
8k text + 20 images, ratio=0.9
```

Expected:

```text
no negative text_len in qwen3_vl.py
no phase2 local encoder rows mismatch
prefill reaches decode
```
