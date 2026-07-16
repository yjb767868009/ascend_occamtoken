# Patch for Phase2 Direct Encoder Path

This patch is for deployments that run image encoding in an optimized phase2
worker path before Qwen3.5 model-level `_process_image_input` is called.

## Problem

OccamToken true Stage-I reduces image placeholders first:

```text
2916 placeholders -> 2624 placeholders
```

The optimized phase2 path then runs the vision encoder directly:

```text
visual(pixel_values, grid_thw) -> 2916 embeddings
```

If phase2 validates row counts before the model-level OccamToken pruning patch
runs, it sees:

```text
actual encoder rows = 2916
expected rows from mm_position.get_num_embeds() = 2624
```

and fails before `_patched_process_image_input` can prune.

## Fix

Apply OccamToken Stage-I true pruning immediately after `_encode_local_images`
returns per-image encoder outputs and before `torch.cat` / row-count validation.

## Minimal Internal Change

In your internal `patch_mm_opt/phase2.py`, add:

```python
from vllm_ascend.occamtoken.phase2 import prune_phase2_local_image_outputs
```

Then in `_run_local_encode`, change:

```python
local_outputs = _encode_local_images(
    self, model, processor, my_pil_images, my_image_indices,
)

local_cat = torch.cat([out.contiguous() for out in local_outputs], dim=0)
```

to:

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

Keep the existing row-count validation:

```python
expected_rows = grouped_output_lens[tp_rank]

if int(local_cat.shape[0]) != int(expected_rows):
    raise RuntimeError(...)
```

After this patch, the validation should compare:

```text
2624 == 2624
```

instead of:

```text
2916 != 2624
```

## Why This Location

`_encode_local_images` already returns `local_outputs` split per image, so the
helper can prune each image independently and check the result against
`output_sizes[global_image_idx]`.

This avoids relying on the later Qwen3.5 `_process_image_input` hook, which does
not run before the optimized phase2 row-count validation.

## Expected Runtime Settings

Use true Stage-I mode:

```bash
export VLLM_ASCEND_OCCAMTOKEN_ENABLE=1
export VLLM_ASCEND_OCCAMTOKEN_IMPL=true
export VLLM_ASCEND_OCCAMTOKEN_STAGE=stage1
export VLLM_ASCEND_OCCAMTOKEN_STAGE1_RATIO=0.9
export VLLM_ASCEND_OCCAMTOKEN_STRICT=1
```

For fixed budget mode:

```bash
export VLLM_ASCEND_OCCAMTOKEN_STAGE=fixed
export VLLM_ASCEND_OCCAMTOKEN_TARGET_TOKENS=<target_tokens>
```

## Failure Signal

If placeholder pruning and encoder pruning disagree, the helper raises:

```text
OccamToken phase2 prune size mismatch
```

This is intentional. A mismatch means phase2 would later fail at merge or M-RoPE
alignment, so it should not be silently ignored in performance tests.
