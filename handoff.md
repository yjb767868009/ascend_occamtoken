# Ascend OccamToken Handoff

This is the current handoff for the Qwen3.5 true Stage-I visual token pruning work.

## Goal

Improve Qwen3.5 multimodal RAG prefill performance on vLLM Ascend / AscendCloud workloads with long text and many images, for example:

- about 8k-10k text tokens
- 20 images
- image tokens reduced by Stage-I pruning, for example 2916 -> 2624 at ratio 0.9

The current goal is only:

```text
Stage-I true image-token removal
```

Stage-II is intentionally disabled in true-removal mode.

## Repository

GitHub:

```text
git@github.com:yjb767868009/ascend_occamtoken.git
```

Local repo:

```text
/home/yujubo/ascend_occamtoken
```

Local vLLM Ascend checkout used during development:

```text
/home/yujubo/vllm_ascend
```

Install into a vLLM Ascend checkout with:

```bash
bash scripts/install_into_vllm_ascend.sh <VLLM_ASCEND_CHECKOUT>
```

The installer copies:

```text
src/ascend_occamtoken/* -> <VLLM_ASCEND_CHECKOUT>/vllm_ascend/occamtoken/
patches/platform/patch_occamtoken.py -> <VLLM_ASCEND_CHECKOUT>/vllm_ascend/patch/platform/
patches/worker/patch_occamtoken_qwen35.py -> <VLLM_ASCEND_CHECKOUT>/vllm_ascend/patch/worker/
```

## Runtime Flags

Use this for true Stage-I pruning:

```bash
export VLLM_ASCEND_OCCAMTOKEN_ENABLE=1
export VLLM_ASCEND_OCCAMTOKEN_IMPL=true
export VLLM_ASCEND_OCCAMTOKEN_STAGE=stage1
export VLLM_ASCEND_OCCAMTOKEN_STAGE1_RATIO=0.9
export VLLM_ASCEND_OCCAMTOKEN_STRICT=1
export VLLM_ASCEND_OCCAMTOKEN_LOG_STATS=1
```

For fixed-budget true pruning:

```bash
export VLLM_ASCEND_OCCAMTOKEN_STAGE=fixed
export VLLM_ASCEND_OCCAMTOKEN_TARGET_TOKENS=<target_tokens>
```

For masked quality-only ablations:

```bash
export VLLM_ASCEND_OCCAMTOKEN_IMPL=masked
```

## Current Implementation

Important files:

```text
src/ascend_occamtoken/config.py
src/ascend_occamtoken/pruning.py
src/ascend_occamtoken/phase2.py
src/ascend_occamtoken/mrope.py
patches/worker/patch_occamtoken_qwen35.py
patches/platform/patch_occamtoken.py
docs/phase2_direct_encoder_patch.md
docs/testing_handoff.md
```

### Masked Mode

Masked mode keeps the visual sequence length unchanged and replaces pruned visual embeddings with a mean or zero vector.

It is useful for quality ablation only. It should not be used for performance claims.

### True Mode

True mode reduces image placeholder count and image embedding count.

Data flow for stock vLLM/vLLM Ascend path:

```text
_patched_get_prompt_updates
  -> reduces image placeholders per image

vision encoder
  -> produces original image embeddings

_patched_process_image_input
  -> prune_stage1_true per image

_patched_embed_input_ids
  -> merge shortened placeholders with shortened embeddings
```

Data flow for optimized phase2/direct-encoder path:

```text
_patched_get_prompt_updates
  -> reduces image placeholders per image

phase2 direct visual(pixel_values, grid_thw)
  -> produces original image embeddings

prune_phase2_local_image_outputs
  -> prune original encoder outputs before phase2 row-count validation
```

## What Is Deliberately Not Implemented

True Stage-II is not implemented.

In true mode:

```text
stage2: no true pruning
full: Stage-I true pruning only
```

This is intentional. Stage-II depends on text/query information and happens too late for the current scheduler/metadata path. Implementing it safely would require deeper scheduler and attention metadata changes.

MoE internals are not modified.

Reason:

- pruning happens before the language model
- MoE sees a shorter token sequence
- router/expert logic does not need special handling

## Known Issue 1: M-RoPE Negative text_len

Observed failure:

```text
ValueError: all elements of broadcast shape must be non-negative
qwen3_vl.py::_get_mrope_input_positions
text_len = offset - st
```

Root cause:

```text
placeholder count was shortened, for example 88 -> 79
but stock Qwen3VL M-RoPE still used image_grid_thw original count, for example 88
st advanced too far
next image offset became smaller than st
text_len became negative
```

Fix:

```text
src/ascend_occamtoken/mrope.py
```

It provides:

```python
from vllm_ascend.occamtoken.mrope import install_mrope_patch

install_mrope_patch()
```

The patch changes Qwen3VL M-RoPE position initialization so image actual token count is derived from the actual placeholder span in `input_tokens`, not from original `image_grid_thw`.

Important: in optimized phase2 / AscendCloud deployments, this must run before `_init_mrope_positions()`.

If a crash still points to stock:

```text
site-packages/vllm/model_executor/models/qwen3_vl.py::_get_mrope_input_positions
```

then `install_mrope_patch()` was loaded too late or not loaded in that worker process.

## Known Issue 2: Phase2 Direct Encoder Row Mismatch

Observed failure:

```text
local encoder rows mismatch
got 2916
expected rows=2624
```

Root cause:

```text
_patched_get_prompt_updates reduced placeholder count
phase2 direct encoder produced original image embeddings
phase2 validated row count before _patched_process_image_input could prune
```

Fix:

```text
src/ascend_occamtoken/phase2.py
```

It provides:

```python
from vllm_ascend.occamtoken.phase2 import prune_phase2_local_image_outputs
```

Use it in the internal phase2 direct encoder path after `_encode_local_images()` and before `torch.cat` / row-count validation:

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

See:

```text
docs/phase2_direct_encoder_patch.md
```

## Required Internal Phase2 Imports

In the internal `patch_mm_opt/phase2.py` or equivalent plugin module, add at import time:

```python
from vllm_ascend.occamtoken.mrope import install_mrope_patch
from vllm_ascend.occamtoken.phase2 import prune_phase2_local_image_outputs

install_mrope_patch()
```

Then call `prune_phase2_local_image_outputs()` inside `_run_local_encode()` after `_encode_local_images()` returns.

## Multi-Image Consistency

For multi-image requests, placeholder and embedding pruning must remain per-image aligned.

Current true path tracks per-image budgets:

```text
_patched_get_prompt_updates
  -> appends (original_tokens, budget) per image

_patched_process_image_input
  -> pops the expected budget per image and checks actual pruned length
```

If mismatch happens, one warning is printed:

```text
[occamtoken] true Stage-I multi-image budget warning: ...
```

In strict mode, unexpected processor structure causes fast failure instead of silent fallback.

## Current Limitations

1. M-RoPE positions are length-correct, but not yet top-k-index-correct.

   Stage-I selects top-k visual embeddings, but current M-RoPE patch assigns positions to the first N grid positions. This avoids crashes and length mismatches, but for better accuracy we should eventually propagate keep indices into M-RoPE position generation.

2. Phase2 direct encoder helper assumes `local_outputs` is already split per image.

   This matches the provided internal `_encode_local_images()` implementation:

   ```python
   sizes = (grid_thw.prod(-1) // merge_size // merge_size).tolist()
   batch_outputs = list(image_embeds.split(sizes))
   local_outputs.extend(batch_outputs)
   ```

3. True Stage-II is not implemented.

4. MoE is not specially patched.

## Recommended Next Test

Use a smaller multi-image request first:

```text
8k text + 2 images
ratio=0.9
```

Then scale to:

```text
8k text + 20 images
ratio=0.9
```

Expected success signals:

```text
no qwen3_vl.py negative text_len crash
no phase2 local encoder rows mismatch
[occamtoken] logs show original > kept
prefill proceeds to decode
```

If the negative `text_len` crash still happens, first verify that the internal phase2/plugin process imported and ran:

```python
install_mrope_patch()
```

before `_init_mrope_positions()`.

If the phase2 row mismatch still happens, verify that:

```python
prune_phase2_local_image_outputs()
```

is called before:

```python
local_cat = torch.cat(...)
expected_rows = grouped_output_lens[tp_rank]
```

## Latest Known Commits

Important commits in order:

```text
5d8aec2 Add Stage-I true sparse image pruning
081c156 Harden true sparse prompt replacement
e8445d6 Guard true sparse multi-image alignment
a23f6c3 Fix prompt updates keyword signature
e948a41 Fix true sparse M-RoPE image positions
92fde54 Add phase2 direct encoder pruning helper
```

The handoff may be newer than this list; always check:

```bash
git log --oneline -8
```
