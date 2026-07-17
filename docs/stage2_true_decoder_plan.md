# True Stage-II Decoder-Layer Removal Plan

This note defines the next implementation target for real OccamToken Stage-II on
Qwen3.5. It intentionally rejects the previous shortcuts:

- Stage-II-lite masking in `embed_input_ids()`
- physically reducing `full,true` to `TARGET_RATIO` using Stage-I scores only
- text-aware pruning immediately after the vision encoder

Those paths either do not remove tokens or do not use decoder-layer
text-visual interaction.

## Target Semantics

Stage-I remains the early candidate generator:

```text
vision encoder output -> Stage-I visual saliency -> shorter visual candidates
```

True Stage-II must run after the LLM has mixed text and visual information:

```text
Stage-I candidates + text tokens
  -> LLM layers 0..K
  -> compute query-conditioned image-token scores from hidden states
  -> physically drop low-score image tokens
  -> LLM layers K+1..end run on the shorter sequence
```

The Stage-II score should be based on decoder hidden states, for example:

```python
text_h = hidden_states[text_mask]
image_h = hidden_states[image_mask]

text_h = select_text_window(
    text_h,
    max_text_tokens=config.max_text_tokens,
    question_tail_tokens=config.question_tail_tokens,
)

scores = normalize(image_h) @ normalize(text_h).T
keep_image = topk(scores.max(dim=-1), target_budget)
```

## Why Phase2 Is Not Enough

The internal phase2/direct-encoder path can see image embeddings and can reduce
rows before row-count validation. However, it runs before decoder layers, so it
cannot use the LLM hidden states that encode query-image interaction.

Using text embeddings or prompt tokens at phase2 time is only a Stage-II-lite
approximation. It is not the requested paper-like Stage-II.

## Required Runtime State Changes

Decoder-layer true removal cannot only slice `hidden_states`. The following
runtime structures must remain consistent:

```text
hidden_states
residual
positions / M-RoPE positions
input_ids or token-type masks used after pruning
attention metadata
slot_mapping
KV cache write locations
per-request scheduled token counts
multimodal embedding order and image-token masks
```

If `hidden_states` is shortened while attention metadata still describes the old
sequence, the next attention layer can read/write the wrong KV slots or crash.

## Minimal Safe Implementation Path

Implement a constrained prototype first. Do not try to support every vLLM path
in the first patch.

Initial constraints:

```text
prefill only
single request
no PCP
no chunked prefill
no prefix cache reuse for the pruned prompt
no speculative decode
M-RoPE enabled but rebuilt after pruning
```

The first prototype should fail fast when these constraints are not met.

## Recommended Architecture

Use a runner-controlled two-segment prefill, not a model-only tensor slice.

The central idea is:

```text
segment A:
  run layers 0..K with Stage-I candidate visual tokens
  keep normal vLLM metadata
  collect hidden_states after layer K

score:
  compute decoder-hidden-state text-image relevance
  produce a flattened keep_mask over the current prefill sequence

metadata rebuild:
  shrink the current prefill sequence according to keep_mask
  rebuild positions / M-RoPE / slot_mapping / attention metadata

segment B:
  resume layers K+1..end with the shortened hidden_states and rebuilt metadata
```

This avoids the bad implementation where `Qwen3_5Model.forward()` simply slices
`hidden_states` while attention metadata still describes the old sequence.

## Concrete First Prototype

The first real implementation should introduce three small pieces.

### 1. Runtime config

Extend `OccamTokenConfig` with:

```python
stage2_layer: int = 4
```

Environment variable:

```bash
VLLM_ASCEND_OCCAMTOKEN_STAGE2_LAYER=4
```

Validation:

```text
stage2_layer must be >= start_layer and < end_layer - 1
```

For the first prototype, true Stage-II is active only when:

```text
ENABLE=1
IMPL=true
STAGE=full
```

### 2. Scoring helper

Add a helper in `src/ascend_occamtoken/pruning.py`:

```python
def stage2_true_keep_mask(
    hidden_states: torch.Tensor,
    *,
    image_mask: torch.Tensor,
    text_mask: torch.Tensor,
    config: OccamTokenConfig,
) -> tuple[torch.Tensor, PruneStats]:
    ...
```

Behavior:

```text
1. image_h = hidden_states[image_mask]
2. text_h = hidden_states[text_mask]
3. text_h = select_text_window(text_h, ...)
4. scores = normalize(image_h) @ normalize(text_h).T
5. keep image top-k according to final visual budget
6. return a flattened keep_mask that keeps:
   - all non-image tokens
   - selected image tokens
```

Budget semantics:

```text
stage1_budget(original_image_tokens): Stage-I candidates entering LLM
final_budget(original_image_tokens): final image tokens after Stage-II
```

The helper must not infer the original token count from the already-pruned
image count unless the caller passes both values. For the first prototype, pass:

```text
original_image_tokens
stage1_image_tokens
target_image_tokens
```

or derive `target_image_tokens` when Stage-I logs/metadata already knows the
original count.

### 3. Model runner / layer-loop split

Patch the execution path around:

```text
vllm_ascend/worker/model_runner_v1.py
Qwen3_5Model / Qwen3NextModel layer loop
```

The recommended prototype API is a patched model method, called by the runner:

```python
model.forward_until_layer(
    input_ids=input_ids,
    positions=positions,
    inputs_embeds=inputs_embeds,
    stop_layer=stage2_layer,
)

model.forward_from_layer(
    hidden_states=pruned_hidden_states,
    residual=pruned_residual,
    positions=pruned_positions,
    start_layer=stage2_layer + 1,
)
```

If adding two public methods is too invasive, use one patched forward with
keyword controls:

```python
model.forward(
    ...,
    occamtoken_stop_layer=stage2_layer,
    occamtoken_start_layer=None,
)
```

and then call it again for the suffix with `occamtoken_start_layer`.

The runner, not the model, owns the pruning transition:

```text
1. call prefix forward
2. compute keep_mask
3. rebuild metadata
4. call suffix forward
```

## Metadata Rebuild Details

For the single-request first prototype, use a flattened keep mask:

```text
keep_mask.shape == (num_scheduled_tokens,)
```

Keep all text tokens:

```text
keep_mask[~is_mm_embed] = True
```

Keep only selected image tokens:

```text
keep_mask[is_mm_embed] = selected_image_mask
```

Then rebuild:

```text
hidden_states = hidden_states[keep_mask]
residual = residual[keep_mask] if residual is not None
input_ids = input_ids[keep_mask]
positions = positions[:, keep_mask] for M-RoPE
is_mm_embed = is_mm_embed[keep_mask]
slot_mapping = slot_mapping[keep_mask]
query_lens[0] = keep_mask.sum()
num_scheduled_tokens[req_id] = keep_mask.sum()
```

After that, rebuild attention metadata by calling the same runner metadata path
used before model execution, but with the shortened counts:

```text
_build_attention_metadata(...)
```

Do not reuse the old metadata object.

## Fail-Fast Conditions

The first implementation should raise `RuntimeError` instead of silently
falling back when any of these are true:

```text
num_reqs != 1
pcp_size > 1
shift_computed_tokens != 0
chunked prefill is active
speculative decode is active
prefix cache hit covers any image token
positions is not M-RoPE shape (3, N)
is_mm_embed is missing or has a different length from hidden_states
```

This keeps the first implementation honest. Once the single-request path works,
relax these constraints one by one.

## Why This Needs Runner Work

The decoder layer itself receives only:

```text
hidden_states
residual
positions
```

Attention kernels also depend on metadata built outside the model:

```text
query_start_loc
seq_lens
slot_mapping
block tables
common attention metadata
per-layer attention metadata
```

Therefore model-only pruning is unsafe. The model can compute Stage-II scores,
but the runner must own the actual sequence shortening.

## Implementation Checklist

1. Add config:

```text
VLLM_ASCEND_OCCAMTOKEN_STAGE2_LAYER
```

2. Add scoring helper:

```text
stage2_true_keep_mask(...)
```

3. Patch Qwen3.5/Qwen3Next layer loop to support prefix/suffix execution.

4. Patch model runner to:

```text
detect full,true Stage-II
validate fail-fast constraints
run prefix layers
score image tokens
build keep_mask
shorten hidden/residual/input_ids/positions/is_mm_embed/slot_mapping
rebuild attention metadata
run suffix layers
```

5. Add logs:

```text
[occamtoken] stage=stage2_true layer=4 original=512 kept=256 retention=0.5000
```

6. Add smoke tests:

```text
single request
1 image
no PCP
no chunked prefill
stage1_ratio=0.25
target_ratio=0.125
```

7. Only after that, extend to:

```text
multi-image single request
chunked prefill
PCP
batched requests
global cross-image budget
```

## Proposed Patch Points

### 1. Worker-level metadata owner

The pruning decision must be applied where attention metadata can be rebuilt.
For vLLM Ascend this is closer to:

```text
vllm_ascend/worker/model_runner_v1.py
```

not only:

```text
vllm/model_executor/models/qwen3_5.py
```

The model layer can compute scores, but the model runner owns schedule and KV
metadata.

### 2. Decoder-layer hook

Patch the Qwen3.5 layer loop around:

```text
Qwen3_5Model.forward()
```

which inherits the loop from:

```text
vllm/model_executor/models/qwen3_next.py::Qwen3NextModel.forward
```

Hook after configured layer `K`:

```text
VLLM_ASCEND_OCCAMTOKEN_STAGE2_LAYER=4
```

At that point, compute selected image-token indices from decoder hidden states.

### 3. Metadata rebuild

After selecting tokens, rebuild the current prefill batch metadata so that the
next layer sees a shorter sequence. This must update at least:

```text
positions
mrope_positions
slot_mapping
attention metadata / common attention metadata
num scheduled tokens for the current request
```

For the first constrained version, only support a single request so the mapping
is a simple boolean keep mask over the flattened prefill sequence.

## Configuration

Add only after the metadata path is implemented:

```bash
export VLLM_ASCEND_OCCAMTOKEN_STAGE=full
export VLLM_ASCEND_OCCAMTOKEN_IMPL=true
export VLLM_ASCEND_OCCAMTOKEN_STAGE1_RATIO=0.25
export VLLM_ASCEND_OCCAMTOKEN_TARGET_RATIO=0.125
export VLLM_ASCEND_OCCAMTOKEN_STAGE2_LAYER=4
export VLLM_ASCEND_OCCAMTOKEN_STRICT=1
```

Interpretation:

```text
STAGE1_RATIO: visual candidates entering LLM
TARGET_RATIO: final visual tokens after decoder-layer Stage-II
```

Example:

```text
original image tokens = 2048
Stage-I candidates = 512
Stage-II final = 256
```

## Expected Logs

Stage-I:

```text
[occamtoken] stage=stage1_true original=2048 kept=512 ...
```

Stage-II:

```text
[occamtoken] stage=stage2_true_layer4 original=512 kept=256 ...
```

## Non-goals For The First Prototype

Do not initially implement:

```text
multi-request batches
PCP
chunked prefill
global cross-request budget
decode-time pruning
```

Once the single-request path is correct, extend in this order:

```text
1. multi-image single request
2. chunked prefill
3. PCP
4. batched requests
5. global cross-image budget allocation
```
