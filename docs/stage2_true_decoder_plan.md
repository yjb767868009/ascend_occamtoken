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
