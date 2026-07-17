# Handoff: True Stage-II Direction Reset

This file records only the delta from the previous commit/stage. Full project
context stays in `README.md` and `docs/`.

## Changed In This Increment

The two previous Stage-II attempts are being reverted:

```text
345c4dc Enable Stage-II-lite after true Stage-I
f0b9fe5 Use final budget for full true removal
```

Reason:

```text
Stage-II-lite used text embeddings but did not physically remove tokens.
Final-budget full true removal physically removed tokens but still used Stage-I
visual scores, not decoder-layer text-image interaction.
```

The code returns to the earlier stable behavior:

```text
stage1,true: true Stage-I image-token removal
full,true: Stage-I true removal only; true Stage-II is still not implemented
```

## New Design Note

Added:

```text
docs/stage2_true_decoder_plan.md
```

This document defines the correct target for true Stage-II:

```text
run Stage-I candidates through LLM layers 0..K
compute text-image relevance from decoder hidden states
physically drop low-score image tokens
rebuild positions / M-RoPE / attention metadata / slot mapping
run layers K+1..end on the shorter sequence
```

## Important Correction

Do not implement true Stage-II in `phase2.py` after the vision encoder. That
location can see image embeddings, but it has not seen LLM hidden states or
decoder text-image interaction. At best it is a text-embedding approximation,
not paper-like Stage-II.

Do not implement true Stage-II only inside `Qwen3_5Model.forward()` by slicing
`hidden_states`. The next attention layer still uses vLLM metadata built for the
old sequence length. The pruning operation must be coordinated with the worker
or attention metadata owner.

## Implementation Plan Status

The implementation plan has been expanded in:

```text
docs/stage2_true_decoder_plan.md
```

It now specifies a runner-controlled two-segment prefill:

```text
1. run layers 0..K with Stage-I visual candidates
2. score image tokens from decoder hidden states
3. build a flattened keep_mask
4. rebuild positions / M-RoPE / slot_mapping / attention metadata
5. run layers K+1..end on the shortened sequence
```

The implementation owner should not hand this design work to the testing agent.
The testing agent should only validate the final patch once the runner and layer
loop changes are implemented.

## First Prototype Constraints

```text
prefill only
single request
no PCP
no chunked prefill
no prefix cache reuse
M-RoPE rebuilt after pruning
```

Patch point must be around:

```text
vllm_ascend/worker/model_runner_v1.py
Qwen3_5Model / Qwen3NextModel layer loop
```

The model can compute Stage-II scores from hidden states, but model runner logic
must rebuild sequence metadata before the remaining layers execute.

Do not mark Stage-II complete until logs show:

```text
[occamtoken] stage=stage2_true layer=... original=... kept=...
```

and profiler/token counts confirm that layers after the Stage-II layer run on
the shortened sequence.

## Current Code Prototype

Added an initial decoder-layer prototype:

```text
patches/worker/patch_occamtoken_stage2_decoder.py
```

It installs:

```text
Qwen3_5Model.forward_until_layer(...)
Qwen3_5Model.forward_from_layer(...)
Qwen3_5ForConditionalGeneration.forward(...)
```

For `stage=full, impl=true`, it runs the text model through
`VLLM_ASCEND_OCCAMTOKEN_STAGE2_LAYER`, computes `stage2_true_keep_mask()` from
decoder hidden states, slices hidden/residual/positions, and runs the remaining
layers on the shortened sequence.

Important: this is the first runnable prototype. If the next attention layer
fails with metadata or slot-mapping shape errors, the same scoring and layer
split must be moved one level up into `model_runner_v1.py` so attention metadata
is rebuilt between the prefix and suffix forwards.
