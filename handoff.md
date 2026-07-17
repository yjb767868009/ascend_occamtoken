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

## Next Implementation Target

First prototype constraints:

```text
prefill only
single request
no PCP
no chunked prefill
no prefix cache reuse
M-RoPE rebuilt after pruning
```

Patch point should be around:

```text
vllm_ascend/worker/model_runner_v1.py
Qwen3_5Model / Qwen3NextModel layer loop
```

The model can compute Stage-II scores from hidden states, but model runner logic
must rebuild sequence metadata before the remaining layers execute.
