# Qwen3.5 OccamToken Experiment Plan

## Goal

Reproduce and adapt the OccamToken idea for the current Qwen3.5 multimodal RAG workload:

- Model: Qwen3.5 multimodal model served through vLLM/vLLM Ascend.
- Input shape: about 10k text tokens plus about 2k visual tokens.
- Target: reduce prefill latency, KV cache memory, peak memory, and end-to-end latency while preserving answer quality.
- Implementation constraint: do not directly modify `<VLLM_CHECKOUT>`. All code changes should be applied through the vLLM Ascend patch mechanism, especially the `<VLLM_ASCEND_CHECKOUT>/vllm_ascend/patch/platform` entry path.

Local source versions:

- `<VLLM_CHECKOUT>`: `v0.19.1`
- `<VLLM_ASCEND_CHECKOUT>`: `v0.19.1rc1`

## Paper Baseline

Paper: OccamToken: Efficient VLM Inference with Training-Free and Budget-Adaptive Token Pruning.

OccamToken has two stages:

- Stage I: image-adaptive redundancy pruning at the vision encoder output.
- Stage II: register-anchored query relevance pruning inside the language model.

Important paper numbers to reproduce directionally:

| Setting | Model | Retention | Accuracy |
|---|---|---:|---:|
| Full OccamToken | Qwen3-VL 8B | 22.2% | 95.6 RelAcc |
| Full OccamToken | Qwen3-VL 8B | 11.1% | 92.0 RelAcc |
| w/o Stage-I | Qwen3-VL 8B | 22.2% | 95.2 RelAcc |
| w/o Stage-I | Qwen3-VL 8B | 11.1% | 91.8 RelAcc |
| Fixed budget | Qwen3-VL 8B | 22.2% | 93.1 RelAcc |
| Fixed budget | Qwen3-VL 8B | 11.1% | 88.7 RelAcc |

The paper does not provide a clean Stage-I-only table. We should add that experiment because it is important for engineering decisions.

## Repository Constraints

Do not edit files under:

- `<VLLM_CHECKOUT>`

Use patch-style integration under:

- `<VLLM_ASCEND_CHECKOUT>/vllm_ascend/patch/platform`
- `<VLLM_ASCEND_CHECKOUT>/vllm_ascend/patch/worker`

Current relevant files:

- `vllm_ascend/patch/platform/__init__.py`
- `vllm_ascend/patch/worker/__init__.py`
- `vllm_ascend/patch/worker/patch_qwen3_5.py`
- `vllm_ascend/patch/worker/patch_qwen3vl.py`
- `vllm_ascend/patch/worker/patch_multimodal_merge.py`

Qwen3.5 in vLLM already reuses Qwen3-VL multimodal infrastructure:

- `Qwen3_5ForConditionalGeneration` inherits from `Qwen3VLForConditionalGeneration`.
- Image embeddings are produced by `self.visual(pixel_values, grid_thw=grid_thw)`.
- Visual embeddings are split per image in `_process_image_input`.
- Text and visual embeddings are merged in `embed_input_ids` via `_merge_multimodal_embeddings`.

This makes the Qwen3.5 implementation path close to Qwen3-VL, but the current code explicitly sets:

- `supports_multimodal_pruning = False`
- `self.is_multimodal_pruning_enabled = False`

OccamToken should not depend on vLLM's built-in EVS pruning path at first. Treat it as a separate experimental patch.

## Proposed Patch Design

### Entry Point

Add a platform-level patch entry guarded by environment variables:

- New file in vLLM Ascend: `vllm_ascend/patch/platform/patch_occamtoken.py`
- Source maintained here: `patches/platform/patch_occamtoken.py`
- Import it from `vllm_ascend/patch/platform/__init__.py` only when enabled.

Suggested environment variables:

```bash
VLLM_ASCEND_OCCAMTOKEN_ENABLE=1
VLLM_ASCEND_OCCAMTOKEN_IMPL=true         # masked | true
VLLM_ASCEND_OCCAMTOKEN_STAGE=full        # off | stage1 | stage2 | full | fixed
VLLM_ASCEND_OCCAMTOKEN_TARGET_RATIO=0.125
VLLM_ASCEND_OCCAMTOKEN_STAGE1_RATIO=0.25
VLLM_ASCEND_OCCAMTOKEN_STAGE2_LAYER=11
VLLM_ASCEND_OCCAMTOKEN_LOG_STATS=1
```

The platform patch should import a worker/model patch module only when the feature is enabled.

### Worker Patch

Add a worker patch module:

- Source maintained here: `patches/worker/patch_occamtoken_qwen35.py`
- Target in vLLM Ascend: `vllm_ascend/patch/worker/patch_occamtoken_qwen35.py`
- Utility package target in vLLM Ascend: `vllm_ascend/occamtoken`

Patch targets:

- `Qwen3_5ForConditionalGeneration._process_image_input`
- `Qwen3_5ForConditionalGeneration.embed_input_ids`
- `Qwen3VLMultiModalProcessor._get_prompt_updates`
- optional later target: selected Qwen3.5 decoder layer forward for Stage-II pruning.

Implementation should be incremental:

1. Phase A: fixed-budget pruning after image embeddings are created.
2. Phase B: Stage-I-only dynamic image pruning.
3. Phase C: Stage-II-only query-aware masked pruning for quality ablation only.
4. Phase D: keep `impl=true` as Stage-I true removal only until Stage-I performance is validated.

## Important Implementation Detail

The hardest engineering problem is placeholder alignment.

vLLM's multimodal merge expects:

```text
number of visual embeddings == number of visual placeholder tokens
```

If we physically remove visual embeddings after the processor has already inserted placeholder tokens, `_merge_multimodal_embeddings` will fail unless we also update the placeholder mask/token sequence.

Therefore the first implementation should use one of two safe approaches:

### Option 1: Masked Embedding Replacement

Keep the same visual token count, but replace pruned visual embeddings with a cheap neutral vector:

- zero vector,
- register/reference vector,
- mean retained visual embedding,
- or repeated nearest retained embedding.

Pros:

- Easy to integrate.
- No tokenizer/placeholder surgery.
- Good for validating accuracy effects.

Cons:

- Does not reduce LLM sequence length.
- Little or no TTFT/KV benefit.
- Only useful as a correctness baseline.

### Option 2: True Token Removal

Remove visual embeddings and also shrink multimodal placeholder positions before merge.

Pros:

- Actually reduces LLM prefill cost and KV cache.
- Required for the performance goal.

Cons:

- More invasive.
- Must update `is_multimodal`, replacement token positions, M-RoPE positions, and possibly request metadata.
- Needs careful testing with multi-image RAG inputs.

Current status:

- Option 1 is implemented for all stages.
- Option 2 is implemented for image-token fixed/Stage-I pruning.
- In `full` with `VLLM_ASCEND_OCCAMTOKEN_IMPL=true`, Stage-I is true token removal and Stage-II is intentionally a no-op.
- True Stage-II token removal is not implemented because it happens after text embeddings are available and would require scheduler/attention metadata-safe changes.

## Experimental Matrix

### Core Ablation

Run these configurations on the same Qwen3.5 model and same dataset:

| ID | Method | Query-aware | Early prune | Target visual tokens |
|---|---|---|---|---:|
| A0 | Full visual tokens | no | no | 2000 |
| A1 | Fixed top-k | no | yes | 512 |
| A2 | Fixed top-k | no | yes | 256 |
| A3 | Fixed top-k | no | yes | 128 |
| B1 | Stage-I only | no | yes | dynamic, avg 512 |
| B2 | Stage-I only | no | yes | dynamic, avg 256 |
| B3 | Stage-I only | no | yes | dynamic, avg 128 |
| C1 | Stage-II only | yes | no | dynamic, avg 512 |
| C2 | Stage-II only | yes | no | dynamic, avg 256 |
| C3 | Stage-II only | yes | no | dynamic, avg 128 |
| D1 | Stage-I + Stage-II | yes | yes | dynamic, avg 512 |
| D2 | Stage-I + Stage-II | yes | yes | dynamic, avg 256 |
| D3 | Stage-I + Stage-II | yes | yes | dynamic, avg 128 |

This explicitly answers the missing paper question:

- Is Stage-I-only strong?
- Is Stage-II the real accuracy-preserving component?
- Does full two-stage pruning justify the additional complexity?

## Metrics

### Quality Metrics

Use task-specific scoring:

- exact match / F1 for extractive answers,
- LLM-as-judge only as secondary,
- citation correctness for RAG answers,
- image-grounded evidence correctness,
- refusal / hallucination rate.

Report quality as:

```text
relative_accuracy = pruned_accuracy / full_token_accuracy
```

Also report absolute accuracy deltas.

### Performance Metrics

Record these for every request:

- visual token count before pruning,
- visual token count after Stage-I,
- final visual token count after Stage-II,
- total input tokens,
- TTFT,
- vision encoder time,
- LLM prefill time,
- decode time,
- output tokens/s,
- end-to-end latency,
- peak NPU memory,
- KV cache memory estimate,
- batch size / concurrency throughput.

Primary success metrics:

- TTFT reduction,
- prefill latency reduction,
- peak memory reduction,
- no more than 2% absolute quality drop in normal-image RAG,
- no more than 5% relative quality drop in high-risk OCR/chart groups unless gated by fallback.

## Execution Steps

1. Run baseline with unmodified vLLM Ascend.
2. Implement masked pruning smoke test.
3. Implement fixed-budget true token removal.
4. Add dynamic Stage-I.
5. Add Stage-II-lite query-aware scoring.
6. Run full ablation matrix.
7. Decide whether Stage-II-paper-like decoder-attention pruning is worth implementing.

## Decision Criteria

Adopt the patch if:

- TTFT improves by at least 25% at the target workload,
- peak memory or KV memory estimate improves by at least 25%,
- normal-image RAG quality drop is within 2% absolute,
- OCR/chart groups have a conservative fallback with acceptable quality,
- implementation remains fully isolated in vLLM Ascend patch modules.

Do not adopt full two-stage pruning if:

- Stage-I-only gives nearly the same quality with much lower complexity,
- Stage-II-lite gives no clear quality advantage,
- true token removal requires broad vLLM scheduler changes.
