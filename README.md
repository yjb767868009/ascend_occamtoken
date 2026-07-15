# Ascend OccamToken

Experimental OccamToken-style visual token pruning for Qwen3.5 on vLLM Ascend.

This repository is intentionally separate from:

- `<VLLM_CHECKOUT>`
- `<VLLM_ASCEND_CHECKOUT>`

The implementation target is patch-based integration with vLLM Ascend. We should not directly modify vLLM source code.

## Goals

- Reproduce OccamToken-style visual token pruning on Qwen3.5.
- Compare fixed pruning, Stage-I only, Stage-II only, and full two-stage pruning.
- Measure quality, TTFT, prefill latency, KV/memory impact, and end-to-end latency on RAG workloads.

## Layout

- `docs/experiment_plan.md`: experiment and implementation plan.
- `src/ascend_occamtoken`: reusable pruning utilities; the install script copies these into `vllm_ascend/occamtoken`.
- `patches`: patch modules intended to be copied or symlinked into `vllm_ascend`.
- `benchmarks`: benchmark and evaluation scripts.

## Install Into vLLM Ascend

```bash
bash scripts/install_into_vllm_ascend.sh <VLLM_ASCEND_CHECKOUT>
```

Enable the patch at runtime:

```bash
export VLLM_ASCEND_OCCAMTOKEN_ENABLE=1
export VLLM_ASCEND_OCCAMTOKEN_IMPL=true  # masked | true
export VLLM_ASCEND_OCCAMTOKEN_STAGE=stage1  # fixed | stage1 | stage2 | full
export VLLM_ASCEND_OCCAMTOKEN_TARGET_RATIO=0.125
export VLLM_ASCEND_OCCAMTOKEN_STAGE1_RATIO=0.25
export VLLM_ASCEND_OCCAMTOKEN_LOG_STATS=1
```

Current implementation status:

- Implemented: masked pruning for `fixed`, `stage1`, `stage2`, and `full`.
- Implemented: true image-token removal for `fixed`, `stage1`, and the Stage-I part of `full`.
- Stage-II in `VLLM_ASCEND_OCCAMTOKEN_IMPL=true` is intentionally a no-op for now.
- Not yet implemented: true Stage-II query-aware token removal.

Masked pruning keeps the visual sequence length unchanged and replaces pruned embeddings with a mean or zero vector. It is intended for quality ablation only; it should not be expected to improve TTFT or KV memory yet.

True pruning reduces image placeholder tokens in the multimodal processor and
returns the same number of pruned image embeddings from the vision path. This is
the mode to use for prefill/KV/TTFT experiments:

```bash
export VLLM_ASCEND_OCCAMTOKEN_IMPL=true
MODEL_PATH=<QWEN3_5_MODEL_PATH> bash benchmarks/run_occamtoken_matrix.sh stage1-256
```

In `full` with `true`, Stage-I performs true token removal and Stage-II does
nothing. This keeps the first performance milestone clean: measure real Stage-I
sparsity before adding any query-aware late-stage pruning.

Example matrix entry:

```bash
MODEL_PATH=<QWEN3_5_MODEL_PATH> bash benchmarks/run_occamtoken_matrix.sh full-256
```
