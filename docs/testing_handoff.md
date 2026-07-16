# Testing Handoff for Ascend OccamToken

This document is for the next testing agent that will run the Qwen3.5/VLLM Ascend experiments on the company server.

## Current Repository State

- Repository: `ascend_occamtoken`
- Target upstreams used during local development:
  - vLLM: `v0.19.1`
  - vLLM Ascend: `v0.19.1rc1`
- Integration style:
  - Do not edit vLLM source directly.
  - Install this experiment into a vLLM Ascend checkout through the patch mechanism.
  - The installer copies files into `vllm_ascend/occamtoken`, `vllm_ascend/patch/platform`, and `vllm_ascend/patch/worker`.

Important: the current implementation has two modes:

- `VLLM_ASCEND_OCCAMTOKEN_IMPL=masked`
  - Keeps the visual token sequence length unchanged.
  - Replaces pruned visual token embeddings with either retained-token mean or zero.
  - Useful for quality and ablation validation.
- `VLLM_ASCEND_OCCAMTOKEN_IMPL=true`
  - Reduces image placeholder tokens in the multimodal processor.
  - Returns the same reduced number of image embeddings from the Qwen3.5 vision path.
  - This is the mode to use for TTFT/prefill/KV experiments.

Current true sparse support is Stage-I/fixed image-token removal only. Stage-II is intentionally a no-op when `VLLM_ASCEND_OCCAMTOKEN_IMPL=true`.

## What Has Been Implemented

Main files:

- `src/ascend_occamtoken/config.py`
  - Environment-backed configuration.
- `src/ascend_occamtoken/pruning.py`
  - Token scoring, masked pruning, and true Stage-I pruning helpers.
  - Supports masked fixed, Stage-I, Stage-II-lite, and full modes.
  - Supports true fixed/Stage-I image token removal.
  - Does not run Stage-II in true mode.
- `src/ascend_occamtoken/logging.py`
  - Lightweight stderr stats logging.
- `patches/platform/patch_occamtoken.py`
  - Patch entry point loaded through vLLM Ascend platform patch initialization.
- `patches/worker/patch_occamtoken_qwen35.py`
  - Monkey patch for Qwen3.5 multimodal embeddings path.
  - Patches `Qwen3VLMultiModalProcessor._get_prompt_updates` so true mode shrinks image placeholders before scheduling.
- `scripts/install_into_vllm_ascend.sh`
  - Direct copy installer.
- `benchmarks/run_occamtoken_matrix.sh`
  - Minimal serve-command matrix for smoke experiments.
- `docs/experiment_plan.md`
  - Experiment design and ablation plan.
- `tests/test_pruning.py`
  - Unit tests for pruning helper behavior.

## Required Server Setup

The company server should already have compatible Ascend runtime, vLLM, vLLM Ascend, PyTorch, and model weights available.

Expected inputs:

- A vLLM checkout compatible with `v0.19.1`.
- A vLLM Ascend checkout compatible with `v0.19.1rc1` or the target deployment branch.
- A Qwen3.5 multimodal model path.
- Test prompts that include approximately:
  - 10k text tokens
  - 2k image tokens
  - RAG-style long context

Do not hardcode company paths or model paths in this repo. Pass paths through environment variables or command arguments.

## Install Into vLLM Ascend

From this repo:

```bash
bash scripts/install_into_vllm_ascend.sh <VLLM_ASCEND_CHECKOUT>
```

If no argument is provided, the script uses:

```bash
${VLLM_ASCEND_CHECKOUT:-${HOME}/vllm_ascend}
```

After install, confirm that this block exists in:

```text
<VLLM_ASCEND_CHECKOUT>/vllm_ascend/patch/platform/__init__.py
```

Expected block:

```python
# OccamToken experiment patch. Installed from an external experiment checkout.
if os.getenv("VLLM_ASCEND_OCCAMTOKEN_ENABLE", "0").lower() in ("1", "true", "yes", "on"):
    import vllm_ascend.patch.platform.patch_occamtoken  # noqa
```

Also confirm these copied files exist:

```text
<VLLM_ASCEND_CHECKOUT>/vllm_ascend/occamtoken/config.py
<VLLM_ASCEND_CHECKOUT>/vllm_ascend/occamtoken/pruning.py
<VLLM_ASCEND_CHECKOUT>/vllm_ascend/occamtoken/logging.py
<VLLM_ASCEND_CHECKOUT>/vllm_ascend/patch/platform/patch_occamtoken.py
<VLLM_ASCEND_CHECKOUT>/vllm_ascend/patch/worker/patch_occamtoken_qwen35.py
```

## Environment Variables

Main switch:

```bash
export VLLM_ASCEND_OCCAMTOKEN_ENABLE=1
```

Implementation mode:

```bash
export VLLM_ASCEND_OCCAMTOKEN_IMPL=true    # true image-token removal
export VLLM_ASCEND_OCCAMTOKEN_IMPL=masked  # shape-preserving quality ablation
```

Formal true-sparse tests should use:

```bash
export VLLM_ASCEND_OCCAMTOKEN_STRICT=1
```

With strict mode enabled, unexpected processor/output structures fail fast. With
strict mode disabled, the patch prints one fallback diagnostic and uses vLLM's
original image replacement for that request.

If an optimized phase2/direct-encoder path is used, install the M-RoPE patch at
plugin import time:

```python
from vllm_ascend.occamtoken.mrope import install_mrope_patch

install_mrope_patch()
```

If the worker still crashes inside stock `qwen3_vl.py::_get_mrope_input_positions`
with negative `text_len`, this patch was not loaded early enough.

Modes:

```bash
export VLLM_ASCEND_OCCAMTOKEN_STAGE=off
export VLLM_ASCEND_OCCAMTOKEN_STAGE=fixed
export VLLM_ASCEND_OCCAMTOKEN_STAGE=stage1
export VLLM_ASCEND_OCCAMTOKEN_STAGE=stage2
export VLLM_ASCEND_OCCAMTOKEN_STAGE=full
```

Budget controls:

```bash
export VLLM_ASCEND_OCCAMTOKEN_TARGET_RATIO=0.25
export VLLM_ASCEND_OCCAMTOKEN_TARGET_TOKENS=256
export VLLM_ASCEND_OCCAMTOKEN_STAGE1_RATIO=0.5
export VLLM_ASCEND_OCCAMTOKEN_STAGE1_TOKENS=512
export VLLM_ASCEND_OCCAMTOKEN_MIN_TOKENS=64
```

Scoring and replacement:

```bash
export VLLM_ASCEND_OCCAMTOKEN_REPLACEMENT=mean
export VLLM_ASCEND_OCCAMTOKEN_STAGE1_SCORER=norm
export VLLM_ASCEND_OCCAMTOKEN_STAGE2_SCORER=text_similarity
export VLLM_ASCEND_OCCAMTOKEN_MAX_TEXT_TOKENS=512
export VLLM_ASCEND_OCCAMTOKEN_QUESTION_TAIL_TOKENS=128
```

Stats:

```bash
export VLLM_ASCEND_OCCAMTOKEN_LOG_STATS=1
```

Expected stderr line format:

```text
[occamtoken] stage=<mode> original=<n> kept=<k> pruned=<p> retention=<r> replacement=<mean|zero>
```

## First Smoke Test

Run with the patch disabled first:

```bash
export VLLM_ASCEND_OCCAMTOKEN_ENABLE=0
MODEL_PATH=<QWEN3_5_MODEL_PATH> bash benchmarks/run_occamtoken_matrix.sh off
```

Then run enabled but no pruning:

```bash
export VLLM_ASCEND_OCCAMTOKEN_ENABLE=1
MODEL_PATH=<QWEN3_5_MODEL_PATH> bash benchmarks/run_occamtoken_matrix.sh off
```

Then run a conservative masked-pruning mode:

```bash
export VLLM_ASCEND_OCCAMTOKEN_IMPL=masked
MODEL_PATH=<QWEN3_5_MODEL_PATH> bash benchmarks/run_occamtoken_matrix.sh stage1-512
```

Then run a conservative true-pruning mode:

```bash
export VLLM_ASCEND_OCCAMTOKEN_IMPL=true
export VLLM_ASCEND_OCCAMTOKEN_STRICT=1
MODEL_PATH=<QWEN3_5_MODEL_PATH> bash benchmarks/run_occamtoken_matrix.sh stage1-256
```

If the model fails to start, inspect whether the patch was loaded and whether the target class name for Qwen3.5 has changed in the installed vLLM version.

## Suggested Experiment Matrix

Start with functional and quality smoke tests:

| Mode | Target | Purpose |
| --- | --- | --- |
| off | full tokens | baseline |
| fixed | 512 visual tokens | fixed-budget sanity |
| stage1 | 512 visual tokens | Stage-I-only ablation |
| stage2 | 512 visual tokens | masked Stage-II-lite-only ablation |
| full | Stage-I 1024, final 512 | masked combined ablation, or true Stage-I-only if `impl=true` |
| fixed | 256 visual tokens | aggressive fixed-budget sanity |
| stage1 | 256 visual tokens | aggressive Stage-I-only ablation |
| full | Stage-I 512, final 256 | aggressive combined ablation |

For the user's target case, also test around the expected 2k image-token input:

| Original Visual Tokens | Kept Tokens | Retention |
| --- | --- | --- |
| 2048 | 1024 | 50.0% |
| 2048 | 768 | 37.5% |
| 2048 | 512 | 25.0% |
| 2048 | 384 | 18.75% |
| 2048 | 256 | 12.5% |

Report all quality metrics together with actual logged retention. Do not rely only on configured target ratios.

For performance claims, compare only `impl=true` runs against the baseline. `impl=masked` runs should be reported as quality ablations only.

## Metrics to Collect

Performance:

- TTFT
- end-to-end latency
- tokens/sec
- peak HBM memory
- KV-cache usage
- prefill time
- decode time
- batch size and concurrency

Quality:

- Exact task score if there is an internal evaluation set.
- For RAG:
  - answer correctness
  - citation/grounding correctness
  - hallucination rate
  - failure examples under high pruning

Logging:

- Keep the `[occamtoken]` stderr lines.
- Keep vLLM request metrics.
- Record model path as a redacted label, not the actual internal path, if logs will be shared externally.

## Expected Results and Interpretation

For `impl=masked`:

- Accuracy changes are meaningful.
- Stage-I-only versus Stage-I plus Stage-II-lite comparisons are meaningful.
- TTFT and memory speedup are not expected to be large.
- Any large observed speedup should be treated suspiciously until true token count and KV-cache size are verified.

For `impl=true`:

- Fixed and Stage-I image tokens should be physically reduced before scheduling.
- TTFT, prefill time, and KV-cache pressure should improve if image tokens are a meaningful part of the prompt.
- Stage-II is intentionally disabled. `stage2` true is equivalent to no pruning, and `full` true is Stage-I true pruning only.

If Stage-I-only is already close to full Stage-I plus Stage-II-lite, that supports the user's concern that Stage-II may contribute less than claimed for this workload. If Stage-I-only is clearly worse, then the paper should have shown the missing Stage-I-only result because it would strengthen the Stage-II claim.

## Known Risks

- The patch targets the local Qwen3.5 class shape from vLLM `v0.19.1`. If the company server uses a different fork or branch, class names and method names may differ.
- This implementation assumes visual token embeddings can be identified from multimodal embedding flow. Verify with logging on the actual server.
- Replacement with mean embeddings in masked mode may distort attention in a way that differs from true pruning.
- Stage-II-lite currently exists only for masked ablation. It is intentionally disabled in true mode.
- True Stage-II token removal would require scheduler/metadata-safe changes beyond the current patch.

## Recommended Next Development Step

Recommended next development after this handoff:

1. Verify fixed/Stage-I true image-token removal on the company Ascend server.
2. Compare masked pruning versus true removal at identical kept-token budgets.
3. Verify placeholder alignment and M-RoPE behavior for multi-image prompts.
4. Measure whether Stage-I-only true pruning gives enough speedup and quality.
5. Only then consider true Stage-II token removal with scheduler/attention metadata changes.

Do not start with full dynamic Stage-I plus Stage-II true removal. Fixed top-k is the safest way to expose shape, position, and placeholder bugs first.

## Pre-Push Hygiene

Before pushing any follow-up changes:

```bash
find . -path "*/__pycache__" -o -name "*.pyc"
git status --short
```

Also run the team's standard credential scanner before pushing. Do not commit internal model paths, company server paths, credentials, or raw logs.
