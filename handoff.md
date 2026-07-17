# Handoff: Stage-II-Lite Increment

This file records only the delta from the previous commit/stage. Full project
context stays in `README.md` and `docs/`.

## Changed In This Increment

`patches/worker/patch_occamtoken_qwen35.py` now allows Stage-II-lite to run even
when `VLLM_ASCEND_OCCAMTOKEN_IMPL=true`:

```python
if config.stage2_active():
    ...
    pruned, item_stats = prune_stage2_masked(...)
```

Effect:

```text
stage=full, impl=true:
  Stage-I: true image-token removal before scheduling/merge.
  Stage-II-lite: query-aware masked pruning on the remaining visual embeddings.
```

`README.md` was updated to state this explicitly.

`src/ascend_occamtoken/config.py` adds `OccamTokenConfig.stage2_budget()`.
For `stage=full, impl=true` with ratio-based budgets, Stage-II-lite maps the
final target ratio back onto the already-shortened Stage-I sequence:

```text
stage1_ratio=0.25, target_ratio=0.125
original 2048 -> Stage-I 512 -> Stage-II-lite budget 256
```

This avoids accidentally applying `target_ratio` twice:

```text
wrong: 2048 -> 512 -> 64
```

## Why This Was Added

The previous true/full path was Stage-I true removal only; Stage-II was skipped
in true mode. This made it impossible to test whether query-aware Stage-II helps
accuracy after Stage-I on the user's Qwen3.5 RAG workload.

This increment gives a low-risk Stage-II ablation:

```text
stage1 true only:
  performance gain from shorter visual sequence

full true:
  same Stage-I sequence length, plus Stage-II-lite quality reranking/masking
```

So compare `stage=stage1, impl=true` against `stage=full, impl=true` at the same
`STAGE1_RATIO`.

## Important Limitation

This is not true Stage-II token removal. Stage-II-lite runs inside
`embed_input_ids()`, after vLLM has already scheduled the prompt and built the
multimodal mask. At that point the code can safely replace/mask embeddings, but
cannot shorten `input_ids`, `is_multimodal`, M-RoPE positions, KV allocation, or
scheduler metadata.

Expected logs for `stage=full, impl=true`:

```text
[occamtoken] stage=stage1_true ...
[occamtoken] stage=stage2_masked ...
```

For actual true Stage-II token removal, the next implementation must move
query-aware pruning into a scheduler/PCP-safe point where the selected visual
positions, M-RoPE positions, and multimodal embedding rows are updated together.
