# Handoff: Full True Removal Increment

This file records only the delta from the previous commit/stage. Full project
context stays in `README.md` and `docs/`.

## Changed In This Increment

`stage=full, impl=true` now performs physical visual token removal to the final
target budget.

Before this change:

```text
full,true:
  Stage-I true removal to STAGE1_RATIO
  Stage-II-lite masked pruning to TARGET_RATIO
  scheduled/KV length stayed at Stage-I length
```

After this change:

```text
full,true:
  prompt image placeholders use TARGET_RATIO / TARGET_TOKENS
  image embeddings are pruned to the same final budget
  scheduled/KV length follows the final budget
```

Example:

```text
original image tokens = 2048
STAGE1_RATIO = 0.25
TARGET_RATIO = 0.125

stage1,true -> 512 physical image tokens
full,true   -> 256 physical image tokens
```

## Code Changes

`src/ascend_occamtoken/config.py`

- Added `OccamTokenConfig.true_image_budget()`.
- `stage1,true` still uses `stage1_budget()`.
- `full,true` uses `final_budget()`.

`src/ascend_occamtoken/pruning.py`

- Added `prune_true_image_tokens()`.
- It uses the true physical budget and logs `stage=full_true` for full mode.

`patches/worker/patch_occamtoken_qwen35.py`

- Prompt replacement now uses `config.true_image_budget(num_tokens)`.
- `_patched_process_image_input()` now uses `prune_true_image_tokens()`.
- Stage-II masked pruning is disabled again for true sparse mode, because
  `full,true` is now physically shortened to the final budget.

`src/ascend_occamtoken/phase2.py`

- The phase2 direct-encoder helper now uses `prune_true_image_tokens()` so its
  output row count matches the final placeholder count in `full,true`.

## Important Limitation

This is true removal to the final budget, but it is not late query-aware true
removal. The query-aware Stage-II scorer needs text embeddings, which are
available only after vLLM has already scheduled the prompt and allocated token
metadata. Removing tokens at that point requires scheduler/request-state changes:

```text
input_ids
positions / M-RoPE positions
is_multimodal mask
slot mapping / KV metadata
multimodal embedding rows
```

The current increment gives the requested physical token reduction for
`full,true` without attempting unsafe late sequence mutation.

## Test Expectation

For `stage=full, impl=true`, expect logs like:

```text
[occamtoken] stage=full_true original=... kept=...
```

Do not expect `stage=stage2_masked` in true mode after this change.
