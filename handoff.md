# Handoff: Phase2 Logging Increment

This file records only the delta from the previous commit/stage. Full project
context stays in `README.md` and `docs/`.

## Changed In This Increment

`src/ascend_occamtoken/phase2.py` now logs per-image Stage-I true pruning stats
from the direct encoder helper:

```python
pruned, item_stats = prune_stage1_true(image_embeds, config)
stats.append(item_stats)
log_stats(stats)
```

With `VLLM_ASCEND_OCCAMTOKEN_LOG_STATS=1`, each phase2 direct-encoder cache miss
should emit one line per locally encoded image:

```text
[occamtoken] stage=stage1 original=... kept=... retention=...
```

`tests/test_pruning.py` adds coverage that two local image outputs produce two
`[occamtoken]` log lines.

## Why This Was Added

In the stock worker path, `_patched_process_image_input()` already logged stats.
The internal phase2 direct-encoder path used `prune_phase2_local_image_outputs()`
but previously discarded `_stats`, so successful phase2 pruning could be silent.

This made multi-image tests ambiguous: seeing only one `[occamtoken]` line did
not prove that only one image was pruned.

## Important Debug Note

If `[occamtoken]` appears only once and later requests are silent, first check
whether later requests hit the multimodal encoder cache. In
`vllm_ascend/worker/pcp_utils.py`, scheduled encoder inputs are skipped when
`mm_feature.identifier` is already in `encoder_cache`. Cache hits reuse the
stored encoder output and do not call `prune_phase2_local_image_outputs()` again.

Temporary phase2 debug print:

```python
print(
    f"[occamtoken-debug] phase2 local_outputs={len(local_outputs)} "
    f"my_image_indices={my_image_indices} "
    f"output_sizes={[output_sizes[i] for i in my_image_indices]}",
    flush=True,
)
```

Temporary PCP scheduling debug print:

```python
print(
    f"[occamtoken-debug] mm_hash={mm_hash} "
    f"cache_hit={mm_hash in encoder_cache}",
    flush=True,
)
```

When changing pruning ratio or token budget, restart the worker or clear the
multimodal encoder cache. The cache key may be image-derived, so cached encoder
outputs can hide a new OccamToken config.
