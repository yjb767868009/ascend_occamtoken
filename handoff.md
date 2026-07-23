# Handoff: Runner-Controlled True Stage-II Increment

This file records only the delta from the previous commit. Full background stays
in `README.md` and `docs/`.

## New Increment: Qwen3-VL Dedicated Patch Path

Added a separate source-side patch directory for Qwen3-VL:

```text
patches/qwen3_vl/patch_occamtoken_qwen3vl.py
```

It reuses the common pruning / logging / M-RoPE helpers, but keeps the model
patch code separate from the Qwen3.5 patch.

The platform entry point now selects the patch by target model:

```text
VLLM_ASCEND_OCCAMTOKEN_TARGET_MODEL=qwen3_5   # default
VLLM_ASCEND_OCCAMTOKEN_TARGET_MODEL=qwen3_vl
```

For Qwen3-VL, the first supported path is image true Stage-I removal. The
Qwen3.5 Stage-II runner path remains isolated behind the Qwen3.5 target model
selection.

## Changed In This Increment

Added a runner-level true Stage-II path:

```text
patches/worker/patch_occamtoken_runner.py
```

It patches `NPUModelRunner` instead of doing a model-only slice:

```text
1. run Qwen3.5 layers 0..K with normal metadata
2. compute Stage-II keep_mask from decoder hidden states
3. compact hidden/residual/positions
4. update runner token layout and KV slot mapping
5. rebuild suffix attention metadata
6. run layers K+1..end under the compact metadata context
```

`patch_occamtoken_stage2_decoder.py` now only installs layer split helpers:

```text
Qwen3_5Model.forward_until_layer(...)
Qwen3_5Model.forward_from_layer(...)
```

The previous model-only `Qwen3_5ForConditionalGeneration.forward` Stage-II hook
was removed.

## Metadata Handling Added

The new helper `update_metadata_after_drop` updates compact state after token
drop:

```text
forward context:
  token_drop_applied
  new_cu_seqlens / new_cu_seqlens_cpu
  new_seq_lens / new_seq_lens_cpu
  new_total_tokens
  new_max_seq_len
  slot_mapping_out

FA-style metadata:
  slot_mapping
  seq_lens / seq_lens_cpu / seq_lens_list
  query_start_loc
  actual_seq_lengths_q
  max_query_len
  max_seq_len
  num_actual_tokens
  num_actual_tokens_pcp_padded

GDN:
  suffix metadata is rebuilt through vLLM Ascend builders after compacting
  runner query_start_loc, gdn_query_start_loc, seq_lens, and slot_mapping.
  The patch does not synthesize causal_conv1d fields manually.

qwen3_5_mtp / MTP:
  logits_indices are remapped from old token coordinates to compact coordinates.
  spec_decode_common_attn_metadata is replaced with compact metadata before the
  drafter path runs.
```

Mixed prefill+decode batches are handled by separating compact query lengths
from KV sequence lengths:

```text
prefill rows: seq_len = num_computed + compact_query_len
decode rows:  seq_len keeps the existing KV seq_len
```

## MTP Compact Plan Variables

The runner patch publishes these module-level variables after token drop:

```text
_kept_indices
_new_cu_seqlens
_new_seq_lens
_new_max_seq_len
```

Before the MTP proposer path runs, the patch uses them to overwrite
`common_attn_metadata`:

```text
query_start_loc
query_start_loc_cpu
seq_lens
seq_lens_cpu
_seq_lens_cpu
num_actual_tokens
max_query_len
max_seq_len
actual_seq_lengths_q
num_input_tokens
```

After `propose_draft_token_ids()` returns or raises, the four variables are set
back to `None` to avoid polluting the next step.

The user-facing config is:

```text
--speculative_config '{"method": "qwen3_5_mtp", "num_speculative_tokens": 1}'
```

In vLLM Ascend internals this is expected to reach the `method == "mtp"` branch
inside `AscendEagleProposer`. The class name includes "Eagle", but this test is
for Qwen3.5 MTP, not EAGLE.

The patch intentionally raises if compact speculative metadata exists but the
runner method is not `mtp` or `qwen3_5_mtp`.

## Deployment

`scripts/install_into_vllm_ascend.sh` now copies:

```text
patches/worker/patch_occamtoken_runner.py
```

The patch was installed into:

```text
/home/yujubo/vllm_ascend
```

and compiled successfully there.

## Test Methodology

Do not simplify the test matrix first. The next testing agent should run the
final intended scenario directly, then fix each failure in place until that same
scenario passes.

Final scenario:

Run with:

```text
VLLM_ASCEND_OCCAMTOKEN_ENABLE=1
VLLM_ASCEND_OCCAMTOKEN_IMPL=true
VLLM_ASCEND_OCCAMTOKEN_STAGE=full
VLLM_ASCEND_OCCAMTOKEN_STAGE2_LAYER=4
VLLM_ASCEND_OCCAMTOKEN_LOG_STATS=1
--speculative_config '{"method": "qwen3_5_mtp", "num_speculative_tokens": 1}'
```

Input:

```text
8k text + 20 images
Qwen3.5
```

Do not temporarily disable MTP, GDN, mixed full/GDN layers, or true Stage-II as a
way to make the test pass. Temporary toggles are acceptable only to localize a
failure, not as the final fix.

For every failure, classify it before editing:

```text
A. placeholder / image embedding / M-RoPE span mismatch
B. Stage-II keep_mask or hidden/residual/positions compact mismatch
C. suffix FA attention metadata / slot_mapping mismatch
D. suffix GDN metadata mismatch
E. qwen3_5_mtp proposer metadata / logits index mismatch
F. postprocess / sampling / prompt logprobs mismatch
G. graph / communication / MoE shape mismatch
```

Fix policy:

```text
1. Identify which component still uses S1 token space after Stage-II.
2. Move that component to S2 compact token space or rebuild its metadata.
3. Add an invariant near the boundary so the same class of bug fails early.
4. Re-run the same final scenario without reducing the configuration.
```

## Known Uncertainties To Watch

The biggest risk is not the Stage-II score. The risk is that some downstream
component silently keeps using S1 token coordinates after the suffix forward has
switched to S2.

Most suspicious areas:

```text
1. MTP proposer:
   - qwen3_5_mtp is expected to appear internally as method == "mtp".
   - The class name may be AscendEagleProposer, but this is the MTP branch.
   - Watch logits_indices, token_indices_to_sample, target_positions,
     target_hidden_states, and common_attn_metadata.

2. slot_mapping:
   - Suffix FA/GDN layers must write KV/cache using S2 slot_mapping.
   - slot_mapping[:compact_total] should be valid and should not describe S1.

3. M-RoPE positions:
   - hidden_states and positions must have the same compact token dimension.
   - runner.positions and mrope_positions.gpu must not retain S1 length for
     suffix layers.

4. GDN:
   - A single GDN layer sees a fixed token set; the danger is not dynamic length
     inside one layer.
   - The danger is a suffix GDN layer accidentally using prefix/S1 metadata.
   - GDN causal_conv1d metadata should be rebuilt by the existing builder from
     compact common metadata, not partially hand-written.

5. mixed prefill+decode:
   - compact query lengths and KV seq_lens are different concepts.
   - prefill rows use num_computed + compact_query_len.
   - decode rows keep their existing KV seq_len.

6. prompt logprobs:
   - If enabled, hidden_states[:num_scheduled_tokens] now refers to compact S2
     tokens, not the original full prompt.

7. graph / communication / MoE:
   - Suffix context must publish S2 num_tokens.
   - If any collective or graph path still uses S1 num_tokens, expect shape
     mismatch or silent wrong results.
```

Recommended invariants:

```text
suffix hidden_states.shape[0] == compact_total
suffix positions token dimension == compact_total
suffix common_attn_metadata.query_start_loc[-1] == compact_total
suffix per-layer metadata.num_actual_tokens == compact_total
suffix logits_indices.max() < compact_total
MTP common_attn_metadata.query_start_loc_cpu == _new_cu_seqlens
MTP common_attn_metadata.seq_lens_cpu == _new_seq_lens
MTP common_attn_metadata.max_query_len == max(diff(_new_cu_seqlens))
compact metadata variables are None after propose_draft_token_ids returns
```

Success criteria:

```text
stage2_true_layer... logs appear after Stage-I logs
suffix layers run with compact num_tokens
8k + 20 images does not trigger negative M-RoPE text_len
qwen3_5_mtp proposes from compact query_start_loc and seq_lens
no compact metadata variables leak into the next request or step
the final scenario passes without disabling core paths
```
