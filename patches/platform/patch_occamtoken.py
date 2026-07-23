"""Platform entry point for OccamToken target-model experiments.

Copy this file to:
    <VLLM_ASCEND_CHECKOUT>/vllm_ascend/patch/platform/patch_occamtoken.py

Then import it from vllm_ascend.patch.platform.__init__ behind an environment
guard. The actual model monkey patches live in the worker patch module.
"""

import os

if os.getenv("VLLM_ASCEND_OCCAMTOKEN_ENABLE", "0").lower() in {
    "1",
    "true",
    "yes",
    "on",
}:
    target_model = os.getenv("VLLM_ASCEND_OCCAMTOKEN_TARGET_MODEL", "qwen3_5").lower()
    if target_model in {"qwen3_vl", "qwen3vl"}:
        import vllm_ascend.patch.worker.patch_occamtoken_qwen3vl  # noqa
    elif target_model in {"qwen3_5", "qwen35"}:
        import vllm_ascend.patch.worker.patch_occamtoken_qwen35  # noqa
        import vllm_ascend.patch.worker.patch_occamtoken_stage2_decoder  # noqa
        import vllm_ascend.patch.worker.patch_occamtoken_runner  # noqa
    else:
        raise ValueError(
            "Unsupported VLLM_ASCEND_OCCAMTOKEN_TARGET_MODEL="
            f"{target_model!r}. Expected qwen3_5 or qwen3_vl."
        )
