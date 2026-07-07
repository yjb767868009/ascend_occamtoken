"""Platform entry point for Qwen3.5 OccamToken experiments.

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
    import vllm_ascend.patch.worker.patch_occamtoken_qwen35  # noqa
