"""Runtime configuration for OccamToken experiments."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Literal


StageMode = Literal["off", "fixed", "stage1", "stage2", "full"]
ImplementationMode = Literal["masked", "true"]
ReplacementMode = Literal["zero", "mean"]
Stage1Scorer = Literal["norm", "mean_similarity"]
Stage2Scorer = Literal["text_similarity"]


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.lower() in {"1", "true", "yes", "on"}


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    return float(raw)


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    return int(raw)


def _env_str(name: str, default: str) -> str:
    raw = os.getenv(name)
    return default if raw is None or raw == "" else raw


@dataclass(frozen=True)
class OccamTokenConfig:
    """Environment-backed OccamToken settings.

    ``masked`` keeps the visual token count unchanged and replaces pruned
    embeddings. ``true`` reduces image placeholders and image embeddings before
    scheduling/merge, so the language model sees fewer visual tokens.
    """

    enabled: bool = False
    stage: StageMode = "off"
    implementation: ImplementationMode = "masked"
    target_ratio: float = 0.125
    target_tokens: int = 0
    stage1_ratio: float = 0.25
    stage1_tokens: int = 0
    min_tokens: int = 16
    replacement: ReplacementMode = "mean"
    stage1_scorer: Stage1Scorer = "norm"
    stage2_scorer: Stage2Scorer = "text_similarity"
    stage2_layer: int = 4
    max_text_tokens: int = 2048
    question_tail_tokens: int = 512
    log_stats: bool = False
    strict: bool = False

    @classmethod
    def from_env(cls) -> "OccamTokenConfig":
        enabled = _env_bool("VLLM_ASCEND_OCCAMTOKEN_ENABLE", False)
        stage = _env_str("VLLM_ASCEND_OCCAMTOKEN_STAGE", "off").lower()
        if stage not in {"off", "fixed", "stage1", "stage2", "full"}:
            raise ValueError(f"Unsupported VLLM_ASCEND_OCCAMTOKEN_STAGE={stage!r}")

        implementation = _env_str("VLLM_ASCEND_OCCAMTOKEN_IMPL", "masked").lower()
        if implementation not in {"masked", "true"}:
            raise ValueError(
                "Unsupported VLLM_ASCEND_OCCAMTOKEN_IMPL="
                f"{implementation!r}"
            )

        replacement = _env_str("VLLM_ASCEND_OCCAMTOKEN_REPLACEMENT", "mean").lower()
        if replacement not in {"zero", "mean"}:
            raise ValueError(
                "Unsupported VLLM_ASCEND_OCCAMTOKEN_REPLACEMENT="
                f"{replacement!r}"
            )

        stage1_scorer = _env_str("VLLM_ASCEND_OCCAMTOKEN_STAGE1_SCORER", "norm").lower()
        if stage1_scorer not in {"norm", "mean_similarity"}:
            raise ValueError(
                "Unsupported VLLM_ASCEND_OCCAMTOKEN_STAGE1_SCORER="
                f"{stage1_scorer!r}"
            )

        stage2_scorer = _env_str(
            "VLLM_ASCEND_OCCAMTOKEN_STAGE2_SCORER", "text_similarity"
        ).lower()
        if stage2_scorer not in {"text_similarity"}:
            raise ValueError(
                "Unsupported VLLM_ASCEND_OCCAMTOKEN_STAGE2_SCORER="
                f"{stage2_scorer!r}"
            )

        return cls(
            enabled=enabled,
            stage=stage,  # type: ignore[arg-type]
            implementation=implementation,  # type: ignore[arg-type]
            target_ratio=_env_float("VLLM_ASCEND_OCCAMTOKEN_TARGET_RATIO", 0.125),
            target_tokens=_env_int("VLLM_ASCEND_OCCAMTOKEN_TARGET_TOKENS", 0),
            stage1_ratio=_env_float("VLLM_ASCEND_OCCAMTOKEN_STAGE1_RATIO", 0.25),
            stage1_tokens=_env_int("VLLM_ASCEND_OCCAMTOKEN_STAGE1_TOKENS", 0),
            min_tokens=_env_int("VLLM_ASCEND_OCCAMTOKEN_MIN_TOKENS", 16),
            replacement=replacement,  # type: ignore[arg-type]
            stage1_scorer=stage1_scorer,  # type: ignore[arg-type]
            stage2_scorer=stage2_scorer,  # type: ignore[arg-type]
            stage2_layer=_env_int("VLLM_ASCEND_OCCAMTOKEN_STAGE2_LAYER", 4),
            max_text_tokens=_env_int("VLLM_ASCEND_OCCAMTOKEN_MAX_TEXT_TOKENS", 2048),
            question_tail_tokens=_env_int(
                "VLLM_ASCEND_OCCAMTOKEN_QUESTION_TAIL_TOKENS", 512
            ),
            log_stats=_env_bool("VLLM_ASCEND_OCCAMTOKEN_LOG_STATS", False),
            strict=_env_bool("VLLM_ASCEND_OCCAMTOKEN_STRICT", False),
        )

    def active(self) -> bool:
        return self.enabled and self.stage != "off"

    def stage1_active(self) -> bool:
        return self.active() and self.stage in {"fixed", "stage1", "full"}

    def stage2_active(self) -> bool:
        return self.active() and self.stage in {"stage2", "full"}

    def stage1_budget(self, num_tokens: int) -> int:
        if self.stage == "fixed":
            return self.final_budget(num_tokens)
        return _budget(
            num_tokens,
            tokens=self.stage1_tokens,
            ratio=self.stage1_ratio,
            min_tokens=self.min_tokens,
        )

    def final_budget(self, num_tokens: int) -> int:
        return _budget(
            num_tokens,
            tokens=self.target_tokens,
            ratio=self.target_ratio,
            min_tokens=self.min_tokens,
        )

    def true_sparse_active(self) -> bool:
        return self.active() and self.implementation == "true"

    def true_stage1_active(self) -> bool:
        return self.true_sparse_active() and self.stage1_active()


def _budget(num_tokens: int, *, tokens: int, ratio: float, min_tokens: int) -> int:
    if num_tokens <= 0:
        return 0
    if tokens > 0:
        budget = tokens
    else:
        budget = int(round(num_tokens * ratio))
    budget = max(min_tokens, budget)
    return min(num_tokens, budget)
