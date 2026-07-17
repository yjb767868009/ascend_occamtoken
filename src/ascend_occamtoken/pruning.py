"""Tensor pruning helpers for OccamToken-style experiments."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .config import OccamTokenConfig


@dataclass(frozen=True)
class PruneStats:
    stage: str
    original_tokens: int
    kept_tokens: int

    @property
    def retention(self) -> float:
        if self.original_tokens == 0:
            return 1.0
        return self.kept_tokens / self.original_tokens


def topk_indices(scores: torch.Tensor, budget: int) -> torch.Tensor:
    """Return sorted top-k indices for stable token order."""
    num_tokens = int(scores.shape[0])
    if budget >= num_tokens:
        return torch.arange(num_tokens, device=scores.device)
    if budget <= 0:
        return torch.empty(0, dtype=torch.long, device=scores.device)
    _, indices = torch.topk(scores, k=budget, largest=True, sorted=False)
    return indices.sort().values


def stage1_scores(embeddings: torch.Tensor, config: OccamTokenConfig) -> torch.Tensor:
    """Score visual embeddings without text-query information."""
    embeddings_f = embeddings.float()
    if config.stage1_scorer == "norm":
        return torch.linalg.vector_norm(embeddings_f, dim=-1)

    normalized = F.normalize(embeddings_f, dim=-1)
    ref = F.normalize(embeddings_f.mean(dim=0, keepdim=True), dim=-1)
    return (normalized @ ref.transpose(0, 1)).squeeze(-1).abs()


def stage2_scores(
    visual_embeddings: torch.Tensor,
    text_embeddings: torch.Tensor,
    config: OccamTokenConfig,
) -> torch.Tensor:
    """Score visual embeddings with query/text information."""
    del config
    if text_embeddings.numel() == 0:
        return torch.linalg.vector_norm(visual_embeddings.float(), dim=-1)

    visual = F.normalize(visual_embeddings.float(), dim=-1)
    text = F.normalize(text_embeddings.float(), dim=-1)
    scores = visual @ text.transpose(0, 1)
    return scores.max(dim=-1).values


def mask_pruned_embeddings(
    embeddings: torch.Tensor,
    keep_indices: torch.Tensor,
    *,
    replacement: str,
) -> torch.Tensor:
    """Keep tensor shape, replacing pruned embeddings.

    This is the smoke-test mode. It does not reduce sequence length, but it lets
    us validate pruning quality without touching vLLM placeholder accounting.
    """
    num_tokens = int(embeddings.shape[0])
    if keep_indices.numel() >= num_tokens:
        return embeddings

    keep_mask = torch.zeros(num_tokens, dtype=torch.bool, device=embeddings.device)
    keep_mask[keep_indices] = True
    output = embeddings.clone()

    if replacement == "zero":
        fill = torch.zeros_like(output[:1])
    elif keep_indices.numel() > 0:
        fill = embeddings[keep_indices].mean(dim=0, keepdim=True)
    else:
        fill = torch.zeros_like(output[:1])

    output[~keep_mask] = fill.to(dtype=output.dtype)
    return output


def prune_stage1_masked(
    embeddings: torch.Tensor,
    config: OccamTokenConfig,
) -> tuple[torch.Tensor, PruneStats]:
    keep = stage1_keep_indices(embeddings, config)
    pruned = mask_pruned_embeddings(
        embeddings,
        keep,
        replacement=config.replacement,
    )
    return pruned, PruneStats("stage1_masked", int(embeddings.shape[0]), int(keep.numel()))


def stage1_keep_indices(
    embeddings: torch.Tensor,
    config: OccamTokenConfig,
) -> torch.Tensor:
    budget = config.stage1_budget(int(embeddings.shape[0]))
    scores = stage1_scores(embeddings, config)
    return topk_indices(scores, budget)


def prune_stage1_true(
    embeddings: torch.Tensor,
    config: OccamTokenConfig,
) -> tuple[torch.Tensor, PruneStats]:
    keep = stage1_keep_indices(embeddings, config)
    pruned = embeddings.index_select(0, keep)
    return pruned, PruneStats("stage1_true", int(embeddings.shape[0]), int(keep.numel()))


def prune_stage2_masked(
    visual_embeddings: torch.Tensor,
    text_embeddings: torch.Tensor,
    config: OccamTokenConfig,
) -> tuple[torch.Tensor, PruneStats]:
    budget = config.stage2_budget(int(visual_embeddings.shape[0]))
    scores = stage2_scores(visual_embeddings, text_embeddings, config)
    keep = topk_indices(scores, budget)
    pruned = mask_pruned_embeddings(
        visual_embeddings,
        keep,
        replacement=config.replacement,
    )
    return pruned, PruneStats("stage2_masked", int(visual_embeddings.shape[0]), int(keep.numel()))


def select_text_window(
    text_embeddings: torch.Tensor,
    *,
    max_text_tokens: int,
    question_tail_tokens: int,
) -> torch.Tensor:
    """Select a bounded text window for Stage-II-lite scoring.

    RAG prompts can be long; using all 10k text tokens for visual relevance is
    expensive and can be noisy. The first version keeps the tail, which usually
    contains the user question and final instruction.
    """
    if text_embeddings.shape[0] <= max_text_tokens:
        return text_embeddings
    tail = min(question_tail_tokens, max_text_tokens, int(text_embeddings.shape[0]))
    return text_embeddings[-tail:]
