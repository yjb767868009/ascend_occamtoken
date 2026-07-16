import torch

from ascend_occamtoken.config import OccamTokenConfig
from ascend_occamtoken.phase2 import prune_phase2_local_image_outputs
from ascend_occamtoken.pruning import (
    prune_stage1_masked,
    prune_stage1_true,
    prune_stage2_masked,
    select_text_window,
)


def test_stage1_masked_keeps_shape_and_changes_some_tokens():
    config = OccamTokenConfig(enabled=True, stage="stage1", stage1_ratio=0.25, min_tokens=1)
    embeddings = torch.arange(32, dtype=torch.float32).view(8, 4)

    pruned, stats = prune_stage1_masked(embeddings, config)

    assert pruned.shape == embeddings.shape
    assert stats.original_tokens == 8
    assert stats.kept_tokens == 2
    assert not torch.equal(pruned, embeddings)


def test_stage1_true_reduces_shape():
    config = OccamTokenConfig(
        enabled=True,
        stage="stage1",
        implementation="true",
        stage1_ratio=0.25,
        min_tokens=1,
    )
    embeddings = torch.arange(32, dtype=torch.float32).view(8, 4)

    pruned, stats = prune_stage1_true(embeddings, config)

    assert pruned.shape == (2, 4)
    assert stats.original_tokens == 8
    assert stats.kept_tokens == 2


def test_fixed_stage_uses_target_budget_for_first_pass():
    config = OccamTokenConfig(
        enabled=True,
        stage="fixed",
        implementation="true",
        target_tokens=3,
        stage1_tokens=7,
        min_tokens=1,
    )

    assert config.stage1_budget(8) == 3


def test_phase2_helper_prunes_each_local_image_to_output_size():
    config = OccamTokenConfig(
        enabled=True,
        stage="stage1",
        implementation="true",
        stage1_ratio=0.5,
        min_tokens=1,
    )
    outputs = [
        torch.arange(16, dtype=torch.float32).view(4, 4),
        torch.arange(24, dtype=torch.float32).view(6, 4),
    ]

    pruned = prune_phase2_local_image_outputs(
        outputs,
        my_image_indices=[2, 4],
        output_sizes=[0, 0, 2, 0, 3],
        config=config,
    )

    assert [item.shape[0] for item in pruned] == [2, 3]


def test_stage2_masked_uses_text_similarity():
    config = OccamTokenConfig(enabled=True, stage="stage2", target_tokens=1, min_tokens=1)
    visual = torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    text = torch.tensor([[0.0, 1.0]])

    pruned, stats = prune_stage2_masked(visual, text, config)

    assert pruned.shape == visual.shape
    assert stats.kept_tokens == 1
    assert torch.equal(pruned[1], visual[1])


def test_select_text_window_prefers_tail():
    text = torch.arange(20, dtype=torch.float32).view(10, 2)
    selected = select_text_window(text, max_text_tokens=4, question_tail_tokens=3)
    assert selected.shape == (3, 2)
    assert torch.equal(selected, text[-3:])
