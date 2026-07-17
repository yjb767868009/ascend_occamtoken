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


def test_phase2_helper_logs_each_pruned_image(monkeypatch, capsys):
    monkeypatch.setenv("VLLM_ASCEND_OCCAMTOKEN_LOG_STATS", "1")
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

    prune_phase2_local_image_outputs(
        outputs,
        my_image_indices=[0, 1],
        output_sizes=[2, 3],
        config=config,
    )

    captured = capsys.readouterr()
    assert captured.err.count("[occamtoken]") == 2
    assert "original=4 kept=2" in captured.err
    assert "original=6 kept=3" in captured.err


def test_stage2_masked_uses_text_similarity():
    config = OccamTokenConfig(enabled=True, stage="stage2", target_tokens=1, min_tokens=1)
    visual = torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    text = torch.tensor([[0.0, 1.0]])

    pruned, stats = prune_stage2_masked(visual, text, config)

    assert pruned.shape == visual.shape
    assert stats.kept_tokens == 1
    assert torch.equal(pruned[1], visual[1])


def test_stage2_masked_can_run_after_stage1_true():
    config = OccamTokenConfig(
        enabled=True,
        stage="full",
        implementation="true",
        stage1_ratio=0.5,
        target_ratio=0.25,
        min_tokens=1,
    )
    visual = torch.tensor(
        [
            [0.0, 1.0],
            [0.0, 2.0],
            [3.0, 0.0],
            [4.0, 0.0],
        ]
    )
    text = torch.tensor([[1.0, 0.0]])

    stage1, stage1_stats = prune_stage1_true(visual, config)
    stage2, stage2_stats = prune_stage2_masked(stage1, text, config)

    assert stage1.shape == (2, 2)
    assert stage1_stats.kept_tokens == 2
    assert stage2.shape == stage1.shape
    assert stage2_stats.original_tokens == 2
    assert stage2_stats.kept_tokens == 1


def test_true_full_stage2_ratio_is_relative_to_original_budget():
    config = OccamTokenConfig(
        enabled=True,
        stage="full",
        implementation="true",
        stage1_ratio=0.25,
        target_ratio=0.125,
        min_tokens=1,
    )

    assert config.stage1_budget(2048) == 512
    assert config.stage2_budget(512) == 256


def test_select_text_window_prefers_tail():
    text = torch.arange(20, dtype=torch.float32).view(10, 2)
    selected = select_text_window(text, max_text_tokens=4, question_tail_tokens=3)
    assert selected.shape == (3, 2)
    assert torch.equal(selected, text[-3:])
