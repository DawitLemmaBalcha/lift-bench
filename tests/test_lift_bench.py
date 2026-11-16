"""Tests for LIFT-Bench."""

import importlib
import math
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from inspect_ai import eval
from inspect_ai.scorer import Target
from pytest_inspect_evals.assertions import assert_eval_success

from inspect_lift_bench import lift_bench, record_to_sample

# The package's __init__ exports the task function under the module's name, so
# resolve the module through importlib rather than attribute access.
task_module = importlib.import_module("inspect_lift_bench.tasks")

RECORD = {
    "dataset_name": "short_doc_few_constraints",
    "dataset_id": "l3_d2_sh3_sv2_snv2",
    "topic": "Philip Johnson",
    "example_id": "87d735ab12e99c44afb0fd3a560ee149",
    "document_outline": "# Philip Johnson",
    "metadata": {},
    "split": "test",
    "prompt": [{"role": "user", "content": "Write a markdown document about Philip Johnson."}],
    "info": {
        "evaluations": [
            {
                "metric_name": "lexical_match",
                "weight": 1.0,
                "evaluation_id": "e1",
            }
        ]
    },
}


def _state(completion: str, metadata: dict | None = None) -> SimpleNamespace:
    return SimpleNamespace(
        output=SimpleNamespace(completion=completion),
        metadata=metadata if metadata is not None else {"evaluations": []},
    )


def test_record_to_sample() -> None:
    sample = record_to_sample(RECORD)
    assert sample.id == "short_doc_few_constraints/87d735ab12e99c44afb0fd3a560ee149"
    assert isinstance(sample.input, list)
    first = sample.input[0]
    assert isinstance(first.content, str)
    assert first.content == "Write a markdown document about Philip Johnson."
    assert sample.metadata is not None
    assert sample.metadata["config"] == "short_doc_few_constraints"
    assert sample.metadata["topic"] == "Philip Johnson"
    assert sample.metadata["evaluations"][0]["metric_name"] == "lexical_match"


def test_all_configs_by_default() -> None:
    assert task_module._resolve_configs(None) == list(task_module.CONFIGS)


def test_single_and_multiple_configs() -> None:
    assert task_module._resolve_configs("long_doc_few_constraints") == ["long_doc_few_constraints"]
    assert task_module._resolve_configs(["flat_doc_some_constraints"]) == [
        "flat_doc_some_constraints"
    ]


def test_unknown_config_raises() -> None:
    with pytest.raises(ValueError, match="Unknown LIFT-Bench config"):
        task_module._resolve_configs("not_a_config")


async def test_scorer_returns_the_weighted_score(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_reward = AsyncMock(
        return_value={
            "score": 0.5,
            "agg_eval_scores": [1.0, 0.0],
            "agg_eval_ids": ["a", "b"],
        }
    )
    monkeypatch.setattr(task_module, "constraint_reward", fake_reward)
    scorer = task_module.lift_bench_scorer(judge_model="judge/model")
    score = await scorer(
        _state("draft", {"evaluations": [{"metric_name": "x"}], "config": "cfg"}),
        Target(""),
    )

    assert score is not None
    assert score.value == pytest.approx(0.5)
    assert score.metadata["config"] == "cfg"
    assert score.metadata["agg_eval_ids"] == ["a", "b"]

    fake_reward.assert_awaited_once()
    assert fake_reward.await_args is not None
    kwargs = fake_reward.await_args.kwargs
    assert kwargs["completion"] == [{"role": "assistant", "content": "draft"}]
    assert kwargs["info"]["evaluations"] == [{"metric_name": "x"}]
    assert kwargs["judge_model"] == "judge/model"


async def test_scorer_leaves_a_failed_check_unscored(monkeypatch: pytest.MonkeyPatch) -> None:
    from lift_bench.hierarchical_drafting.metrics.errors import EvaluationError

    monkeypatch.setattr(
        task_module,
        "constraint_reward",
        AsyncMock(
            side_effect=EvaluationError(
                "judge failed", agg_eval_scores=[1.0, None], agg_eval_ids=["a", "b"]
            )
        ),
    )
    scorer = task_module.lift_bench_scorer()
    score = await scorer(
        _state("draft", {"evaluations": [{"metric_name": "x"}], "config": "cfg"}),
        Target(""),
    )

    assert score is not None
    assert math.isnan(score.value)
    assert score.reason == "scoring_failed"
    assert score.metadata["agg_eval_ids"] == ["a", "b"]


async def test_empty_completion_scores_zero_and_stays_in_the_denominator() -> None:
    scorer = task_module.lift_bench_scorer()
    score = await scorer(_state("   "), Target(""))

    assert score is not None
    assert score.value == pytest.approx(0.0)
    assert score.reason == "no_response"


@pytest.mark.dataset_download
@pytest.mark.huggingface
def test_dataset_split_loads_at_the_pinned_revision() -> None:
    dataset = task_module.load_dataset(["short_doc_few_constraints"])
    assert len(dataset) == 85
    sample = dataset[0]
    assert sample.metadata is not None
    assert sample.metadata["config"] == "short_doc_few_constraints"
    assert sample.metadata["evaluations"]


@pytest.mark.dataset_download
@pytest.mark.huggingface
def test_end_to_end_with_mock_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        task_module,
        "constraint_reward",
        AsyncMock(
            return_value={
                "score": 0.25,
                "agg_eval_scores": [0.25],
                "agg_eval_ids": ["e1"],
            }
        ),
    )
    [log] = eval(
        tasks=lift_bench(config="short_doc_few_constraints"),
        model="mockllm/model",
        limit=1,
    )
    assert_eval_success(log, expected_sample_count=1)
    assert log.samples is not None
    scores = log.samples[0].scores
    assert scores is not None
    assert scores["lift_bench_scorer"].value == pytest.approx(0.25)


def test_version_comes_from_eval_yaml() -> None:
    assert task_module.EVAL_VERSION == "1-A"
