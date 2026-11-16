"""LIFT-Bench: scoped instruction-following for document drafting."""

from typing import Any

from inspect_ai import Task, task
from inspect_ai.dataset import Dataset, MemoryDataset, Sample
from inspect_ai.scorer import Score, Scorer, grouped, mean, scorer, stderr
from inspect_ai.solver import generate
from inspect_evals_utils.huggingface import hf_dataset
from inspect_evals_utils.metadata import load_eval_version
from lift_bench.hierarchical_drafting.metrics.errors import EvaluationError
from lift_bench.hierarchical_drafting.metrics.registry import constraint_reward

# The version comes from eval.yaml, the file beside this module.
EVAL_VERSION = load_eval_version(__file__)

DATASET_PATH = "ai21labs/LIFTBench"
"""The released benchmark dataset."""

DATASET_REVISION = "7320d3271387297c14acd32d9584b88dccdd1f01"
"""The commit the benchmark was registered at, so scores are comparable across runs."""

CONFIGS = (
    "short_doc_few_constraints",
    "short_doc_some_constraints",
    "short_doc_many_constraints",
    "short_doc_numerous_constraints",
    "long_doc_few_constraints",
    "long_doc_some_constraints",
    "long_doc_many_constraints",
    "long_doc_numerous_constraints",
    "flat_doc_some_constraints",
    "flat_doc_many_constraints",
)
"""The ten published splits, factored by document size, constraint density and scoping."""

JUDGE_MODEL: str | None = None
"""The judge the paper used, or ``None`` to use the model named by each task's checks."""


def record_to_sample(record: dict[str, Any]) -> Sample:
    """Convert one LIFT-Bench record into an Inspect sample.

    The model's input is the task's chat prompt; the constraint specifications live
    in ``metadata`` because the scorer needs them and the model must not see them.
    """
    config = record["dataset_name"]
    prompt = record["prompt"]
    return Sample(
        id=f"{config}/{record['example_id']}",
        input=prompt,
        target="",
        metadata={
            "config": config,
            "dataset_id": record["dataset_id"],
            "topic": record["topic"],
            "document_outline": record["document_outline"],
            "evaluations": record["info"]["evaluations"],
            "prompt": prompt,
        },
    )


def load_dataset(configs: list[str]) -> Dataset:
    """Load the given splits of the benchmark at its pinned revision."""
    samples: list[Sample] = []
    for config in configs:
        split = hf_dataset(
            path=DATASET_PATH,
            name=config,
            split="test",
            revision=DATASET_REVISION,
            sample_fields=record_to_sample,
        )
        samples.extend(split)
    return MemoryDataset(samples=samples, name="lift_bench")


@scorer(metrics=[mean(), stderr(), grouped(mean(), "config")])
def lift_bench_scorer(judge_model: str | None = None) -> Scorer:
    """Score a drafted document against the task's scoped constraints.

    Every check in the task's ``evaluations`` runs, and the score is the weight of
    the checks it passes: verifiable checks run in code, judged checks call an
    OpenAI-compatible endpoint configured by ``OPENAI_BASE_URL`` and
    ``OPENAI_API_KEY``. A check that cannot be evaluated leaves the sample
    unscored, rather than silently contributing a zero to the mean.
    """

    async def score(state: Any, target: Any) -> Score:
        completion = state.output.completion
        if not completion.strip():
            # The eval scores an empty draft as zero rather than dropping it, so
            # a model that returns nothing still counts in the denominator.
            return Score(value=0.0, reason="no_response", explanation=completion)

        evaluations = state.metadata.get("evaluations", [])
        try:
            result = await constraint_reward(
                completion=[{"role": "assistant", "content": completion}],
                info={"evaluations": evaluations},
                prompt=state.metadata.get("prompt"),
                judge_model=judge_model,
            )
        except EvaluationError as exc:
            return Score.unscored(
                reason="scoring_failed",
                explanation=f"constraint checks could not all be evaluated: {exc}",
                metadata={
                    "config": state.metadata.get("config"),
                    "agg_eval_scores": exc.agg_eval_scores,
                    "agg_eval_ids": exc.agg_eval_ids,
                },
            )

        return Score(
            value=float(result["score"]),
            explanation=(
                f"{len(result['agg_eval_scores'])} scoped constraint(s); "
                f"weighted score {result['score']:.4f}"
            ),
            metadata={
                "config": state.metadata.get("config"),
                "agg_eval_scores": result["agg_eval_scores"],
                "agg_eval_ids": result["agg_eval_ids"],
            },
        )

    return score


@task
def lift_bench(
    config: str | list[str] | None = None,
    judge_model: str | None = JUDGE_MODEL,
) -> Task:
    """LIFT-Bench: scoped instruction-following for document drafting.

    Args:
        config: The split to run, a list of splits, or ``None`` for all ten.
        judge_model: The model that scores judged constraints. ``None`` uses the
            model named by each task's checks, which is the paper's judge.
    """
    return Task(
        dataset=load_dataset(_resolve_configs(config)),
        solver=generate(),
        scorer=lift_bench_scorer(judge_model=judge_model),
        version=EVAL_VERSION,
    )


def _resolve_configs(config: str | list[str] | None) -> list[str]:
    if config is None:
        return list(CONFIGS)
    configs = [config] if isinstance(config, str) else list(config)
    unknown = [name for name in configs if name not in CONFIGS]
    if unknown:
        raise ValueError(
            f"Unknown LIFT-Bench config(s): {', '.join(unknown)}. "
            f"Valid configs: {', '.join(CONFIGS)}."
        )
    return configs
