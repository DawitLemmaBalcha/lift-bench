# LIFT-Bench: Inspect implementation

[LIFT-Bench](https://arxiv.org/abs/2610.28819) measures whether a model can follow
constraints that are scoped to individual elements of a document outline, rather than
to the response as a whole. A task gives a topic and a nested outline of headers,
paragraphs, code blocks, tables and lists, and attaches constraints to specific
elements of it.

This repository is an [Inspect AI](https://inspect.aisi.org.uk/) implementation of the
benchmark. It is laid out for the
[Inspect Evals register](https://github.com/UKGovernmentBEIS/inspect_evals/blob/main/register/README.md);
the submission is pending a run with two models.

- Paper: [arXiv:2610.28819](https://arxiv.org/abs/2610.28819)
- Dataset: [ai21labs/LIFTBench](https://huggingface.co/datasets/ai21labs/LIFTBench) (Apache-2.0)
- Reference implementation: [AI21Labs/lift-bench](https://github.com/AI21Labs/lift-bench) (Apache-2.0)

## What it measures

Each constraint is scored only against the element it is attached to, so the benchmark
separates "a paragraph in French somewhere" from "section 2.1 in French". Constraint
density is a generation parameter rather than an afterthought. A task mixes
verifiable checks, run in code, with judged content requirements, run by an LLM judge,
into one weighted score.

The benchmark is factored into ten splits by document size and constraint density:

| Split | Documents | Constraints | Test tasks |
| --- | --- | --- | --- |
| `short_doc_few_constraints` | short | few | 85 |
| `short_doc_some_constraints` | short | some | 85 |
| `short_doc_many_constraints` | short | many | 85 |
| `short_doc_numerous_constraints` | short | numerous | 85 |
| `long_doc_few_constraints` | long | few | 66 |
| `long_doc_some_constraints` | long | some | 66 |
| `long_doc_many_constraints` | long | many | 66 |
| `long_doc_numerous_constraints` | long | numerous | 66 |
| `flat_doc_some_constraints` | flat | some | 90 |
| `flat_doc_many_constraints` | flat | many | 90 |
| **Total** | | | **784** |

## Dataset

The task loads each split from `ai21labs/LIFTBench` at the pinned revision
`7320d3271387297c14acd32d9584b88dccdd1f01`, so a change upstream cannot move the
samples without a change here. One record becomes one `Sample`:

- `input`: the record's chat `prompt` (the topic, the outline, and the constraints);
- `metadata`: the split name, topic, outline, and the `evaluations` specification;
- `id`: `<split>/<example_id>`, which is unique across splits;
- `target`: empty, because the score is the weighted constraint score rather than a
  single gold answer.

## Scoring

The scorer runs the task's scoped checks with the reference implementation's
`constraint_reward`. Verifiable checks (`lexical_count`, `lexical_match`,
`parser_evaluator`, `emoji_count`, `case_type`, `ifbench`, `target_language`) run in
process. `light_absolute_critic` calls an OpenAI-compatible endpoint, configured by
`OPENAI_BASE_URL` and `OPENAI_API_KEY`, with the judge named by the task's checks or by
the `judge_model` task parameter.

`Score.value` is the weighted sum of the checks that pass, between 0 and 1. The task
reports `mean` and `stderr` over all samples, plus a `grouped(mean)` metric keyed by
`config` for the ten per-split scores. A check that cannot be evaluated leaves the
sample unscored rather than contributing a silent zero, and an empty completion scores
zero with `reason="no_response"` so it stays in the denominator.

## Usage

```bash
uv sync
uv run inspect eval inspect_lift_bench/lift_bench --model openai/gpt-5-mini --limit 10
```

Run one split, or a subset:

```bash
uv run inspect eval inspect_lift_bench/lift_bench \
  --model openai/gpt-5-mini \
  -T config=long_doc_few_constraints
```

Override the judge (keeping the default judge makes scores comparable with the paper):

```bash
uv run inspect eval inspect_lift_bench/lift_bench \
  --model openai/gpt-5-mini \
  -T judge_model=gpt-5-mini
```

View logs with `uv run inspect view`. Put `INSPECT_EVAL_MODEL`, `OPENAI_API_KEY` and any
other provider keys in a `.env` file (git-ignored) to avoid repeating them.

## Differences from the reference implementation

| What changed | Why | Expected effect on scores |
| --- | --- | --- |
| The ten splits are one Inspect task, selected with `config`. | Inspect runs one dataset per task; a single task makes per-split metrics comparable in one log. | None; each sample is scored exactly as its split's task. |
| The task calls the reference `constraint_reward` rather than reimplementing the checks. | The verifiers and judge protocol are the benchmark; forking them invites drift. | None while the pinned revision is used. |
| The judge is configurable. | The paper's judge (`google/gemma-4-31B-it`) is not available on every endpoint. | Non-zero if the judge differs; report the judge with results. |
| A failed check marks the sample unscored. | A missing verdict is not a wrong answer; a zero would bias the mean down silently. | Excluded samples are visible as `unscored` in the log. |

## Results

Fill in after the two log runs used for the register submission. Report the model, the
judge, the split, and the weighted score.

## Architecture

```text
ai21labs/LIFTBench (pinned revision)
  -> record_to_sample: prompt -> Sample.input, evaluations -> Sample.metadata
  -> generate() (single turn)
  -> lift_bench_scorer: constraint_reward over scoped checks
       verifiable checks: in process
       light_absolute_critic: OpenAI-compatible judge endpoint
  -> Score.value: weighted constraint score
```

## Layout

```text
src/inspect_lift_bench/tasks.py        the task, dataset loading, record_to_sample and scorer
src/inspect_lift_bench/eval.yaml       title, description, contributors, version, tasks
tests/test_lift_bench.py               unit, dataset and end-to-end tests
```

The package is named `inspect_lift_bench` rather than `lift_bench` so it does not shadow
the reference implementation, which the scorer imports.

## Checks and tests

```bash
make check   # ruff, basedpyright, inspect-evals-lint, generated README sections, pre-commit
make test    # pytest
```

Dataset-download and end-to-end tests are marked `dataset_download` and `huggingface`;
they run when the network and the Hub are available.
