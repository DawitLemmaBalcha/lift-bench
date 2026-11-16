"""Run LIFT-Bench through OpenCode Go.

OpenCode Go serves Anthropic- and OpenAI-compatible endpoints that require an
``x-opencode-session`` routing header. This script supplies it to the model under
test and to the judge, which otherwise talks to the endpoint through its own
OpenAI client.

Usage:
    OPENCODE_API_KEY=... uv run python scripts/run_opencode_go.py \
        --model claude-haiku-5-5 --judge deepseek-v4.1-flash

Leave out ``--model``/``--judge`` to use those defaults. ``--limit`` runs a
subset; ``--config`` selects one split instead of all ten.
"""

from __future__ import annotations

import argparse
import os
import sys
import uuid

from inspect_ai import eval
from inspect_ai.model import get_model
from lift_bench.hierarchical_drafting.utils import inference
from openai import AsyncOpenAI

from inspect_lift_bench import lift_bench

OPENCODE_BASE = "https://opencode.ai/inference/go"
ANTHROPIC_URL = f"{OPENCODE_BASE}/anthropic"
OPENAI_URL = f"{OPENCODE_BASE}/openai/v1"


def _api_key() -> str:
    key = os.environ.get("OPENCODE_API_KEY") or os.environ.get("OPENCODE_GO_API_KEY")
    if not key:
        sys.exit("Set OPENCODE_API_KEY (or OPENCODE_GO_API_KEY) to an OpenCode Go key.")
    return key


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="claude-haiku-5-5", help="model under test")
    parser.add_argument("--judge", default="deepseek-v4.1-flash", help="judge model")
    parser.add_argument("--config", default=None, help="one split; default all ten")
    parser.add_argument("--limit", type=int, default=None, help="samples to run")
    args = parser.parse_args()

    key = _api_key()
    session = f"lift-bench-{uuid.uuid4()}"
    headers = {"x-opencode-session": session}

    model = get_model(
        f"anthropic/{args.model}",
        base_url=ANTHROPIC_URL,
        api_key=key,
        default_headers=headers,
    )

    def opencode_client() -> AsyncOpenAI:
        return AsyncOpenAI(
            base_url=OPENAI_URL,
            api_key=key,
            max_retries=0,
            default_headers=headers,
        )

    # The judge builds its own client; route it through the same session.
    inference._get_client = opencode_client  # type: ignore[assignment]

    [log] = eval(
        lift_bench(config=args.config, judge_model=args.judge),
        model=model,
        limit=args.limit,
        display="plain",
    )
    print(f"status: {log.status}")
    print(f"log: {log.location}")


if __name__ == "__main__":
    main()
