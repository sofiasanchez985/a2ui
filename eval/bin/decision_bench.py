#!/usr/bin/env python3
# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""CLI tool for benchmarking A2UI decision models against golden plans."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
import sys
from typing import Any

from a2ui.core.basic_catalog import BasicCatalog
from a2ui.inference_formats.decision import DecisionSynthesizer, SynthesisConfig
from a2ui_eval.decision_bench import (
    BenchmarkSummary,
    load_plan_cases,
    render_markdown,
    run_benchmark,
)
from a2ui_eval.decision_common import (
    DECISION_BACKEND_NAMES,
    build_decision_backend,
)
from a2ui_eval.shared.utils import GIT_ROOT
from dotenv import load_dotenv

load_dotenv()
load_dotenv(Path.home() / ".env")

DEFAULT_CASES_PATH = GIT_ROOT / "eval" / "datasets" / "decision_plans" / "v1.yaml"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark decision backend latency, accuracy, and calibration.",
    )
    parser.add_argument(
        "--backend",
        choices=list(DECISION_BACKEND_NAMES),
        default=None,
        help=(
            "Decision backend to evaluate (mock, jev, systemone, or step0;"
            " required unless A2UI_DECISION_BACKEND is set)."
        ),
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Optional model identifier passed to the backend.",
    )
    parser.add_argument(
        "--endpoint-url",
        type=str,
        default=None,
        help="Optional endpoint URL override for remote backends.",
    )
    parser.add_argument(
        "--calibration",
        type=str,
        default=None,
        help="Optional path to a calibration profile JSON file.",
    )
    parser.add_argument(
        "--cases",
        type=str,
        default=str(DEFAULT_CASES_PATH),
        help="Path to YAML cases file (default: eval/datasets/decision_plans/v1.yaml).",
    )
    parser.add_argument(
        "--split",
        type=str,
        default=None,
        help="Optional dataset split filter (tuning or heldout).",
    )
    parser.add_argument(
        "--tags",
        type=str,
        default="smoke",
        help="Comma-separated filter tags (default: smoke).",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=3,
        help="Number of times to run each case (default: 3).",
    )
    parser.add_argument(
        "--include-unverified",
        action="store_true",
        help="Include cases with verified=false.",
    )
    parser.add_argument(
        "--fit-temperature",
        action="store_true",
        help="Fit temperature T to minimize Expected Calibration Error.",
    )
    parser.add_argument(
        "--json",
        type=str,
        default=None,
        dest="json_path",
        help="Optional path to write raw JSON summary.",
    )
    parser.add_argument(
        "--markdown",
        action="store_true",
        default=True,
        help="Render summary table in Markdown to stdout (default: True).",
    )
    return parser.parse_args(argv)


def _serialize_summary(summary: BenchmarkSummary) -> dict[str, Any]:
    """Serializes `summary` into a JSON-compatible dictionary."""
    return {
        "backend_kind": summary.backend_kind,
        "model_id": summary.model_id,
        "cases_count": summary.cases_count,
        "total_runs": summary.total_runs,
        "family_accuracy": summary.family_accuracy,
        "archetype_accuracy": summary.archetype_accuracy,
        "block_precision": summary.block_precision,
        "block_recall": summary.block_recall,
        "abstain_rate": summary.abstain_rate,
        "fallback_rate": summary.fallback_rate,
        "round_trips_per_surface": summary.round_trips_per_surface,
        "decision_latency_p50": summary.decision_latency_p50,
        "decision_latency_p95": summary.decision_latency_p95,
        "host_overhead_p50": summary.host_overhead_p50,
        "host_overhead_p95": summary.host_overhead_p95,
        "decision_calls_mean": summary.decision_calls_mean,
        "brier_family": summary.brier_family,
        "ece_family": summary.ece_family,
        "brier_archetype": summary.brier_archetype,
        "ece_archetype": summary.ece_archetype,
        "brier_gates": summary.brier_gates,
        "ece_gates": summary.ece_gates,
        "fitted_temperature": summary.fitted_temperature,
        "cases": [
            {
                "name": cs.case.name,
                "family_accuracy": cs.family_accuracy,
                "archetype_accuracy": cs.archetype_accuracy,
                "block_precision": cs.block_precision,
                "block_recall": cs.block_recall,
                "abstain_rate": cs.abstain_rate,
                "fallback_rate": cs.fallback_rate,
                "round_trips_per_surface": cs.round_trips_per_surface,
                "decision_latency_p50": cs.decision_latency_p50,
                "host_overhead_p50": cs.host_overhead_p50,
            }
            for cs in summary.case_scores
        ],
    }


async def _async_main(args: argparse.Namespace) -> int:
    cases_path = Path(args.cases)
    if not cases_path.is_absolute():
        cases_path = GIT_ROOT / cases_path

    cases = load_plan_cases(
        cases_path,
        tags=args.tags,
        split=args.split,
        include_unverified=args.include_unverified,
    )
    if not cases:
        sys.stderr.write(f"No cases found matching criteria in {cases_path}\n")
        return 1

    cal_path = Path(args.calibration) if args.calibration else None
    backend = build_decision_backend(
        args.backend,
        endpoint_url=args.endpoint_url,
        model=args.model,
        calibration=cal_path,
    )
    synth = DecisionSynthesizer(
        [BasicCatalog("v1.0")],
        backend,
        config=SynthesisConfig(),
    )

    try:
        summary = await run_benchmark(
            synth,
            cases,
            repeat=args.repeat,
            fit_temp=args.fit_temperature,
        )
    finally:
        await backend.aclose()

    if args.markdown:
        label = f"{summary.backend_kind}/{summary.model_id}"
        print(render_markdown(summary, backend_name=label))

    if args.json_path:
        out_path = Path(args.json_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(_serialize_summary(summary), indent=2),
            encoding="utf-8",
        )

    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    return asyncio.run(_async_main(args))


if __name__ == "__main__":
    sys.exit(main())
