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

"""Seeds decision plan cases from recorded Inspect AI evaluation logs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

from a2ui_eval.shared.utils import GIT_ROOT
import yaml

BIN_DIR = Path(__file__).parent.resolve()
if str(BIN_DIR) not in sys.path:
    sys.path.insert(0, str(BIN_DIR))

from report_evals import load_log_data

DECISION_PLANS_DIR = (GIT_ROOT / "eval" / "datasets" / "decision_plans").resolve()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Seed unverified decision plan cases from evaluation logs.",
    )
    parser.add_argument(
        "logs",
        nargs="+",
        help="Paths to .eval log files or directories containing them.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=str(DECISION_PLANS_DIR / "seeded.yaml"),
        help="Output YAML file path (must reside under eval/datasets/decision_plans/).",
    )
    return parser.parse_args(argv)


def _validate_out_path(out_path: Path) -> Path:
    resolved = out_path.resolve()
    try:
        resolved.relative_to(DECISION_PLANS_DIR)
    except ValueError as exc:
        raise ValueError(
            f"Output file must be inside {DECISION_PLANS_DIR}, got {resolved}"
        ) from exc
    return resolved


def _parse_block_dict(raw_block: Any) -> dict[str, Any] | None:
    """Parses `raw_block` (a JSON string or dict) into a mapping, if valid."""
    if isinstance(raw_block, str):
        try:
            parsed = json.loads(raw_block)
        except json.JSONDecodeError:
            return None
        return parsed if isinstance(parsed, dict) else None
    if isinstance(raw_block, dict):
        return raw_block
    return None


def _normalize_blocks_list(raw_blocks: Any) -> list[str]:
    """Normalizes `decision_blocks` metadata into a list of block IDs."""
    if isinstance(raw_blocks, str):
        return [b.strip() for b in raw_blocks.split(",") if b.strip()]
    if isinstance(raw_blocks, list):
        return [str(b) for b in raw_blocks if b]
    return []


def _sample_to_case_stub(
    sample: dict[str, Any],
    *,
    log_file: Path,
    existing_names: set[str],
) -> dict[str, Any] | None:
    """Builds an unverified `decision_plans` stub from a single logged sample."""
    meta = sample.get("metadata") or {}
    block_data = _parse_block_dict(meta.get("decision_block"))
    arch = meta.get("decision_archetype")
    if block_data is None or not arch or arch == "unknown":
        return None

    sample_id = str(sample.get("id") or sample.get("name") or "sample")
    case_name = f"{sample_id}_{arch}"
    if case_name in existing_names:
        return None

    family = meta.get("decision_family") or meta.get("decision_stage") or "monitor"
    if family == "archetype":
        family = "monitor"

    stub: dict[str, Any] = {
        "name": case_name,
        "intent": block_data.get("intent") or sample.get("input") or "",
        "payload": block_data.get("payload") or {},
        "expected": {
            "family": family,
            "archetypes": [arch],
            "blocks": {
                "required": _normalize_blocks_list(meta.get("decision_blocks")),
            },
            "action_gate": "pass",
        },
        "split": "tuning",
        "tags": ["seeded"],
        "verified": False,
        "source": {
            "log_path": str(log_file),
            "sample_id": sample_id,
        },
    }
    actions = block_data.get("actions")
    if isinstance(actions, dict) and actions:
        stub["actions"] = actions
    mutable_paths = (
        block_data.get("mutable_paths") or block_data.get("mutablePaths") or []
    )
    if mutable_paths:
        stub["mutable_paths"] = list(mutable_paths)
    exclude = block_data.get("exclude")
    if exclude:
        stub["exclude"] = list(exclude)
    layout = block_data.get("layout")
    if layout:
        stub["layout"] = list(layout)
    return stub


def _collect_eval_files(log_paths: list[str | Path]) -> list[Path]:
    """Expands file and directory paths into `.eval` log files."""
    files: list[Path] = []
    for log_path_str in log_paths:
        log_path = Path(log_path_str)
        candidates = (
            list(log_path.glob("**/*.eval")) if log_path.is_dir() else [log_path]
        )
        files.extend(f for f in candidates if f.is_file())
    return files


def _load_existing_cases(out_path: Path) -> tuple[list[dict[str, Any]], set[str]]:
    """Loads existing YAML cases and returns `(cases, existing_names)`."""
    if not out_path.is_file():
        return [], set()
    with open(out_path, "r", encoding="utf-8") as f:
        existing_cases: list[dict[str, Any]] = yaml.safe_load(f) or []
    names = {
        str(item["name"])
        for item in existing_cases
        if isinstance(item, dict) and "name" in item
    }
    return existing_cases, names


def seed_cases_from_logs(
    log_paths: list[str | Path],
    out_file: str | Path,
) -> int:
    """Reads eval logs and seeds decision plan cases into out_file."""
    out_path = _validate_out_path(Path(out_file))
    existing_cases, existing_names = _load_existing_cases(out_path)
    new_cases: list[dict[str, Any]] = []

    for log_file in _collect_eval_files(log_paths):
        try:
            log_data = load_log_data(str(log_file))
        except Exception as exc:  # pylint: disable=broad-exception-caught
            sys.stderr.write(f"Warning: could not load {log_file}: {exc}\n")
            continue
        for sample in log_data.get("samples") or []:
            stub = _sample_to_case_stub(
                sample, log_file=log_file, existing_names=existing_names
            )
            if stub is not None:
                existing_names.add(str(stub["name"]))
                new_cases.append(stub)

    if not new_cases and not existing_cases:
        return 0

    all_cases = existing_cases + new_cases
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(all_cases, f, sort_keys=False)

    return len(new_cases)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        added = seed_cases_from_logs(args.logs, args.out)
        print(f"Seeded {added} new case(s) into {args.out}")
        return 0
    except Exception as exc:
        sys.stderr.write(f"Error: {exc}\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
