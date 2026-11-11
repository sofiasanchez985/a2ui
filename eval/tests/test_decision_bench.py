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

"""Unit tests for decision_bench, calibrate_decision_backend, and seed_decision_plans."""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys
from unittest.mock import patch

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../bin")))

from a2ui.core.basic_catalog import BasicCatalog
from a2ui.inference_formats.decision import Calibration, DecisionSynthesizer
from a2ui_eval.decision_bench import (
    PlanCase,
    brier_score,
    expected_calibration_error,
    fit_temperature,
    load_plan_cases,
    render_markdown,
    run_benchmark,
    run_case,
)
from a2ui_eval.decision_common import is_transcrypt_ciphertext
from a2ui_eval.shared.utils import GIT_ROOT
import calibrate_decision_backend
from decision_fixtures import (
    HOST_ANSWERS,
    HOST_DECISION_BLOCK,
    TRANSFER_ANSWERS,
    scripted_backend,
)
import pytest
from seed_decision_plans import seed_cases_from_logs

CASES_FILE = GIT_ROOT / "eval" / "datasets" / "decision_plans" / "v1.yaml"
SCHEMA_FILE = GIT_ROOT / "eval" / "datasets" / "decision_plans" / "schema.json"

_HOST_PLAN_CASE = PlanCase(
    name="host_health",
    intent=HOST_DECISION_BLOCK.intent,
    payload=dict(HOST_DECISION_BLOCK.payload),
    mutable_paths=list(HOST_DECISION_BLOCK.mutable_paths),
    expected_family="monitor",
    expected_archetypes=["dashboard", "metric_grid"],
    required_blocks=["headline", "facts"],
    expected_action_gate="pass",
    split="tuning",
    verified=True,
    tags=["smoke"],
)


def _write_synthetic_tuning_yaml(path: Path) -> Path:
    path.write_text(
        """
- name: host_health
  intent: "Show CPU utilization for prod-1"
  payload:
    host: "prod-1"
    cpu: 0.78
  expected:
    family: "monitor"
    archetypes: ["dashboard", "metric_grid"]
    blocks:
      required: ["headline", "facts"]
    action_gate: "pass"
  split: "tuning"
  tags: ["smoke"]
  verified: true

- name: wire_transfer
  intent: "Review wire transfer to Acme Corp"
  payload:
    recipient: "Acme Corp"
    amount: 1500
    internalTrace: "omit-me"
  actions:
    send: "Confirm Transfer"
    cancel: "Cancel"
  exclude: ["/internalTrace"]
  expected:
    family: "act"
    archetypes: ["confirm", "checkout", "review"]
    blocks:
      required: ["actions"]
    action_gate: "pass"
  split: "tuning"
  tags: ["smoke"]
  verified: true
""",
        encoding="utf-8",
    )
    return path


def test_synthetic_schema_validation(tmp_path: Path) -> None:
    sample_yaml = _write_synthetic_tuning_yaml(tmp_path / "plans.yaml")
    synthetic_cases = load_plan_cases(sample_yaml, schema_path=SCHEMA_FILE)
    assert len(synthetic_cases) == 2
    assert synthetic_cases[0].expected_family == "monitor"
    assert synthetic_cases[0].split == "tuning"
    assert synthetic_cases[0].is_heldout is False
    assert [a.id for a in synthetic_cases[1].action_specs()] == ["send", "cancel"]
    assert synthetic_cases[1].exclude == ["/internalTrace"]


def test_real_v1_yaml_validation_or_skip_if_encrypted() -> None:
    if not CASES_FILE.is_file():
        pytest.skip("Decision plan dataset file is absent.")
    if is_transcrypt_ciphertext(CASES_FILE):
        pytest.skip(
            "Decision plan dataset is encrypted with Transcrypt; unlock to run"
            " real dataset validation."
        )

    cases = load_plan_cases(CASES_FILE, schema_path=SCHEMA_FILE)
    assert len(cases) == 16
    for case in cases:
        assert case.verified is True
        assert case.split == "tuning"
        assert case.is_heldout is False
        assert "smoke" in case.tags
        assert "actions" not in case.payload
        assert "/actions" not in case.mutable_paths
        assert len(case.expected_archetypes) > 0
        assert case.expected_family in {
            "monitor",
            "act",
            "input",
            "browse",
            "inspect",
            "follow",
        }


def test_load_plan_cases_rejects_ciphertext(tmp_path: Path) -> None:
    encrypted_file = tmp_path / "encrypted.yaml"
    encrypted_file.write_bytes(b"U2FsdGVkX1+ciphertextbytes")
    with pytest.raises(RuntimeError, match="encrypted with Transcrypt"):
        load_plan_cases(encrypted_file)


@pytest.mark.asyncio
async def test_scripted_backend_host_case_scores_perfect() -> None:
    backend = scripted_backend(HOST_ANSWERS, model_id="mock-v1")
    synth = DecisionSynthesizer([BasicCatalog("v1.0")], backend)

    score = await run_case(synth, _HOST_PLAN_CASE, repeat=1)
    assert score.family_accuracy == 1.0
    assert score.archetype_accuracy == 1.0
    assert score.block_recall == 1.0
    assert score.block_precision > 0.0
    assert score.abstain_rate == 0.0
    assert score.fallback_rate == 0.0
    assert score.round_trips_per_surface == 1.0
    assert score.action_gate_match is True


@pytest.mark.asyncio
async def test_run_benchmark_reports_backend_model_round_trips_and_fallback(
    tmp_path: Path,
) -> None:
    sample_yaml = _write_synthetic_tuning_yaml(tmp_path / "plans.yaml")
    cases = load_plan_cases(sample_yaml, schema_path=SCHEMA_FILE)

    # Abstain on family so the synthesizer falls back to facts.
    backend = scripted_backend(
        {"family": ("none", 0.95)},
        model_id="mock-uncalibrated",
    )
    synth = DecisionSynthesizer([BasicCatalog("v1.0")], backend)

    summary = await run_benchmark(synth, cases, repeat=1)
    assert summary.backend_kind == "mock"
    assert summary.model_id == "mock-uncalibrated"
    assert summary.cases_count == 2
    assert summary.fallback_rate == 1.0
    assert summary.round_trips_per_surface == 1.0

    md = render_markdown(summary)
    assert "Backend: `mock` (model `mock-uncalibrated`)" in md
    assert "Fallback rate: 100.0%" in md
    assert "Round trips per surface: 1.00" in md


@pytest.mark.asyncio
async def test_scripted_transfer_actions_case_scores_perfect(tmp_path: Path) -> None:
    sample_yaml = _write_synthetic_tuning_yaml(tmp_path / "plans.yaml")
    cases = load_plan_cases(sample_yaml, schema_path=SCHEMA_FILE)
    transfer_case = cases[1]

    backend = scripted_backend(TRANSFER_ANSWERS, model_id="mock-act")
    synth = DecisionSynthesizer([BasicCatalog("v1.0")], backend)
    score = await run_case(synth, transfer_case, repeat=1)
    assert score.family_accuracy == 1.0
    assert score.archetype_accuracy == 1.0
    assert score.block_recall == 1.0
    assert score.fallback_rate == 0.0


@pytest.mark.asyncio
async def test_array_only_abstention_records_abstain_without_crashing() -> None:
    array_case = PlanCase(
        name="resort_list",
        intent="Browse available ski resorts",
        payload={"resorts": [{"name": "Alta", "openRuns": 70}]},
        expected_family="browse",
        expected_archetypes=["feed", "card_list"],
        required_blocks=["collection"],
        expected_action_gate="pass",
        split="tuning",
        verified=True,
    )
    backend = scripted_backend(
        {"family": ("none", 0.95)},
        model_id="mock-abstain",
    )
    synth = DecisionSynthesizer([BasicCatalog("v1.0")], backend)
    score = await run_case(synth, array_case, repeat=1)
    assert score.abstain_rate == 1.0
    assert score.fallback_rate == 0.0
    assert score.round_trips_per_surface == 1.0


@pytest.mark.asyncio
async def test_deliberately_wrong_script_scores_zero() -> None:
    wrong_answers = dict(HOST_ANSWERS)
    wrong_answers["family"] = ("input", 0.99)
    backend = scripted_backend(wrong_answers)
    synth = DecisionSynthesizer([BasicCatalog("v1.0")], backend)

    score = await run_case(synth, _HOST_PLAN_CASE, repeat=1)
    assert score.family_accuracy == 0.0


def test_brier_and_ece_computation() -> None:
    wrong_pairs = [(1.0, 0.0), (1.0, 0.0)]
    assert brier_score(wrong_pairs) == 1.0
    assert expected_calibration_error(wrong_pairs, num_bins=10) == 1.0

    calibrated_pairs = [(0.5, 1.0), (0.5, 0.0)]
    assert brier_score(calibrated_pairs) == 0.25
    assert expected_calibration_error(calibrated_pairs, num_bins=10) == 0.0

    pairs = [(0.1, 0.0), (0.9, 1.0)]
    assert pytest.approx(brier_score(pairs)) == (0.01 + 0.01) / 2
    assert pytest.approx(expected_calibration_error(pairs, num_bins=10)) == 0.1


def test_fit_temperature_returns_1_for_calibrated_inputs() -> None:
    calibrated_data = [(0.8, 1.0)] * 8 + [(0.8, 0.0)] * 2
    t = fit_temperature(calibrated_data)
    assert t == 1.0
    assert fit_temperature(calibrated_data, candidates=(-1.0, 0.0, 1.0)) == 1.0


def test_calibrate_decision_backend_writes_loadable_profile(
    tmp_path: Path,
) -> None:
    sample_yaml = _write_synthetic_tuning_yaml(tmp_path / "tuning.yaml")
    cal_out = tmp_path / "cal.json"

    rc = calibrate_decision_backend.main([
        "--backend",
        "mock",
        "--model",
        "mock",
        "--cases",
        str(sample_yaml),
        "--out",
        str(cal_out),
    ])
    assert rc == 0
    assert cal_out.is_file()

    loaded = Calibration.load(cal_out, backend_kind="mock", model_id="mock")
    assert loaded.temperature > 0.0


def test_calibrate_decision_backend_refuses_heldout_cases(
    tmp_path: Path,
) -> None:
    heldout_split_yaml = tmp_path / "heldout_split.yaml"
    heldout_split_yaml.write_text(
        """
- name: secret_eval_case
  intent: "Show CPU utilization"
  payload:
    cpu: 0.5
  expected:
    family: "monitor"
    archetypes: ["dashboard"]
  split: "heldout"
  tags: ["smoke"]
  verified: true
""",
        encoding="utf-8",
    )
    cal_out = tmp_path / "should_not_exist.json"

    with pytest.raises(ValueError, match="held-out"):
        calibrate_decision_backend.load_tuning_cases(
            heldout_split_yaml, schema_path=SCHEMA_FILE
        )

    rc_split = calibrate_decision_backend.main([
        "--backend",
        "mock",
        "--model",
        "mock",
        "--cases",
        str(heldout_split_yaml),
        "--out",
        str(cal_out),
    ])
    assert rc_split == 1
    assert not cal_out.exists()

    heldout_tag_yaml = tmp_path / "heldout_tag.yaml"
    heldout_tag_yaml.write_text(
        """
- name: tagged_heldout_case
  intent: "Show CPU utilization"
  payload:
    cpu: 0.5
  expected:
    family: "monitor"
    archetypes: ["dashboard"]
  tags: ["smoke", "heldout"]
  verified: true
""",
        encoding="utf-8",
    )
    rc_tag = calibrate_decision_backend.main([
        "--backend",
        "mock",
        "--model",
        "mock",
        "--cases",
        str(heldout_tag_yaml),
        "--out",
        str(cal_out),
    ])
    assert rc_tag == 1
    assert not cal_out.exists()


def test_compute_bayes_optimal_threshold() -> None:
    assert calibrate_decision_backend.compute_bayes_optimal_threshold(
        1.0, 20.0
    ) == pytest.approx(0.95)
    assert calibrate_decision_backend.compute_bayes_optimal_threshold(25.0, 20.0) == 0.0
    with pytest.raises(ValueError, match="non-negative"):
        calibrate_decision_backend.compute_bayes_optimal_threshold(-0.1, 10.0)
    with pytest.raises(ValueError, match="strictly positive"):
        calibrate_decision_backend.compute_bayes_optimal_threshold(1.0, 0.0)


def test_plan_case_from_dict_handles_explicit_nulls() -> None:
    case = PlanCase.from_dict({
        "name": "null_case",
        "intent": "test",
        "payload": None,
        "mutable_paths": None,
        "tags": None,
        "expected": {
            "family": None,
            "archetypes": None,
            "blocks": None,
            "action_gate": None,
        },
    })
    assert case.payload == {}
    assert case.expected_archetypes == []
    assert case.required_blocks == []
    assert case.expected_action_gate == "pass"


def test_seed_script_creates_unverified_stubs_skipped_by_default(
    tmp_path: Path,
) -> None:
    fake_log = {
        "samples": [{
            "id": "sample_123",
            "input": "Show server stats",
            "metadata": {
                "decision_block": json.dumps({
                    "intent": "Show server stats",
                    "payload": {"server": "web-1", "load": 0.42},
                    "actions": {"restart": "Restart Server"},
                    "mutablePaths": ["/load"],
                }),
                "decision_archetype": "dashboard",
                "decision_stage": "monitor",
                "decision_blocks": ["headline_row", "facts_card"],
            },
        }]
    }

    target_seeded = (
        GIT_ROOT / "eval" / "datasets" / "decision_plans" / "test_tmp_seeded.yaml"
    )
    if target_seeded.is_file():
        target_seeded.unlink()

    fake_log_file = tmp_path / "test.eval"
    fake_log_file.write_text("{}", encoding="utf-8")

    try:
        with patch("seed_decision_plans.load_log_data", return_value=fake_log):
            added = seed_cases_from_logs([fake_log_file], target_seeded)
            assert added == 1

        loaded_default = load_plan_cases(
            target_seeded, include_unverified=False, schema_path=SCHEMA_FILE
        )
        assert len(loaded_default) == 0

        loaded_all = load_plan_cases(
            target_seeded, include_unverified=True, schema_path=SCHEMA_FILE
        )
        assert len(loaded_all) == 1
        assert loaded_all[0].name == "sample_123_dashboard"
        assert loaded_all[0].verified is False
        assert loaded_all[0].split == "tuning"
        assert [a.id for a in loaded_all[0].action_specs()] == ["restart"]
        assert loaded_all[0].mutable_paths == ["/load"]
        assert loaded_all[0].expected_family == "monitor"
        assert loaded_all[0].expected_archetypes == ["dashboard"]
        assert loaded_all[0].required_blocks == ["headline_row", "facts_card"]
    finally:
        if target_seeded.is_file():
            target_seeded.unlink()


@pytest.mark.asyncio
async def test_collect_calibration_pairs_preserves_alignment_on_empty_score() -> None:
    from a2ui_eval.decision_bench import CaseScore

    synth = DecisionSynthesizer(
        [BasicCatalog("v1.0")], backend=scripted_backend(HOST_ANSWERS)
    )
    host_score = await run_case(synth, _HOST_PLAN_CASE)
    empty_case = PlanCase(
        name="empty_case",
        intent="Empty",
        payload={},
        expected_family="inform",
        expected_archetypes=["article"],
        required_blocks=["nonexistent_block"],
    )
    pairs_with_empty = calibrate_decision_backend._collect_calibration_pairs([
        CaseScore(case=empty_case, repeats=1, evaluations=[]),
        host_score,
    ])
    pairs_direct = calibrate_decision_backend._collect_calibration_pairs([
        host_score,
    ])
    assert pairs_with_empty == pairs_direct
