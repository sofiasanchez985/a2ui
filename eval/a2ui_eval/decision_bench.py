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

"""Golden plan evaluation and latency benchmark for A2UI decision models."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
import json
import math
from pathlib import Path
import time
from typing import Any

from a2ui.inference_formats.decision import (
    ActionSpec,
    BackendCapabilities,
    Calibration,
    DecisionAnswer,
    DecisionBackend,
    DecisionBatch,
    DecisionQuestion,
    DecisionRecord,
    DecisionSynthesizer,
    QuestionKind,
    SynthesisOutput,
    SynthesisStage,
)
from a2ui.parser import A2uiCompilationError
import jsonschema
import yaml

from .decision_common import is_transcrypt_ciphertext


@dataclass(frozen=True)
class PlanCase:
    """A single ground-truth decision case."""

    name: str
    intent: str
    payload: dict[str, Any]
    expected_family: str
    expected_archetypes: list[str]
    actions: dict[str, Any] = field(default_factory=dict)
    exclude: list[str] = field(default_factory=list)
    replace: bool = False
    mutable_paths: list[str] = field(default_factory=list)
    layout: list[str] = field(default_factory=list)
    title: str | None = None
    summary: str | None = None
    required_blocks: list[str] = field(default_factory=list)
    forbidden_blocks: list[str] = field(default_factory=list)
    expected_action_gate: str = "pass"
    split: str | None = None
    tags: list[str] = field(default_factory=list)
    verified: bool = True
    source: dict[str, Any] | None = None

    @property
    def is_heldout(self) -> bool:
        """Whether this case belongs to the held-out evaluation split."""
        if self.split and self.split.strip().lower() == "heldout":
            return True
        lowered = {t.strip().lower() for t in self.tags}
        return "heldout" in lowered or "split:heldout" in lowered

    def action_specs(self) -> tuple[ActionSpec, ...]:
        """Parses `self.actions` map entries into `ActionSpec` instances."""
        return tuple(
            ActionSpec.from_block_value(str(act_id), val)
            for act_id, val in self.actions.items()
        )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> PlanCase:
        expected = data.get("expected") or {}
        blocks = expected.get("blocks") or {}
        raw_split = data.get("split")
        return cls(
            name=str(data["name"]),
            intent=str(data["intent"]),
            payload=dict(data.get("payload") or {}),
            expected_family=str(expected.get("family") or ""),
            expected_archetypes=list(expected.get("archetypes") or []),
            actions=dict(data.get("actions") or {}),
            exclude=list(data.get("exclude") or []),
            replace=bool(data.get("replace", False)),
            mutable_paths=list(data.get("mutable_paths") or []),
            layout=list(data.get("layout") or []),
            title=data.get("title"),
            summary=data.get("summary"),
            required_blocks=list(blocks.get("required") or []),
            forbidden_blocks=list(blocks.get("forbidden") or []),
            expected_action_gate=str(expected.get("action_gate") or "pass"),
            split=str(raw_split).strip() if raw_split is not None else None,
            tags=list(data.get("tags") or []),
            verified=bool(data.get("verified", True)),
            source=data.get("source"),
        )


def _parse_tag_filter(tags: Sequence[str] | str | None) -> set[str]:
    """Normalizes an optional tag filter string or sequence into a set."""
    if isinstance(tags, str):
        return {t.strip() for t in tags.split(",") if t.strip()}
    if tags:
        return {str(t).strip() for t in tags if str(t).strip()}
    return set()


def _case_matches_filters(
    case: PlanCase,
    *,
    include_unverified: bool,
    target_tags: set[str],
    split: str | None,
) -> bool:
    """Checks whether `case` passes verification, tag, and split filters."""
    if not include_unverified and not case.verified:
        return False
    if target_tags and not any(t in target_tags for t in case.tags):
        return False
    if split is not None:
        wanted = split.strip().lower()
        if wanted == "heldout":
            return case.is_heldout
        if wanted == "tuning":
            return not case.is_heldout and (
                case.split is None or case.split.lower() == "tuning"
            )
        return (case.split or "").lower() == wanted
    return True


def load_plan_cases(
    cases_path: str | Path,
    *,
    tags: Sequence[str] | str | None = None,
    split: str | None = None,
    include_unverified: bool = False,
    schema_path: str | Path | None = None,
) -> list[PlanCase]:
    """Loads and validates decision plan cases from a decrypted YAML file."""
    path = Path(cases_path)
    if not path.is_file():
        raise FileNotFoundError(f"Decision cases file not found: {path}")
    if is_transcrypt_ciphertext(path):
        raise RuntimeError(
            f"Decision cases file {path} is encrypted with Transcrypt; unlock"
            " datasets with `eval/bin/transcrypt` first."
        )

    schema_file = Path(schema_path) if schema_path else path.parent / "schema.json"
    schema = (
        json.loads(schema_file.read_text(encoding="utf-8"))
        if schema_file.is_file()
        else None
    )
    raw_cases = yaml.safe_load(path.read_text(encoding="utf-8")) or []
    if schema is not None:
        jsonschema.validate(instance=raw_cases, schema=schema)

    target_tags = _parse_tag_filter(tags)
    parsed = [
        PlanCase.from_dict(item) for item in raw_cases if isinstance(item, Mapping)
    ]
    return [
        case
        for case in parsed
        if _case_matches_filters(
            case,
            include_unverified=include_unverified,
            target_tags=target_tags,
            split=split,
        )
    ]


def is_action_gate_blocked(output: SynthesisOutput) -> bool:
    """Checks whether the synthesis run was blocked by an action gate."""
    if output.stage is not SynthesisStage.ABSTAIN_FALLBACK:
        return False
    return any(
        record.question.id == "archetype_status" and record.outcome == "action_gate"
        for record in output.trace
    )


@dataclass(frozen=True)
class CaseEvaluation:
    """Evaluation metrics for a single synthesis execution."""

    output: SynthesisOutput
    wall_time_ms: float
    family_correct: bool
    archetype_correct: bool
    block_precision: float
    block_recall: float
    forbidden_block_count: int
    action_gate_outcome: str
    abstained: bool
    used_fallback: bool = False

    @property
    def host_overhead_ms(self) -> float:
        return max(0.0, self.wall_time_ms - self.output.decision_latency_ms)


@dataclass
class CaseScore:
    """Aggregated score for a case across repeats."""

    case: PlanCase
    repeats: int
    evaluations: list[CaseEvaluation] = field(default_factory=list)

    @property
    def family_accuracy(self) -> float:
        if not self.evaluations:
            return 0.0
        return sum(1.0 for e in self.evaluations if e.family_correct) / len(
            self.evaluations
        )

    @property
    def archetype_accuracy(self) -> float:
        if not self.evaluations:
            return 0.0
        return sum(1.0 for e in self.evaluations if e.archetype_correct) / len(
            self.evaluations
        )

    @property
    def block_precision(self) -> float:
        if not self.evaluations:
            return 0.0
        return sum(e.block_precision for e in self.evaluations) / len(self.evaluations)

    @property
    def block_recall(self) -> float:
        if not self.evaluations:
            return 0.0
        return sum(e.block_recall for e in self.evaluations) / len(self.evaluations)

    @property
    def abstain_rate(self) -> float:
        if not self.evaluations:
            return 0.0
        return sum(1.0 for e in self.evaluations if e.abstained) / len(self.evaluations)

    @property
    def fallback_rate(self) -> float:
        if not self.evaluations:
            return 0.0
        return sum(1.0 for e in self.evaluations if e.used_fallback) / len(
            self.evaluations
        )

    @property
    def action_gate_match(self) -> bool:
        if not self.evaluations:
            return False
        expected = self.case.expected_action_gate
        if expected == "any":
            return True
        return all(e.action_gate_outcome == expected for e in self.evaluations)

    @property
    def decision_latency_p50(self) -> float:
        latencies = sorted(e.output.decision_latency_ms for e in self.evaluations)
        return _percentile(latencies, 0.50)

    @property
    def decision_latency_p95(self) -> float:
        latencies = sorted(e.output.decision_latency_ms for e in self.evaluations)
        return _percentile(latencies, 0.95)

    @property
    def host_overhead_p50(self) -> float:
        overheads = sorted(e.host_overhead_ms for e in self.evaluations)
        return _percentile(overheads, 0.50)

    @property
    def host_overhead_p95(self) -> float:
        overheads = sorted(e.host_overhead_ms for e in self.evaluations)
        return _percentile(overheads, 0.95)

    @property
    def round_trips_per_surface(self) -> float:
        if not self.evaluations:
            return 0.0
        return sum(e.output.decision_calls for e in self.evaluations) / len(
            self.evaluations
        )

    @property
    def mean_decision_calls(self) -> float:
        return self.round_trips_per_surface


def _percentile(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        return 0.0
    idx = (len(sorted_values) - 1) * q
    low = int(math.floor(idx))
    high = int(math.ceil(idx))
    if low == high:
        return sorted_values[low]
    weight = idx - low
    return sorted_values[low] * (1.0 - weight) + sorted_values[high] * weight


def brier_score(pairs: Sequence[tuple[float, float]]) -> float:
    """Computes the mean squared error (Brier score) for probability pairs."""
    if not pairs:
        return 0.0
    return sum((p - y) ** 2 for p, y in pairs) / len(pairs)


def _bin_bounds_match(p: float, low: float, high: float, is_last: bool) -> bool:
    """Returns whether probability `p` falls inside `[low, high)` (or `[low, high]`)."""
    if is_last:
        return low <= p <= high or math.isclose(p, high)
    return low <= p < high


def expected_calibration_error(
    pairs: Sequence[tuple[float, float]], num_bins: int = 10
) -> float:
    """Computes Expected Calibration Error (ECE) across `num_bins` equal bins."""
    if not pairs or num_bins <= 0:
        return 0.0
    bin_size = 1.0 / num_bins
    total = len(pairs)
    ece = 0.0
    for b in range(num_bins):
        low = b * bin_size
        high = (b + 1) * bin_size
        in_bin = [
            (p, y)
            for p, y in pairs
            if _bin_bounds_match(p, low, high, b == num_bins - 1)
        ]
        if not in_bin:
            continue
        bin_count = len(in_bin)
        avg_conf = sum(p for p, _ in in_bin) / bin_count
        avg_acc = sum(y for _, y in in_bin) / bin_count
        ece += (bin_count / total) * abs(avg_acc - avg_conf)
    return ece


TEMPERATURE_GRID = (
    0.25,
    0.50,
    0.75,
    1.00,
    1.25,
    1.50,
    1.75,
    2.00,
    2.25,
    2.50,
    2.75,
    3.00,
    3.25,
    3.50,
    3.75,
    4.00,
)


def _rescale_binary_probability(p: float, cal: Calibration) -> float:
    """Applies `Calibration` temperature scaling to a binary probability `p`."""
    p_clamped = min(max(p, 1e-6), 1.0 - 1e-6)
    scaled = cal.apply({"pos": p_clamped, "neg": 1.0 - p_clamped})
    return scaled["pos"]


def fit_temperature(
    calibration_data: Sequence[tuple[float, float]],
    candidates: Sequence[float] | None = None,
) -> float:
    """Finds temperature `T` in `candidates` that minimizes Expected Calibration Error."""
    if not calibration_data:
        return 1.0
    grid = candidates or TEMPERATURE_GRID
    best_temp = 1.0
    best_ece = float("inf")

    for temp in grid:
        if temp <= 0:
            continue
        cal = Calibration(temperature=float(temp))
        rescaled = [
            (_rescale_binary_probability(p, cal), y) for p, y in calibration_data
        ]
        ece = expected_calibration_error(rescaled)
        if ece < best_ece or (math.isclose(ece, best_ece) and temp == 1.0):
            best_ece = ece
            best_temp = float(temp)

    return best_temp


@dataclass
class BenchmarkSummary:
    """Complete summary of a golden plan benchmark run."""

    cases_count: int
    total_runs: int
    backend_kind: str
    model_id: str
    family_accuracy: float
    archetype_accuracy: float
    block_precision: float
    block_recall: float
    abstain_rate: float
    fallback_rate: float
    round_trips_per_surface: float
    decision_latency_p50: float
    decision_latency_p95: float
    host_overhead_p50: float
    host_overhead_p95: float
    decision_calls_mean: float
    brier_family: float
    ece_family: float
    brier_archetype: float
    ece_archetype: float
    brier_gates: float
    ece_gates: float
    fitted_temperature: float | None = None
    case_scores: list[CaseScore] = field(default_factory=list)


def _observed_family_and_prob(
    answers_by_id: Mapping[str, DecisionAnswer], output: SynthesisOutput
) -> tuple[str, float]:
    """Extracts `(observed_family, probability)` from answers or `layout_plan`."""
    if "family" in answers_by_id:
        return answers_by_id["family"].top()
    if output.layout_plan is not None:
        return output.layout_plan.family, 1.0
    return "", 1.0


def _observed_archetype_and_prob(
    answers_by_id: Mapping[str, DecisionAnswer],
    output: SynthesisOutput,
    observed_family: str,
) -> tuple[str, float]:
    """Extracts `(observed_archetype, probability)` from `layout_plan` or answers."""
    if output.layout_plan is not None and output.layout_plan.archetype:
        arch_ans = answers_by_id.get(f"archetype:{output.layout_plan.family}")
        prob = arch_ans.top()[1] if arch_ans is not None else output.confidence
        return output.layout_plan.archetype, prob
    fam_arch_id = f"archetype:{observed_family}"
    if fam_arch_id in answers_by_id:
        return answers_by_id[fam_arch_id].top()
    for qid, ans in answers_by_id.items():
        if qid.startswith("archetype:"):
            return ans.top()
    return "", 1.0


def _block_precision_and_recall(
    case: PlanCase, output: SynthesisOutput
) -> tuple[float, float, int]:
    """Computes `(precision, recall, forbidden_count)` for planned blocks."""
    observed = (
        {b.block_id for b in output.layout_plan.blocks}
        if output.layout_plan is not None
        else set()
    )
    required_set = set(case.required_blocks)
    forbidden_set = set(case.forbidden_blocks)
    if required_set:
        recall = len(required_set & observed) / len(required_set)
        precision = len(required_set & observed) / len(observed) if observed else 0.0
    else:
        recall = 1.0
        precision = 1.0
    forbidden_count = len(forbidden_set & observed)
    if forbidden_count > 0:
        precision = max(0.0, precision - (forbidden_count * 0.25))
    return precision, recall, forbidden_count


def evaluate_single_run(
    case: PlanCase, output: SynthesisOutput, wall_time_ms: float
) -> tuple[CaseEvaluation, list[tuple[str, float, float]]]:
    """Evaluates one synthesis output against `case` expectations."""
    answers_by_id = {ans.question_id: ans for ans in output.decision_answers}
    observed_family, family_prob = _observed_family_and_prob(answers_by_id, output)
    family_correct = observed_family.lower() == case.expected_family.lower()

    observed_archetype, archetype_prob = _observed_archetype_and_prob(
        answers_by_id, output, observed_family
    )
    archetype_correct = observed_archetype.lower() in {
        a.lower() for a in case.expected_archetypes
    }

    precision, recall, forbidden_count = _block_precision_and_recall(case, output)
    is_blocked = is_action_gate_blocked(output)
    action_gate_outcome = "abstain" if is_blocked else "pass"
    used_fallback = output.fallback_source is not None
    abstained = output.stage is SynthesisStage.ABSTAIN_FALLBACK or not output.messages

    eval_result = CaseEvaluation(
        output=output,
        wall_time_ms=wall_time_ms,
        family_correct=family_correct,
        archetype_correct=archetype_correct,
        block_precision=precision,
        block_recall=recall,
        forbidden_block_count=forbidden_count,
        action_gate_outcome=action_gate_outcome,
        abstained=abstained,
        used_fallback=used_fallback,
    )

    required_set = set(case.required_blocks)
    cal_points: list[tuple[str, float, float]] = [
        ("family", family_prob, 1.0 if family_correct else 0.0),
        ("archetype", archetype_prob, 1.0 if archetype_correct else 0.0),
    ]
    for qid, ans in answers_by_id.items():
        if qid.startswith("gate:"):
            block_id = qid.removeprefix("gate:")
            cal_points.append(
                ("gate", ans.yes, 1.0 if block_id in required_set else 0.0)
            )

    return eval_result, cal_points


class _BackendCallRecorder(DecisionBackend):
    """Wraps a `DecisionBackend` to retain answers and timings across abstentions."""

    def __init__(self, inner: DecisionBackend) -> None:
        self._inner = inner
        self.answers: list[DecisionAnswer] = []
        self.calls: int = 0
        self.latency_ms: float = 0.0

    @property
    def backend_kind(self) -> str:
        return self._inner.backend_kind

    @property
    def model_id(self) -> str:
        return self._inner.model_id

    @property
    def capabilities(self) -> BackendCapabilities:
        return self._inner.capabilities

    async def answer(self, batch: DecisionBatch) -> tuple[DecisionAnswer, ...]:
        t0 = time.perf_counter()
        result = await self._inner.answer(batch)
        self.latency_ms += (time.perf_counter() - t0) * 1000.0
        self.calls += 1
        self.answers.extend(result)
        return result

    async def aclose(self) -> None:
        await self._inner.aclose()

    async def aclose_current_loop(self) -> None:
        await self._inner.aclose_current_loop()

    async def __aenter__(self) -> _BackendCallRecorder:
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.aclose()


def _abstained_compilation_output(
    recorder: _BackendCallRecorder, exc: A2uiCompilationError
) -> SynthesisOutput:
    """Builds an abstained `SynthesisOutput` when `synthesize` had no fallback."""
    trace: tuple[DecisionRecord, ...] = ()
    if "(action_gate)" in str(exc):
        trace = (
            DecisionRecord(
                question=DecisionQuestion(
                    id="archetype_status",
                    kind=QuestionKind.BOOLEAN,
                    prompt="Action confidence gate",
                    allow_none=False,
                ),
                outcome="action_gate",
            ),
        )
    return SynthesisOutput(
        stage=SynthesisStage.ABSTAIN_FALLBACK,
        messages=[],
        confidence=0.0,
        entropy=0.0,
        fallback_reason=str(exc),
        backend_kind=recorder.backend_kind,
        model_id=recorder.model_id,
        trace=trace,
        decision_latency_ms=recorder.latency_ms,
        decision_calls=recorder.calls,
        decision_answers=tuple(recorder.answers),
    )


async def _synthesize_plan_case(
    synth: DecisionSynthesizer, case: PlanCase
) -> tuple[SynthesisOutput, float]:
    """Runs `synth.synthesize` on `case` and returns `(output, wall_time_ms)`."""
    original_backend = synth.backend
    recorder = _BackendCallRecorder(original_backend)
    synth.backend = recorder
    start = time.perf_counter()
    try:
        output = await synth.synthesize(
            case.intent,
            case.payload,
            actions=case.action_specs(),
            exclude=case.exclude,
            mutable_paths=case.mutable_paths,
            layout=case.layout or None,
            title=case.title,
            summary=case.summary,
        )
        if not output.decision_answers and recorder.answers:
            output = replace(output, decision_answers=tuple(recorder.answers))
    except A2uiCompilationError as exc:
        output = _abstained_compilation_output(recorder, exc)
    finally:
        synth.backend = original_backend
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return output, elapsed_ms


async def run_case(
    synth: DecisionSynthesizer,
    case: PlanCase,
    *,
    repeat: int = 1,
) -> CaseScore:
    """Executes a decision synthesis case `repeat` times and scores results."""
    score = CaseScore(case=case, repeats=repeat)
    for _ in range(repeat):
        output, elapsed_ms = await _synthesize_plan_case(synth, case)
        evaluation, _ = evaluate_single_run(case, output, elapsed_ms)
        score.evaluations.append(evaluation)
    return score


def _empty_benchmark_summary(synth: DecisionSynthesizer) -> BenchmarkSummary:
    """Returns a zeroed `BenchmarkSummary` when no cases were run."""
    return BenchmarkSummary(
        cases_count=0,
        total_runs=0,
        backend_kind=synth.backend.backend_kind,
        model_id=synth.backend.model_id,
        family_accuracy=0.0,
        archetype_accuracy=0.0,
        block_precision=0.0,
        block_recall=0.0,
        abstain_rate=0.0,
        fallback_rate=0.0,
        round_trips_per_surface=0.0,
        decision_latency_p50=0.0,
        decision_latency_p95=0.0,
        host_overhead_p50=0.0,
        host_overhead_p95=0.0,
        decision_calls_mean=0.0,
        brier_family=0.0,
        ece_family=0.0,
        brier_archetype=0.0,
        ece_archetype=0.0,
        brier_gates=0.0,
        ece_gates=0.0,
    )


async def _run_case_with_calibration(
    synth: DecisionSynthesizer, case: PlanCase, repeat: int
) -> tuple[CaseScore, list[tuple[str, float, float]]]:
    """Executes `case` `repeat` times and returns `(score, calibration_records)`."""
    score = CaseScore(case=case, repeats=repeat)
    records: list[tuple[str, float, float]] = []
    for _ in range(repeat):
        output, elapsed_ms = await _synthesize_plan_case(synth, case)
        evaluation, cal_points = evaluate_single_run(case, output, elapsed_ms)
        score.evaluations.append(evaluation)
        records.extend(cal_points)
    return score, records


def _group_calibration_pairs(
    records: Sequence[tuple[str, float, float]],
) -> dict[str, list[tuple[float, float]]]:
    """Groups `(kind, prob, target)` records by `kind`."""
    grouped: dict[str, list[tuple[float, float]]] = {
        "family": [],
        "archetype": [],
        "gate": [],
    }
    for kind, prob, target in records:
        if kind in grouped:
            grouped[kind].append((prob, target))
    return grouped


def _summarize_evaluations(
    synth: DecisionSynthesizer,
    cases_count: int,
    case_scores: list[CaseScore],
    all_evaluations: list[CaseEvaluation],
    calibration_records: Sequence[tuple[str, float, float]],
    *,
    fit_temp: bool,
) -> BenchmarkSummary:
    """Computes `BenchmarkSummary` metrics over `all_evaluations`."""
    total_runs = len(all_evaluations)
    latencies = sorted(e.output.decision_latency_ms for e in all_evaluations)
    overheads = sorted(e.host_overhead_ms for e in all_evaluations)
    round_trips = sum(e.output.decision_calls for e in all_evaluations) / total_runs
    fam_hits = sum(1.0 for e in all_evaluations if e.family_correct)
    arch_hits = sum(1.0 for e in all_evaluations if e.archetype_correct)
    abstains = sum(1.0 for e in all_evaluations if e.abstained)
    fallbacks = sum(1.0 for e in all_evaluations if e.used_fallback)

    grouped = _group_calibration_pairs(calibration_records)
    family_pairs = grouped["family"]
    arch_pairs = grouped["archetype"]
    gate_pairs = grouped["gate"]
    fitted_t = (
        fit_temperature(family_pairs + arch_pairs + gate_pairs) if fit_temp else None
    )

    return BenchmarkSummary(
        cases_count=cases_count,
        total_runs=total_runs,
        backend_kind=synth.backend.backend_kind,
        model_id=synth.backend.model_id,
        family_accuracy=fam_hits / total_runs,
        archetype_accuracy=arch_hits / total_runs,
        block_precision=sum(e.block_precision for e in all_evaluations) / total_runs,
        block_recall=sum(e.block_recall for e in all_evaluations) / total_runs,
        abstain_rate=abstains / total_runs,
        fallback_rate=fallbacks / total_runs,
        round_trips_per_surface=round_trips,
        decision_latency_p50=_percentile(latencies, 0.50),
        decision_latency_p95=_percentile(latencies, 0.95),
        host_overhead_p50=_percentile(overheads, 0.50),
        host_overhead_p95=_percentile(overheads, 0.95),
        decision_calls_mean=round_trips,
        brier_family=brier_score(family_pairs),
        ece_family=expected_calibration_error(family_pairs),
        brier_archetype=brier_score(arch_pairs),
        ece_archetype=expected_calibration_error(arch_pairs),
        brier_gates=brier_score(gate_pairs),
        ece_gates=expected_calibration_error(gate_pairs),
        fitted_temperature=fitted_t,
        case_scores=case_scores,
    )


async def run_benchmark(
    synth: DecisionSynthesizer,
    cases: Sequence[PlanCase],
    *,
    repeat: int = 3,
    fit_temp: bool = False,
) -> BenchmarkSummary:
    """Runs the benchmark across `cases` and summarizes accuracy, latency, and calibration."""
    case_scores: list[CaseScore] = []
    all_evaluations: list[CaseEvaluation] = []
    calibration_records: list[tuple[str, float, float]] = []

    for case in cases:
        score, records = await _run_case_with_calibration(synth, case, repeat)
        case_scores.append(score)
        all_evaluations.extend(score.evaluations)
        calibration_records.extend(records)

    if not all_evaluations:
        return _empty_benchmark_summary(synth)

    return _summarize_evaluations(
        synth,
        len(cases),
        case_scores,
        all_evaluations,
        calibration_records,
        fit_temp=fit_temp,
    )


def _format_case_row(cs: CaseScore) -> str:
    """Formats one per-case row in the Markdown benchmark report."""
    fam_mark = "pass" if cs.family_accuracy >= 1.0 else f"{cs.family_accuracy:.0%}"
    arch_mark = (
        "pass" if cs.archetype_accuracy >= 1.0 else f"{cs.archetype_accuracy:.0%}"
    )
    return (
        f"| `{cs.case.name}` | {fam_mark} | {arch_mark} |"
        f" {cs.block_recall:.0%} | {cs.block_precision:.0%} |"
        f" {cs.round_trips_per_surface:.2f} | {cs.decision_latency_p50:.1f} |"
        f" {cs.fallback_rate:.0%} |"
    )


def render_markdown(summary: BenchmarkSummary, backend_name: str | None = None) -> str:
    """Renders a Markdown report from `summary`."""
    header_label = backend_name or f"{summary.backend_kind}/{summary.model_id}"
    profile_key = f"{summary.backend_kind}/{summary.model_id}"
    lines: list[str] = [
        f"## Decision model benchmark summary: `{header_label}`",
        "",
        f"- Backend: `{summary.backend_kind}` (model `{summary.model_id}`)",
        f"- Cases evaluated: {summary.cases_count} ({summary.total_runs} runs)",
        f"- Family accuracy: {summary.family_accuracy * 100.0:.1f}%",
        f"- Archetype accuracy: {summary.archetype_accuracy * 100.0:.1f}%",
        f"- Block precision: {summary.block_precision * 100.0:.1f}%",
        f"- Block recall: {summary.block_recall * 100.0:.1f}%",
        f"- Fallback rate: {summary.fallback_rate * 100.0:.1f}%",
        f"- Abstain rate: {summary.abstain_rate * 100.0:.1f}%",
        "",
        "### Latency and round trips",
        "",
        (
            f"- Decision latency (p50 / p95): {summary.decision_latency_p50:.1f} ms"
            f" / {summary.decision_latency_p95:.1f} ms"
        ),
        (
            f"- Host overhead (p50 / p95): {summary.host_overhead_p50:.1f} ms /"
            f" {summary.host_overhead_p95:.1f} ms"
        ),
        f"- Round trips per surface: {summary.round_trips_per_surface:.2f}",
        "",
        "### Calibration",
        "",
        "| Question kind | Brier score | ECE (10-bin) |",
        "| :--- | :--- | :--- |",
        f"| Family | {summary.brier_family:.4f} | {summary.ece_family:.4f} |",
        f"| Archetype | {summary.brier_archetype:.4f} | {summary.ece_archetype:.4f} |",
        f"| Gates | {summary.brier_gates:.4f} | {summary.ece_gates:.4f} |",
    ]
    if summary.fitted_temperature is not None:
        profile_json = json.dumps(
            {profile_key: {"temperature": summary.fitted_temperature}}
        )
        lines.extend([
            "",
            "### Fitted calibration profile",
            "",
            f"Profile entry: `{profile_json}`",
        ])

    lines.extend([
        "",
        "### Case breakdown",
        "",
        (
            "| Case | Family | Archetype | Recall | Prec | Round trips |"
            " Backend ms | Fallback |"
        ),
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
        *(_format_case_row(cs) for cs in summary.case_scores),
    ])
    return "\n".join(lines)
