# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Shared contracts for the format playground: modes, settings, and metrics.

This module has no ADK or A2A imports so the renderers, commands, and tests can
depend on it without pulling in a model runtime.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any, Protocol

from a2a.types import Part


class RenderMode(str, Enum):
    """Inference mode a turn is rendered through."""

    DECISION = "decision"
    EXPRESS = "express"
    DIRECT = "direct"
    TOOL = "tool"

    def __str__(self) -> str:
        return self.value


CHAT_MODES: tuple[RenderMode, ...] = (
    RenderMode.DECISION,
    RenderMode.EXPRESS,
    RenderMode.DIRECT,
)


@dataclass(frozen=True)
class PlaygroundSettings:
    """Per-conversation state the slash commands change."""

    mode: RenderMode = RenderMode.DECISION
    compare: bool = False
    """When True, the next render goes through all three modes, then resets."""
    turn: int = 0
    """Number of render requests so far; drives the per-turn surface ids."""

    def next_turn(self) -> PlaygroundSettings:
        """The settings for the next render request."""
        return replace(self, turn=self.turn + 1)


@dataclass(frozen=True)
class LlmTurn:
    """One model call: its text and what it cost."""

    text: str
    prompt_tokens: int = 0
    output_tokens: int = 0
    latency_ms: float = 0.0


@dataclass(frozen=True)
class LlmToolTurn:
    """One model call that invokes a tool: the tool called and what it cost."""

    tool_name: str | None = None
    arguments: dict[str, Any] = field(default_factory=dict)
    result: Any = None
    prompt_tokens: int = 0
    output_tokens: int = 0
    latency_ms: float = 0.0
    text: str = ""


class LlmCaller(Protocol):
    """Runs one instruction-plus-message turn against a language model.

    `instruction_key` identifies the system instruction so an implementation
    can keep one configured agent per key. `session_id=None` means a fresh
    session with no history (stage 2); a string keeps conversation history
    under that id (stage 1).
    """

    async def __call__(
        self,
        *,
        instruction_key: str,
        instruction: str,
        user_text: str,
        session_id: str | None,
    ) -> LlmTurn:
        ...

    async def call_with_tools(
        self,
        *,
        instruction_key: str,
        instruction: str,
        user_text: str,
        session_id: str | None,
        tools: Sequence[Any],
    ) -> LlmToolTurn:
        ...


def format_decision_ms(ms: float) -> str:
    """Formats a decision-model latency in milliseconds (or seconds when >= 1000 ms)."""
    if ms < 1.0:
        return f"{ms:.2f} ms"
    if ms < 1000.0:
        return f"{ms:.0f} ms"
    return f"{ms / 1000.0:.2f} s ({ms:.0f} ms)"


@dataclass(frozen=True)
class TurnMetrics:
    """What one render cost, shown in the footer under every surface."""

    mode: RenderMode
    backend: str
    """Decision `backend_kind` (`mock`, `jev`, `step0`, ...) or `llm` for stage-2 modes."""
    llm_calls: int
    latency_ms: float
    prompt_tokens: int = 0
    output_tokens: int = 0
    archetype: str | None = None
    confidence: float | None = None
    block_count: int | None = None
    fallback: str | None = None
    """A `FallbackSource` value, or `validation_retry` when stage 2 needed a retry."""
    fallback_reason: str | None = None
    """Why the decision fallback chain ran (`SynthesisOutput.fallback_reason`)."""
    stage: str | None = None
    """The `SynthesisStage` value (`template_select`, `archetype`, `abstain_fallback`)."""
    decision_latency_ms: float | None = None
    """Wall-clock milliseconds spent waiting on the decision model."""
    decision_calls: int | None = None
    """Number of decision model `answer()` calls."""
    template_id: str | None = None
    """Registered template id when `stage == 'template_select'`."""
    model_id: str | None = None
    """The decision backend's pinned model id."""
    operation: str | None = None
    """The `SurfaceOperation` the decision render emitted."""
    cache_hit: bool = False
    """Whether a cached plan answered without a decision backend call."""
    fit_score: float | None = None
    """How well the chosen archetype fits the payload (`LayoutPlan.fit_score`)."""

    def path_summary(self) -> str | None:
        """Describes whether rendering used the decision model, a template, or a fallback."""
        conf = f" ({self.confidence:.2f})" if self.confidence is not None else ""
        if self.stage == "template_select":
            tpl = f" {self.template_id}" if self.template_id else ""
            return f"path: templates · template{tpl}{conf}"
        if self.stage == "archetype":
            arch = f"archetype {self.archetype}" if self.archetype else "archetype"
            if self.block_count is None:
                return f"path: decision model · {arch}{conf}"
            blocks = f"{self.block_count} block{'' if self.block_count == 1 else 's'}"
            return f"path: decision model · {arch}{conf} · {blocks}"
        if self.stage == "abstain_fallback":
            source = self.fallback or "abstained"
            reason = f": {self.fallback_reason}" if self.fallback_reason else ""
            return f"path: fallback ({source}) · decision model abstained{reason}"
        return None

    def backend_label(self) -> str:
        """`backend_kind` plus the pinned model id when there is one."""
        if self.model_id and self.model_id != self.backend:
            return f"{self.backend} ({self.model_id})"
        return self.backend

    def _timing(self) -> str:
        if self.decision_latency_ms is None:
            return f"{self.latency_ms / 1000.0:.1f} s"
        dec_time = f"decision {format_decision_ms(self.decision_latency_ms)}"
        if self.llm_calls > 0:
            return f"{dec_time} (total {self.latency_ms / 1000.0:.1f} s)"
        return dec_time

    def _call_label(self) -> str:
        plural = "" if self.llm_calls == 1 else "s"
        if self.mode == RenderMode.TOOL and self.llm_calls > 0:
            return f"{self.llm_calls} LLM call{plural} (tool call{plural} only)"
        return f"{self.llm_calls} LLM call{plural}"

    def footer(self) -> str:
        """One line summarizing the turn, fields separated by ` · `."""
        parts: list[str] = [
            f"**{self.mode}**",
            self.backend_label(),
            self._call_label(),
            self._timing(),
        ]
        if self.prompt_tokens or self.output_tokens:
            parts.append(f"{self.prompt_tokens} in / {self.output_tokens} out tokens")
        if self.operation and self.operation != "create":
            parts.append(f"surface {self.operation}")
        if self.cache_hit:
            parts.append("cache hit")
        if self.fit_score is not None:
            parts.append(f"fit {self.fit_score:.2f}")
        path = self.path_summary()
        parts.extend([path] if path is not None else self._plan_parts())
        return " · ".join(parts)

    def _plan_parts(self) -> list[str]:
        """Archetype, block count, and fallback when no decision stage is known."""
        parts: list[str] = []
        if self.archetype is not None:
            conf = f" ({self.confidence:.2f})" if self.confidence is not None else ""
            parts.append(f"archetype {self.archetype}{conf}")
        if self.block_count is not None:
            parts.append(
                f"{self.block_count} block{'' if self.block_count == 1 else 's'}"
            )
        if self.fallback is not None:
            parts.append(f"fallback: {self.fallback}")
        return parts

    def plus_stage1(self, stage1: LlmTurn) -> TurnMetrics:
        """Folds the stage-1 call into these metrics."""
        return replace(
            self,
            llm_calls=self.llm_calls + 1,
            latency_ms=self.latency_ms + stage1.latency_ms,
            prompt_tokens=self.prompt_tokens + stage1.prompt_tokens,
            output_tokens=self.output_tokens + stage1.output_tokens,
        )


@dataclass(frozen=True)
class RenderResult:
    """The outcome of rendering one block through one mode."""

    parts: list[Part] = field(default_factory=list)
    """A2UI data parts for the rendered surface; empty when `error` is set."""
    metrics: TurnMetrics | None = None
    trace_lines: tuple[str, ...] = ()
    """Decision only: one line per planner question, for `/trace`."""
    error: str | None = None
    """Why rendering failed, if it did."""
    decision_answers: tuple[Any, ...] = ()
    output: Any = None

    @property
    def ok(self) -> bool:
        """Whether the render produced a surface."""
        return self.error is None and bool(self.parts)


MAIN_SURFACE_ID = "main"
"""The surface decision and tool renders reuse, so later turns update it in place."""


def surface_id_for(settings: PlaygroundSettings, mode: RenderMode | None = None) -> str:
    """The surface id a render of `mode` targets.

    Decision and tool renders reuse `MAIN_SURFACE_ID` across turns. Express and
    direct renders, which a model writes from scratch, and every surface of a
    `/compare` turn get per-turn ids.
    """
    base = f"turn-{settings.turn}"
    if settings.compare:
        return f"{base}-{mode}" if mode is not None else base
    if mode in (RenderMode.DECISION, RenderMode.TOOL):
        return MAIN_SURFACE_ID
    return base


def status_surface_id(settings: PlaygroundSettings, suffix: str) -> str:
    """The per-turn id of a status or footer surface."""
    return f"turn-{settings.turn}-{suffix}"


def as_mode(value: Any) -> RenderMode | None:
    """Parses a user-typed mode name, or None when it is not one."""
    try:
        return RenderMode(str(value).strip().lower())
    except ValueError:
        return None
