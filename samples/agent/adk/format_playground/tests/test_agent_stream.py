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

"""End-to-end tests for `PlaygroundAgent.stream` with a scripted model."""

from __future__ import annotations

import json
from typing import Any

from a2a.types import DataPart, Part, TextPart
import pytest

from a2ui.inference_formats.decision import DeterministicMockBackend
from agent import PlaygroundAgent
from fakes import FakeLlmCaller
from modes import LlmToolTurn, RenderMode
from renderers import Renderers
from stage1 import UI_ACTION_PREFIX
from test_renderers import EXPRESS_BAD, EXPRESS_OK, RESORT_PAYLOAD, direct_json_reply

pytestmark = pytest.mark.asyncio

STAGE1_BLOCK = (
    "<a2ui>"
    + json.dumps({"intent": "Show the resort overview", "payload": RESORT_PAYLOAD})
    + "</a2ui>"
)


def messages_of(parts: list[Part]) -> list[dict[str, Any]]:
    return [p.root.data for p in parts if isinstance(p.root, DataPart)]


def surface_ids(parts: list[Part]) -> list[str]:
    ids: list[str] = []
    for msg in messages_of(parts):
        create = msg.get("createSurface")
        if isinstance(create, dict):
            ids.append(str(create["surfaceId"]))
    return ids


def _component_text(comp: dict[str, Any]) -> str | None:
    """The `text` of a `Text` component in either the v0.9 or v1.0 shape."""
    kind = comp.get("component")
    props = kind.get("Text") if isinstance(kind, dict) else comp
    text = props.get("text") if isinstance(props, dict) else None
    return text if isinstance(text, str) else None


def _components_of(msg: dict[str, Any]) -> list[dict[str, Any]]:
    """Components carried by a `createSurface` (v1.0) or `updateComponents` message."""
    found: list[dict[str, Any]] = []
    for key in ("createSurface", "updateComponents"):
        body = msg.get(key)
        if isinstance(body, dict):
            found.extend(body.get("components", []))
    return found


def status_text(parts: list[Part]) -> str:
    """Joins every `Text` literal in the parts, in order."""
    texts = [
        text
        for msg in messages_of(parts)
        for comp in _components_of(msg)
        if (text := _component_text(comp)) is not None
    ]
    return "\n".join(texts)


async def collect(
    agent: PlaygroundAgent, query: str, session: str, version: str | None = "0.9"
) -> list[Part]:
    items = [item async for item in agent.stream(query, session, version)]
    assert len(items) == 1 and items[0]["is_task_complete"] is True
    return items[0]["parts"]


@pytest.fixture
def agent(scripted_llm: FakeLlmCaller) -> PlaygroundAgent:
    renderers = Renderers(versions=["0.9", "1.0"], backend=DeterministicMockBackend())
    return PlaygroundAgent(
        "http://localhost:10010",
        llm=scripted_llm,
        renderers=renderers,
        model_name="fake-model",
    )


async def test_agent_card_advertises_both_versions(agent: PlaygroundAgent) -> None:
    uris = [e.uri for e in agent.agent_card.capabilities.extensions or []]
    assert len(uris) == 2
    assert agent.agent_card.skills[0].id == "render_ui"


async def test_help_mode_and_trace_before_any_render(
    agent: PlaygroundAgent, monkeypatch: pytest.MonkeyPatch
) -> None:
    help_parts = await collect(agent, "/help", "s1")
    assert surface_ids(help_parts) == ["turn-0-status"]
    text = status_text(help_parts)
    assert "/compare" in text and "`mock`" in text and "`fake-model`" in text
    assert "/backend" in text

    mode_parts = await collect(agent, "/mode express", "s1")
    assert "Mode set to `express`" in status_text(mode_parts)
    unknown = await collect(agent, "/sparkle", "s1")
    assert "Unknown command `/sparkle`" in status_text(unknown)

    monkeypatch.setenv("JEV_API_KEY", "vcp_test_key")
    backend_parts = await collect(agent, "/backend jev", "s1")
    assert "Decision backend set to `jev`." in status_text(backend_parts)
    nameless = await collect(agent, "/backend", "s1")
    assert "`/backend` needs a backend name" in status_text(nameless)
    remix = await collect(agent, "/remix", "s1")
    assert "Unknown command `/remix`" in status_text(remix)
    reset_parts = await collect(agent, "/backend mock", "s1")
    assert "Decision backend set to `mock`." in status_text(reset_parts)

    trace_parts = await collect(agent, "/trace", "s1")
    assert "No decision turn yet" in status_text(trace_parts)

    text_only = await collect(agent, "hello", "s1", version=None)
    assert isinstance(text_only[0].root, TextPart)


async def test_decision_turn_renders_surface_footer_and_trace(
    agent: PlaygroundAgent, scripted_llm: FakeLlmCaller
) -> None:
    scripted_llm.script("stage1", STAGE1_BLOCK)
    parts = await collect(agent, "I want to see the resort", "s2", "1.0")
    ids = surface_ids(parts)
    assert ids == ["main", "turn-1-decision-footer"]
    footer = status_text(parts)
    assert "**decision** · mock · 1 LLM call" in footer
    assert "archetype" in footer
    stage1_call = scripted_llm.calls_for("stage1")[-1]
    assert stage1_call["session_id"] == "s2"
    assert "Format playground" in str(stage1_call["instruction"])

    trace = await collect(agent, "/trace", "s2", "1.0")
    assert "`family`" in status_text(trace)
    assert "**Backend**: `mock`" in status_text(trace)


async def test_decision_mode_needs_a_v10_client(
    agent: PlaygroundAgent, scripted_llm: FakeLlmCaller
) -> None:
    scripted_llm.script("stage1", STAGE1_BLOCK)
    parts = await collect(agent, "I want to see the resort", "s2-v09", "0.9")
    assert surface_ids(parts) == ["turn-1-decision-footer"]
    assert "v1.0 only" in status_text(parts)


async def test_second_turn_updates_the_main_surface_in_place(
    agent: PlaygroundAgent, scripted_llm: FakeLlmCaller
) -> None:
    different_shape_block = (
        "<a2ui>"
        + json.dumps(
            {"intent": "Check host health", "payload": {"host": "prod-1", "cpu": 0.78}}
        )
        + "</a2ui>"
    )
    scripted_llm.script("stage1", STAGE1_BLOCK, STAGE1_BLOCK, different_shape_block)
    first = await collect(agent, "show the resort", "s-reuse", "1.0")
    second = await collect(agent, "show it again", "s-reuse", "1.0")
    third = await collect(agent, "check host health", "s-reuse", "1.0")
    assert surface_ids(first) == ["main", "turn-1-decision-footer"]
    # Turn 2 has the same shape, so PlanCache emits only updateDataModel (`cache hit`).
    assert surface_ids(second) == ["turn-2-decision-footer"]
    data_updates = [
        m
        for m in messages_of(second)
        if m.get("updateDataModel", {}).get("surfaceId") == "main"
    ]
    assert data_updates
    assert not [
        m
        for m in messages_of(second)
        if m.get("updateComponents", {}).get("surfaceId") == "main"
    ]
    assert "cache hit" in status_text(second)
    # Turn 3 changes payload shape, so it re-plans and emits updateComponents (`update_root`).
    assert surface_ids(third) == ["turn-3-decision-footer"]
    comp_updates = [
        m
        for m in messages_of(third)
        if m.get("updateComponents", {}).get("surfaceId") == "main"
    ]
    assert comp_updates
    assert "surface update_root" in status_text(third)
    assert "INTEGRITY_ERROR" not in json.dumps(messages_of(third))


async def test_express_turn_uses_fresh_stage2_session(
    agent: PlaygroundAgent, scripted_llm: FakeLlmCaller
) -> None:
    await collect(agent, "/mode express", "s3")
    scripted_llm.script("stage1", STAGE1_BLOCK)
    scripted_llm.script("express:0.9", EXPRESS_OK.replace("turn-1", "turn-1"))
    parts = await collect(agent, "show the resort", "s3")
    assert surface_ids(parts) == ["turn-1", "turn-1-express-footer"]
    assert "**express** · llm · 2 LLM calls" in status_text(parts)
    stage2_call = scripted_llm.calls_for("express:0.9")[-1]
    assert stage2_call["session_id"] is None
    assert "Casa Verde" in str(stage2_call["user_text"])


async def test_compare_renders_three_surfaces_and_resets(
    agent: PlaygroundAgent, scripted_llm: FakeLlmCaller
) -> None:
    await collect(agent, "/compare", "s4", "1.0")
    scripted_llm.script("stage1", STAGE1_BLOCK, STAGE1_BLOCK)
    scripted_llm.script("express:1.0", EXPRESS_OK.replace("turn-1", "turn-1-express"))
    scripted_llm.script(
        "direct:1.0",
        direct_json_reply(agent.renderers, "1.0").replace("turn-1", "turn-1-direct"),
    )
    parts = await collect(agent, "compare the resort", "s4", "1.0")
    assert surface_ids(parts) == [
        "turn-1-decision",
        "turn-1-decision-footer",
        "turn-1-express",
        "turn-1-express-footer",
        "turn-1-direct",
        "turn-1-direct-footer",
        "turn-1-compare",
    ]
    summary = status_text(parts)
    assert "**Compare**" in summary and "`decision`: 1 call" in summary
    assert "`express`: 2 calls" in summary and "`direct`: 2 calls" in summary
    assert len(scripted_llm.calls_for("stage1")) == 1

    # The next turn is back to the single default mode.
    again = await collect(agent, "again", "s4", "1.0")
    assert surface_ids(again) == ["main", "turn-2-decision-footer"]


async def test_prose_from_stage1_and_stage2_failures_never_raise(
    agent: PlaygroundAgent, scripted_llm: FakeLlmCaller
) -> None:
    scripted_llm.script("stage1", "I would show a card with the resort name.")
    prose = await collect(agent, "show the resort", "s5")
    assert surface_ids(prose) == ["turn-1-status"]
    assert "could not turn that into a decision block" in status_text(prose)

    await collect(agent, "/mode express", "s5")
    scripted_llm.script("stage1", STAGE1_BLOCK)
    scripted_llm.script("express:0.9", EXPRESS_BAD, EXPRESS_BAD)
    failed = await collect(agent, "show the resort", "s5")
    assert surface_ids(failed) == ["turn-2-express-footer"]
    text = status_text(failed)
    assert "3 LLM calls" in text and "Render failed" in text


async def test_ui_action_text_goes_through_stage1(
    agent: PlaygroundAgent, scripted_llm: FakeLlmCaller
) -> None:
    scripted_llm.script("stage1", STAGE1_BLOCK)
    action_text = "User interacted with the UI: action `submit` with context {}."
    parts = await collect(agent, action_text, "s6", "1.0")
    assert surface_ids(parts)[0] == "main"
    assert scripted_llm.calls_for("stage1")[-1]["user_text"] == action_text
    assert action_text.startswith(UI_ACTION_PREFIX)
    assert RenderMode.DECISION.value in status_text(parts)


async def test_default_mode_starts_new_session_in_configured_mode(
    scripted_llm: FakeLlmCaller,
) -> None:
    renderers = Renderers(versions=["0.9"], backend=DeterministicMockBackend())
    express_agent = PlaygroundAgent(
        "http://localhost:10010",
        llm=scripted_llm,
        renderers=renderers,
        model_name="fake-model",
        default_mode=RenderMode.EXPRESS,
    )
    help_parts = await collect(express_agent, "/help", "s-default-express")
    assert "Active mode: `express`" in status_text(help_parts)


async def test_tool_mode_renders_surface_with_tool_result(
    agent: PlaygroundAgent, scripted_llm: FakeLlmCaller
) -> None:
    await collect(agent, "/mode tool", "s-tool-1")
    scripted_llm.script_tool(
        "tool_mode",
        LlmToolTurn(
            tool_name="get_host_health",
            arguments={"hostname": "prod-01"},
            result={"hostname": "prod-01", "cpu": "98%", "status": "critical"},
            prompt_tokens=150,
            output_tokens=30,
            latency_ms=120.0,
        ),
    )
    parts = await collect(agent, "check host 01 health", "s-tool-1", "1.0")
    ids = surface_ids(parts)
    assert ids == ["main", "turn-1-tool-footer"]
    footer = status_text(parts)
    assert "**tool**" in footer
    assert "1 LLM call (tool call only)" in footer


async def test_tool_mode_plain_text_reply_when_no_tool_called(
    agent: PlaygroundAgent, scripted_llm: FakeLlmCaller
) -> None:
    await collect(agent, "/mode tool", "s-tool-2")
    scripted_llm.script_tool(
        "tool_mode",
        LlmToolTurn(
            tool_name=None,
            arguments={},
            result=None,
            text="I can only check host health or ski resorts.",
            prompt_tokens=100,
            output_tokens=20,
            latency_ms=80.0,
        ),
    )
    parts = await collect(agent, "what is the capital of France?", "s-tool-2")
    assert surface_ids(parts) == ["turn-1-status", "turn-1-tool-footer"]
    assert "I can only check host health" in status_text(parts)


async def test_rate_command_records_preference_and_summarizes(
    agent: PlaygroundAgent,
    scripted_llm: FakeLlmCaller,
    tmp_path: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ratings_file = tmp_path / "ratings.jsonl"
    monkeypatch.setattr("agent.RATINGS_PATH", ratings_file)

    # Rate before turn
    unrated = await collect(agent, "/rate decision", "s-rate")
    assert "No previous turn to rate" in status_text(unrated)

    # Valid turn
    scripted_llm.script("stage1", STAGE1_BLOCK)
    await collect(agent, "show resort", "s-rate")

    # Rate invalid mode
    bad_mode = await collect(agent, "/rate glitter", "s-rate")
    assert "Unknown mode `glitter`" in status_text(bad_mode)

    # Rate mode not rendered in previous turn
    unrendered_mode = await collect(agent, "/rate express", "s-rate")
    assert "was not rendered in the previous turn" in status_text(unrendered_mode)

    # Rate with comment
    rated = await collect(agent, "/rate decision Very fast and clean", "s-rate")
    assert (
        "Recorded preference for `decision` with comment: Very fast and clean"
        in status_text(rated)
    )

    # Show ratings
    summary = await collect(agent, "/ratings", "s-rate")
    assert "(1 total vote)" in status_text(summary)
    assert "`decision`: 1 (100.0%)" in status_text(summary)


async def test_action_events_route_by_id_with_context() -> None:
    from agent_executor import action_query, build_query, classify_data_part

    event = {
        "version": "1.0",
        "action": {
            "name": "save_profile",
            "surfaceId": "main",
            "sourceComponentId": "actions_action_button_0",
            "timestamp": "2026-11-10T00:00:00Z",
            "context": {"email": "alex@example.com", "name": "Alex"},
        },
    }
    kind, action = classify_data_part(event)
    assert kind == "action"
    query = action_query(action)
    assert query.startswith(UI_ACTION_PREFIX)
    assert "action `save_profile` on surface `main`" in query
    assert '"email": "alex@example.com"' in query
    parts = [Part(root=DataPart(data=event))]
    assert build_query(parts) == query


async def test_set_backend_awaits_old_factory_aclose_and_skips_non_str_part_text() -> (
    None
):
    from types import SimpleNamespace
    from agent import _append_part_texts, _collect_tool_event_parts

    local_renderers = Renderers(versions=["1.0"], backend=DeterministicMockBackend())
    closed = False

    async def fake_aclose() -> None:
        nonlocal closed
        closed = True

    local_renderers._decision_factory.aclose = fake_aclose  # type: ignore[method-assign]
    await local_renderers.set_backend(DeterministicMockBackend())
    assert closed is True

    chunks: list[str] = []
    mixed_parts = [
        SimpleNamespace(text="hello "),
        SimpleNamespace(text=None),
        SimpleNamespace(text=42),
        SimpleNamespace(),
        SimpleNamespace(text="world"),
    ]
    _append_part_texts(mixed_parts, chunks)
    assert "".join(chunks) == "hello world"

    tool_chunks: list[str] = []
    name, args, resp = _collect_tool_event_parts(
        mixed_parts, tool_chunks, None, {}, None
    )
    assert "".join(tool_chunks) == "hello world"
    assert (name, args, resp) == (None, {}, None)


async def test_adk_llm_caller_runner_for_constructs_llm_agent_with_and_without_tools() -> (
    None
):
    from agent import AdkLlmCaller
    from tools import TOOLS

    caller = AdkLlmCaller("gemini-3.6-flash", "format_playground")
    plain_runner = caller._runner_for("stage1", "Return a decision block.")
    assert plain_runner.agent.tools == []
    assert plain_runner.agent.generate_content_config is not None
    assert plain_runner.agent.generate_content_config.thinking_config is not None
    assert (
        plain_runner.agent.generate_content_config.thinking_config.thinking_budget
        == 1000
    )

    tool_runner = caller._runner_for("tool_mode", "Call a tool.", tools=TOOLS)
    assert len(tool_runner.agent.tools) == len(TOOLS)


async def test_direct_tool_command_renders_without_llm_calls(
    agent: PlaygroundAgent, scripted_llm: FakeLlmCaller
) -> None:
    parts = await collect(agent, "/tool host", "s-direct-tool", version="1.0")
    assert scripted_llm.calls == []
    assert surface_ids(parts) == ["main", "turn-1-tool-footer"]
    footer = status_text(parts)
    assert "0 LLM calls" in footer

    cached = await collect(agent, "/tool host", "s-direct-tool", version="1.0")
    assert scripted_llm.calls == []
    assert "cache hit" in status_text(cached)

    missing = await collect(agent, "/tool", "s-direct-tool", version="1.0")
    assert "Usage: `/tool <name>`" in status_text(missing)

    unknown = await collect(agent, "/tool bogus", "s-direct-tool", version="1.0")
    assert "Unknown tool `bogus`" in status_text(unknown)
