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

"""End-to-end synthesis with a custom catalog that ships no patterns."""

from __future__ import annotations

from typing import Any

import pytest

from a2ui.core import Catalog
from a2ui.inference_formats.decision import (
    DecisionSynthesizer,
    FallbackSource,
    SurfaceOperation,
    SurfaceTemplate,
    SynthesisStage,
    TemplateRegistry,
)
from a2ui.parser import A2uiCompilationError
from a2ui.parser.messages import to_message_dicts
from a2ui.utils import validate_payload

from .layout_fixtures import mock_backend

FINANCE_ID = "https://example.com/catalogs/finance/v1/catalog.json"
QUOTE = {"ticker": "GOOG", "price": 182.5}


def _finance_catalog(protocol_version: str = "1.0") -> Catalog:
    """A custom catalog with no Basic components and no patterns file."""
    return Catalog.from_json({
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "catalogId": FINANCE_ID,
        "protocolVersion": protocol_version,
        "components": {
            "TickerQuoteCard": {
                "type": "object",
                "description": "Displays a live equity quote.",
                "properties": {
                    "component": {"const": "TickerQuoteCard"},
                    "ticker": {"$ref": "common_types.json#/$defs/DynamicString"},
                    "price": {"$ref": "common_types.json#/$defs/DynamicNumber"},
                },
                "required": ["component", "ticker", "price"],
            },
        },
        "functions": {},
    })


def _quote_registry() -> TemplateRegistry:
    registry = TemplateRegistry()
    registry.register(
        SurfaceTemplate(
            template_id="quote_tpl",
            description="Approved equity quote card",
            surface_id="default_surface",
            catalog_id=FINANCE_ID,
            protocol_version="1.0",
            components=(
                {
                    "id": "root",
                    "component": "TickerQuoteCard",
                    "ticker": {"@path": "/ticker"},
                    "price": {"@path": "/price"},
                },
            ),
            required_pointers=("/ticker", "/price"),
        )
    )
    return registry


def _components(messages: Any) -> list[dict[str, Any]]:
    for d in to_message_dicts(messages):
        if "updateComponents" in d:
            return list(d["updateComponents"]["components"])
        if "components" in d.get("createSurface", {}):
            return list(d["createSurface"]["components"])
    return []


def test_custom_catalogs_must_target_v10() -> None:
    with pytest.raises(ValueError, match="v1.0 only"):
        DecisionSynthesizer([_finance_catalog("0.9")], mock_backend())


@pytest.mark.parametrize(
    "operation", [SurfaceOperation.CREATE, SurfaceOperation.UPDATE_ROOT]
)
@pytest.mark.asyncio
async def test_template_question_renders_without_any_patterns(
    operation: SurfaceOperation,
) -> None:
    catalog = _finance_catalog()
    backend = mock_backend(**{"template": ("quote_tpl", 0.95)})
    synth = DecisionSynthesizer([catalog], backend, registry=_quote_registry())

    out = await synth.synthesize("Show the GOOG quote", QUOTE, operation=operation)

    assert out.stage is SynthesisStage.TEMPLATE_SELECT
    assert out.selected_template_id == "quote_tpl"
    assert out.layout_plan is not None and out.layout_plan.template_id == "quote_tpl"
    assert out.decision_calls == 1
    assert backend.batches[0].questions[0].id == "template"
    assert _components(out.messages)[0]["component"] == "TickerQuoteCard"
    assert out.operation is operation
    validate_payload([catalog], out.messages)


@pytest.mark.asyncio
async def test_template_question_below_threshold_falls_back() -> None:
    catalog = _finance_catalog()
    synth = DecisionSynthesizer(
        [catalog],
        mock_backend(**{"template": ("compose", 0.95)}),
        registry=_quote_registry(),
    )
    out = await synth.synthesize(
        "Show the GOOG quote",
        QUOTE,
        fallback_dsl='root = TickerQuoteCard(ticker="GOOG", price=182.5)',
    )
    assert out.stage is SynthesisStage.ABSTAIN_FALLBACK
    assert out.fallback_source is FallbackSource.DSL


@pytest.mark.asyncio
async def test_no_patterns_and_no_templates_needs_a_fallback() -> None:
    catalog = _finance_catalog()
    synth = DecisionSynthesizer([catalog], mock_backend())
    with pytest.raises(A2uiCompilationError, match="no eligible archetype"):
        await synth.synthesize("Show the GOOG quote", QUOTE)

    out = await synth.synthesize(
        "Show the GOOG quote",
        QUOTE,
        fallback_dsl='root = TickerQuoteCard(ticker="GOOG", price=182.5)',
    )
    assert out.fallback_source is FallbackSource.DSL
    assert out.fallback_reason == "no eligible archetype"
    assert _components(out.messages)[0]["component"] == "TickerQuoteCard"
    validate_payload([catalog], out.messages)
