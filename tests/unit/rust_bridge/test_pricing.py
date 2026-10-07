from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Final, cast  # noqa: TID251  # models external registrations with unrelated metadata

import pytest
from pydantic import JsonValue, TypeAdapter

from litellm.litellm_core_utils.llm_cost_calc.utils import parse_prompt_tokens_details
from litellm.rust_bridge import pricing
from litellm.rust_bridge.configuration import Decision
from litellm.types.utils import ModelInfo, PromptTokensDetailsWrapper, Usage

_JSON: Final = TypeAdapter(dict[str, JsonValue])


@dataclass
class RecordedCalculator:
    calls: tuple[tuple[Mapping[str, JsonValue], Mapping[str, JsonValue]], ...] = ()

    def __call__(self, pricing_json: bytes, request_json: bytes) -> object:
        self.calls += ((_JSON.validate_json(pricing_json), _JSON.validate_json(request_json)),)
        return (84.0, 20.0)


@pytest.fixture(autouse=True)
def reset_pricing_binding() -> Iterator[None]:
    pricing.CATALOG_COST.reset()
    yield
    pricing.CATALOG_COST.reset()


@pytest.mark.parametrize("selected", tuple(Decision))
def test_native_projection_ignores_unrelated_model_metadata(selected: Decision) -> None:
    recorded: Final = RecordedCalculator()
    pricing.CATALOG_COST.override(recorded)
    usage: Final = Usage(
        prompt_tokens=100,
        completion_tokens=10,
        prompt_tokens_details=PromptTokensDetailsWrapper(text_tokens=40, cached_tokens=40, cache_write_tokens=20),
    )
    supplied: Final = cast(  # cast-ok: external model registrations can carry unrelated metadata
        ModelInfo,
        {
            "input_cost_per_token": 1.0,
            "output_cost_per_token": 2.0,
            "cache_read_input_token_cost": 0.1,
            "cache_creation_input_token_cost": 2.0,
            "unrelated_metadata": object(),
        },
    )

    result: Final = pricing.calculate_cost(
        model_info=supplied,
        custom_llm_provider="openai",
        usage=usage,
        prompt=parse_prompt_tokens_details(usage),
        completion=None,
        service_tier=None,
        billed_at=datetime(2026, 1, 5, 16, 30, 0, 123456, tzinfo=timezone.utc),
        threshold_is_inclusive=False,
        selected=selected,
    )

    if selected is Decision.PYTHON:
        assert result is None
        assert recorded.calls == ()
        return

    assert result == (84.0, 20.0)
    assert len(recorded.calls) == 1
    fields, request = recorded.calls[0]
    assert "unrelated_metadata" not in fields
    assert fields["cache_creation_input_token_cost"] == 2.0
    assert (request["cache_write_5m_tokens"], request["cache_write_1h_tokens"]) == (20, 0)
