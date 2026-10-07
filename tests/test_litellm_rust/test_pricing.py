from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from math import isclose
from typing import Final

import pytest
from pydantic import JsonValue, TypeAdapter

import litellm
from litellm._internal_context import pinned_billing_time
from litellm.litellm_core_utils.llm_cost_calc.utils import generic_cost_per_token
from litellm.rust_bridge import configuration, pricing
from litellm.types.utils import CacheCreationTokenDetails, ModelInfo, ModelResponse, PromptTokensDetailsWrapper, Usage
from tests.test_litellm_rust.support.isolation import rebound

pytestmark = pytest.mark.requires_rust_extension

MODEL: Final = "native-pricing-fixture"
BILLED_AT: Final = datetime(2026, 1, 5, 16, 30, 0, 123456, tzinfo=timezone.utc)
_JSON: Final = TypeAdapter(dict[str, JsonValue])


@dataclass
class RecordedCalculator:
    native: pricing.CatalogCostCalculator
    calls: tuple[tuple[Mapping[str, JsonValue], Mapping[str, JsonValue]], ...] = ()

    def __call__(self, pricing_json: bytes, request_json: bytes) -> object:
        self.calls += ((_JSON.validate_json(pricing_json), _JSON.validate_json(request_json)),)
        return self.native(pricing_json, request_json)


@pytest.fixture(autouse=True)
def reset_pricing_binding() -> Iterator[None]:
    configuration.reset_rust_configuration()
    pricing.CATALOG_COST.reset()
    yield
    pricing.CATALOG_COST.reset()
    configuration.reset_rust_configuration()


@pytest.fixture
def calculator() -> RecordedCalculator:
    native: Final = pricing.CATALOG_COST.load()
    assert native is not None, "Build the native extension before exercising the shared cost calculator"
    recorded: Final = RecordedCalculator(native)
    pricing.CATALOG_COST.override(recorded)
    return recorded


def unexpected_calculation(pricing_json: bytes, request_json: bytes) -> object:
    pytest.fail("Unsupported usage must keep the existing provider calculator")


@pytest.fixture
def unused_calculator() -> RecordedCalculator:
    recorded: Final = RecordedCalculator(unexpected_calculation)
    pricing.CATALOG_COST.override(recorded)
    return recorded


@pytest.fixture
def model_info() -> Iterator[ModelInfo]:
    fields: Final[ModelInfo] = {
        "litellm_provider": "openai",
        "mode": "chat",
        "input_cost_per_token": 1.0,
        "output_cost_per_token": 2.0,
        "cache_read_input_token_cost": 0.1,
        "cache_creation_input_token_cost": 2.0,
        "cache_creation_input_token_cost_above_1hr": 4.0,
    }
    with rebound(litellm, "model_cost", {**litellm.model_cost, MODEL: fields}):
        litellm.utils._invalidate_model_cost_lowercase_map()
        litellm.get_model_info.cache_clear()
        try:
            yield litellm.get_model_info(model=MODEL, custom_llm_provider="openai")
        finally:
            litellm.get_model_info.cache_clear()
            litellm.utils._invalidate_model_cost_lowercase_map()


def cache_usage(durations: tuple[int | None, int | None] | None = None) -> Usage:
    return Usage(
        prompt_tokens=100,
        completion_tokens=10,
        total_tokens=110,
        prompt_tokens_details=PromptTokensDetailsWrapper(
            cached_tokens=40,
            cache_write_tokens=20,
            cache_creation_token_details=(
                CacheCreationTokenDetails(
                    ephemeral_5m_input_tokens=durations[0], ephemeral_1h_input_tokens=durations[1]
                )
                if durations is not None
                else None
            ),
        ),
    )


@pytest.mark.parametrize(
    ("durations", "five", "one", "expected"),
    (
        pytest.param(None, 20, 0, 104.0, id="gateway-default-five-minute"),
        pytest.param((15, 5), 15, 5, 114.0, id="mixed-durations"),
        pytest.param((None, 20), 0, 20, 144.0, id="one-hour-only"),
        pytest.param((20, None), 20, 0, 104.0, id="five-minute-only"),
    ),
)
def test_gateway_completion_cost_uses_shared_native_cache_pricing(
    model_info: ModelInfo,
    calculator: RecordedCalculator,
    durations: tuple[int | None, int | None] | None,
    five: int,
    one: int,
    expected: float,
) -> None:
    response: Final = ModelResponse(model=MODEL, usage=cache_usage(durations))
    with pinned_billing_time(BILLED_AT):
        native: Final = litellm.completion_cost(completion_response=response, model=MODEL, custom_llm_provider="openai")
        pricing.CATALOG_COST.override(None)
        fallback: Final = litellm.completion_cost(
            completion_response=response, model=MODEL, custom_llm_provider="openai"
        )

    assert isclose(native, expected)
    assert isclose(fallback, expected)
    assert len(calculator.calls) == 1
    fields, request = calculator.calls[0]
    assert fields["cache_creation_input_token_cost_above_1hr"] == model_info.get(
        "cache_creation_input_token_cost_above_1hr"
    )
    assert request == {
        "prompt_tokens": 100,
        "completion_tokens": 10,
        "cache_read_tokens": 40,
        "cache_write_tokens": 20,
        "cache_write_5m_tokens": five,
        "cache_write_1h_tokens": one,
        "reasoning_tokens": 0,
        "service_tier": None,
        "billed_at_ns": 1767630600123456000,
        "threshold_is_inclusive": False,
    }


def test_native_gateway_cost_keeps_deployment_prices_provider_policy_and_regional_uplifts(
    model_info: ModelInfo, calculator: RecordedCalculator
) -> None:
    effective: Final[ModelInfo] = {
        **model_info,
        "input_cost_per_token_above_100k_tokens": 5.0,
        "output_cost_per_token_above_100k_tokens": 6.0,
        "regional_processing_uplift_multiplier_eu": 1.1,
        "regional_endpoint_uplift_multiplier": 1.2,
    }
    result: Final = generic_cost_per_token(
        MODEL,
        Usage(
            prompt_tokens=100_000,
            completion_tokens=10,
            prompt_tokens_details=PromptTokensDetailsWrapper(cached_tokens=40, cache_write_tokens=20),
        ),
        "xai",
        model_info=effective,
        data_residency="eu",
        vertex_location="us-east5",
        current_time=BILLED_AT,
    )

    expected: Final = (((100_000 - 60) * 5 + 40 * 0.1 + 20 * 2) * 1.1 * 1.2, 10 * 6 * 1.1 * 1.2)
    assert all(isclose(actual, cost) for actual, cost in zip(result, expected, strict=True))
    assert len(calculator.calls) == 1
    fields, request = calculator.calls[0]
    assert fields["litellm_provider"] == "xai"
    assert request["threshold_is_inclusive"] is True


def test_gateway_structured_tier_keeps_free_hourly_cache_writes(
    model_info: ModelInfo, calculator: RecordedCalculator
) -> None:
    effective: Final[ModelInfo] = {
        **model_info,
        "tiered_pricing": [
            {
                "range": [0, 200],
                "input_cost_per_token": 1.0,
                "output_cost_per_token": 2.0,
                "cache_read_input_token_cost": 0.1,
                "cache_creation_input_token_cost": 2.0,
                "cache_creation_input_token_cost_above_1hr": 0.0,
            }
        ],
    }

    result: Final = generic_cost_per_token(
        MODEL, cache_usage((0, 20)), "openai", model_info=effective, current_time=BILLED_AT
    )
    pricing.CATALOG_COST.override(None)
    fallback: Final = generic_cost_per_token(
        MODEL, cache_usage((0, 20)), "openai", model_info=effective, current_time=BILLED_AT
    )

    assert result == fallback == (44.0, 20.0)
    assert len(calculator.calls) == 1


@pytest.mark.parametrize(
    ("usage", "expected"),
    (
        pytest.param(
            Usage(prompt_tokens=100, completion_tokens=10, prompt_tokens_details={"audio_tokens": 20}),
            (180.0, 20.0),
            id="audio-billing-stays-with-provider-policy",
        ),
        pytest.param(
            Usage(prompt_tokens=100, completion_tokens=10, prompt_tokens_details={"text_tokens": 50}),
            (50.0, 20.0),
            id="explicit-text-count-differs-from-total",
        ),
        pytest.param(cache_usage((5, 5)), (74.0, 20.0), id="duration-total-disagrees"),
    ),
)
def test_gateway_shapes_outside_native_text_cost_keep_existing_billing(
    model_info: ModelInfo, unused_calculator: RecordedCalculator, usage: Usage, expected: tuple[float, float]
) -> None:
    result: Final = generic_cost_per_token(
        MODEL, usage, "openai", model_info={**model_info, "input_cost_per_audio_token": 5.0}, current_time=BILLED_AT
    )

    assert all(isclose(actual, cost) for actual, cost in zip(result, expected, strict=True))
    assert unused_calculator.calls == ()


def test_gateway_cost_stays_available_without_native_extension(model_info: ModelInfo) -> None:
    pricing.CATALOG_COST.override(None)

    result: Final = generic_cost_per_token(
        MODEL, cache_usage(), "openai", model_info=model_info, current_time=BILLED_AT
    )

    assert result == (84.0, 20.0)


@pytest.mark.parametrize("switch", ("process", "environment"))
def test_gateway_operator_opt_out_keeps_python_cost(
    monkeypatch: pytest.MonkeyPatch, model_info: ModelInfo, unused_calculator: RecordedCalculator, switch: str
) -> None:
    if switch == "process":
        monkeypatch.delenv("LITELLM_RUST", raising=False)
        configuration.rust(False)
    else:
        monkeypatch.setenv("LITELLM_RUST", "false")

    result: Final = generic_cost_per_token(
        MODEL, cache_usage(), "openai", model_info=model_info, current_time=BILLED_AT
    )

    assert result == (84.0, 20.0)
    assert unused_calculator.calls == ()
