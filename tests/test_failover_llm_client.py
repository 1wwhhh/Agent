from __future__ import annotations

from collections.abc import AsyncIterator

import pytest

from app.schemas.llm import LLMRequest, LLMResponse, LLMResponseChunk
from app.tools.failover_client import FailoverLLMClient
from app.tools.llm_client import CircuitBreakerConfig, LLMClient, LLMClientError


class StubLLMClient(LLMClient):
    def __init__(self, outcomes, **kwargs) -> None:
        super().__init__(**kwargs)
        self.outcomes = list(outcomes)
        self.calls = 0

    async def _generate(self, request: LLMRequest) -> LLMResponse:
        self.calls += 1
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    async def _stream(self, request: LLMRequest) -> AsyncIterator[LLMResponseChunk]:
        raise NotImplementedError


@pytest.mark.asyncio
async def test_failover_client_uses_secondary_provider_when_primary_fails():
    primary = StubLLMClient(
        outcomes=[LLMClientError("primary failure")],
        model_name="primary",
        max_retries=0,
    )
    secondary = StubLLMClient(
        outcomes=[LLMResponse(text="secondary ok", model_name="secondary")],
        model_name="secondary",
        max_retries=0,
    )
    client = FailoverLLMClient(clients=[primary, secondary], provider_names=["primary", "secondary"])

    response = await client.generate(LLMRequest(prompt="hello"))

    assert response.text == "secondary ok"
    assert response.raw_response["selected_provider"] == "secondary"
    assert response.raw_response["attempted_providers"] == ["primary", "secondary"]
    assert primary.calls == 1
    assert secondary.calls == 1


@pytest.mark.asyncio
async def test_failover_client_skips_open_circuit_breaker_provider():
    primary = StubLLMClient(
        outcomes=[LLMClientError("primary failure")],
        model_name="primary",
        max_retries=0,
        circuit_breaker_config=CircuitBreakerConfig(failure_threshold=1, reset_timeout_seconds=60),
    )
    secondary = StubLLMClient(
        outcomes=[
            LLMResponse(text="secondary first", model_name="secondary"),
            LLMResponse(text="secondary second", model_name="secondary"),
        ],
        model_name="secondary",
        max_retries=0,
    )
    client = FailoverLLMClient(clients=[primary, secondary], provider_names=["primary", "secondary"])

    first = await client.generate(LLMRequest(prompt="hello"))
    second = await client.generate(LLMRequest(prompt="hello again"))

    assert first.text == "secondary first"
    assert second.text == "secondary second"
    assert primary.calls == 1
    assert secondary.calls == 2
