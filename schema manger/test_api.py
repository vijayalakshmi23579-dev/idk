"""test_api.py — mock litellm so we don't hit the network."""
import asyncio
import io
import logging
import os
from unittest.mock import AsyncMock, MagicMock, patch
import sys


class MockRateLimitError(Exception):
    status_code = 429

    def __init__(self, message, headers=None):
        super().__init__(message)
        self.headers = headers or {}


class MockNotFoundError(Exception):
    status_code = 404


class MockOverloadError(Exception):
    status_code = 529


class MockContextWindowExceededError(Exception):
    pass


class MockAPIError(Exception):
    pass


class MockAPIConnectionError(Exception):
    pass


class MockBadRequestError(Exception):
    pass

mock_litellm = MagicMock()
mock_litellm.acompletion = AsyncMock()
mock_litellm.RateLimitError = MockRateLimitError
mock_litellm.NotFoundError = MockNotFoundError
mock_litellm.ContextWindowExceededError = MockContextWindowExceededError
mock_litellm.APIError = MockAPIError
mock_litellm.APIConnectionError = MockAPIConnectionError
mock_litellm.supports_reasoning = MagicMock(return_value=False)
sys.modules["litellm"] = mock_litellm

mock_dotenv = MagicMock()
sys.modules["dotenv"] = mock_dotenv

import api

from api import (
    stream, complete, should_enable_thinking,
    ToolCallAccumulator, _freeze_system_prompt,
    _apply_cache_control_to_messages, _strip_excess_media,
    _model_supports_thinking,
    get_active_model, set_active_model, list_models,
    ACTIVE_MODEL, FALLBACK_MODELS, _sync_provider_credentials,
    ProviderRequestError,
)
from messages import create_user_message


def mock_chunk(text=None, reasoning=None, tool_calls=None, finish=None, usage=None):
    c = MagicMock()
    c.choices = [MagicMock()]
    d = MagicMock()
    d.content = text
    d.reasoning_content = reasoning
    d.tool_calls = tool_calls
    c.choices[0].delta = d
    c.choices[0].finish_reason = finish
    c.usage = usage
    return c


def mock_usage(in_tok=100, out_tok=20, cache_read=50, cache_create=0):
    u = MagicMock()
    u.prompt_tokens = in_tok
    u.completion_tokens = out_tok
    u.cache_read_input_tokens = cache_read
    u.cache_creation_input_tokens = cache_create
    return u


async def stream_yields(response_chunks):
    for c in response_chunks:
        yield c


async def main():
    # ── Thinking detection (no model arg → defaults to ACTIVE_MODEL) ──
    assert should_enable_thinking([create_user_message("think about this")]) is False  # ACTIVE_MODEL may not support
    # Force a model that supports thinking
    assert should_enable_thinking(
        [create_user_message("think about this")], "claude-sonnet-4-5"
    ) is True
    assert should_enable_thinking(
        [create_user_message("plan the refactor")], "claude-sonnet-4-5"
    ) is True
    assert should_enable_thinking(
        [create_user_message("ultrathink")], "claude-sonnet-4-5"
    ) is True
    assert should_enable_thinking(
        [create_user_message("read the file")], "claude-sonnet-4-5"
    ) is False
    assert should_enable_thinking(
        [create_user_message("x" * 600)], "claude-sonnet-4-5"
    ) is True
    # Model that doesn't support thinking → always False
    assert should_enable_thinking(
        [create_user_message("think about this")], "groq/llama-3.3-70b-versatile"
    ) is False
    print("✅ Thinking detection OK")

    # ── Frozen system prompt ──
    assert _freeze_system_prompt("") == []
    blocks = _freeze_system_prompt("You are helpful")
    assert len(blocks) == 1
    assert blocks[0]["cache_control"] == {"type": "ephemeral"}
    print("✅ System prompt freezing OK")

    # ── Cache control on user + assistant messages ──
    msgs = [
        {"role": "user", "content": [{"type": "text", "text": "first"}]},
        {"role": "assistant", "content": [{"type": "text", "text": "reply"}]},
        {"role": "user", "content": [{"type": "text", "text": "second"}]},
    ]
    tagged = _apply_cache_control_to_messages([m.copy() for m in msgs])
    assert "cache_control" in tagged[2]["content"][-1]
    assert "cache_control" in tagged[1]["content"][-1]
    assert "cache_control" not in tagged[0]["content"][-1]
    print("✅ Cache control on messages OK")

    # ── Thinking blocks don't get cache_control ──
    msgs = [{"role": "assistant", "content": [
        {"type": "text", "text": "answer"},
        {"type": "thinking", "thinking": "reasoning"},
    ]}]
    tagged = _apply_cache_control_to_messages([m.copy() for m in msgs])
    assert "cache_control" not in tagged[0]["content"][-1]
    print("✅ Thinking block cache exclusion OK")

    # ── Media stripping ──
    big_msgs = [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": f"img{i}"}} for i in range(50)
    ]}]
    stripped = _strip_excess_media([m.copy() for m in big_msgs], limit=20)
    media_count = sum(1 for b in stripped[0]["content"] if b.get("type") == "image_url")
    assert media_count == 20
    print("✅ Media stripping OK")

    # ── Model capability check ──
    assert _model_supports_thinking("claude-sonnet-4-5") is True
    assert _model_supports_thinking("deepseek-r1") is True
    assert _model_supports_thinking("llama-3.3-70b") is False
    print("✅ Model capability check OK")

    # ── Tool call accumulator ──
    acc = ToolCallAccumulator()
    tc1 = MagicMock(index=0, id="call_abc")
    tc1.function = MagicMock(name="Bash", arguments='{"comm')
    tc2 = MagicMock(index=0, id=None)
    tc2.function = MagicMock(name=None, arguments='and": "ls"}')
    acc.feed([tc1])
    acc.feed([tc2])
    assembled = acc.assemble()
    assert len(assembled) == 1
    assert assembled[0]["input"] == {"command": "ls"}
    print("✅ Tool call accumulator OK")

    # ── Malformed args ──
    acc = ToolCallAccumulator()
    tc = MagicMock(index=0, id="call_x")
    tc.function = MagicMock(name="X", arguments='{bad json')
    acc.feed([tc])
    assert acc.assemble()[0]["input"] == {}
    print("✅ Malformed args handled OK")

    # ── Streaming text ──
    mock_litellm.acompletion.reset_mock()
    mock_litellm.acompletion.return_value = stream_yields([
        mock_chunk(text="Hello "),
        mock_chunk(text="world"),
        mock_chunk(finish="stop", usage=mock_usage()),
    ])
    events = []
    async for e in stream([create_user_message("hi")], enable_thinking=False):
        events.append(e)
    text_events = [e for e in events if e["type"] == "text_delta"]
    assert len(text_events) == 2
    stop = next(e for e in events if e["type"] == "message_stop")
    assert stop["stop_reason"] == "end_turn"
    print("✅ Streaming text OK")

    # ── Abort closes a pending provider stream without emitting partial tools ──
    class AbortSignal:
        def __init__(self):
            self.aborted = False
            self.event = asyncio.Event()

        async def wait(self):
            await self.event.wait()

        def abort(self):
            self.aborted = True
            self.event.set()

    async def stalled_response():
        yield mock_chunk(text="partial response")
        await asyncio.Event().wait()

    mock_litellm.acompletion.reset_mock()
    mock_litellm.acompletion.return_value = stalled_response()
    abort_signal = AbortSignal()
    events = []
    async for event in stream(
        [create_user_message("wait")],
        enable_thinking=False,
        signal=abort_signal,
    ):
        events.append(event)
        if event["type"] == "text_delta":
            abort_signal.abort()
    assert [event["text"] for event in events if event["type"] == "text_delta"] == [
        "partial response",
    ]
    stop = next(event for event in events if event["type"] == "message_stop")
    assert stop["stop_reason"] == "interrupted"
    assert not any(event["type"] == "tool_use" for event in events)
    print("✅ Abort closes provider stream cleanly OK")

    # ── Streaming tool call ──
    mock_litellm.acompletion.reset_mock()
    tc1 = MagicMock(index=0, id="call_x")
    tc1.function = MagicMock(name="Bash", arguments='{"command"')
    tc2 = MagicMock(index=0, id=None)
    tc2.function = MagicMock(name=None, arguments=': "ls"}')
    mock_litellm.acompletion.return_value = stream_yields([
        mock_chunk(tool_calls=[tc1]),
        mock_chunk(tool_calls=[tc2]),
        mock_chunk(finish="tool_calls", usage=mock_usage()),
    ])
    events = []
    async for e in stream([create_user_message("list files")], enable_thinking=False):
        events.append(e)
    tool_events = [e for e in events if e["type"] == "tool_use"]
    assert len(tool_events) == 1
    assert tool_events[0]["input"] == {"command": "ls"}
    stop = next(e for e in events if e["type"] == "message_stop")
    assert stop["stop_reason"] == "tool_use"
    print("✅ Streaming tool call OK")

    # ── Streaming thinking ──
    mock_litellm.acompletion.reset_mock()
    mock_litellm.acompletion.return_value = stream_yields([
        mock_chunk(reasoning="let me think..."),
        mock_chunk(reasoning="done"),
        mock_chunk(text="Answer", finish="stop"),
    ])
    events = []
    async for e in stream([create_user_message("think")], enable_thinking=True):
        events.append(e)
    thinking_events = [e for e in events if e["type"] == "thinking_delta"]
    assert len(thinking_events) == 2
    print("✅ Streaming thinking OK")

    # ── Provider-wide request concurrency cap ──
    active_requests = 0
    max_active_requests = 0
    release_requests = asyncio.Event()
    reached_limit = asyncio.Event()

    async def bounded_completion(**_kwargs):
        nonlocal active_requests, max_active_requests
        active_requests += 1
        max_active_requests = max(max_active_requests, active_requests)
        if active_requests == 2:
            reached_limit.set()
        await release_requests.wait()
        active_requests -= 1
        return MagicMock()

    mock_litellm.acompletion.reset_mock()
    mock_litellm.acompletion.side_effect = bounded_completion
    with patch.dict(os.environ, {"AGENT_PROVIDER_MAX_CONCURRENCY": "2"}):
        requests = [
            asyncio.create_task(api._call_with_fallback(
                {"model": "groq/test-model", "messages": []},
                enable_thinking=False,
            ))
            for _ in range(5)
        ]
        await asyncio.wait_for(reached_limit.wait(), timeout=1)
        assert active_requests == 2
        release_requests.set()
        await asyncio.gather(*requests)
    assert max_active_requests == 2
    print("✅ Provider concurrency cap OK")

    # ── Provider RPM/TPM buckets use provider-specific overrides ──
    with patch.dict(os.environ, {
        "AGENT_PROVIDER_MAX_RPM": "60",
        "AGENT_PROVIDER_MAX_RPM_GROQ": "120",
        "AGENT_PROVIDER_MAX_TPM": "6000",
        "AGENT_PROVIDER_MAX_TPM_GROQ": "9000",
    }):
        limiter = api._provider_rate_limiter("groq/test-model")
        assert limiter.rpm == 120
        assert limiter.tpm == 9000
        reservation = await limiter.acquire(8500)
        assert reservation.tokens == 8500
        await reservation.settle(5000)

    mock_litellm.acompletion.reset_mock()
    with patch.dict(os.environ, {"AGENT_PROVIDER_MAX_TPM_GROQ": "1"}):
        try:
            async for _ in stream(
                [create_user_message("request exceeds configured token capacity")],
                model="groq/test-model",
                max_tokens=8,
                enable_thinking=False,
            ):
                pass
            raise AssertionError("expected configured TPM capacity failure")
        except ProviderRequestError as error:
            assert error.category == "rate_limited"
    assert mock_litellm.acompletion.call_count == 0
    fake_clock = [0.0]

    async def advance_rate_clock(seconds):
        fake_clock[0] += seconds

    with patch("api.time.monotonic", lambda: fake_clock[0]), patch(
        "api.asyncio.sleep", advance_rate_clock,
    ):
        rpm_limiter = api._ProviderRateLimiter("test-provider", rpm=1, tpm=0)
        await rpm_limiter.acquire(0)
        await rpm_limiter.acquire(0)
    assert fake_clock[0] == 60.0
    print("✅ Provider RPM/TPM overrides and capacity enforcement OK")

    # ── 429 honors Retry-After and retries the same model ──
    mock_litellm.acompletion.reset_mock()
    rate_limit = mock_litellm.RateLimitError(
        "rate limited", headers={"Retry-After": "7"},
    )
    mock_litellm.acompletion.side_effect = [
        rate_limit,
        stream_yields([mock_chunk(text="after wait", finish="stop")]),
    ]
    events = []
    with patch("api._retry_sleep", new_callable=AsyncMock) as retry_sleep:
        with patch.dict(os.environ, {
        "LITELLM_FALLBACK_MODELS": "groq/test-approved-fallback",
        }):
            async for e in stream(
                [create_user_message("hi")], enable_thinking=False,
            ):
                events.append(e)
    text_events = [e for e in events if e["type"] == "text_delta"]
    assert text_events[0]["text"] == "after wait"
    assert mock_litellm.acompletion.call_count == 2
    assert all(
        call.kwargs["model"] == get_active_model()
        for call in mock_litellm.acompletion.call_args_list
    )
    retry_sleep.assert_awaited_once_with(7.0)
    assert any(
        event.get("area") == "provider" and event.get("status") == "retrying"
        and event.get("delay_seconds") == 7.0
        for event in events
    )
    print("✅ 429 Retry-After handling OK")

    # ── Hard quota 429 fails without retries or model switching ──
    mock_litellm.acompletion.reset_mock()
    mock_litellm.acompletion.side_effect = MockRateLimitError(
        "spending limit exceeded",
    )
    with patch("api._retry_sleep", new_callable=AsyncMock) as retry_sleep:
        with patch.dict(os.environ, {
            "LITELLM_FALLBACK_MODELS": "groq/test-approved-fallback",
        }):
            try:
                async for _ in stream(
                    [create_user_message("hi")], enable_thinking=False,
                ):
                    pass
                raise AssertionError("expected hard quota failure")
            except ProviderRequestError as error:
                assert "rate_limited" in str(error)
    assert mock_litellm.acompletion.call_count == 1
    retry_sleep.assert_not_awaited()
    print("✅ Hard quota 429 fails fast OK")

    # ── Over-budget Retry-After fails instead of retrying early ──
    mock_litellm.acompletion.reset_mock()
    mock_litellm.acompletion.side_effect = MockRateLimitError(
        "temporary rate limit", headers={"Retry-After": "3600"},
    )
    with patch("api._retry_sleep", new_callable=AsyncMock) as retry_sleep:
        try:
            async for _ in stream(
                [create_user_message("hi")], enable_thinking=False,
            ):
                pass
            raise AssertionError("expected Retry-After budget failure")
        except ProviderRequestError as error:
            assert "rate_limited" in str(error)
    assert mock_litellm.acompletion.call_count == 1
    retry_sleep.assert_not_awaited()
    print("✅ Retry-After budget respected OK")

    # ── 404 does not retry the same model ──
    mock_litellm.acompletion.reset_mock()
    not_found = MockNotFoundError("unknown model secret=do-not-log")
    not_found.request_id = "req-404-safe"
    mock_litellm.acompletion.side_effect = not_found
    log_output = io.StringIO()
    log_handler = logging.StreamHandler(log_output)
    api.logger.addHandler(log_handler)
    failure_events = []
    try:
        async for event in stream(
            [create_user_message("hi")],
            model="nvidia_nim/missing-model",
            enable_thinking=False,
        ):
            failure_events.append(event)
        raise AssertionError("expected model-not-found failure")
    except ProviderRequestError as error:
        assert "model_or_route_not_found" in str(error)
        assert "req-404-safe" in str(error)
        assert "secret=do-not-log" not in str(error)
    finally:
        api.logger.removeHandler(log_handler)
        log_handler.close()
    assert mock_litellm.acompletion.call_count == 1
    assert "req-404-safe" in log_output.getvalue()
    assert "secret=do-not-log" not in log_output.getvalue()
    assert any(
        event.get("area") == "model" and event.get("status") == "failed"
        and event.get("request_id") == "req-404-safe"
        for event in failure_events
    )
    print("✅ 404 fails without same-model retry OK")

    # ── 404 can use an explicitly configured alternate model ──
    mock_litellm.acompletion.reset_mock()
    mock_litellm.acompletion.side_effect = [
        MockNotFoundError("unknown model"),
        stream_yields([mock_chunk(text="alternate", finish="stop")]),
    ]
    events = [event async for event in stream(
        [create_user_message("hi")],
        model="nvidia_nim/missing-model",
        fallback_model="groq/approved-model",
        enable_thinking=False,
    )]
    assert mock_litellm.acompletion.call_count == 2
    assert mock_litellm.acompletion.call_args_list[0].kwargs["model"] == "nvidia_nim/missing-model"
    assert mock_litellm.acompletion.call_args_list[1].kwargs["model"] == "groq/approved-model"
    assert any(event.get("text") == "alternate" for event in events)
    assert next(event for event in events if event["type"] == "message_start")["model"] == "groq/approved-model"
    assert any(
        event.get("area") == "provider" and event.get("status") == "fallback"
        and event.get("alternate_model") == "groq/approved-model"
        for event in events
    )
    print("✅ 404 explicit alternate routing OK")

    # ── 529 retries boundedly, then uses configured alternate ──
    mock_litellm.acompletion.reset_mock()
    mock_litellm.acompletion.side_effect = [
        MockOverloadError("provider overloaded"),
        MockOverloadError("provider overloaded"),
        MockOverloadError("provider overloaded"),
        stream_yields([mock_chunk(text="recovered", finish="stop")]),
    ]
    with patch("api._retry_sleep", new_callable=AsyncMock) as retry_sleep:
        events = [event async for event in stream(
            [create_user_message("hi")],
            model="nvidia_nim/model-a",
            fallback_model="groq/model-b",
            enable_thinking=False,
            query_source="subagent_task",
        )]
    assert mock_litellm.acompletion.call_count == 4
    assert [call.kwargs["model"] for call in mock_litellm.acompletion.call_args_list] == [
        "nvidia_nim/model-a", "nvidia_nim/model-a", "nvidia_nim/model-a",
        "groq/model-b",
    ]
    assert retry_sleep.await_count == 2
    assert any(event.get("text") == "recovered" for event in events)
    print("✅ 529 bounded retry + alternate OK")

    # ── Permanent model errors do not trigger fallback attempts ──
    mock_litellm.acompletion.reset_mock()
    mock_litellm.acompletion.side_effect = MockBadRequestError("unknown model")
    with patch.dict(os.environ, {
        "LITELLM_FALLBACK_MODELS": "groq/test-approved-fallback",
    }):
        try:
            async for _ in stream([create_user_message("hi")], enable_thinking=False):
                pass
            raise AssertionError("expected the model error to propagate")
        except ProviderRequestError as error:
            assert "provider_error" in str(error)
    assert mock_litellm.acompletion.call_count == 1
    print("✅ Permanent errors do not fall back OK")

    with patch.dict(os.environ, {"NVIDIA_API_KEY": "test-only"}, clear=True):
        _sync_provider_credentials()
        assert os.environ["NVIDIA_NIM_API_KEY"] == "test-only"
        assert "nvidia_nim/nvidia/nemotron-3.5-bolt-e4b" in list_models()
        assert "nvidia_nim/meta/llama-3.3-70b-instruct" not in list_models()
    print("✅ NVIDIA key alias OK")

    with patch.dict(os.environ, {"GEMINI_API_KEY": "test-only"}, clear=True):
        models = list_models()
        assert "gemini/gemini-3.8-flash" in models
        assert not any("gemini-2.5" in model for model in models)
    print("✅ Current Gemini catalog OK")

    with patch.dict(os.environ, {"GROQ_API_KEY": "test-only"}, clear=True):
        models = list_models()
        assert "groq/openai/gpt-oss-120b" in models
        assert "groq/openai/gpt-oss-20b" in models
        assert not any("llama-3." in model for model in models)
    print("✅ Current Groq catalog OK")

    # ── Non-streaming complete ──
        # ── Non-streaming complete ──
    mock_litellm.acompletion.reset_mock()
    mock_litellm.acompletion.side_effect = None        # ← clear exhausted list
    resp = MagicMock()
    resp.choices = [MagicMock()]
    resp.choices[0].message.content = "Summary of conversation"
    mock_litellm.acompletion.return_value = resp
    result = await complete([create_user_message("summarize")])
    assert result == "Summary of conversation"
    print("✅ Non-streaming complete OK")

    mock_litellm.acompletion.reset_mock()
    missing_response = MockNotFoundError("missing primary model")
    mock_litellm.acompletion.side_effect = [missing_response, resp]
    result = await complete(
        [create_user_message("summarize")],
        model="nvidia_nim/missing-model",
        fallback_model="groq/approved-model",
    )
    assert result == "Summary of conversation"
    assert [call.kwargs["model"] for call in mock_litellm.acompletion.call_args_list] == [
        "nvidia_nim/missing-model", "groq/approved-model",
    ]
    print("✅ Non-streaming explicit fallback OK")

    # ── Model switching ──
    original = get_active_model()
    set_active_model("gemini/gemini-2.5-flash")
    assert get_active_model() == "gemini/gemini-2.5-flash"
    set_active_model("nvidia/nemotron-3.5-bolt-e4b")
    assert get_active_model() == "nvidia_nim/nvidia/nemotron-3.5-bolt-e4b"
    set_active_model(original)
    print("✅ Model switching OK")

    print(f"\n🎉 All api.py tests passed ({len(FALLBACK_MODELS)} fallbacks)")


asyncio.run(main())