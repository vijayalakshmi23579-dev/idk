"""test_query_primitives.py — verifies query_config, query_transitions, query_deps."""
import asyncio
import os
from unittest.mock import patch

from query_config import build_query_config, QueryConfig, QueryGates
from query_transitions import Continue, Terminal
from query_deps import production_deps, NoopDeps, QueryDeps, _production_call_model


async def main():
    # ── QueryConfig gates ──
    os.environ.pop("CLAUDE_CODE_DISABLE_STREAMING_TOOL_EXECUTION", None)
    os.environ.pop("CLAUDE_CODE_EMIT_TOOL_USE_SUMMARIES", None)
    os.environ.pop("USER_TYPE", None)
    cfg = build_query_config(session_id="test-1")
    assert cfg.session_id == "test-1"
    assert cfg.gates.streaming_tool_execution is True       # default ON
    assert cfg.gates.emit_tool_use_summaries is False       # default OFF
    assert cfg.gates.is_ant is False
    assert cfg.gates.fast_mode_enabled is True

    os.environ["CLAUDE_CODE_DISABLE_STREAMING_TOOL_EXECUTION"] = "1"
    os.environ["CLAUDE_CODE_EMIT_TOOL_USE_SUMMARIES"] = "1"
    os.environ["USER_TYPE"] = "ant"
    cfg2 = build_query_config()
    assert cfg2.gates.streaming_tool_execution is False
    assert cfg2.gates.emit_tool_use_summaries is True
    assert cfg2.gates.is_ant is True
    for k in ("CLAUDE_CODE_DISABLE_STREAMING_TOOL_EXECUTION",
              "CLAUDE_CODE_EMIT_TOOL_USE_SUMMARIES", "USER_TYPE"):
        os.environ.pop(k, None)
    print("✅ QueryConfig gates OK")

    # ── Continue: all 7 reasons construct ──
    for reason in ("next_turn", "collapse_drain_retry", "reactive_compact_retry",
                   "max_output_tokens_escalate", "max_output_tokens_recovery",
                   "token_budget_continuation", "stop_hook_blocking"):
        c = Continue(reason=reason)
        assert c.reason == reason
    c = Continue(reason="collapse_drain_retry", committed=3)
    assert c.committed == 3
    c = Continue(reason="max_output_tokens_recovery", attempt=2)
    assert c.attempt == 2
    print("✅ Continue (7 reasons) OK")

    # ── Terminal: all 10 reasons construct ──
    for reason in ("completed", "aborted_streaming", "aborted_tools",
                   "blocking_limit", "prompt_too_long", "image_error",
                   "model_error", "max_turns", "hook_stopped",
                   "stop_hook_prevented"):
        t = Terminal(reason=reason)
        assert t.reason == reason
    t = Terminal(reason="model_error", error=RuntimeError("boom"))
    assert isinstance(t.error, RuntimeError)
    t = Terminal(reason="max_turns", turn_count=5)
    assert t.turn_count == 5
    print("✅ Terminal (10 reasons) OK")

    # ── production_deps shape ──
    deps = production_deps()
    assert callable(deps.call_model)
    assert callable(deps.microcompact)
    assert callable(deps.autocompact)
    assert callable(deps.uuid)
    assert deps.uuid() != deps.uuid()
    print("✅ production_deps OK")

    # ── Production adapter preserves route metadata ──
    from messages import create_user_message

    captured = {}

    async def fake_stream(**kwargs):
        captured.update(kwargs)
        yield {
            "type": "runtime_activity", "area": "model",
            "status": "started", "model": kwargs.get("model"),
        }
        yield {"type": "message_stop", "stop_reason": "end_turn", "usage": {}}

    with patch("api.stream", fake_stream):
        result = [message async for message in _production_call_model(
            messages=[create_user_message("route test")],
            system_prompt="test",
            options={
                "model": "nvidia_nim/model-a",
                "fallback_model": "groq/model-b",
                "query_source": "subagent_task",
                "max_output_tokens_override": 4096,
            },
        )]
    assert captured["model"] == "nvidia_nim/model-a"
    assert captured["fallback_model"] == "groq/model-b"
    assert captured["query_source"] == "subagent_task"
    assert captured["max_tokens"] == 4096
    assert result[0]["type"] == "runtime_activity"
    assert result[-1].stop_reason == "end_turn"
    print("✅ Production adapter preserves route metadata")

    from api import ProviderRequestError

    async def failing_stream(**_kwargs):
        if False:
            yield {}
        raise ProviderRequestError(
            "provider=nvidia_nim model=nvidia_nim/missing category=model_or_route_not_found status=404",
            category="model_or_route_not_found",
            status_code=404,
            model="nvidia_nim/missing",
        )

    with patch("api.stream", failing_stream):
        failure = [message async for message in _production_call_model(
            messages=[create_user_message("route error")],
            system_prompt="test",
            options={"model": "nvidia_nim/missing"},
        )]
    assert failure[-1].is_api_error is True
    assert failure[-1].api_error == "model_or_route_not_found"
    assert "status=404" in failure[-1].content
    print("✅ Provider failure becomes a query error message")

    async def overflow_stream(**_kwargs):
        if False:
            yield {}
        raise ProviderRequestError(
            "provider=gemini model=gemini/model category=context_overflow status=400",
            category="context_overflow",
            status_code=400,
            model="gemini/model",
        )

    with patch("api.stream", overflow_stream):
        overflow = [message async for message in _production_call_model(
            messages=[create_user_message("too much context")],
            system_prompt="test",
            options={"model": "gemini/model"},
        )]
    assert overflow[-1].is_api_error is True
    assert overflow[-1].api_error == "invalid_request"
    assert "too long" in overflow[-1].content.lower()
    print("✅ Context overflow remains eligible for compaction")

    # ── NoopDeps yields scripted responses ──
    from messages import create_assistant_message
    r1 = create_assistant_message(content="hello")
    r2 = create_assistant_message(content="world")
    nd = NoopDeps(responses=[r1, r2])
    first = [m async for m in nd.call_model()]
    second = [m async for m in nd.call_model()]
    assert len(first) == len(second) == 1
    assert first[0].content == "hello"
    assert second[0].content == "world"
    assert nd.uuid() == "uuid-1"
    assert nd.uuid() == "uuid-2"
    print("✅ NoopDeps OK")

    # ── microcompact / autocompact pass-through ──
    out = await nd.microcompact([], None, "test")
    assert out["messages"] == []
    out = await nd.autocompact([], None, {}, "test", None, 0)
    assert out["compaction_result"] is None
    assert out["consecutive_failures"] is None
    print("✅ NoopDeps compaction passthrough OK")

    print("\n🎉 All query primitives tests passed")


asyncio.run(main())