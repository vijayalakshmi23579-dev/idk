"""test_query.py — no network. Scripted responses exercise every transition."""
import asyncio
from dataclasses import dataclass

from messages import (
    AssistantMessage, ToolUseBlock, create_assistant_message,
    create_compact_boundary_message, create_user_message,
)
from permission_types import PermissionPassthroughDecision
from query import (
    QueryParams, query, get_context_window_for_model,
    get_max_output_tokens_for_model,
)
from query_deps import NoopDeps, QueryDeps
from compact import CompactionResult
from query_transitions import Terminal, Continue
from registry import register, clear
from state import AppState, ToolPermissionContext, FileStateCache
from tool import ToolUseContext, ToolResult, build_tool
from pydantic import BaseModel


class In(BaseModel):
    x: int = 0


async def _passthrough(_args, _ctx):
    return PermissionPassthroughDecision(message="ok")


def make_ctx():
    state = AppState(
        messages=[],
        tool_permission_context=ToolPermissionContext(mode="bypassPermissions"),
        read_file_cache=FileStateCache(),
    )
    return ToolUseContext(
        get_app_state=lambda: state,
        set_app_state=lambda f: None,
    )


def make_tool(name):
    async def call(args, ctx, can=None, parent=None, on_progress=None):
        return ToolResult(data={"y": args.x * 2})

    return build_tool(
        name=name, description="x", prompt="x", input_schema=In,
        call=call, max_result_size_chars=1000,
        map_tool_result_to_tool_result_block_param=lambda o, i: {
            "type": "tool_result", "tool_use_id": i, "content": str(o["y"]),
        },
        render_tool_use_message=lambda i, o: "",
        check_permissions=_passthrough,
    )


async def _collect(agen):
    """Collect all events from an async generator."""
    events = []
    async for e in agen:
        events.append(e)
    return events


def _terminal(events):
    for e in reversed(events):
        if isinstance(e, Terminal):
            return e
    return None


async def main():
    clear()
    register(make_tool("Doubler"))

    # ── 1. Model info: known + fallback ──
    assert get_context_window_for_model("claude-sonnet-4-5") == 200_000
    assert get_context_window_for_model("gemini/gemini-2.5-flash") == 1_048_576
    assert get_context_window_for_model("groq/llama-3.3-70b-versatile") == 131_072
    assert get_context_window_for_model("unknown-model") == 200_000  # default
    assert get_max_output_tokens_for_model("claude-sonnet-4-5") > 0
    print("✅ Model info OK")

    # ── 2. Simple text response → completed ──
    deps = NoopDeps(responses=[
        create_assistant_message(content="Hello there", stop_reason="end_turn"),
    ])
    params = QueryParams(
        messages=[create_user_message("hi")],
        system_prompt="You are helpful",
        tool_use_context=make_ctx(),
        deps=deps.as_deps(),
    )
    events = await _collect(query(params))
    term = _terminal(events)
    assert term is not None and term.reason == "completed"
    print("✅ Text response → completed OK")

    # ── 3. Tool call → next_turn → completed ──
    deps = NoopDeps(responses=[
        create_assistant_message(
            tool_calls=[ToolUseBlock(id="tc1", name="Doubler", input={"x": 5})],
            stop_reason="tool_use",
        ),
        create_assistant_message(content="Done: 10", stop_reason="end_turn"),
    ])
    params = QueryParams(
        messages=[create_user_message("double 5")],
        system_prompt="You are helpful",
        tool_use_context=make_ctx(),
        deps=deps.as_deps(),
    )
    events = await _collect(query(params))
    term = _terminal(events)
    assert term is not None and term.reason == "completed"
    tool_results = [e for e in events if getattr(e, "type", None) == "tool_result"]
    assert len(tool_results) == 1
    assert "10" in tool_results[0].content
    print("✅ Tool call → next_turn → completed OK")

    # ── 4. Max turns check ──
    deps = NoopDeps(responses=[
        create_assistant_message(
            tool_calls=[ToolUseBlock(id="t1", name="Doubler", input={"x": 1})],
            stop_reason="tool_use",
        ),
        create_assistant_message(
            tool_calls=[ToolUseBlock(id="t2", name="Doubler", input={"x": 2})],
            stop_reason="tool_use",
        ),
        create_assistant_message(content="never reached", stop_reason="end_turn"),
    ])
    params = QueryParams(
        messages=[create_user_message("go")],
        system_prompt="",
        tool_use_context=make_ctx(),
        deps=deps.as_deps(),
        max_turns=1,
    )
    events = await _collect(query(params))
    term = _terminal(events)
    assert term is not None and term.reason == "max_turns"
    assert term.turn_count == 2
    print("✅ max_turns → Terminal(turn_count) OK")

    # ── 5. Token budget continuation ──
    budget_calls = {"n": 0}
    def get_budget():
        budget_calls["n"] += 1
        return 100_000  # big budget
    def get_tokens():
        return 10_000 + budget_calls["n"] * 20_000
    deps = NoopDeps(responses=[
        create_assistant_message(content="turn 1", stop_reason="end_turn"),
        create_assistant_message(content="turn 2", stop_reason="end_turn"),
        create_assistant_message(content="turn 3 done", stop_reason="end_turn"),
    ])
    params = QueryParams(
        messages=[create_user_message("start")],
        system_prompt="",
        tool_use_context=make_ctx(),
        deps=deps.as_deps(),
        get_current_turn_token_budget=get_budget,
        get_turn_output_tokens=get_tokens,
    )
    events = await _collect(query(params))
    term = _terminal(events)
    assert term is not None and term.reason == "completed"
    assert budget_calls["n"] >= 2, f"Expected continuation, got {budget_calls['n']}"
    print("✅ Token budget continuation OK")

    # ── 6. Abort during streaming ──
    class _AbortCtx:
        aborted = True
    deps = NoopDeps(responses=[
        create_assistant_message(content="unreachable", stop_reason="end_turn"),
    ])
    ctx = make_ctx()
    ctx.abort_controller = _AbortCtx()
    params = QueryParams(
        messages=[create_user_message("go")],
        system_prompt="",
        tool_use_context=ctx,
        deps=deps.as_deps(),
    )
    events = await _collect(query(params))
    term = _terminal(events)
    assert term is not None and term.reason == "aborted_streaming"
    print("✅ Abort during streaming OK")

    # ── 7. Empty response ──
    deps = NoopDeps(responses=[
        create_assistant_message(content=None, stop_reason="end_turn"),
    ])
    params = QueryParams(
        messages=[create_user_message("hi")],
        system_prompt="",
        tool_use_context=make_ctx(),
        deps=deps.as_deps(),
    )
    events = await _collect(query(params))
    term = _terminal(events)
    assert term is not None and term.reason == "completed"
    print("✅ Empty response → completed OK")

    # ── 8. Compaction publishes the bounded transcript to the caller ──
    base_deps = NoopDeps(responses=[
        create_assistant_message(content="continued", stop_reason="end_turn"),
    ]).as_deps()
    compact_result = CompactionResult(
        boundary_marker=create_compact_boundary_message("auto", 100_000),
        summary_messages=[create_user_message("earlier work", is_compact_summary=True)],
        attachments=[],
        hook_results=[],
    )

    async def return_compaction(*_args, **_kwargs):
        return {"compaction_result": compact_result, "consecutive_failures": 0}

    deps = QueryDeps(
        call_model=base_deps.call_model,
        microcompact=base_deps.microcompact,
        autocompact=return_compaction,
        uuid=base_deps.uuid,
    )
    params = QueryParams(
        messages=[create_user_message("old, large transcript")],
        system_prompt="",
        tool_use_context=make_ctx(),
        deps=deps,
    )
    events = await _collect(query(params))
    assert _terminal(events).reason == "completed"
    assert params.messages[0].subtype == "compact_boundary"
    assert any(getattr(m, "is_compact_summary", False) for m in params.messages)
    assert not any(
        getattr(m, "content", None) == "old, large transcript"
        for m in params.messages
    )
    print("✅ Compaction publishes bounded transcript OK")

    # ── 9. Rejected tool → completed (denial message to model) ──
    deps = NoopDeps(responses=[
        create_assistant_message(
            tool_calls=[ToolUseBlock(id="t1", name="Doubler", input={"x": 1})],
            stop_reason="tool_use",
        ),
        create_assistant_message(content="OK, I'll stop", stop_reason="end_turn"),
    ])
    params = QueryParams(
        messages=[create_user_message("try")],
        system_prompt="",
        tool_use_context=make_ctx(),
        deps=deps.as_deps(),
        can_use_tool=lambda _d: _async_false(),
    )
    events = await _collect(query(params))
    term = _terminal(events)
    assert term is not None and term.reason == "completed"
    print("✅ Rejected tool → completed OK")

    print("\n🎉 All query.py tests passed")


async def _async_false():
    return False


asyncio.run(main())