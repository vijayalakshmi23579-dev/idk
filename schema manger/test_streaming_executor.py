"""Regression tests for streaming tool execution lifecycle."""
import asyncio

from pydantic import BaseModel

from permission_types import PermissionPassthroughDecision
from messages import ProgressMessage, ToolResultMessage
from registry import clear, register
from state import AppState, FileStateCache, ToolPermissionContext
from streaming_executor import StreamingToolExecutor
from tool import ToolResult, ToolUseContext, build_tool


class ToolInput(BaseModel):
    label: str
    delay: float


async def _passthrough(_input, _context):
    return PermissionPassthroughDecision(message="allowed")


def _make_tool(name: str, safe: bool = True, emit_progress: bool = False):
    async def call(input, _context, _can_use_tool=None, _parent=None, on_progress=None):
        if emit_progress and on_progress is not None:
            on_progress({"type": "test_progress", "output": input.label})
        await asyncio.sleep(input.delay)
        return ToolResult(data={"label": input.label})

    return build_tool(
        name=name,
        description=name,
        prompt=name,
        input_schema=ToolInput,
        call=call,
        max_result_size_chars=1000,
        map_tool_result_to_tool_result_block_param=lambda data, tool_use_id: {
            "type": "tool_result",
            "tool_use_id": tool_use_id,
            "content": data["label"],
        },
        render_tool_use_message=lambda _input, _options: name,
        check_permissions=_passthrough,
        is_read_only=lambda _input: safe,
        is_concurrency_safe=lambda _input: safe,
    )


async def _test_parallel_tools_survive_first_completion():
    clear()
    fast_tool = register(_make_tool("Fast"))
    slow_tool = register(_make_tool("Slow"))
    state = AppState(
        messages=[],
        tool_permission_context=ToolPermissionContext(mode="bypassPermissions"),
        read_file_cache=FileStateCache(),
    )
    context = ToolUseContext(
        get_app_state=lambda: state,
        set_app_state=lambda _update: None,
    )
    executor = StreamingToolExecutor([fast_tool, slow_tool], None, context)

    executor.add_tool(
        {"id": "fast-id", "name": "Fast", "input": {"label": "fast", "delay": 0.01}},
        None,
    )
    executor.add_tool(
        {"id": "slow-id", "name": "Slow", "input": {"label": "slow", "delay": 0.08}},
        None,
    )

    results = [
        update.message
        async for update in executor.get_remaining_results()
        if update.message is not None
    ]

    assert [result.content for result in results] == ["fast", "slow"]
    assert all(not result.is_error for result in results)
    clear()


async def _test_mutating_tools_wait_for_release_and_discard():
    clear()
    calls = []
    tool = _make_tool("Mutation", safe=False)
    original_call = tool.call

    async def track_call(*args):
        calls.append(True)
        return await original_call(*args)

    tool.call = track_call
    register(tool)
    state = AppState(
        messages=[],
        tool_permission_context=ToolPermissionContext(mode="bypassPermissions"),
        read_file_cache=FileStateCache(),
    )
    context = ToolUseContext(get_app_state=lambda: state)
    executor = StreamingToolExecutor([tool], None, context)
    executor.add_tool(
        {"id": "mutation-id", "name": "Mutation", "input": {"label": "write", "delay": 0}},
        None,
    )

    assert calls == []
    executor.discard()
    assert calls == []

    released_executor = StreamingToolExecutor([tool], None, context)
    released_executor.add_tool(
        {"id": "released-id", "name": "Mutation", "input": {"label": "write", "delay": 0}},
        None,
    )
    assert calls == []
    released_executor.release_deferred_tools()
    results = [
        update.message
        async for update in released_executor.get_remaining_results()
        if isinstance(update.message, ToolResultMessage)
    ]
    assert calls == [True]
    assert len(results) == 1 and not results[0].is_error
    clear()


async def _test_progress_is_emitted_separately():
    clear()
    tool = register(_make_tool("Progress", emit_progress=True))
    state = AppState(
        messages=[],
        tool_permission_context=ToolPermissionContext(mode="bypassPermissions"),
        read_file_cache=FileStateCache(),
    )
    context = ToolUseContext(get_app_state=lambda: state)
    executor = StreamingToolExecutor([tool], None, context)
    executor.add_tool(
        {"id": "progress-id", "name": "Progress", "input": {"label": "working", "delay": 0.02}},
        None,
    )

    messages = [
        update.message
        async for update in executor.get_remaining_results()
        if update.message is not None
    ]
    assert any(isinstance(message, ProgressMessage) for message in messages)
    assert any(
        isinstance(message, ToolResultMessage) and message.content == "working"
        for message in messages
    )
    clear()


if __name__ == "__main__":
    asyncio.run(_test_parallel_tools_survive_first_completion())
    asyncio.run(_test_mutating_tools_wait_for_release_and_discard())
    asyncio.run(_test_progress_is_emitted_separately())
    print("Streaming executor tests passed")