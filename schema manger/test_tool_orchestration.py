"""test_tool_orchestration.py — no network. Verifies partition + batch execution."""
import asyncio
import os
import time
from pydantic import BaseModel

from state import AppState, ToolPermissionContext, FileStateCache
from tool import ToolUseContext, ToolResult, build_tool
from permission_types import PermissionPassthroughDecision
from registry import register, clear
from tool_orchestration import (
    partition_tool_calls, run_tools, run_tools_serially,
    run_tools_concurrently, get_max_tool_use_concurrency, MessageUpdate,
)


class In(BaseModel):
    value: int = 0
    sleep: float = 0.1


async def _passthrough(_args, _ctx):
    return PermissionPassthroughDecision(message="ok")


def make_ctx():
    state = AppState(
        messages=[],
        tool_permission_context=ToolPermissionContext(mode="bypassPermissions"),
        read_file_cache=FileStateCache(),
    )
    in_progress = set()
    def setter(f):
        nonlocal in_progress
        in_progress = f(in_progress)
    return ToolUseContext(
        get_app_state=lambda: state,
        set_app_state=lambda f: None,
        set_in_progress_tool_use_ids=setter,
    )


_order: list[str] = []


def make_tool(name, safe, sleep=0.1):
    async def call(args, ctx, can=None, parent=None, on_progress=None):
        await asyncio.sleep(args.sleep)
        return ToolResult(
            data={"v": args.value},
            context_modifier=lambda c: (_order.append(name), c)[1],
        )

    return build_tool(
        name=name, description="x", prompt="x", input_schema=In,
        call=call, max_result_size_chars=1000,
        map_tool_result_to_tool_result_block_param=lambda o, i: {
            "type": "tool_result", "tool_use_id": i, "content": str(o["v"]),
        },
        render_tool_use_message=lambda i, o: "",
        check_permissions=_passthrough,
        is_concurrency_safe=lambda args: safe,
        is_read_only=lambda args: safe,
    )


async def main():
    clear()
    register(make_tool("SafeA", safe=True))
    register(make_tool("SafeB", safe=True))
    register(make_tool("Serial", safe=False))

    # ── Env var parsing ──
    os.environ.pop("CLAUDE_CODE_MAX_TOOL_USE_CONCURRENCY", None)
    assert get_max_tool_use_concurrency() == 10
    os.environ["CLAUDE_CODE_MAX_TOOL_USE_CONCURRENCY"] = "5"
    assert get_max_tool_use_concurrency() == 5
    os.environ["CLAUDE_CODE_MAX_TOOL_USE_CONCURRENCY"] = "not-a-number"
    assert get_max_tool_use_concurrency() == 10
    os.environ.pop("CLAUDE_CODE_MAX_TOOL_USE_CONCURRENCY", None)
    print("✅ get_max_tool_use_concurrency OK")

    ctx = make_ctx()

    # ── Partition: empty ──
    assert partition_tool_calls([], ctx) == []
    print("✅ Partition empty OK")

    # ── Partition: two safe → one batch ──
    b = partition_tool_calls([
        {"id": "1", "name": "SafeA", "input": {"value": 1}},
        {"id": "2", "name": "SafeB", "input": {"value": 2}},
    ], ctx)
    assert len(b) == 1 and b[0].is_concurrency_safe
    assert len(b[0].blocks) == 2
    print("✅ Partition two-safe OK")

    # ── Partition: safe, serial, safe → 3 batches ──
    b = partition_tool_calls([
        {"id": "1", "name": "SafeA", "input": {"value": 1}},
        {"id": "2", "name": "Serial", "input": {"value": 2}},
        {"id": "3", "name": "SafeB", "input": {"value": 3}},
    ], ctx)
    assert len(b) == 3
    assert b[0].is_concurrency_safe and b[1].is_concurrency_safe is False and b[2].is_concurrency_safe
    print("✅ Partition safe/serial/safe OK")

    # ── Partition: two serials → 2 batches (no merge) ──
    b = partition_tool_calls([
        {"id": "1", "name": "Serial", "input": {"value": 1}},
        {"id": "2", "name": "Serial", "input": {"value": 2}},
    ], ctx)
    assert len(b) == 2 and all(not x.is_concurrency_safe for x in b)
    print("✅ Partition two-serial (no merge) OK")

    # ── Partition: unknown tool → unsafe ──
    b = partition_tool_calls([{"id": "1", "name": "Ghost", "input": {}}], ctx)
    assert len(b) == 1 and b[0].is_concurrency_safe is False
    print("✅ Partition unknown-tool fail-closed OK")

    # ── Partition: schema parse failure → unsafe ──
    b = partition_tool_calls([
        {"id": "1", "name": "SafeA", "input": {"value": "not-int"}},
    ], ctx)
    assert len(b) == 1 and b[0].is_concurrency_safe is False
    print("✅ Partition parse-failure fail-closed OK")

    # ── Partition: is_concurrency_safe raises → unsafe ──
    def raising(_args):
        raise RuntimeError("boom")

    async def call_noop(args, ctx, can=None, parent=None, on_progress=None):
        return ToolResult(data={"v": 0})

    register(build_tool(
        name="Thrower", description="x", prompt="x", input_schema=In,
        call=call_noop, max_result_size_chars=1000,
        map_tool_result_to_tool_result_block_param=lambda o, i: {"content": "x"},
        render_tool_use_message=lambda i, o: "",
        check_permissions=_passthrough,
        is_concurrency_safe=raising,
    ))
    b = partition_tool_calls([{"id": "1", "name": "Thrower", "input": {"value": 1}}], ctx)
    assert b[0].is_concurrency_safe is False
    print("✅ Partition throwing is_concurrency_safe fail-closed OK")

    # ── run_tools: 3 safe → concurrent, fast ──
    _order.clear()
    start = time.time()
    updates: list[MessageUpdate] = []
    async for u in run_tools([
        {"id": "c1", "name": "SafeA", "input": {"value": 1, "sleep": 0.1}},
        {"id": "c2", "name": "SafeB", "input": {"value": 2, "sleep": 0.1}},
        {"id": "c3", "name": "SafeA", "input": {"value": 3, "sleep": 0.1}},
    ], [], None, make_ctx()):
        updates.append(u)
    elapsed = time.time() - start
    msgs = [u for u in updates if u.message is not None]
    assert len(msgs) == 3
    assert elapsed < 0.25, f"Expected ~0.1s parallel, got {elapsed:.2f}s"
    print(f"✅ 3 safe concurrent in {elapsed:.2f}s OK")

    # ── Modifiers applied in BLOCK order after batch ──
    # _order records modifier application order; must be c1, c2, c3 (block order)
    assert _order == ["SafeA", "SafeB", "SafeA"], f"Order was {_order}"
    print("✅ Modifiers applied in block order OK")

    # ── run_tools: serial + safe → serialized ──
    start = time.time()
    updates = []
    async for u in run_tools([
        {"id": "s1", "name": "Serial", "input": {"value": 1, "sleep": 0.1}},
        {"id": "s2", "name": "SafeA", "input": {"value": 2, "sleep": 0.1}},
    ], [], None, make_ctx()):
        updates.append(u)
    elapsed = time.time() - start
    assert len([u for u in updates if u.message]) == 2
    assert 0.18 < elapsed < 0.35, f"Expected ~0.2s serialized, got {elapsed:.2f}s"
    print(f"✅ Serial + safe in {elapsed:.2f}s OK")

    # ── Serial path applies modifiers immediately ──
    _order.clear()
    updates = []
    async for u in run_tools_serially(
        [
            {"id": "z1", "name": "Serial", "input": {"value": 1, "sleep": 0.01}},
            {"id": "z2", "name": "Serial", "input": {"value": 2, "sleep": 0.01}},
        ],
        [], None, make_ctx(),
    ):
        updates.append(u)
    assert _order == ["Serial", "Serial"]
    print("✅ Serial applies modifiers immediately OK")

    # ── Empty run_tools ──
    updates = [u async for u in run_tools([], [], None, make_ctx())]
    assert updates == []
    print("✅ Empty run_tools OK")

    print("\n🎉 All tool_orchestration.py tests passed")


asyncio.run(main())