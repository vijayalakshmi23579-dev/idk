"""test_execute.py — no network. Mock tools exercise every pipeline stage."""
import asyncio
from pydantic import BaseModel

from execute import execute_tool, ExecutionResult, CANCEL_MESSAGE
from streaming_executor import _override_abort_controller
from state import AppState, ToolPermissionContext, FileStateCache
from tool import ToolUseContext, ToolResult, build_tool, ValidationResult
from permission_types import PermissionPassthroughDecision
from registry import register, clear


class In(BaseModel):
    x: int = 0


async def approve(_decision):
    return True


async def deny(_decision):
    return False


async def _passthrough(_args, _ctx):
    return PermissionPassthroughDecision(message="check")


def make_ctx(mode="bypassPermissions", allow=None, deny=None, abort_controller=None):
    state = AppState(
        messages=[],
        tool_permission_context=ToolPermissionContext(
            mode=mode,
            always_allow_rules=allow or {},
            always_deny_rules=deny or {},
        ),
        read_file_cache=FileStateCache(),
    )
    return ToolUseContext(
        get_app_state=lambda: state,
        set_app_state=lambda f: None,
        abort_controller=abort_controller,
    )


def make_tool(name, call_fn, check_result=None, validate_fn=None, backfill=None):
    async def _check(args, ctx):
        if check_result is not None:
            return check_result
        return PermissionPassthroughDecision(message="check")

    return build_tool(
        name=name,
        description="x",
        prompt="x",
        input_schema=In,
        call=call_fn,
        max_result_size_chars=1000,
        map_tool_result_to_tool_result_block_param=lambda o, i: {
            "type": "tool_result", "tool_use_id": i, "content": str(o),
        },
        render_tool_use_message=lambda i, o: "",
        check_permissions=_check,
        validate_input=validate_fn or (lambda inp, ctx: ValidationResult(result=True)),
        backfill_observable_input=backfill,
    )


class _Aborted:
    aborted = True
    reason = "interrupt"


async def main():
    clear()

    original_context = make_ctx()
    copied_context = _override_abort_controller(original_context, None)
    assert original_context._loop_guard is copied_context._loop_guard
    print("✅ Streaming context preserves loop-guard state OK")

    async def call_double(args, ctx, can=None, parent=None, on_progress=None):
        return ToolResult(data={"y": args.x * 2})

    register(make_tool("Doubler", call_double))

    # ── 1. Happy path ──
    r = await execute_tool({"id": "t1", "name": "Doubler", "input": {"x": 21}}, make_ctx())
    assert r.is_error is False
    assert "'y': 42" in r.result_message.content
    assert r.result_block["tool_use_id"] == "t1"
    print("✅ Happy path OK")

    # ── 2. Unknown tool ──
    r = await execute_tool({"id": "t2", "name": "Nope", "input": {}}, make_ctx())
    assert r.is_error is True and "Unknown tool" in r.result_message.content
    print("✅ Unknown tool OK")

    # ── 3. Invalid input (schema) ──
    r = await execute_tool(
        {"id": "t3", "name": "Doubler", "input": {"x": "not-an-int"}}, make_ctx(),
    )
    assert r.is_error is True and "Invalid input" in r.result_message.content
    print("✅ Invalid input OK")

    # ── 4. Validation failure (model-audience) ──
    async def call_noop(args, ctx, can=None, parent=None, on_progress=None):
        return ToolResult(data={})

    register(make_tool(
        "Validated", call_noop,
        validate_fn=lambda inp, ctx: ValidationResult(
            result=False, message="x must be positive", error_code=7,
        ),
    ))
    r = await execute_tool({"id": "t4", "name": "Validated", "input": {"x": 5}}, make_ctx())
    assert r.is_error is True
    assert "x must be positive" in r.result_message.content
    assert "error_code=7" in r.result_message.content
    print("✅ Validation failure OK")

    # ── 5. Permission deny (rule) ──
    r = await execute_tool(
        {"id": "t5", "name": "Doubler", "input": {"x": 5}},
        make_ctx(mode="default", deny={"userSettings": ["Doubler"]}),
    )
    assert r.is_error is True
    assert "denied" in r.result_message.content.lower()
    assert "IMPORTANT" in r.result_message.content  # workaround guidance
    print("✅ Permission deny OK")

    # ── 6. Ask approved via can_use_tool ──
    r = await execute_tool(
        {"id": "t6", "name": "Doubler", "input": {"x": 5}},
        make_ctx(mode="default"),
        can_use_tool=approve,
    )
    assert r.is_error is False
    assert "'y': 10" in r.result_message.content
    print("✅ Ask approved OK")

    # ── 7. Ask denied via can_use_tool ──
    r = await execute_tool(
        {"id": "t7", "name": "Doubler", "input": {"x": 5}},
        make_ctx(mode="default"),
        can_use_tool=deny,
    )
    assert r.is_error is True and "IMPORTANT" in r.result_message.content
    print("✅ Ask denied OK")

    # ── 8. Tool execution raises → error to model ──
    async def call_boom(args, ctx, can=None, parent=None, on_progress=None):
        raise RuntimeError("boom")

    register(make_tool("Boom", call_boom))
    r = await execute_tool(
        {"id": "t8", "name": "Boom", "input": {}},
        make_ctx(mode="default"),
        can_use_tool=approve,
    )
    assert r.is_error is True
    assert "boom" in r.result_message.content
    print("✅ Execution error OK")

    # ── 9. Abort check ──
    r = await execute_tool(
        {"id": "t9", "name": "Doubler", "input": {"x": 1}},
        make_ctx(abort_controller=_Aborted()),
    )
    assert r.is_error is True
    assert r.result_message.content == CANCEL_MESSAGE
    print("✅ Abort check OK")

    # ── 10. Context modifier returned ──
    def modifier(ctx):
        return ctx

    async def call_mod(args, ctx, can=None, parent=None, on_progress=None):
        return ToolResult(data={"y": 1}, context_modifier=modifier)

    register(make_tool("Modifier", call_mod))
    r = await execute_tool({"id": "t10", "name": "Modifier", "input": {}}, make_ctx())
    assert r.is_error is False
    assert r.context_modifier is modifier
    print("✅ Context modifier OK")

    # ── 11. backfill_observable_input is called ──
    called = {"n": 0}

    def backfill(inp):
        called["n"] += 1
        inp["injected"] = True

    register(make_tool("Backfilled", call_double, backfill=backfill))
    r = await execute_tool({"id": "t11", "name": "Backfilled", "input": {"x": 3}}, make_ctx())
    assert r.is_error is False
    assert called["n"] == 1
    print("✅ backfill_observable_input OK")

    print("\n🎉 All execute.py tests passed")


asyncio.run(main())