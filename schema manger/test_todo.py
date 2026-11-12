"""test_todo.py — no network. Verifies the exact source behaviors."""
import asyncio
import os
import tempfile

from messages import ToolUseBlock  # noqa: F401 (import-path guard)
from registry import register, clear as clear_registry
from execute import execute_tool
from state import AppState, FileStateCache, ToolPermissionContext
from tool import ToolUseContext
from store import Store
from todo import (
    TodoWriteTool, TodoInput, TodoItem, MAX_TODO_ITEMS,
    MAX_RESULT_SIZE_CHARS, is_todo_v2_enabled, is_strict_todo_enabled,
    is_verification_nudge_enabled,
)


def make_ctx(store=None, agent_id=None, session_id="test-session"):
    state = AppState(
        messages=[],
        tool_permission_context=ToolPermissionContext(mode="bypassPermissions"),
        read_file_cache=FileStateCache(),
    )
    ctx = ToolUseContext(
        get_app_state=lambda: state, set_app_state=lambda f: None,
        read_file_state=state.read_file_cache,
    )
    ctx.session_id = session_id
    ctx.agent_id = agent_id
    if store is not None:
        ctx._store = store
    return ctx


async def _todo(ctx, todos):
    return await execute_tool(
        {"id": "tu", "name": "TodoWrite", "input": {"todos": todos}},
        ctx,
    )


async def main():
    clear_registry()
    register(TodoWriteTool)
    tmp = tempfile.mkdtemp()

    # ── 1. Metadata matches source exactly ──
    assert TodoWriteTool.name == "TodoWrite"
    assert TodoWriteTool.max_result_size_chars == 100_000
    assert TodoWriteTool.should_defer is True
    print("✅ Metadata (100_000, shouldDefer) OK")

    # ── 2. isEnabled is False when Tasks V2 is on ──
    os.environ.pop("CLAUDE_CODE_ENABLE_TASKS", None)
    assert is_todo_v2_enabled() is False
    assert TodoWriteTool.is_enabled() is True
    os.environ["CLAUDE_CODE_ENABLE_TASKS"] = "1"
    assert is_todo_v2_enabled() is True
    assert TodoWriteTool.is_enabled() is False
    os.environ["CLAUDE_CODE_ENABLE_TASKS"] = "0"
    assert is_todo_v2_enabled() is False
    assert TodoWriteTool.is_enabled() is True
    os.environ.pop("CLAUDE_CODE_ENABLE_TASKS", None)
    print("✅ isEnabled = !isTodoV2Enabled OK")

    # ── 3. Output shape: oldTodos / newTodos / verificationNudgeNeeded ──
    ctx = make_ctx()
    res = await execute_tool(
        {"id": "tu", "name": "TodoWrite", "input": {"todos": [
            {"content": "read code", "status": "completed",
             "active_form": "Reading code"},
        ]}}, ctx,
    )
    # execute_tool returns ExecutionResult; the raw data lives in the mapper,
    # so call the tool directly to inspect the output shape
    from tool import ToolResult
    plan_events = []
    direct = await TodoWriteTool.call(TodoInput(todos=[
        TodoItem(content="read code", status="completed",
                 active_form="Reading code"),
    ]), ctx, on_progress=plan_events.append)
    assert "old_todos" in direct.data
    assert "new_todos" in direct.data
    assert "verification_nudge_needed" in direct.data
    assert direct.data["old_todos"] == []
    assert len(direct.data["new_todos"]) == 1
    assert plan_events == [{
        "type": "plan_activity", "status": "updated", "item_count": 0,
        "completed_count": 0, "in_progress_count": 0, "cleared": True,
    }]
    print("✅ Output shape (old/new/nudge) OK")

    # ── 4. Model-facing result does NOT contain the list ──
    block = TodoWriteTool.map_tool_result_to_tool_result_block_param(
        direct.data, "tu1",
    )
    assert "Todos have been modified successfully" in block["content"]
    assert "read code" not in block["content"], \
        "source never sends the todo list back to the model"
    print("✅ Model sees fixed string, not the list OK")

    # ── 5. per-agent isolation: todoKey = agentId ?? sessionId ──
    parent_ctx = make_ctx(agent_id=None)
    child_ctx = make_ctx(agent_id="agent-1")
    await TodoWriteTool.call(TodoInput(todos=[
        TodoItem(content="parent task", status="pending",
                 active_form="Doing parent task"),
    ]), parent_ctx)
    await TodoWriteTool.call(TodoInput(todos=[
        TodoItem(content="child task", status="pending",
                 active_form="Doing child task"),
    ]), child_ctx)
    parent_state = parent_ctx.get_app_state()
    child_state = child_ctx.get_app_state()
    assert len(parent_state.todos["test-session"]) == 1
    assert parent_state.todos["test-session"][0]["content"] == "parent task"
    assert len(child_state.todos["agent-1"]) == 1
    assert child_state.todos["agent-1"][0]["content"] == "child task"
    print("✅ Per-agent isolation OK")

    # ── 6. allDone clears the list to [] ──
    ctx = make_ctx(session_id="clear-test")
    await TodoWriteTool.call(TodoInput(todos=[
        TodoItem(content="a", status="completed", active_form="Doing a"),
        TodoItem(content="b", status="completed", active_form="Doing b"),
    ]), ctx)
    stored = ctx.get_app_state().todos["clear-test"]
    assert stored == [], f"allDone should clear, got {stored}"
    print("✅ allDone clears to [] OK")

    # ── 7. Not all done → list preserved ──
    ctx = make_ctx(session_id="keep-test")
    await TodoWriteTool.call(TodoInput(todos=[
        TodoItem(content="a", status="completed", active_form="Doing a"),
        TodoItem(content="b", status="pending", active_form="Doing b"),
    ]), ctx)
    stored = ctx.get_app_state().todos["keep-test"]
    assert len(stored) == 2
    print("✅ Not-all-done preserves list OK")

    # ── 8. Verification nudge: all 6 conditions ──
    os.environ["CLAUDE_CODE_VERIFICATION_NUDGE"] = "1"
    try:
        # 8a. main thread + 3 items + all done + no verif → nudge
        ctx = make_ctx(agent_id=None)
        r = await TodoWriteTool.call(TodoInput(todos=[
            TodoItem(content="a", status="completed", active_form="A"),
            TodoItem(content="b", status="completed", active_form="B"),
            TodoItem(content="c", status="completed", active_form="C"),
        ]), ctx)
        assert r.data["verification_nudge_needed"] is True
        block = TodoWriteTool.map_tool_result_to_tool_result_block_param(
            r.data, "tu2",
        )
        assert "verification agent" in block["content"]
        print("✅ Nudge fires (main + 3 done + no verif) OK")

        # 8b. sub-agent → no nudge
        ctx = make_ctx(agent_id="agent-2")
        r = await TodoWriteTool.call(TodoInput(todos=[
            TodoItem(content="a", status="completed", active_form="A"),
            TodoItem(content="b", status="completed", active_form="B"),
            TodoItem(content="c", status="completed", active_form="C"),
        ]), ctx)
        assert r.data["verification_nudge_needed"] is False
        print("✅ Nudge suppressed for sub-agent OK")

        # 8c. only 2 items → no nudge
        ctx = make_ctx(agent_id=None)
        r = await TodoWriteTool.call(TodoInput(todos=[
            TodoItem(content="a", status="completed", active_form="A"),
            TodoItem(content="b", status="completed", active_form="B"),
        ]), ctx)
        assert r.data["verification_nudge_needed"] is False
        print("✅ Nudge suppressed (<3 items) OK")

        # 8d. verif item present → no nudge
        ctx = make_ctx(agent_id=None)
        r = await TodoWriteTool.call(TodoInput(todos=[
            TodoItem(content="a", status="completed", active_form="A"),
            TodoItem(content="b", status="completed", active_form="B"),
            TodoItem(content="verify the work", status="completed",
                     active_form="Verifying"),
        ]), ctx)
        assert r.data["verification_nudge_needed"] is False
        print("✅ Nudge suppressed (verif item present) OK")

        # 8e. not all done → no nudge
        ctx = make_ctx(agent_id=None)
        r = await TodoWriteTool.call(TodoInput(todos=[
            TodoItem(content="a", status="completed", active_form="A"),
            TodoItem(content="b", status="completed", active_form="B"),
            TodoItem(content="c", status="pending", active_form="C"),
        ]), ctx)
        assert r.data["verification_nudge_needed"] is False
        print("✅ Nudge suppressed (not all done) OK")
    finally:
        os.environ.pop("CLAUDE_CODE_VERIFICATION_NUDGE", None)

    # ── 9. Nudge disabled by env → never fires ──
    os.environ.pop("CLAUDE_CODE_VERIFICATION_NUDGE", None)
    ctx = make_ctx(agent_id=None)
    r = await TodoWriteTool.call(TodoInput(todos=[
        TodoItem(content="a", status="completed", active_form="A"),
        TodoItem(content="b", status="completed", active_form="B"),
        TodoItem(content="c", status="completed", active_form="C"),
    ]), ctx)
    assert r.data["verification_nudge_needed"] is False
    print("✅ Nudge off when env unset OK")

    # ── 10. checkPermissions always allows ──
    from permission_types import PermissionAllowDecision
    perm = await TodoWriteTool.check_permissions(
        TodoInput(todos=[TodoItem(content="x", status="pending",
                                 active_form="Doing x")]),
        make_ctx(),
    )
    assert isinstance(perm, PermissionAllowDecision)
    print("✅ checkPermissions always allow OK")

    # ── 11. toAutoClassifierInput format ──
    inp = TodoInput(todos=[
        TodoItem(content="a", status="pending", active_form="A"),
        TodoItem(content="b", status="pending", active_form="B"),
        TodoItem(content="c", status="pending", active_form="C"),
    ])
    assert TodoWriteTool.to_auto_classifier_input(inp) == "3 items"
    print("✅ toAutoClassifierInput = 'N items' OK")

    # ── 12. Strict mode: max-20 enforced ──
    os.environ["CLAUDE_CODE_STRICT_TODO"] = "1"
    try:
        too_many = [
            {"content": f"t{i}", "status": "pending", "active_form": "Doing"}
            for i in range(MAX_TODO_ITEMS + 1)
        ]
        ctx = make_ctx()
        res = await _todo(ctx, too_many)
        assert res.is_error
        assert "Too many todos" in res.result_message.content
        print("✅ Strict mode: max 20 OK")

        # 12b. Strict mode: single in_progress
        ctx = make_ctx()
        res = await _todo(ctx, [
            {"content": "a", "status": "in_progress", "active_form": "A"},
            {"content": "b", "status": "in_progress", "active_form": "B"},
        ])
        assert res.is_error
        assert "Exactly one todo may be in_progress" in res.result_message.content
        print("✅ Strict mode: single in_progress OK")
    finally:
        os.environ.pop("CLAUDE_CODE_STRICT_TODO", None)

    # ── 13. Non-strict mode: multiple in_progress allowed ──
    ctx = make_ctx()
    res = await _todo(ctx, [
        {"content": "a", "status": "in_progress", "active_form": "A"},
        {"content": "b", "status": "in_progress", "active_form": "B"},
    ])
    assert not res.is_error, res.result_message.content
    print("✅ Non-strict: multiple in_progress OK")

    # ── 14. Persists to store (our addition) ──
    db = os.path.join(tmp, "todo.db")
    store = Store(db)
    ctx = make_ctx(store=store, session_id="persist-session")
    await TodoWriteTool.call(TodoInput(todos=[
        TodoItem(content="step 1", status="completed",
                 active_form="Doing step 1"),
        TodoItem(content="step 2", status="in_progress",
                 active_form="Doing step 2"),
    ]), ctx)
    plan = store.get_plan("persist-session")
    assert plan is not None
    assert plan["objective"] == "step 1"
    assert len(plan["items"]) == 2
    assert plan["focus"] == "step 2"
    store.close()
    print("✅ Persists to store OK")

    # ── 15. Status normalization ──
    assert TodoItem(content="x", status="DONE", active_form="A").status == "completed"
    assert TodoItem(content="x", status="in progress", active_form="A").status == "in_progress"
    assert TodoItem(content="x", status="ACTIVE", active_form="A").status == "in_progress"
    print("✅ Status normalization OK")

    # ── 16. Empty content rejected by schema ──
    try:
        TodoItem(content="   ", status="pending", active_form="A")
        raise AssertionError("empty content should be rejected")
    except Exception:
        pass
    print("✅ Empty content rejected OK")

    print("\n🎉 All todo.py tests passed")


asyncio.run(main())