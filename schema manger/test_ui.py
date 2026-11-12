"""Headless smoke tests for the Textual agent UI."""
import asyncio
import os

os.environ["AGENT_STORE_PATH"] = ":memory:"

from textual.widgets import Button, Header, Input

import ui
from messages import ProgressMessage, ToolResultMessage
from registry import find
from ui import (
    AgentApp, PermissionScreen, PlanPanel, StatusBar, StreamingMarkdown,
    SystemNotice, ToolCard, UserMessage,
)


async def main():
    app = AgentApp()
    async with app.run_test(size=(110, 32)) as pilot:
        assert app.query_one(Header)
        composer = app.query_one("#composer", Input)
        assert app.query_one(StatusBar)
        assert app.query_one(PlanPanel)
        assert find("Read") is not None
        assert find("Write") is not None
        assert find("Bash") is not None
        assert app.ctx._store.count_messages(app.ctx.session_id) == 0
        app._append_widget(UserMessage("hello from user"))
        streaming = StreamingMarkdown()
        app._append_widget(streaming)
        streaming.append_text("assistant response")
        streaming.finish()
        await pilot.pause()
        assert streaming._buffer == "assistant response"
        composer.value = "/tools"
        await pilot.press("enter")
        await pilot.pause()
        assert any("Write" in notice._text for notice in app.query(SystemNotice))

        card = ToolCard("tool-1", "Bash", "echo test")
        app._tool_cards["tool-1"] = card
        app._append_widget(card)
        await pilot.pause()
        await app._dispatch_event(ProgressMessage(
            tool_use_id="tool-1",
            parent_tool_use_id="",
            data={"type": "bash_progress", "output": "still running"},
        ))
        assert card._progress_lines == ["still running"]
        await app._dispatch_event(ProgressMessage(
            tool_use_id="task-tool",
            parent_tool_use_id="",
            data={
                "type": "task_lifecycle", "task_id": "inspect",
                "status": "succeeded", "needs": [], "duration_ms": 12,
            },
        ))
        assert any("Task inspect: succeeded" in notice._text
                   for notice in app.query(SystemNotice))
        await app._dispatch_event(ToolResultMessage(
            tool_use_id="tool-1", content="done", is_error=False,
        ))
        assert "DONE" in card.title

        app.ctx._store.save_plan(app.ctx.session_id, "UI test", [
            {"content": "verify rendering", "status": "in_progress"},
        ], "verify rendering")
        app._refresh_plan()
        assert app.query_one(PlanPanel).title == "PLAN  0/1"

        await app._dispatch_event({
            "type": "runtime_activity", "area": "memory",
            "status": "injected", "source": "project_files", "count": 1,
            "names": ["Deployment Guide"],
        })
        assert any("Memory context injected" in notice._text
                   for notice in app.query(SystemNotice))
        await app._dispatch_event(ProgressMessage(
            tool_use_id="todo-tool",
            parent_tool_use_id="",
            data={
                "type": "plan_activity", "status": "updated",
                "item_count": 3, "completed_count": 1,
                "in_progress_count": 1, "cleared": False,
            },
        ))
        assert any("Plan updated: 3 items" in notice._text
                   for notice in app.query(SystemNotice))
        await app._dispatch_event({
            "type": "runtime_activity", "area": "provider",
            "status": "fallback", "model": "nvidia_nim/model-a",
            "alternate_model": "groq/model-b", "category": "provider_overloaded",
        })
        assert any("Provider fallback" in notice._text
                   for notice in app.query(SystemNotice))

        first_permission = app.run_worker(
            app._can_use_tool(
                type("Decision", (), {"message": "Allow the first worker?"})(),
            ),
            group="permission-test",
            exclusive=False,
        )
        await pilot.pause()
        assert isinstance(app.screen, PermissionScreen)
        app.screen.query_one("#allow", Button)
        app.screen.query_one("#deny", Button)
        second_permission = app.run_worker(
            app._can_use_tool(
                type("Decision", (), {"message": "Allow the second worker?"})(),
            ),
            group="permission-test",
            exclusive=False,
        )
        await pilot.pause()
        assert isinstance(app.screen, PermissionScreen)
        await pilot.press("y")
        assert await first_permission.wait() is True
        await pilot.pause()
        assert isinstance(app.screen, PermissionScreen)
        app.screen.query_one("#allow", Button)
        app.screen.query_one("#deny", Button)
        await pilot.press("n")
        assert await second_permission.wait() is False
        await pilot.pause()
        assert app.query_one(StatusBar).activity == "Ready"

        await pilot.press("ctrl+c")
        await pilot.pause()
        assert not app.is_running
        await pilot.pause(0.1)
    query_started = asyncio.Event()
    query_closed = asyncio.Event()

    async def stalled_query(_params):
        query_started.set()
        try:
            await asyncio.Event().wait()
        finally:
            query_closed.set()
            if False:
                yield None

    original_query = ui.query
    ui.query = stalled_query
    app = AgentApp()
    try:
        async with app.run_test(size=(80, 24)) as pilot:
            app._start_turn("close the interface")
            await asyncio.wait_for(query_started.wait(), timeout=1)
            await pilot.press("ctrl+c")
            await pilot.pause(0.1)
            assert query_closed.is_set()
            assert not app.is_running
    finally:
        ui.query = original_query
    print("Textual UI mounts with registered tools and session store")


if __name__ == "__main__":
    asyncio.run(main())
