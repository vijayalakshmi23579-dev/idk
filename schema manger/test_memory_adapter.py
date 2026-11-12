"""Offline tests for injected context and persistent memory tools."""
import asyncio
import os
import tempfile
import time
from unittest.mock import patch

from memdir import write_memory
from memory_injector import build_memory_injection
from memory_tools import (
    MemoryExpandInput,
    MemoryExpandTool,
    MemorySearchInput,
    MemorySearchTool,
)
from compact import microcompact
from messages import (
    ToolResultMessage, ToolUseBlock, create_assistant_message, create_user_message,
)
from store import Store
from tool import ToolUseContext


async def main():
    with tempfile.TemporaryDirectory() as temporary:
        memory_dir = os.path.join(temporary, "memory")
        os.makedirs(memory_dir)
        memory_path = write_memory(
            memory_dir, "Deployment Guide", "deployment process", "project",
            "Deploy from the release branch after tests pass.",
        )
        old_time = time.time() - 2 * 24 * 60 * 60
        os.utime(memory_path, (old_time, old_time))

        store = Store(os.path.join(temporary, "history.db"))
        session_id = "adapter-test"
        source_id = store.append_message(
            session_id, "user", "deployment release process",
        )
        summary_id = store.append_summary(
            session_id, 0, "Prior deployment discussion", source_msg_ids=[source_id],
        )
        store.save_plan(session_id, "Release deployment", [
            {"content": "Run tests", "status": "in_progress"},
        ], focus="Run tests")

        context = ToolUseContext()
        context._store = store
        context.session_id = session_id
        context.project_root = temporary
        messages = [create_user_message("How does deployment work?")]
        injection_events = []

        with patch.dict(os.environ, {
            "CLAUDE_COWORK_MEMORY_PATH_OVERRIDE": memory_dir,
            "CLAUDE_MEMORY_LLM_SELECT": "0",
        }):
            injected = await build_memory_injection(
                messages, context, on_event=injection_events.append,
            )
            assert "Plan: Release deployment" in injected
            assert "Prior deployment discussion" in injected
            assert "Deployment Guide" in injected
            assert "Freshness warning" in injected

            second_injection = await build_memory_injection(messages, context)
            assert "Deployment Guide" not in second_injection
        assert {event["source"] for event in injection_events} == {
            "plan", "conversation_summary", "project_files",
        }
        assert all("content" not in event for event in injection_events)
        assert "Deployment Guide" in next(
            event["names"] for event in injection_events
            if event["source"] == "project_files"
        )

        with patch.dict(os.environ, {"CLAUDE_CODE_DISABLE_AUTO_MEMORY": "1"}):
            disabled_injection = await build_memory_injection(messages, context)
            assert "Plan: Release deployment" in disabled_injection
            assert "Deployment Guide" not in disabled_injection

        search_result = await MemorySearchTool.call(
            MemorySearchInput(query="deployment"), context,
        )
        search_events = []
        search_result = await MemorySearchTool.call(
            MemorySearchInput(query="deployment"), context,
            on_progress=search_events.append,
        )
        assert search_result.data and search_result.data[0]["id"] == source_id
        assert [event["status"] for event in search_events] == ["started", "completed"]
        assert search_events[-1]["result_count"] == 1
        assert "query" not in search_events[-1]

        expand_events = []
        expand_result = await MemoryExpandTool.call(
            MemoryExpandInput(node_id=summary_id), context,
            on_progress=expand_events.append,
        )
        assert expand_result.data[0]["content"] == "deployment release process"
        assert [event["status"] for event in expand_events] == ["started", "completed"]
        assert "content" not in expand_events[-1]

        from main import _persist_query_messages

        context._stored_message_ids = []
        context._summary_source_offset = 0
        _persist_query_messages(context, [
            create_user_message("new deployment question"),
            create_assistant_message(content="new deployment answer"),
            create_user_message("Updated deployment context", is_compact_summary=True),
        ])
        updated_root = store.get_root_summary(session_id)
        assert updated_root["depth"] == 1
        expanded_root = store.expand(updated_root["id"])
        assert [message["content"] for message in expanded_root] == [
            "deployment release process",
            "new deployment question",
            "new deployment answer",
        ]

        old_tool_messages = []
        for index in range(4):
            tool_id = f"read-{index}"
            old_tool_messages.extend([
                create_assistant_message(tool_calls=[ToolUseBlock(
                    id=tool_id, name="Read", input={"file_path": f"f{index}.py"},
                )]),
                ToolResultMessage(tool_use_id=tool_id, content=f"original result {index}"),
            ])
        _persist_query_messages(context, old_tool_messages)
        compacted = await microcompact(old_tool_messages, context)
        tombstone = compacted["messages"][1]
        assert "MemoryExpand" in tombstone.content
        message_id = store.get_message_id_by_tool_use_id(session_id, "read-0")
        recovered = await MemoryExpandTool.call(
            MemoryExpandInput(node_id=message_id, kind="message"), context,
        )
        assert recovered.data[0]["content"] == "original result 0"
        foreign_id = store.append_message("other-session", "user", "private note")
        foreign = await MemoryExpandTool.call(
            MemoryExpandInput(node_id=foreign_id, kind="message"), context,
        )
        assert foreign.data == []

        active_view = store.get_active_view_messages(session_id)
        stored_count = store.count_messages(session_id)
        _persist_query_messages(context, active_view)
        assert store.count_messages(session_id) == stored_count

        store.close()
    print("Memory injection and search/expand tools OK")


if __name__ == "__main__":
    asyncio.run(main())