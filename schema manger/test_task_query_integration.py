"""Exercise parent Task selection and child routing through the query adapter."""
import asyncio
import json
import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import api
from agent_tasks import AgentTasksTool
from messages import ToolResultMessage, create_user_message
from query import QueryParams, query
from registry import clear, register
from state import AppState, FileStateCache, ToolPermissionContext
from tool import ToolUseContext
from tools.read import ReadTool


def make_context() -> ToolUseContext:
    state = AppState(
        messages=[],
        tool_permission_context=ToolPermissionContext(mode="bypassPermissions"),
        read_file_cache=FileStateCache(),
    )
    context = ToolUseContext(
        get_app_state=lambda: state,
        set_app_state=lambda _update: None,
        read_file_state=state.read_file_cache,
    )
    context.model_override = "adapter-test/lead-model"
    context.project_root = os.getcwd()
    context.session_id = "task-query-integration"
    context.rendered_system_prompt = "Test system prompt"
    return context


def message_text(message) -> str:
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            block.get("text", "") if isinstance(block, dict) else str(block)
            for block in content
        )
    return ""


async def main() -> None:
    task_input = {
        "tasks": [
            {"id": "alpha", "instruction": "Find alpha", "complexity": 1},
            {"id": "beta", "instruction": "Find beta", "complexity": 3},
            {"id": "gamma", "instruction": "Find gamma", "complexity": 2},
            {
                "id": "join",
                "instruction": "Combine the findings",
                "needs": ["alpha", "beta", "gamma"],
                "complexity": 5,
            },
        ],
        "estimate": {
            "estimated_steps": 16,
            "sequential_dependency": False,
            "independent_branches": 3,
            "read_heavy": True,
            "disjoint_writes": False,
            "uncertain_steps": False,
        },
    }
    calls: list[dict] = []
    parent_calls = 0
    worker_prompts: dict[str, str] = {}

    async def scripted_stream(**kwargs):
        nonlocal parent_calls
        model = kwargs["model"]
        source = kwargs["query_source"]
        tool_names = {tool.name for tool in kwargs["tools"]}
        calls.append({"model": model, "source": source, "tools": tool_names})

        if source != "subagent_task":
            parent_calls += 1
            if parent_calls == 1:
                assert tool_names == {"Task", "Read"}
                yield {
                    "type": "tool_use",
                    "id": "parent-task-call",
                    "name": "Task",
                    "input": task_input,
                }
                yield {
                    "type": "message_stop",
                    "stop_reason": "tool_use",
                    "usage": {"output_tokens": 30},
                }
                return

            tool_results = [
                message for message in kwargs["messages"]
                if isinstance(message, ToolResultMessage)
            ]
            assert any(
                message.tool_use_id == "parent-task-call"
                and '"status": "succeeded"' in message.content
                for message in tool_results
            )
            yield {"type": "text_delta", "text": "Delegation complete"}
            yield {
                "type": "message_stop",
                "stop_reason": "end_turn",
                "usage": {"output_tokens": 3},
            }
            return

        assert model in {
            "adapter-test/worker-small",
            "adapter-test/worker-mid",
            "adapter-test/lead-model",
        }
        assert tool_names == {"Read"}
        prompt = "\n".join(message_text(message) for message in kwargs["messages"])
        task_id = next(
            task for task in ("alpha", "beta", "gamma", "join")
            if f"Task ID: {task}" in prompt
        )
        worker_prompts[task_id] = prompt
        yield {"type": "text_delta", "text": f"{task_id} findings"}
        yield {
            "type": "message_stop",
            "stop_reason": "end_turn",
            "usage": {"output_tokens": 4},
        }

    clear()
    register(AgentTasksTool)
    register(ReadTool)
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            tier_config = Path(temp_dir) / "tiers.json"
            tier_config.write_text(json.dumps({
                "version": 1,
                "tiers": {
                    "T1": {"model": "adapter-test/worker-small"},
                    "T2": {"model": "adapter-test/worker-mid"},
                    "T3": {"model": "ACTIVE_MODEL"},
                },
            }), encoding="utf-8")
            with patch.dict(os.environ, {
                "AGENT_WORKER_TIERS_CONFIG": str(tier_config),
            }), patch.object(api, "stream", scripted_stream):
                events = [event async for event in query(QueryParams(
                    messages=[create_user_message("Delegate independent work")],
                    system_prompt="Test system prompt",
                    tool_use_context=make_context(),
                    max_turns=4,
                ))]
    finally:
        clear()

    assert parent_calls == 2, (parent_calls, calls)
    assert calls[0] == {
        "model": "adapter-test/lead-model",
        "source": "repl_main_thread",
        "tools": {"Task", "Read"},
    }
    assert "alpha findings" in worker_prompts["join"]
    assert "beta findings" in worker_prompts["join"]
    assert "gamma findings" in worker_prompts["join"]
    assert calls.count({
        "model": "adapter-test/worker-small",
        "source": "subagent_task",
        "tools": {"Read"},
    }) == 2
    assert calls.count({
        "model": "adapter-test/worker-mid",
        "source": "subagent_task",
        "tools": {"Read"},
    }) == 1
    assert calls.count({
        "model": "adapter-test/lead-model",
        "source": "subagent_task",
        "tools": {"Read"},
    }) == 1
    task_events = [
        event.data for event in events
        if getattr(event, "type", None) == "progress"
        and event.data.get("type") == "task_lifecycle"
    ]
    assert ("join", "started") in {
        (event["task_id"], event["status"]) for event in task_events
    }
    assert ("join", "succeeded") in {
        (event["task_id"], event["status"]) for event in task_events
    }
    assert any(
        getattr(event, "reason", None) == "completed" for event in events
    )
    print("Task query integration and model-agnostic tier routing passed")


if __name__ == "__main__":
    asyncio.run(main())