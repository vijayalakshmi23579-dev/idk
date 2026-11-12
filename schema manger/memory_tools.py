"""Read-only tools for searching and expanding the persistent memory DAG."""
from __future__ import annotations

import json
from typing import Literal

from pydantic import BaseModel, Field

from permission_types import PermissionAllowDecision
from tool import ToolResult, ToolUseContext, build_tool


MAX_MEMORY_TOOL_RESULT_CHARS = 20_000


class MemorySearchInput(BaseModel):
    query: str = Field(min_length=1, description="Full-text search query")
    limit: int = Field(default=10, ge=1, le=50)


class MemoryExpandInput(BaseModel):
    node_id: int = Field(gt=0, description="Summary or message ID to expand")
    kind: Literal["summary", "message"] = "summary"


def _store(context: ToolUseContext):
    store = getattr(context, "_store", None)
    if store is None:
        raise ValueError("Persistent memory is unavailable in this session")
    return store


def _session_id(context: ToolUseContext) -> str:
    session_id = getattr(context, "session_id", None)
    if not session_id:
        raise ValueError("Memory lookup requires a session")
    return session_id


def _emit_memory_progress(on_progress, action: str, status: str, **details) -> None:
    if on_progress is None:
        return
    try:
        on_progress({
            "type": "memory_activity",
            "action": action,
            "status": status,
            **details,
        })
    except Exception:
        pass


async def _search(
    input: MemorySearchInput, context: ToolUseContext, _can_use_tool=None,
    _parent_message=None, on_progress=None,
) -> ToolResult:
    _emit_memory_progress(on_progress, "search", "started")
    try:
        results = _store(context).search(
            input.query, _session_id(context), limit=input.limit,
        )
    except Exception as error:
        _emit_memory_progress(on_progress, "search", "failed", error_type=type(error).__name__)
        raise ValueError(f"Memory search failed: {error}") from error
    _emit_memory_progress(
        on_progress, "search", "completed", result_count=len(results),
        source_ids=[item.get("id") for item in results[:20] if item.get("id") is not None],
    )
    return ToolResult(data=results)


async def _expand(
    input: MemoryExpandInput, context: ToolUseContext, _can_use_tool=None,
    _parent_message=None, on_progress=None,
) -> ToolResult:
    _emit_memory_progress(
        on_progress, "expand", "started", node_id=input.node_id, kind=input.kind,
    )
    try:
        results = _store(context).expand(
            input.node_id,
            kind=input.kind,
            session_id=_session_id(context),
        )
    except Exception as error:
        _emit_memory_progress(
            on_progress, "expand", "failed", node_id=input.node_id,
            kind=input.kind, error_type=type(error).__name__,
        )
        raise ValueError(f"Memory expansion failed: {error}") from error
    _emit_memory_progress(
        on_progress, "expand", "completed", node_id=input.node_id,
        kind=input.kind, result_count=len(results),
    )
    return ToolResult(data=results)


def _map_result(data, tool_use_id: str) -> dict:
    content = json.dumps(data, ensure_ascii=True, default=str)
    if len(content) > MAX_MEMORY_TOOL_RESULT_CHARS:
        content = content[:MAX_MEMORY_TOOL_RESULT_CHARS] + "\n[truncated]"
    return {"type": "tool_result", "tool_use_id": tool_use_id, "content": content}


async def _always_allow(input, context):
    return PermissionAllowDecision(updated_input=input.model_dump())


MemorySearchTool = build_tool(
    name="MemorySearch",
    description="Search this session's persistent conversation memory",
    prompt="Search prior stored messages using SQLite full-text search.",
    input_schema=MemorySearchInput,
    call=_search,
    map_tool_result_to_tool_result_block_param=_map_result,
    render_tool_use_message=lambda _input, _options: "MemorySearch",
    max_result_size_chars=MAX_MEMORY_TOOL_RESULT_CHARS,
    is_read_only=lambda _input: True,
    is_concurrency_safe=lambda _input: True,
    check_permissions=_always_allow,
)

MemoryExpandTool = build_tool(
    name="MemoryExpand",
    description="Recover original messages from a stored summary",
    prompt="Expand a summary node into its exact source messages.",
    input_schema=MemoryExpandInput,
    call=_expand,
    map_tool_result_to_tool_result_block_param=_map_result,
    render_tool_use_message=lambda _input, _options: "MemoryExpand",
    max_result_size_chars=MAX_MEMORY_TOOL_RESULT_CHARS,
    is_read_only=lambda _input: True,
    is_concurrency_safe=lambda _input: True,
    check_permissions=_always_allow,
)


__all__ = [
    "MemorySearchInput", "MemoryExpandInput", "MemorySearchTool",
    "MemoryExpandTool", "MAX_MEMORY_TOOL_RESULT_CHARS",
]