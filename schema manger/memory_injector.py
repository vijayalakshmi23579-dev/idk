"""Assemble a bounded per-turn context reminder from plans and memory."""
from __future__ import annotations

import os
from typing import Any, Callable

from memdir import (
    SelectionContext,
    find_relevant_memories,
    get_memory_dir,
    load_memory_content,
    memory_freshness_note,
    scan_memories,
)
from messages import get_text


MAX_MEMORY_CHARS = 5_000 * 4


def _enabled() -> bool:
    truthy = {"1", "true", "yes", "on"}
    if os.environ.get("CLAUDE_CODE_DISABLE_AUTO_MEMORY", "").lower() in truthy:
        return False
    if os.environ.get("CLAUDE_CODE_SIMPLE", "").lower() in truthy:
        return False
    if (
        os.environ.get("CLAUDE_CODE_REMOTE", "").lower() in truthy
        and not os.environ.get("REMOTE_MEMORY_DIR")
    ):
        return False
    return True


def _last_user_message(messages: list) -> str:
    for message in reversed(messages):
        if getattr(message, "type", None) == "user" and not getattr(message, "is_meta", False):
            return get_text(message) or ""
    return ""


def _recent_tools(messages: list) -> list[str]:
    names: list[str] = []
    for message in reversed(messages):
        if getattr(message, "type", None) != "assistant":
            continue
        for call in reversed(getattr(message, "tool_calls", None) or []):
            name = getattr(call, "name", None)
            if name and name not in names:
                names.append(name)
            if len(names) >= 20:
                return names
    return names


def _format_plan(plan: dict | None) -> str:
    if not plan or not plan.get("items"):
        return ""
    lines = [f"Plan: {plan.get('objective', 'Current work')}"]
    for item in plan["items"]:
        status = item.get("status", "pending").upper()
        lines.append(f"[{status}] {item.get('content', '')}")
    if plan.get("focus"):
        lines.append(f"Current focus: {plan['focus']}")
    return "\n".join(lines)


async def build_memory_injection(
    messages: list,
    context: Any,
    query_source: str = "",
    include_summary: bool = True,
    on_event: Callable[[dict[str, object]], None] | None = None,
) -> str:
    """Return one compact reminder; do not modify conversation history."""
    if query_source in ("compact", "session_memory"):
        return ""

    def emit(source: str, count: int = 0, names: list[str] | None = None) -> None:
        if on_event is None:
            return
        event: dict[str, object] = {
            "type": "runtime_activity",
            "area": "memory",
            "status": "injected",
            "source": source,
            "count": count,
        }
        if names:
            event["names"] = names[:20]
        try:
            on_event(event)
        except Exception:
            pass

    parts: list[str] = []
    store = getattr(context, "_store", None)
    session_id = getattr(context, "session_id", None)
    if store is not None and session_id:
        try:
            stored_plan = store.get_plan(session_id)
            plan = _format_plan(stored_plan)
            if plan:
                parts.append(plan)
                emit("plan", len(stored_plan.get("items", [])))
        except Exception:
            pass
        if include_summary:
            try:
                summary = store.get_root_summary(session_id)
                if summary and summary.get("content"):
                    parts.append("Conversation summary:\n" + summary["content"][:8_000])
                    emit("conversation_summary")
            except Exception:
                pass

        if _enabled():
            try:
                project_root = getattr(context, "project_root", None) or os.getcwd()
                memories = scan_memories(get_memory_dir(project_root))
                selection = getattr(context, "_memory_selection_context", None)
                if selection is None:
                    selection = SelectionContext(user_message="")
                    context._memory_selection_context = selection
                selection.user_message = _last_user_message(messages)
                selection.recently_used_tools = _recent_tools(messages)
                selection.already_read.update(
                    getattr(context, "loaded_nested_memory_paths", None) or set()
                )
                read_cache = getattr(context, "read_file_state", None)
                selection.already_read.update(getattr(read_cache, "files", {}).keys())

                selected = await find_relevant_memories(memories, selection)
                memory_parts: list[str] = []
                selected_names: list[str] = []
                for memory in selected:
                    body = load_memory_content(memory)
                    if not body:
                        continue
                    body = body[:MAX_MEMORY_CHARS]
                    freshness = memory_freshness_note(memory)
                    section = f"Memory: {memory.name}\n{body}"
                    if freshness:
                        section = freshness + "\n" + section
                    memory_parts.append(section)
                    selected_names.append(memory.name)
                    selection.already_surfaced.add(memory.filename)
                    selection.session_bytes_used += memory.size_bytes
                if memory_parts:
                    parts.append("Relevant file memories:\n" + "\n\n".join(memory_parts))
                    emit("project_files", len(memory_parts), selected_names)
            except (OSError, ValueError):
                pass

    return "\n\n".join(parts)


__all__ = ["MAX_MEMORY_CHARS", "build_memory_injection"]