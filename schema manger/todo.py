"""
todo.py — TodoWrite: the externalized plan state.

Port of src/tools/TodoWriteTool/TodoWriteTool.ts.

WHAT THE SOURCE ACTUALLY DOES (verified against three independent analyses):

  Input:  { todos: [{content, status, activeForm}] }
  Output: { oldTodos, newTodos, verificationNudgeNeeded? }

  call({todos}, context):
    todoKey  = context.agentId ?? getSessionId()     ← per-agent isolation
    oldTodos = appState.todos[todoKey] ?? []
    allDone  = todos.every(t => t.status == 'completed')
    newTodos = allDone ? [] : todos                  ← CLEARS on completion
    setAppState(prev => ({...prev, todos: {..., [todoKey]: newTodos}}))

    verificationNudgeNeeded = (
        feature('VERIFICATION_AGENT')
        && getFeatureValue(...)
        && !context.agentId         // main thread only
        && allDone
        && todos.length >= 3
        && !todos.some(t => /verif/i.test(t.content))
    )
    return { oldTodos, newTodos, verificationNudgeNeeded }

  checkPermissions: ALWAYS allow — no permission checks for todo ops.

  mapToolResultToToolResultBlockParam: renders a FIXED success string,
  plus an optional verification nudge. The model NEVER sees oldTodos or
  newTodos. Those exist for the UI only.

DELIBERATE DEVIATIONS (flagged, not hidden):

  1. Disk mirror. Source stores in AppState only (session-scoped). We ALSO
     write to store.save_plan() so the list survives compaction AND session
     restart. Correct for a personal agent that restarts.

  2. Optional strict validation. Source enforces neither max-20 nor
     single-in-progress. We enforce both when CLAUDE_CODE_STRICT_TODO=1,
     because a weak free-tier model needs the hard constraint more than
     Claude does.

  3. Verification nudge fires without GrowthBook. Source gates it on a
     remote flag; we gate it on the same env var that enables thinking,
     so it is testable offline.
"""
from __future__ import annotations

import os
import re
from typing import Literal

from pydantic import BaseModel, Field, field_validator

from messages import ToolUseBlock  # re-export for callers
from permission_types import PermissionAllowDecision
from tool import ToolResult, ToolUseContext, ValidationResult, build_tool


# ═════════════════════════════════════════════════════════════════════════
# Constants — exact from source
# ═════════════════════════════════════════════════════════════════════════

TODO_WRITE_TOOL_NAME = "TodoWrite"
TODO_READ_TOOL_NAME = "TodoRead"

MAX_TODO_ITEMS = 20                 # our strict-mode cap (source has none)
MAX_RESULT_SIZE_CHARS = 100_000     # source
VERIF_PATTERN = re.compile(r"verif", re.IGNORECASE)


# ═════════════════════════════════════════════════════════════════════════
# Schema — exact from source types.ts
# ═════════════════════════════════════════════════════════════════════════

TodoStatus = Literal["pending", "in_progress", "completed"]


class TodoItem(BaseModel):
    content: str = Field(description="The task description")
    status: TodoStatus = Field(
        description="pending | in_progress | completed",
    )
    active_form: str = Field(
        description=(
            "Present continuous form shown in the spinner while this is "
            "in_progress (e.g. 'Running tests' instead of 'Run tests')"
        ),
    )

    @field_validator("active_form", mode="before")
    @classmethod
    def _default_active_form(cls, v, info):
        if v:
            return v
        content = (info.data or {}).get("content") or "Working"
        return content

    @field_validator("status", mode="before")
    @classmethod
    def _normalize_status(cls, v):
        if isinstance(v, str):
            s = v.strip().lower().replace(" ", "_")
            if s in ("pending", "in_progress", "completed"):
                return s
            if s in ("done", "complete"):
                return "completed"
            if s in ("active", "doing", "inprogress", "in-progress"):
                return "in_progress"
        return "pending"

    @field_validator("content", mode="before")
    @classmethod
    def _non_empty_content(cls, v):
        if not isinstance(v, str) or not v.strip():
            raise ValueError("content must be a non-empty string")
        return v.strip()


class TodoInput(BaseModel):
    todos: list[TodoItem] = Field(
        description=(
            "The updated todo list. Pass the COMPLETE list every time — "
            "this replaces the previous list entirely."
        ),
    )


# ═════════════════════════════════════════════════════════════════════════
# Feature gates — source: isTodoV2Enabled() controls isEnabled()
# ═════════════════════════════════════════════════════════════════════════

def is_todo_v2_enabled() -> bool:
    return os.environ.get("CLAUDE_CODE_ENABLE_TASKS", "").strip().lower() in (
        "1", "true", "yes", "on",
    )


def is_strict_todo_enabled() -> bool:
    """Our addition: enforce max-20 and single-in-progress."""
    return os.environ.get("CLAUDE_CODE_STRICT_TODO", "").lower() in (
        "1", "true", "yes", "on",
    )


def is_verification_nudge_enabled() -> bool:
    """Source gates on feature('VERIFICATION_AGENT') + GrowthBook. We gate on env."""
    return os.environ.get("CLAUDE_CODE_VERIFICATION_NUDGE", "").lower() in (
        "1", "true", "yes", "on",
    )


# ═════════════════════════════════════════════════════════════════════════
# Context keys
# ═════════════════════════════════════════════════════════════════════════

def _session_id(ctx: ToolUseContext) -> str:
    sid = getattr(ctx, "session_id", None)
    if sid:
        return sid
    return f"session-{id(ctx)}"


def _todo_key(ctx: ToolUseContext) -> str:
    """Source: todoKey = context.agentId ?? getSessionId().
    Per-agent isolation — a sub-agent's todo list is invisible to the parent."""
    agent_id = getattr(ctx, "agent_id", None)
    return agent_id if agent_id else _session_id(ctx)


def _get_app_state(ctx: ToolUseContext):
    try:
        return ctx.get_app_state()
    except Exception:
        return None


def _get_todos_from_app_state(ctx: ToolUseContext) -> list[dict]:
    state = _get_app_state(ctx)
    if state is None:
        return []
    todos_map = getattr(state, "todos", None) or {}
    return list(todos_map.get(_todo_key(ctx), []) or [])


def _set_todos_in_app_state(ctx: ToolUseContext, todos: list[dict]) -> None:
    state = _get_app_state(ctx)
    if state is None:
        return
    if not hasattr(state, "todos"):
        try:
            state.todos = {}
        except Exception:
            return
    try:
        state.todos[_todo_key(ctx)] = todos
    except Exception:
        pass


# ═════════════════════════════════════════════════════════════════════════
# Store mirror — our addition for cross-session persistence
# ═════════════════════════════════════════════════════════════════════════

def _get_store(ctx: ToolUseContext):
    return getattr(ctx, "_store", None)


def _objective_from_todos(todos: list[dict]) -> str:
    if not todos:
        return "No active work"
    first = todos[0].get("content", "")
    return first if len(first) <= 80 else first[:77] + "..."


# ═════════════════════════════════════════════════════════════════════════
# validate_input — strict mode only (source has NO schema validation)
# ═════════════════════════════════════════════════════════════════════════

def _validate_input(input: TodoInput, ctx: ToolUseContext) -> ValidationResult:
    if not is_strict_todo_enabled():
        return ValidationResult(result=True)

    if len(input.todos) > MAX_TODO_ITEMS:
        return ValidationResult(
            result=False,
            message=(
                f"Too many todos: {len(input.todos)}. Maximum is "
                f"{MAX_TODO_ITEMS}. Consolidate related items."
            ),
            error_code=1,
        )

    in_progress = [t for t in input.todos if t.status == "in_progress"]
    if len(in_progress) > 1:
        return ValidationResult(
            result=False,
            message=(
                f"Exactly one todo may be in_progress at a time; you have "
                f"{len(in_progress)}. Set the others to pending and focus on "
                f"the one you are actively working on."
            ),
            error_code=2,
        )

    return ValidationResult(result=True)


# ═════════════════════════════════════════════════════════════════════════
# call — exact source pipeline
# ═════════════════════════════════════════════════════════════════════════

async def _call(
    input: TodoInput,
    ctx: ToolUseContext,
    can_use_tool=None,
    parent_message=None,
    on_progress=None,
) -> ToolResult:
    # 1. Read old todos for this key
    old_todos = _get_todos_from_app_state(ctx)

    # 2. Normalize the incoming list
    new_todos = [
        {
            "content": t.content,
            "status": t.status,
            "active_form": t.active_form,
        }
        for t in input.todos
    ]

    # 3. allDone → clear to []
    all_done = bool(new_todos) and all(
        t["status"] == "completed" for t in new_todos
    )
    stored_todos = [] if all_done else new_todos

    # 4. Six-condition verification nudge (exact source logic)
    verification_nudge_needed = False
    if (
        is_verification_nudge_enabled()
        and not getattr(ctx, "agent_id", None)   # main thread only
        and all_done
        and len(new_todos) >= 3
        and not any(VERIF_PATTERN.search(t["content"]) for t in new_todos)
    ):
        verification_nudge_needed = True

    # 5. Update AppState (source)
    _set_todos_in_app_state(ctx, stored_todos)
    if on_progress is not None:
        try:
            on_progress({
                "type": "plan_activity",
                "status": "updated",
                "item_count": len(stored_todos),
                "completed_count": sum(
                    todo["status"] == "completed" for todo in stored_todos
                ),
                "in_progress_count": sum(
                    todo["status"] == "in_progress" for todo in stored_todos
                ),
                "cleared": all_done,
            })
        except Exception:
            pass

    # 6. Mirror to store for cross-session survival (our addition)
    store = _get_store(ctx)
    if store is not None:
        focus = next(
            (t["content"] for t in stored_todos if t["status"] == "in_progress"),
            None,
        )
        try:
            store.save_plan(
                _session_id(ctx),
                _objective_from_todos(stored_todos),
                stored_todos,
                focus,
            )
        except Exception:
            pass

    # 7. Return the source's exact output shape
    return ToolResult(data={
        "old_todos": old_todos,
        "new_todos": new_todos,
        "verification_nudge_needed": verification_nudge_needed,
    })


# ═════════════════════════════════════════════════════════════════════════
# map_tool_result — exact source text; model never sees the list
# ═════════════════════════════════════════════════════════════════════════

_BASE_RESULT = (
    "Todos have been modified successfully. Ensure that you continue to use "
    "the todo list to track your progress. Please proceed with the current "
    "tasks if applicable"
)

_NUDGE = (
    "\n\nNOTE: You just closed out 3+ tasks and none of them was a "
    "verification step. Before writing your final summary, spawn the "
    "verification agent to check your work."
)


def _map_result(data: dict, tool_use_id: str) -> dict:
    nudge = data.get("verification_nudge_needed", False)
    content = _BASE_RESULT + (_NUDGE if nudge else "")
    return {
        "type": "tool_result",
        "tool_use_id": tool_use_id,
        "content": content,
    }


# ═════════════════════════════════════════════════════════════════════════
# Permissions — source: ALWAYS allow
# ═════════════════════════════════════════════════════════════════════════

async def _check_permissions(input: TodoInput, ctx: ToolUseContext):
    return PermissionAllowDecision(updated_input=input.model_dump())


# ═════════════════════════════════════════════════════════════════════════
# Prompt
# ═════════════════════════════════════════════════════════════════════════

def _prompt() -> str:
    strict_note = (
        f"\n- Maximum {MAX_TODO_ITEMS} items. Consolidate related work."
        "\n- EXACTLY ONE item may have status 'in_progress' at a time."
        if is_strict_todo_enabled() else ""
    )
    return (
        "Use this tool to create and manage a structured task list for the "
        "current coding session. This helps you track progress, organize "
        "complex tasks, and demonstrate thoroughness to the user.\n\n"
        "Usage:\n"
        "- Pass the COMPLETE updated list every time. This replaces the "
        "previous list entirely.\n"
        "- Status values: 'pending', 'in_progress', 'completed'.\n"
        "- The 'activeForm' is present-continuous tense (e.g. 'Running tests') "
        "and is shown in the spinner.\n"
        "- Ideally you should only have one todo as in_progress at a time."
        f"{strict_note}\n\n"
        "When to use:\n"
        "- Complex multi-step tasks (3+ distinct steps)\n"
        "- Non-trivial tasks requiring careful planning\n"
        "- When the user gives multiple tasks at once\n\n"
        "When NOT to use:\n"
        "- Single, straightforward tasks\n"
        "- Purely conversational requests\n\n"
        "Update the list immediately when you complete a task — do not batch "
        "status changes. Mark a task 'completed' only when it is FULLY done."
    )


# ═════════════════════════════════════════════════════════════════════════
# UI helpers
# ═════════════════════════════════════════════════════════════════════════

def _get_tool_use_summary(inp: dict | None) -> str | None:
    if not inp or not isinstance(inp, dict):
        return None
    todos = inp.get("todos") or []
    if not todos:
        return "Cleared todo list"
    in_progress = next(
        (t for t in todos if t.get("status") == "in_progress"), None
    )
    if in_progress:
        return in_progress.get("active_form") or in_progress.get("content")
    done = sum(1 for t in todos if t.get("status") == "completed")
    return f"{done}/{len(todos)} tasks done"


def _get_activity_description(inp: dict | None) -> str | None:
    s = _get_tool_use_summary(inp)
    return s or "Updating todos"


# ═════════════════════════════════════════════════════════════════════════
# The tool
# ═════════════════════════════════════════════════════════════════════════

TodoWriteTool = build_tool(
    name=TODO_WRITE_TOOL_NAME,
    search_hint="manage task list track progress plan steps todos",
    max_result_size_chars=MAX_RESULT_SIZE_CHARS,   # 100_000
    should_defer=True,
    description="Update the structured todo list",
    prompt=_prompt,
    input_schema=TodoInput,
    call=_call,
    map_tool_result_to_tool_result_block_param=_map_result,
    render_tool_use_message=lambda inp, _opts: "TodoWrite",
    is_read_only=lambda _inp: False,
    is_concurrency_safe=lambda _inp: False,
    to_auto_classifier_input=lambda inp: f"{len(getattr(inp, 'todos', []) or [])} items",
    validate_input=_validate_input,
    check_permissions=_check_permissions,
    get_tool_use_summary=_get_tool_use_summary,
    get_activity_description=_get_activity_description,
    # Source: TodoWrite renders into the todo panel, not the transcript
    render_tool_result_message=None,
    is_enabled=lambda: not is_todo_v2_enabled(),
)


__all__ = [
    "TodoWriteTool", "TodoInput", "TodoItem", "TodoStatus",
    "TODO_WRITE_TOOL_NAME", "MAX_TODO_ITEMS", "MAX_RESULT_SIZE_CHARS",
    "is_todo_v2_enabled", "is_strict_todo_enabled",
    "is_verification_nudge_enabled",
]