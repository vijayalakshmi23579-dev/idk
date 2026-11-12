"""
state.py — session state.

Absorbed from the old context.py. Defines the four state types plus the
mutators that update them:

  - FileStateCache        — LRU cache for file reads (the read-gate)
  - ToolPermissionContext — permission mode + rule lists
  - TaskState             — tracked long-running work
  - AppState              — the mutable session store

Mutators return new state (immutable-style) so the caller can decide
whether to commit changes. Sub-agents get isolated AppState copies.
"""
from __future__ import annotations

import secrets
import time
from dataclasses import dataclass, field
from typing import Any


# ── FileStateCache ────────────────────────────────────────────────────────

@dataclass
class FileStateCache:
    """LRU-ish cache of file reads. The read-gate for FileEditTool."""
    files: dict[str, dict] = field(default_factory=dict)
    max_entries: int = 100

    def get(self, path: str) -> dict | None:
        return self.files.get(path)

    def has(self, path: str) -> bool:
        return path in self.files

    def set(self, path: str, content: str, mtime: float, **extra) -> None:
        self.files[path] = {"content": content, "mtime": mtime, **extra}
        if len(self.files) > self.max_entries:
            oldest = next(iter(self.files))
            del self.files[oldest]

    def invalidate(self, path: str) -> None:
        self.files.pop(path, None)


# ── ToolPermissionContext ─────────────────────────────────────────────────

@dataclass
class ToolPermissionContext:
    """Permission state — used by the gauntlet."""
    mode: str = "default"  # "default" | "plan" | "bypassPermissions" | "auto"
    always_allow_rules: dict[str, list[str]] = field(default_factory=dict)
    always_deny_rules: dict[str, list[str]] = field(default_factory=dict)
    always_ask_rules: dict[str, list[str]] = field(default_factory=dict)
    is_bypass_permissions_mode_available: bool = False
    should_avoid_permission_prompts: bool = False
    await_automated_checks_before_dialog: bool = False


# ── TaskState ─────────────────────────────────────────────────────────────

@dataclass
class TaskState:
    """Tracked long-running work (background bash, sub-agents, dream)."""
    id: str
    type: str  # "local_bash" | "local_agent" | "dream"
    status: str  # "pending" | "running" | "completed" | "failed" | "killed"
    description: str
    output_file: str = ""
    output_offset: int = 0
    notified: bool = False
    start_time: float = 0.0
    end_time: float | None = None
    result: Any = None
    error: str | None = None


# ── AppState ──────────────────────────────────────────────────────────────

@dataclass
class AppState:
    """Mutable session state. Sub-agents see isolated copies."""
    messages: list = field(default_factory=list)
    tool_permission_context: ToolPermissionContext = field(
        default_factory=ToolPermissionContext
    )
    read_file_cache: FileStateCache = field(default_factory=FileStateCache)
    tasks: dict[str, TaskState] = field(default_factory=dict)
    agent_name_registry: dict[str, str] = field(default_factory=dict)
    total_cost_usd: float = 0.0
    session_start_time: float = 0.0
    turn_count: int = 0
    mode: str = "default"


# ── Task ID generation ────────────────────────────────────────────────────

_TASK_ID_PREFIXES = {
    "local_bash": "b",
    "local_agent": "a",
    "dream": "d",
}

_ALPHABET = "0123456789abcdefghijklmnopqrstuvwxyz"


def generate_task_id(task_type: str) -> str:
    """Prefixed task ID (36^8 = 2.8T combos, resists symlink attacks)."""
    prefix = _TASK_ID_PREFIXES.get(task_type, "x")
    return prefix + "".join(secrets.choice(_ALPHABET) for _ in range(8))


# ── Factories ─────────────────────────────────────────────────────────────

def create_task_state(
    task_type: str,
    description: str,
    output_file: str = "",
) -> TaskState:
    return TaskState(
        id=generate_task_id(task_type),
        type=task_type,
        status="pending",
        description=description,
        output_file=output_file,
        start_time=time.time(),
    )


def create_app_state() -> AppState:
    return AppState(
        messages=[],
        tool_permission_context=ToolPermissionContext(),
        read_file_cache=FileStateCache(),
        tasks={},
        agent_name_registry={},
        total_cost_usd=0.0,
        session_start_time=time.time(),
        turn_count=0,
        mode="default",
    )


# ── Mutators (return new state; caller decides to commit) ────────────────

def add_message(state: AppState, message: Any) -> AppState:
    """New state with the message appended."""
    return AppState(
        messages=[*state.messages, message],
        tool_permission_context=state.tool_permission_context,
        read_file_cache=state.read_file_cache,
        tasks=state.tasks,
        agent_name_registry=state.agent_name_registry,
        total_cost_usd=state.total_cost_usd,
        session_start_time=state.session_start_time,
        turn_count=state.turn_count,
        mode=state.mode,
    )


def add_task(state: AppState, task: TaskState) -> AppState:
    return AppState(
        messages=state.messages,
        tool_permission_context=state.tool_permission_context,
        read_file_cache=state.read_file_cache,
        tasks={**state.tasks, task.id: task},
        agent_name_registry=state.agent_name_registry,
        total_cost_usd=state.total_cost_usd,
        session_start_time=state.session_start_time,
        turn_count=state.turn_count,
        mode=state.mode,
    )


def update_task(state: AppState, task_id: str, **updates) -> AppState:
    if task_id not in state.tasks:
        return state
    task = state.tasks[task_id]
    for k, v in updates.items():
        setattr(task, k, v)
    return AppState(
        messages=state.messages,
        tool_permission_context=state.tool_permission_context,
        read_file_cache=state.read_file_cache,
        tasks={**state.tasks, task_id: task},
        agent_name_registry=state.agent_name_registry,
        total_cost_usd=state.total_cost_usd,
        session_start_time=state.session_start_time,
        turn_count=state.turn_count,
        mode=state.mode,
    )


def record_file_read(
    state: AppState,
    path: str,
    content: str,
    mtime: float,
) -> AppState:
    """Record a file read — enables the read-gate for FileEditTool."""
    state.read_file_cache.set(path, content, mtime)
    return state