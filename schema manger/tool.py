"""
tool.py — the tool contract + ToolUseContext.

Extracted from src/Tool.ts. The contract AND the ambient context a tool
receives at call time are defined in the same file, matching Claude Code's
structure (they're tightly coupled).

  - ToolUseContext: what a tool sees at call time (~30 fields)
  - Tool: the 48-field contract every capability satisfies
  - build_tool: fail-closed builder with signature enforcement
"""
from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Literal
from pydantic import BaseModel
from permission_types import PermissionResult, PermissionAllowDecision
# State types — ToolUseContext references these
from state import AppState, FileStateCache, ToolPermissionContext


# ── Default callables for ToolUseContext ──────────────────────────────────

def _default_get_app_state() -> AppState:
    return AppState()


def _default_set_app_state(_f: Callable[[AppState], AppState]) -> None:
    pass


# ── Wire-format types ──────────────────────────────────────────────────────

@dataclass
class ValidationResult:
    """Model-audience coherence check. Returned by validate_input()."""
    result: bool
    message: str | None = None
    error_code: int | None = None




@dataclass
class ToolResult:
    """What a tool's call() returns."""
    data: Any
    new_messages: list = field(default_factory=list)
    # context_modifier is only honored for tools that aren't concurrency-safe.
    # A parallel tool's modifier is silently discarded (race prevention).
    context_modifier: Callable[["ToolUseContext"], "ToolUseContext"] | None = None
    mcp_meta: dict | None = None


# ── ToolUseContext — the ambient world a tool sees ────────────────────────

@dataclass
class ToolUseContext:
    """
    Extracted from Tool.ts. Every field present so tools can rely on it.

    Sub-agent isolation:
      - set_app_state is NO-OP for async sub-agents
      - set_app_state_for_tasks is the escape hatch (reaches root store)
      - local_denial_tracking replaces the global counter that no-ops
      - should_avoid_permission_prompts auto-denies (subagents can't show UI)
      - rendered_system_prompt is frozen at turn start (prompt cache preservation)
    """

    # ── Required core ──
    abort_controller: Any = None
    read_file_state: FileStateCache = field(default_factory=FileStateCache)
    get_app_state: Callable[[], AppState] = _default_get_app_state
    set_app_state: Callable[[Callable[[AppState], AppState]], None] = _default_set_app_state

    # ── Sub-agent escape hatch ──
    set_app_state_for_tasks: Callable[[Callable[[AppState], AppState]], None] | None = None

    # ── Sub-agent identity ──
    agent_id: str | None = None
    agent_type: str | None = None
    allowed_tool_names: frozenset[str] | None = None
    file_scope: Any = None
    project_root: str | None = None
    session_id: str | None = None
    workflow_sandbox_root: str | None = None

    # ── Sub-agent permission degradation ──
    should_avoid_permission_prompts: bool = False
    await_automated_checks_before_dialog: bool = False
    local_denial_tracking: dict | None = None

    # ── Prompt cache preservation ──
    rendered_system_prompt: str | None = None

    # ── Cancellation & progress ──
    set_in_progress_tool_use_ids: Callable | None = None
    set_has_interruptible_tool_in_progress: Callable[[bool], None] | None = None
    set_response_length: Callable | None = None
    on_compact_progress: Callable | None = None
    set_stream_mode: Callable | None = None
    set_conversation_id: Callable | None = None

    # ── UI hooks (REPL only; None in headless) ──
    set_tool_jsx: Callable | None = None
    add_notification: Callable | None = None
    append_system_message: Callable | None = None
    send_os_notification: Callable | None = None
    request_prompt: Callable | None = None
    open_message_selector: Callable | None = None

    # ── File & history state ──
    update_file_history_state: Callable | None = None
    update_attribution_state: Callable | None = None
    preserve_tool_use_results: bool = False

    # ── Memory dedup ──
    nested_memory_attachment_triggers: set[str] | None = None
    loaded_nested_memory_paths: set[str] | None = None
    discovered_skill_names: set[str] | None = None

    # ── Metadata ──
    tool_use_id: str | None = None
    query_tracking: dict | None = None
    tool_decisions: dict | None = None
    file_reading_limits: dict | None = None
    glob_limits: dict | None = None
    content_replacement_state: Any = None
    handle_elicitation: Callable | None = None
    require_can_use_tool: bool = False
    user_modified: bool = False
    messages: list = field(default_factory=list)
    critical_system_reminder: str | None = None


# ── Fail-closed defaults for the Tool contract ────────────────────────────

def _default_not_read_only(_input: Any) -> bool:
    return False


def _default_not_safe(_input: Any) -> bool:
    return False


def _default_skip_classifier(_input: Any) -> str:
    return ""


def _default_valid(_input: Any, _ctx: Any = None) -> ValidationResult:
    return ValidationResult(result=True)


async def _default_allow(input: Any, _ctx: Any = None) -> PermissionResult:
    return PermissionAllowDecision(
        updated_input=input if isinstance(input, dict) else None,
    )


def _default_enabled() -> bool:
    return True


def _default_user_facing_name(_input: dict | None = None) -> str:
    return ""


def _default_get_path(_input: Any) -> str | None:
    return None


# ── The Tool contract ──────────────────────────────────────────────────────

@dataclass(kw_only=True)
class Tool:
    # ═══ IDENTITY ═══
    name: str

    # ═══ SCHEMA ═══
    input_schema: type[BaseModel]

    # ═══ EXECUTION (8 required) ═══
    call: Callable[..., Awaitable[ToolResult]]
    description: Callable[[Any, dict], Awaitable[str]] | str
    prompt: Callable[[dict], Awaitable[str]] | str
    map_tool_result_to_tool_result_block_param: Callable[[Any, str], dict]
    render_tool_use_message: Callable[[dict, dict], str]
    max_result_size_chars: int

    # ═══ RISK (5 defaulted, argument-dependent) ═══
    is_read_only: Callable[[Any], bool] = _default_not_read_only
    is_concurrency_safe: Callable[[Any], bool] = _default_not_safe
    is_destructive: Callable[[Any], bool] = _default_not_safe
    is_open_world: Callable[[Any], bool] = _default_not_safe
    is_enabled: Callable[[], bool] = _default_enabled

    # ═══ VALIDATION (2 defaulted) ═══
    validate_input: Callable[..., ValidationResult] = _default_valid
    check_permissions: Callable[..., Awaitable[PermissionResult]] = _default_allow

    # ═══ CLASSIFIER ═══
    to_auto_classifier_input: Callable[[Any], Any] = _default_skip_classifier

    # ═══ UI DEFAULTS ═══
    user_facing_name: Callable[[dict | None], str] = _default_user_facing_name
    get_path: Callable[[Any], str | None] = _default_get_path

    # ═══ IDENTITY OPTIONAL (9) ═══
    aliases: list[str] | None = None
    search_hint: str = ""
    should_defer: bool = False
    always_load: bool = False
    strict: bool = False
    is_mcp: bool = False
    is_lsp: bool = False
    mcp_info: dict | None = None
    input_json_schema: dict | None = None

    # ═══ BEHAVIOR OPTIONAL (6) ═══
    is_search_or_read_command: Callable[[Any], dict] | None = None
    requires_user_interaction: Callable[[], bool] | None = None
    interrupt_behavior: Callable[[], Literal["cancel", "block"]] | None = None
    inputs_equivalent: Callable[[Any, Any], bool] | None = None
    prepare_permission_matcher: Callable[..., Awaitable[Callable[[str], bool]]] | None = None
    backfill_observable_input: Callable[[dict], None] | None = None

    # ═══ UI OPTIONAL (13) ═══
    user_facing_name_background_color: Callable[[dict | None], str] | None = None
    is_transparent_wrapper: Callable[[], bool] | None = None
    get_tool_use_summary: Callable[[dict | None], str | None] | None = None
    get_activity_description: Callable[[dict | None], str | None] | None = None
    extract_search_text: Callable[[Any], str] | None = None
    is_result_truncated: Callable[[Any], bool] | None = None
    render_tool_result_message: Callable[..., str | None] | None = None
    render_tool_use_tag: Callable[[dict], str | None] | None = None
    render_tool_use_progress_message: Callable[..., str | None] | None = None
    render_tool_use_queued_message: Callable[[], str | None] | None = None
    render_tool_use_rejected_message: Callable[..., str | None] | None = None
    render_tool_use_error_message: Callable[..., str | None] | None = None
    render_grouped_tool_use: Callable[..., str | None] | None = None

    # ═══ CONFIG OPTIONAL ═══
    model_override: str | None = None


# ── Builder ────────────────────────────────────────────────────────────────

_REQUIRED_FIELDS = frozenset({
    "name", "input_schema", "call", "description", "prompt",
    "map_tool_result_to_tool_result_block_param", "render_tool_use_message",
    "max_result_size_chars",
})

_RISK_METHODS = ("is_read_only", "is_concurrency_safe", "is_destructive", "is_open_world")


def build_tool(**kwargs) -> Tool:
    """Build a Tool. Fail-closed enforcement at import time."""
    missing = _REQUIRED_FIELDS - kwargs.keys()
    if missing:
        raise TypeError(
            f"Tool {kwargs.get('name', '?')!r} missing required fields: "
            f"{sorted(missing)}"
        )

    for risk in _RISK_METHODS:
        fn = kwargs.get(risk)
        if fn is None:
            continue
        if not callable(fn):
            raise TypeError(
                f"{kwargs.get('name', '?')}.{risk} must be a callable taking "
                f"`input`, not {type(fn).__name__}"
            )
        try:
            sig = inspect.signature(fn)
        except (TypeError, ValueError):
            continue
        if len(sig.parameters) < 1:
            raise TypeError(
                f"{kwargs.get('name', '?')}.{risk} must accept `input` as its "
                f"first parameter — risk is per-call, not per-tool"
            )

    if "user_facing_name" not in kwargs:
        tool_name = kwargs["name"]
        kwargs["user_facing_name"] = lambda _inp=None, _n=tool_name: _n

    return Tool(**kwargs)


def to_openai_schema(tool: Tool) -> dict:
    if callable(tool.description):
        desc = tool.search_hint or tool.name
    else:
        desc = tool.description

    schema = tool.input_json_schema if tool.input_json_schema else \
             tool.input_schema.model_json_schema()

    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": desc,
            "parameters": schema,
        },
    }