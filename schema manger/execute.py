"""
execute.py — the uniform tool execution pipeline.

Equivalent of src/services/tools/toolExecution.ts (the single-tool path
inside checkPermissionsAndCallTool). Order matches the source:

  find tool → abort check → schema parse → validate_input → backfill
  observable input → permission gauntlet → tool.call → persist large
  result → map to wire block → build ToolResultMessage

Two-audience error split:
  - validate_input fails → error content goes to the MODEL (retryable)
  - permission denied    → error content goes to the MODEL (with workaround)

Deferred (TODO, hooks not built): PreToolUse / PostToolUse / Failure hooks,
resolveHookPermissionDecision, and progress message streaming (needs the
execute_tool_streaming generator — next refactor).
"""
from __future__ import annotations

import json
import logging
import os
import tempfile
from dataclasses import dataclass
from typing import Any

from messages import ToolResultMessage, create_user_message
from permission_types import PermissionAskDecision, PermissionDecision
from permissions import has_permissions_to_use_tool
from registry import find
from tool import Tool, ToolResult, ToolUseContext, ValidationResult

from loop_guard import LoopGuardState, check_loop


logger = logging.getLogger(__name__)


def _guard_state(ctx) -> LoopGuardState:
    if not hasattr(ctx, "_loop_guard"):
        ctx._loop_guard = LoopGuardState()
    return ctx._loop_guard

# Exact strings from src/utils/messages.ts
CANCEL_MESSAGE = (
    "The user doesn't want to take this action right now. STOP what you are "
    "doing and wait for the user to tell you how to proceed."
)

DENIAL_WORKAROUND_GUIDANCE = (
    "IMPORTANT: You *may* attempt to accomplish this action using other tools "
    "that might naturally be used to accomplish this goal. But you *should not* "
    "attempt to work around this denial in malicious ways. You should only try "
    "to work around this restriction in reasonable ways that do not attempt to "
    "bypass the intent behind this denial. If you believe this capability is "
    "essential to complete the user's request, STOP and explain to the user "
    "what you were trying to do and why you need this permission. Let the user "
    "decide how to proceed."
)


# ── Result type ──────────────────────────────────────────────────────────

@dataclass
class ExecutionResult:
    """What execute_tool() returns to the orchestration layer."""
    tool_use_id: str
    tool_name: str
    result_message: ToolResultMessage
    result_block: dict
    is_error: bool
    context_modifier: Any = None


# ── The pipeline ─────────────────────────────────────────────────────────

async def execute_tool(
    tool_use: dict,
    context: ToolUseContext,
    can_use_tool: Any = None,
    parent_message: Any = None,
    on_progress: Any = None,
) -> ExecutionResult:
    """Execute ONE tool call through the full pipeline."""
    tool_use_id = tool_use.get("id", "")
    tool_name = tool_use.get("name", "")
    raw_input = tool_use.get("input", {})

    # ── 1. Find tool ──
    tool = find(tool_name)
    if tool is None:
        return _error_result(
            tool_use_id, tool_name,
            f"Unknown tool: {tool_name}. Available tools: {_list_tool_names()}",
        )

    try:
        enabled = bool(tool.is_enabled())
    except Exception:
        enabled = False
    if not enabled:
        return _error_result(
            tool_use_id, tool_name,
            f"Tool {tool_name} is disabled by runtime policy.",
        )

    allowed_tools = getattr(context, "allowed_tool_names", None)
    if allowed_tools is not None and tool_name not in allowed_tools:
        return _error_result(
            tool_use_id, tool_name,
            f"Tool {tool_name} is not available in this task context.",
        )

    # ── 2. Abort check (before doing any work) ──
    abort = getattr(context, "abort_controller", None)
    if abort is not None and getattr(abort, "aborted", False):
        return _error_result(tool_use_id, tool_name, CANCEL_MESSAGE)
    # ── 2b. Loop guard (before execution) ──
    signal = check_loop(_guard_state(context), tool_name, raw_input)
    if signal.action == "block":
        return _error_result(tool_use_id, tool_name, signal.reason)
    elif signal.action == "warn":
        # Let it run, but note the warning; the caller can surface it
        pass


    # ── 3. Schema parse ──
    try:
        parsed_input = (
            tool.input_schema(**raw_input) if raw_input else tool.input_schema()
        )
    except Exception as e:
        return _error_result(
            tool_use_id, tool_name, f"Invalid input for {tool_name}: {e}",
        )

    file_scope = getattr(context, "file_scope", None)
    if file_scope is not None and tool_name in ("Read", "Edit", "Write"):
        path_error = file_scope.check_access(
            tool_name, getattr(parsed_input, "file_path", ""),
        )
        if path_error:
            return _error_result(tool_use_id, tool_name, path_error)

    # ── 4. validate_input (model-audience coherence check) ──
    try:
        validation: ValidationResult = tool.validate_input(parsed_input, context)
        if not validation.result:
            return _error_result(
                tool_use_id, tool_name,
                f"Validation failed: {validation.message or 'unknown'}",
                error_code=validation.error_code,
            )
    except Exception as e:
        logger.error("Tool %s validate_input raised: %s", tool_name, e)
        return _error_result(tool_use_id, tool_name, f"Validation error: {e}")

    # ── 5. backfill_observable_input (on a COPY — never mutate API input) ──
    if tool.backfill_observable_input is not None:
        try:
            observable_input = dict(raw_input)
            tool.backfill_observable_input(observable_input)
        except Exception as e:
            logger.warning("backfill_observable_input raised for %s: %s", tool_name, e)
            observable_input = raw_input
    else:
        observable_input = raw_input

    # ── 6. Permission gauntlet ──
    # TODO: PreToolUse hooks + resolveHookPermissionDecision when built.
    try:
        decision: PermissionDecision = await has_permissions_to_use_tool(
            tool, observable_input, context,
            assistant_message=parent_message,
            tool_use_id=tool_use_id,
        )
    except Exception as e:
        logger.error("Permission check for %s raised: %s", tool_name, e)
        return _error_result(tool_use_id, tool_name, f"Permission check error: {e}")

    # ── 7. Apply decision ──
    if decision.behavior == "allow":
        final_input = getattr(decision, "updated_input", None) or parsed_input.model_dump()
    elif decision.behavior == "ask":
        approved = await _ask_human(tool_name, decision, context, can_use_tool)
        if not approved:
            return _error_result(
                tool_use_id, tool_name, _build_denial_message(tool_name, decision),
            )
        final_input = getattr(decision, "updated_input", None) or parsed_input.model_dump()
    elif decision.behavior == "deny":
        return _error_result(
            tool_use_id, tool_name, _build_denial_message(tool_name, decision),
        )
    else:
        return _error_result(
            tool_use_id, tool_name,
            f"Unexpected permission behavior: {decision.behavior}",
        )

    if final_input != parsed_input.model_dump():
        try:
            parsed_input = tool.input_schema(**final_input)
        except Exception as e:
            return _error_result(
                tool_use_id, tool_name, f"Updated input invalid: {e}",
            )

    staged_path_alias: tuple[str, str] | None = None
    if (
        file_scope is not None
        and getattr(file_scope, "staging_root", None) is not None
        and tool_name in ("Read", "Edit", "Write")
    ):
        try:
            original_path = getattr(parsed_input, "file_path", "")
            staged_path = file_scope.execution_path(tool_name, original_path)
            original_canonical = file_scope.leases.resolve(original_path)
            parsed_input = parsed_input.model_copy(
                update={"file_path": staged_path},
            )
            staged_path_alias = (original_canonical, staged_path)
        except Exception as e:
            return _error_result(
                tool_use_id, tool_name, f"Could not resolve staged path: {e}",
            )

    # ── 8. Execute ──
    try:
        tool_result: ToolResult = await tool.call(
            parsed_input, context, can_use_tool, parent_message, on_progress,
        )
    except Exception as e:
        logger.error("Tool %s call() raised: %s", tool_name, e)
        error_message = str(e)
        if file_scope is not None and getattr(file_scope, "staging_root", None):
            error_message = error_message.replace(
                file_scope.staging_root + os.sep,
                file_scope.project_root + os.sep,
            )
        return _error_result(
            tool_use_id, tool_name, f"Tool execution failed: {error_message}",
        )

    if staged_path_alias is not None:
        original_path, staged_path = staged_path_alias
        cache = getattr(context, "read_file_state", None)
        staged_state = cache.get(staged_path) if cache is not None else None
        if staged_state is not None:
            extras = {
                key: value for key, value in staged_state.items()
                if key not in ("content", "mtime")
            }
            cache.set(
                original_path,
                staged_state.get("content", ""),
                staged_state.get("mtime", 0),
                **extras,
            )
        tool_result.data = file_scope.externalize_result(tool_result.data)

    # ── 9. Persist large results ──
    data_for_model = await _maybe_persist_large_result(tool, tool_result.data, tool_use_id)

    # ── 10. Map to wire format ──
    try:
        block = tool.map_tool_result_to_tool_result_block_param(
            data_for_model, tool_use_id,
        )
    except Exception as e:
        logger.error("Tool %s map_tool_result raised: %s", tool_name, e)
        return _error_result(tool_use_id, tool_name, f"Result formatting failed: {e}")

    # ── 11. Build result message ──
    result_message = ToolResultMessage(
        tool_use_id=tool_use_id,
        content=_block_content_to_str(block),
        is_error=bool(block.get("is_error", False)),
    )

    result_is_error = bool(block.get("is_error", False))

    return ExecutionResult(
        tool_use_id=tool_use_id,
        tool_name=tool_name,
        result_message=result_message,
        result_block=block,
        is_error=result_is_error,
        context_modifier=tool_result.context_modifier,
    )


# ── Helpers ─────────────────────────────────────────────────────────────

def _error_result(
    tool_use_id: str, tool_name: str, message: str, error_code: int = 0,
) -> ExecutionResult:
    content = message
    if error_code:
        content = f"{message} (error_code={error_code})"
    return ExecutionResult(
        tool_use_id=tool_use_id,
        tool_name=tool_name,
        result_message=ToolResultMessage(
            tool_use_id=tool_use_id, content=content, is_error=True,
        ),
        result_block={
            "type": "tool_result",
            "tool_use_id": tool_use_id,
            "content": content,
            "is_error": True,
        },
        is_error=True,
    )


async def _ask_human(
    tool_name: str,
    decision: PermissionAskDecision,
    context: ToolUseContext,
    can_use_tool: Any,
) -> bool:
    """Ask for approval. can_use_tool → request_prompt → input()."""
    if can_use_tool is not None and callable(can_use_tool):
        try:
            return bool(await can_use_tool(decision))
        except Exception as e:
            logger.error("can_use_tool raised: %s", e)
            return False

    if context.request_prompt is not None:
        try:
            response = await context.request_prompt(
                "permission", f"Allow {tool_name}? {decision.message}",
            )
            if isinstance(response, dict):
                return bool(response.get("approved", False))
            return bool(response)
        except Exception as e:
            logger.error("request_prompt raised: %s", e)
            return False

    if context.should_avoid_permission_prompts:
        return False

    print(f"\n⚠️  Permission required: {tool_name}")
    print(f"   {decision.message}")
    print(f"   [y]es / [n]o: ", end="", flush=True)
    try:
        return input().strip().lower() in ("y", "yes")
    except (EOFError, KeyboardInterrupt):
        return False


def _build_denial_message(tool_name: str, decision: PermissionDecision) -> str:
    base = getattr(decision, "message", "") or (
        f"Permission to use {tool_name} has been denied."
    )
    return f"{base}\n\n{DENIAL_WORKAROUND_GUIDANCE}"


async def _maybe_persist_large_result(tool: Tool, data: Any, tool_use_id: str) -> Any:
    """If serialized data exceeds max_result_size_chars, spill to disk."""
    try:
        data_str = data if isinstance(data, str) else json.dumps(data, default=str)
    except Exception:
        return data
    if len(data_str) <= tool.max_result_size_chars:
        return data
    try:
        fd, path = tempfile.mkstemp(
            prefix=f"tool_result_{tool_use_id}_", suffix=".txt",
        )
        with os.fdopen(fd, "w") as f:
            f.write(data_str)
        preview = data_str[: tool.max_result_size_chars // 2]
        return (
            f"{preview}\n\n... [output truncated — full output at {path}] ...\n\n"
            f"Use Read tool to inspect {path} if needed."
        )
    except Exception as e:
        logger.error("Failed to persist large result: %s", e)
        return data


def _block_content_to_str(block: dict):
    content = block.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        # Image/document blocks must pass through to the model, not flatten to ""
        if any(isinstance(c, dict) and c.get("type") != "text" for c in content):
            return content
        return "\n".join(
            c.get("text", "") for c in content
            if isinstance(c, dict) and c.get("type") == "text"
        )
    return str(content) if content else ""
    

def _list_tool_names() -> str:
    from registry import all_tools
    return ", ".join(t.name for t in all_tools())