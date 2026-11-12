"""
tools/write.py — the Write tool.

Port of src/tools/FileWriteTool/FileWriteTool.ts (177 lines) + prompt.ts (15).

Preserved exactly:
  • maxResultSizeChars=100_000, strict=True
  • toAutoClassifierInput → "${file_path}: ${content}"
  • extractSearchText → '' (deliberate: content would be phantom in update mode)
  • validateInput chain:
      - deny rule on path           → errorCode 1
      - UNC path                    → return True (NTLM defense, no fs ops)
      - ENOENT (new file)           → return True (no read-gate for new files)
      - not read / partial read     → errorCode 2
      - mtime > read timestamp      → errorCode 3
        (Windows fallback: full read + identical content → proceed)
  • call order: expandPath → mkdir(dir) OUTSIDE atomic section →
    snapshot → read meta → TOCTOU → write → update read cache
  • ALWAYS writes LF — source comment: "Write is a full content replacement —
    the model sent explicit line endings in content and meant them."
  • Output: {type:'create', structuredPatch:[], originalFile:null}
            {type:'update', structuredPatch:<hunks>, originalFile:<old>}
  • Wire format: "File created successfully at: {path}"
                 "The file {path} has been updated successfully."

Deferred (flagged):
  • LSP didChange/didSave notification
  • VSCode diff notification
  • Skill directory discovery
  • gitDiff (CLAUDE_CODE_REMOTE + tengu_quartz_lantern)
  • checkTeamMemSecrets (no team memory)
"""
from __future__ import annotations

import asyncio
import os
import shutil
import uuid
from typing import Any

from pydantic import BaseModel, Field, field_validator

from messages import ToolUseBlock  # re-export
from permission_types import (
    PermissionAllowDecision, PermissionAskDecision, SafetyCheckReason,
)
from tool import ToolResult, ToolUseContext, ValidationResult, build_tool
from tools.edit import (
    FILE_UNEXPECTEDLY_MODIFIED_ERROR,
    _file_history_enabled,
    _make_hunks,
    _snapshot_for_undo,
    expand_path,
    get_file_modification_time,
    read_file_for_edit,
    write_text_content,
)


# ═════════════════════════════════════════════════════════════════════════
# Constants
# ═════════════════════════════════════════════════════════════════════════

FILE_WRITE_TOOL_NAME = "Write"

MAX_RESULT_SIZE_CHARS = 100_000

# Source L34-47: the exact errorCode map
ERR_DENIED_DIR = 1
ERR_NOT_READ = 2
ERR_MODIFIED = 3


# ═════════════════════════════════════════════════════════════════════════
# Schema — source L37-40
# ═════════════════════════════════════════════════════════════════════════

class FileWriteInput(BaseModel):
    file_path: str = Field(
        description="The absolute path to the file to write "
                    "(must be absolute, not relative)",
    )
    content: str = Field(description="The content to write to the file")


# ═════════════════════════════════════════════════════════════════════════
# Prompt — prompt.ts L3-15, verbatim
# ═════════════════════════════════════════════════════════════════════════

def _prompt() -> str:
    return (
        "Writes a file to the local filesystem.\n\n"
        "Usage:\n"
        "- This tool will overwrite the existing file if there is one at the "
        "provided path.\n"
        "- If this is an existing file, you MUST use the Read tool first to "
        "read the file's contents. This tool will fail if you did not read "
        "the file first.\n"
        "- Prefer the Edit tool for modifying existing files — it only sends "
        "the diff. Only use this tool to create new files or for complete "
        "rewrites.\n"
        "- NEVER create documentation files (*.md) or README files unless "
        "explicitly requested by the User.\n"
        "- Only use emojis if the user explicitly requests it. Avoid writing "
        "emojis to files unless asked."
    )


# ═════════════════════════════════════════════════════════════════════════
# validateInput — source L73-98, exact gate chain
# ═════════════════════════════════════════════════════════════════════════

def _is_denied_by_rules(full_path: str, ctx: ToolUseContext) -> bool:
    """Source: matchingRuleForInput(fullPath, ctx, 'edit', 'deny').
    Delegates to permission_rules — content-specific deny rules for Edit."""
    try:
        from permission_rules import (
            content_matches_rule_value, parse_rule,
        )
        from permissions import get_deny_rules

        app_state = ctx.get_app_state()
        tpc = app_state.tool_permission_context
        for rule in get_deny_rules(tpc):
            if rule.rule_value.tool_name not in ("Edit", "Write"):
                continue
            content = rule.rule_value.rule_content
            if content is None:
                continue
            if content_matches_rule_value(full_path, rule.rule_value):
                return True
    except (ImportError, AttributeError):
        pass
    return False


def _validate_input(input: FileWriteInput, ctx: ToolUseContext) -> ValidationResult:
    full_path = expand_path(input.file_path)

    # 1. deny rule → errorCode 1 (source L78-81)
    if _is_denied_by_rules(full_path, ctx):
        return ValidationResult(
            result=False,
            message="File is in a directory that is denied by your "
                    "permission settings.",
            error_code=ERR_DENIED_DIR,
        )

    # 2. UNC path → return True without touching the filesystem (source L85)
    if full_path.startswith("\\\\") or full_path.startswith("//"):
        return ValidationResult(result=True)

    # 3. stat — ENOENT means new file, no read-gate (source L88)
    try:
        os.stat(full_path)
    except FileNotFoundError:
        return ValidationResult(result=True)
    except OSError:
        # Permission error on stat — let the gauntlet handle it
        return ValidationResult(result=True)

    # 4. read-gate → errorCode 2 (source L90-92)
    read_state = (
        ctx.read_file_state.get(full_path)
        if ctx.read_file_state else None
    )
    if read_state is None or read_state.get("is_partial_view"):
        return ValidationResult(
            result=False,
            message="File has not been read yet. Read it first before "
                    "writing to it.",
            error_code=ERR_NOT_READ,
        )

    # 5. TOCTOU → errorCode 3 (source L95-97)
    last_write = get_file_modification_time(full_path)
    if last_write > read_state.get("mtime", 0):
        return ValidationResult(
            result=False,
            message="File has been modified since read, either by the user "
                    "or by a linter. Read it again before attempting to "
                    "write it.",
            error_code=ERR_MODIFIED,
        )

    return ValidationResult(result=True)


# ═════════════════════════════════════════════════════════════════════════
# call — source L98-172, exact pipeline
# ═════════════════════════════════════════════════════════════════════════

async def _call(
    input: FileWriteInput,
    ctx: ToolUseContext,
    can_use_tool: Any = None,
    parent_message: Any = None,
    on_progress: Any = None,
) -> ToolResult:
    full_path = expand_path(input.file_path)
    read_state = (
        dict(ctx.read_file_state.get(full_path))
        if ctx.read_file_state and ctx.read_file_state.get(full_path) else None
    )
    original, mtime = await asyncio.to_thread(
        _write_file_sync, full_path, input.content, read_state,
    )

    # Shared context mutations stay on the event-loop thread.
    if ctx.read_file_state:
        ctx.read_file_state.set(
            full_path, input.content, mtime,
            offset=None, limit=None,
        )

    # 7. Output — create vs update (source L160-172)
    if original is not None:
        patch = _make_hunks(original, input.content)
        return ToolResult(data={
            "type": "update",
            "filePath": input.file_path,
            "content": input.content,
            "structuredPatch": patch,
            "originalFile": original,
        })

    return ToolResult(data={
        "type": "create",
        "filePath": input.file_path,
        "content": input.content,
        "structuredPatch": [],
        "originalFile": None,
    })


def _write_file_sync(
    full_path: str,
    content: str,
    read_state: dict | None,
) -> tuple[str | None, int]:
    dir_path = os.path.dirname(full_path)
    if dir_path:
        try:
            os.makedirs(dir_path, exist_ok=True)
        except OSError as error:
            raise RuntimeError(f"Cannot create directory {dir_path}: {error}")

    if _file_history_enabled():
        _snapshot_for_undo(full_path)

    original: str | None = None
    encoding = "utf-8"
    initial_signature = None
    try:
        initial_stat = os.stat(full_path)
    except FileNotFoundError:
        if os.path.lexists(full_path):
            raise RuntimeError(f"Cannot inspect existing file {full_path}.")
    except OSError as error:
        raise RuntimeError(f"Cannot inspect file {full_path}: {error}")
    else:
        initial_signature = _stat_signature(initial_stat)
        try:
            original, exists, encoding, _endings = read_file_for_edit(full_path)
        except OSError as error:
            raise RuntimeError(f"Cannot read existing file {full_path}: {error}")
        if not exists:
            raise RuntimeError(FILE_UNEXPECTEDLY_MODIFIED_ERROR)
        try:
            if _stat_signature(os.stat(full_path)) != initial_signature:
                raise RuntimeError(FILE_UNEXPECTEDLY_MODIFIED_ERROR)
        except OSError as error:
            raise RuntimeError(FILE_UNEXPECTEDLY_MODIFIED_ERROR) from error

    if original is not None:
        last_write = get_file_modification_time(full_path)
        if read_state is None or last_write > read_state.get("mtime", 0):
            is_full_read = (
                read_state is not None
                and read_state.get("offset") is None
                and read_state.get("limit") is None
            )
            content_unchanged = (
                is_full_read and original == read_state.get("content", "")
            )
            if not content_unchanged:
                raise RuntimeError(FILE_UNEXPECTEDLY_MODIFIED_ERROR)
        try:
            if _stat_signature(os.stat(full_path)) != initial_signature:
                raise RuntimeError(FILE_UNEXPECTEDLY_MODIFIED_ERROR)
        except OSError as error:
            raise RuntimeError(FILE_UNEXPECTEDLY_MODIFIED_ERROR) from error
    elif os.path.lexists(full_path):
        raise RuntimeError(
            f"File appeared while writing {full_path}; read it before overwriting."
        )

    # Write replaces the complete text, so preserve the submitted LF endings.
    write_text_content(full_path, content, encoding, "lf")
    return original, get_file_modification_time(full_path)


def _stat_signature(info) -> tuple[int, int, int, int]:
    return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns


# ═════════════════════════════════════════════════════════════════════════
# map_tool_result_to_tool_result_block_param — source L173-177, verbatim
# ═════════════════════════════════════════════════════════════════════════

def _map_result(data: dict, tool_use_id: str) -> dict:
    kind = data.get("type")
    file_path = data.get("filePath", "")
    if kind == "create":
        return {
            "type": "tool_result",
            "tool_use_id": tool_use_id,
            "content": f"File created successfully at: {file_path}",
        }
    return {
        "type": "tool_result",
        "tool_use_id": tool_use_id,
        "content": f"The file {file_path} has been updated successfully.",
    }


# ═════════════════════════════════════════════════════════════════════════
# Permission + UI helpers
# ═════════════════════════════════════════════════════════════════════════

async def _check_permissions(input: FileWriteInput, ctx: ToolUseContext):
    """Delegate to path_safety for protected-path detection. Stage 1a
    (deny rules) and 1b (ask rules) fire BEFORE this."""
    try:
        from path_safety import check_path_safety_for_auto_edit
        safety = check_path_safety_for_auto_edit(input.file_path)
        if not safety.safe:
            return PermissionAskDecision(
                message=safety.message,
                decision_reason=SafetyCheckReason(
                    reason=safety.message,
                    classifier_approvable=safety.classifier_approvable,
                ),
            )
    except ImportError:
        pass
    return PermissionAllowDecision(updated_input=input.model_dump())


def _backfill_observable_input(inp: dict) -> None:
    if isinstance(inp.get("file_path"), str):
        inp["file_path"] = expand_path(inp["file_path"])


async def _prepare_permission_matcher(input: FileWriteInput):
    import fnmatch
    path = input.file_path
    return lambda pattern: fnmatch.fnmatchcase(path, pattern)


def _short_path(path: str) -> str:
    parts = path.replace("\\", "/").split("/")
    return "/".join(parts[-3:]) if len(parts) > 3 else path


def _user_facing_name(inp: dict | None) -> str:
    if not inp or not inp.get("file_path"):
        return "Write"
    return f"Write({_short_path(inp['file_path'])})"


def _get_tool_use_summary(inp: dict | None) -> str | None:
    if not inp or not inp.get("file_path"):
        return None
    return _short_path(inp["file_path"])


def _get_activity_description(inp: dict | None) -> str | None:
    s = _get_tool_use_summary(inp)
    return f"Writing {s}" if s else "Writing file"


# ═════════════════════════════════════════════════════════════════════════
# The tool
# ═════════════════════════════════════════════════════════════════════════

WriteTool = build_tool(
    name=FILE_WRITE_TOOL_NAME,
    search_hint="create or overwrite files",
    max_result_size_chars=MAX_RESULT_SIZE_CHARS,
    strict=True,
    description="Write a file to the local filesystem.",
    prompt=_prompt,
    input_schema=FileWriteInput,
    call=_call,
    map_tool_result_to_tool_result_block_param=_map_result,
    render_tool_use_message=lambda inp, _opts: _user_facing_name(
        inp if isinstance(inp, dict) else None
    ),
    # Risk methods — fail-closed defaults from Tool.ts (same as Edit)
    # is_read_only → false, is_concurrency_safe → false, is_destructive → false
    to_auto_classifier_input=lambda inp: (
        f"{getattr(inp, 'file_path', '')}: {getattr(inp, 'content', '')}"
    ),
    get_path=lambda inp: getattr(inp, "file_path", "") or os.getcwd(),
    user_facing_name=_user_facing_name,
    get_tool_use_summary=_get_tool_use_summary,
    get_activity_description=_get_activity_description,
    validate_input=_validate_input,
    check_permissions=_check_permissions,
    backfill_observable_input=_backfill_observable_input,
    prepare_permission_matcher=_prepare_permission_matcher,
    # extractSearchText → '' deliberately (source L68-72):
    # content would be indexed even in update mode where it isn't rendered
    extract_search_text=lambda _out: "",
)


__all__ = [
    "WriteTool", "FileWriteInput", "FILE_WRITE_TOOL_NAME",
    "MAX_RESULT_SIZE_CHARS", "ERR_DENIED_DIR", "ERR_NOT_READ", "ERR_MODIFIED",
]