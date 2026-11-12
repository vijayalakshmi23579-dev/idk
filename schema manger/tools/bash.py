"""
tools/bash.py — execute shell commands.

Port of src/tools/BashTool/BashTool.tsx (1,413 lines) simplified for a
personal agent. What's preserved:

  • Argument-dependent is_read_only — a per-call classifier, not a flag
  • is_concurrency_safe delegates to is_read_only (source: L~200)
  • Semantic exit codes — grep 1 is "no matches", not an error
  • Destructive command set — for UI warnings, not blocking
  • Output truncation — 30K chars inline, full output to disk if over
  • Timeout — default 120s, capped at 600s
  • Compound command splitting — every subcommand must be read-only
    for the whole command to be read-only
  • Exit code + interpretation appended to the model-facing result

What's deferred (and why):
  • Kernel sandbox — requires vetto or bwrap. Add when you want isolation.
  • Background execution — requires a task store (task.py). Skip initially.
  • Tree-sitter AST parsing — bashSecurity.ts is 2,592 lines. Use a
    shlex + split-on-operators approach.
  • Sed edit simulation — personal agent, low value.
  • LSP diagnostic tracking — no LSP yet.
"""
from __future__ import annotations

import asyncio
import os
import re
import signal
import shlex
import tempfile
from typing import Any

from pydantic import BaseModel, Field, field_validator

from messages import ToolUseBlock  # re-export for callers
from permission_types import PermissionAllowDecision
from tool import ToolResult, ToolUseContext, ValidationResult, build_tool
from path_safety import check_path_safety_for_auto_edit, is_dangerous_removal_path


# ═════════════════════════════════════════════════════════════════════════
# Constants — from source
# ═════════════════════════════════════════════════════════════════════════

BASH_TOOL_NAME = "Bash"

DEFAULT_TIMEOUT_MS = 120_000        # 2 min
MAX_TIMEOUT_MS = 600_000            # 10 min

MAX_RESULT_SIZE_CHARS = 30_000      # 30K chars inline
MAX_OUTPUT_LENGTH = 30_000          # same threshold, source uses this name

# From readOnlyValidation.ts — a command is read-only iff every subcommand
# is on this whitelist. Conservative.
_READ_ONLY_BASE = frozenset({
    "ls", "cat", "head", "tail", "wc", "stat", "file", "strings",
    "grep", "rg", "ag", "ack", "find", "which", "whereis", "locate",
    "echo", "printf", "pwd", "whoami", "date", "uname", "hostname",
    "ps", "top", "df", "du", "free", "env", "printenv",
    "jq", "sort", "uniq", "cut", "tr", "column", "paste",
    "basename", "dirname", "realpath", "readlink",
    "git",  # refined below by _is_git_read_only
})

# git subcommands that are read-only
_READ_ONLY_GIT_SUBCMDS = frozenset({
    "status", "log", "show", "diff", "branch", "remote", "tag",
    "rev-parse", "ls-files", "ls-tree", "blame", "shortlog",
    "describe", "config",  # config --get/--list; conservative
    "fetch",  # network read-only, not mutation
})

# From destructiveCommandWarning.ts — for UI hint, not blocking
_DESTRUCTIVE_COMMANDS = frozenset({
    "rm", "rmdir", "dd", "mkfs", "shred", "truncate",
    "chmod", "chown", "chgrp", "sudo", "su",
    "kill", "killall", "pkill",
    "mv",  # overwrites destination
})

# From commandSemantics.ts — exit codes that are NOT errors
_SEMANTIC_EXIT_CODES: dict[str, dict[int, str]] = {
    "grep":       {1: "no matches found"},
    "rg":         {1: "no matches found"},
    "ag":         {1: "no matches found"},
    "ack":        {1: "no matches found"},
    "find":       {1: "no results"},
    "diff":       {1: "files differ"},
    "git diff":   {1: "differences present"},
    "git grep":   {1: "no matches"},
    "git status": {1: None},  # exit 1 with no message
    "test":       {1: "assertion failed"},
    "pytest":     {1: "tests failed"},
}

_HEREDOC_MARKER = re.compile(
    r"(?<!<)<<(?!<)(-?)\s*(?:'([^']+)'|\"([^\"]+)\"|([^\s;&|]+))"
)


# ═════════════════════════════════════════════════════════════════════════
# Compound-command splitting
# ═════════════════════════════════════════════════════════════════════════

def split_compound_command(command: str) -> list[str]:
    """Split on &&, ||, |, ; outside quotes. Returns trimmed subcommands."""
    parts: list[str] = []
    buf: list[str] = []
    quote: str | None = None
    i = 0
    cmd = command
    while i < len(cmd):
        c = cmd[i]
        if quote:
            buf.append(c)
            if c == quote and (i == 0 or cmd[i - 1] != "\\"):
                quote = None
        elif c in ("'", '"'):
            quote = c
            buf.append(c)
        elif c == ";" or (c in "|&" and i + 1 < len(cmd) and cmd[i + 1] == c):
            parts.append("".join(buf).strip())
            buf = []
            i += 1  # skip the second char of && or ||
        elif c == "|":
            parts.append("".join(buf).strip())
            buf = []
        else:
            buf.append(c)
        i += 1
    if buf:
        parts.append("".join(buf).strip())
    return [p for p in parts if p]


def _strip_heredoc_bodies(command: str) -> str:
    """Keep shell syntax and redirection paths; exclude literal heredoc data."""
    retained: list[str] = []
    delimiters: list[tuple[str, bool]] = []
    for line in command.splitlines():
        if delimiters:
            delimiter, strip_tabs = delimiters[0]
            candidate = line.lstrip("\t") if strip_tabs else line
            if candidate == delimiter:
                delimiters.pop(0)
            continue

        retained.append(line)
        for match in _HEREDOC_MARKER.finditer(line):
            delimiter = next(group for group in match.groups()[1:] if group is not None)
            delimiters.append((delimiter, bool(match.group(1))))
    return "\n".join(retained)


def _tokenize(command: str) -> list[str]:
    try:
        return shlex.split(command, posix=True)
    except ValueError:
        return command.split()


def _base_command(subcommand: str) -> str:
    tokens = _tokenize(subcommand)
    # Skip env-var assignments like FOO=bar
    for tok in tokens:
        if "=" in tok and not tok.startswith("-"):
            continue
        return tok
    return ""


def _is_git_read_only(subcommand: str) -> bool:
    tokens = _tokenize(subcommand)
    # Skip "git" and any -c / --flag before the subcommand
    git_sub: str | None = None
    for tok in tokens[1:]:
        if tok.startswith("-"):
            continue
        git_sub = tok
        break
    return git_sub in _READ_ONLY_GIT_SUBCMDS if git_sub else False


def is_read_only_command(command: str) -> bool:
    """A command is read-only iff EVERY subcommand is on the whitelist."""
    cmd = (command or "").strip()
    if not cmd:
        return False
    subs = split_compound_command(cmd)
    if not subs:
        return False
    for sub in subs:
        base = _base_command(sub)
        if not base:
            return False
        if base not in _READ_ONLY_BASE:
            return False
        if base == "git" and not _is_git_read_only(sub):
            return False
        # Redirects imply writing
        if ">" in sub or "<" in sub:
            return False
    return True


def is_destructive_command(command: str) -> bool:
    for sub in split_compound_command(command or ""):
        if _base_command(sub) in _DESTRUCTIVE_COMMANDS:
            return True
    return False


# ═════════════════════════════════════════════════════════════════════════
# Exit code interpretation — from commandSemantics.ts
# ═════════════════════════════════════════════════════════════════════════

def interpret_exit_code(command: str, code: int) -> str | None:
    if code == 0:
        return None
    subs = split_compound_command(command)
    if not subs:
        return None
    last = subs[-1]
    tokens = _tokenize(last)
    if not tokens:
        return None
    # Try two-word match first ("git diff"), then one-word
    two = " ".join(tokens[:2])
    one = tokens[0]
    table = _SEMANTIC_EXIT_CODES.get(two) or _SEMANTIC_EXIT_CODES.get(one)
    if table and code in table:
        note = table[code]
        return f"exit {code}: {note} (not an error)" if note else f"exit {code}"
    return f"exit {code}"


def _has_semantic_exit_code(command: str, code: int) -> bool:
    subs = split_compound_command(command)
    if not subs:
        return False
    tokens = _tokenize(subs[-1])
    if not tokens:
        return False
    table = _SEMANTIC_EXIT_CODES.get(" ".join(tokens[:2]))
    if table is None:
        table = _SEMANTIC_EXIT_CODES.get(tokens[0])
    return table is not None and code in table


# ═════════════════════════════════════════════════════════════════════════
# Schema
# ═════════════════════════════════════════════════════════════════════════

class BashInput(BaseModel):
    command: str = Field(description="The shell command to execute")
    timeout: int | None = Field(
        default=None, gt=0,
        description=(
            "Optional timeout in milliseconds "
            f"(max {MAX_TIMEOUT_MS}). Defaults to {DEFAULT_TIMEOUT_MS}."
        ),
    )
    description: str | None = Field(
        default=None,
        description=(
            "Clear, concise description of what this command does in active "
            "voice. Never use 'complex' or 'risk' — just describe what it does."
        ),
    )

    @field_validator("timeout", mode="before")
    @classmethod
    def _coerce_timeout(cls, v: Any) -> int | None:
        if v is None:
            return None
        if isinstance(v, str):
            try:
                return int(v)
            except ValueError:
                return None
        return int(v)


# ═════════════════════════════════════════════════════════════════════════
# Prompt — condensed from prompt.ts
# ═════════════════════════════════════════════════════════════════════════

def _prompt() -> str:
    return (
        "Executes a given bash command and returns its output.\n\n"
        "The working directory persists between commands. The shell state "
        "(environment variables, aliases) does not persist — set env vars "
        "inline: `FOO=bar cmd`.\n\n"
        "Usage:\n"
        "- Avoid using this tool to run `find`, `grep`, `cat`, `head`, `tail`, "
        "`sed`, `awk`, or `echo` unless explicitly instructed — prefer the "
        "dedicated Read, Grep, and Edit tools.\n"
        "- Output is truncated at "
        f"{MAX_OUTPUT_LENGTH} characters. If you expect more, write to a file "
        "and Read it.\n"
        "- Default timeout is 2 minutes. Pass `timeout` (milliseconds) for "
        "longer operations.\n"
        "- Chain commands with && when they depend on each other. Use ; only "
        "when you don't care about failure of earlier commands.\n"
        "- Always quote file paths with spaces: `ls \"/path with space\"`.\n"
        f"- The user's working directory is {os.getcwd()}."
    )


# ═════════════════════════════════════════════════════════════════════════
# validate_input — from source L66-149 (simplified: no security layer)
# ═════════════════════════════════════════════════════════════════════════

def _validate_input(input: BashInput, ctx: ToolUseContext) -> ValidationResult:
    cmd = (input.command or "").strip()
    if not cmd:
        return ValidationResult(
            result=False,
            message="Empty command. Provide a shell command to run.",
            error_code=1,
        )

    if input.timeout is not None:
        if input.timeout > MAX_TIMEOUT_MS:
            return ValidationResult(
                result=False,
                message=(
                    f"Timeout {input.timeout}ms exceeds the maximum of "
                    f"{MAX_TIMEOUT_MS}ms."
                ),
                error_code=2,
            )

    # Path-safety check on any paths extracted from the command
    # (basic version — full pathValidation.ts is 382 lines)
    if is_destructive_command(cmd):
        # Extract a rough path argument for the dangerous-removal check
        tokens = _tokenize(cmd)
        if len(tokens) >= 2:
            target = tokens[-1] if not tokens[-1].startswith("-") else None
            if target and is_dangerous_removal_path(target):
                return ValidationResult(
                    result=False,
                    message=(
                        f"Command refuses to run: '{target}' is a dangerous "
                        f"removal path (/, /etc, /usr, etc.). This is a "
                        f"hard safety limit — run it manually if you truly "
                        f"need to."
                    ),
                    error_code=3,
                )

    return ValidationResult(result=True)


# ═════════════════════════════════════════════════════════════════════════
# call — subprocess + timeout + truncation
# ═════════════════════════════════════════════════════════════════════════

async def _call(
    input: BashInput,
    ctx: ToolUseContext,
    can_use_tool: Any = None,
    parent_message: Any = None,
    on_progress: Any = None,
) -> ToolResult:
    cmd = input.command
    timeout_ms = input.timeout or DEFAULT_TIMEOUT_MS
    timeout_s = timeout_ms / 1000.0

    try:
        proc = await asyncio.create_subprocess_shell(
            cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=os.name == "posix",
        )
    except OSError as e:
        return ToolResult(data={
            "stdout": "",
            "stderr": f"Failed to spawn shell: {e}",
            "interrupted": False,
            "exitCode": -1,
            "interpretation": None,
            "truncated": False,
            "fullOutputPath": None,
        })

    stdout_task = asyncio.create_task(_drain_pipe(proc.stdout, on_progress, "stdout"))
    stderr_task = asyncio.create_task(_drain_pipe(proc.stderr, on_progress, "stderr"))
    wait_task = asyncio.create_task(proc.wait())
    abort_controller = getattr(ctx, "abort_controller", None)
    abort_task = (
        asyncio.create_task(_wait_for_abort(abort_controller))
        if abort_controller is not None else None
    )

    try:
        waiters = [wait_task]
        if abort_task is not None:
            waiters.append(abort_task)
        done, _ = await asyncio.wait(
            waiters, timeout=timeout_s, return_when=asyncio.FIRST_COMPLETED,
        )
        interrupted = wait_task not in done
        if interrupted:
            await _terminate_process(proc)
        stdout_b, stderr_b = await asyncio.gather(stdout_task, stderr_task)
    except asyncio.CancelledError:
        await _terminate_process(proc)
        await asyncio.gather(stdout_task, stderr_task, return_exceptions=True)
        raise
    finally:
        if abort_task is not None:
            if not abort_task.done():
                abort_task.cancel()
            await asyncio.gather(abort_task, return_exceptions=True)

    stdout = _decode(stdout_b)
    stderr = _decode(stderr_b)
    exit_code = proc.returncode if proc.returncode is not None else -1
    if interrupted:
        exit_code = -1

    # Combined output for the truncation/disk check
    combined = stdout
    if stderr:
        combined = (stdout + "\n" + stderr) if stdout else stderr

    full_output_path: str | None = None
    truncated = False
    if len(combined) > MAX_OUTPUT_LENGTH:
        truncated = True
        # Persist full output to disk; return a preview
        try:
            fd, full_output_path = tempfile.mkstemp(
                prefix="bash_output_", suffix=".txt",
            )
            with os.fdopen(fd, "w") as f:
                f.write(combined)
        except OSError:
            full_output_path = None
        # Preview: first half of the limit
        preview_len = MAX_OUTPUT_LENGTH // 2
        stdout = stdout[:preview_len]
        stderr = stderr[:preview_len] if stderr else ""

    interpretation = interpret_exit_code(cmd, exit_code)

    return ToolResult(data={
        "command": cmd,
        "stdout": stdout,
        "stderr": stderr,
        "interrupted": interrupted,
        "exitCode": exit_code,
        "interpretation": interpretation,
        "truncated": truncated,
        "fullOutputPath": full_output_path,
        "noOutputExpected": is_silent_command(cmd),
    })


async def _drain_pipe(stream, on_progress, stream_name: str) -> bytes:
    chunks: list[bytes] = []
    while True:
        chunk = await stream.read(8192)
        if not chunk:
            break
        chunks.append(chunk)
        if on_progress is not None:
            try:
                on_progress({
                    "type": "bash_progress",
                    "stream": stream_name,
                    "output": _decode(chunk),
                })
            except Exception:
                pass
    return b"".join(chunks)


async def _wait_for_abort(controller) -> None:
    if getattr(controller, "aborted", False):
        return
    wait = getattr(controller, "wait", None)
    if callable(wait):
        await wait()
        return
    while not getattr(controller, "aborted", False):
        await asyncio.sleep(0.05)


async def _terminate_process(proc) -> None:
    if proc.returncode is not None:
        return
    try:
        if os.name == "posix":
            os.killpg(proc.pid, signal.SIGTERM)
        else:
            proc.terminate()
    except ProcessLookupError:
        pass
    try:
        await asyncio.wait_for(proc.wait(), timeout=1.0)
        return
    except asyncio.TimeoutError:
        pass
    try:
        if os.name == "posix":
            os.killpg(proc.pid, signal.SIGKILL)
        else:
            proc.kill()
    except ProcessLookupError:
        pass
    await proc.wait()


def _decode(raw: bytes) -> str:
    for enc in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


# Commands that produce no stdout on success
_SILENT_COMMANDS = frozenset({
    "mv", "cp", "rm", "rmdir", "mkdir", "chmod", "chown", "chgrp",
    "touch", "ln", "cd", "export", "unset", "wait",
})


def is_silent_command(command: str) -> bool:
    subs = split_compound_command(command or "")
    if not subs:
        return False
    return all(_base_command(s) in _SILENT_COMMANDS for s in subs)


# ═════════════════════════════════════════════════════════════════════════
# map_tool_result_to_tool_result_block_param
# ═════════════════════════════════════════════════════════════════════════

def _map_result(data: dict, tool_use_id: str) -> dict:
    stdout = data.get("stdout", "")
    stderr = data.get("stderr", "")
    interrupted = data.get("interrupted", False)
    exit_code = data.get("exitCode", 0)
    interpretation = data.get("interpretation")
    truncated = data.get("truncated", False)
    full_output_path = data.get("fullOutputPath")
    no_output_expected = data.get("noOutputExpected", False)

    parts: list[str] = []
    if stdout:
        parts.append(stdout.rstrip())
    if stderr:
        parts.append(stderr.rstrip())

    if interrupted:
        parts.append("<error>Command was interrupted before completion</error>")
    elif exit_code != 0:
        line = f"Exit code: {exit_code}"
        if interpretation:
            line += f" ({interpretation})"
        parts.append(line)
    elif not parts and no_output_expected:
        parts.append("(Command completed with no output — expected)")

    if truncated and full_output_path:
        parts.append(
            f"[output truncated at {MAX_OUTPUT_LENGTH} chars — full output "
            f"at {full_output_path}. Use Read to see more.]"
        )

    content = "\n".join(parts) if parts else "(no output)"

    return {
        "type": "tool_result",
        "tool_use_id": tool_use_id,
        "content": content,
        "is_error": interrupted or (
            exit_code != 0 and not _has_semantic_exit_code(
                data.get("command", ""), exit_code,
            )
        ),
    }


# ═════════════════════════════════════════════════════════════════════════
# UI + permission helpers
# ═════════════════════════════════════════════════════════════════════════

async def _check_permissions(input: BashInput, ctx: ToolUseContext):
    """Delegate to path_safety for protected-path detection. The gauntlet's
    stages 1a/1b (deny/ask rules) run BEFORE this — deny rules still win."""
    try:
        command = _strip_heredoc_bodies(input.command)
        for subcommand in split_compound_command(command):
            for token in _tokenize(subcommand)[1:]:
                if token.startswith("-"):
                    continue
                looks_like_path = (
                    token.startswith(("/", "\\", ".", "~"))
                    or any(separator in token for separator in ("/", "\\", ":", "~"))
                )
                if not looks_like_path:
                    continue
                safety = check_path_safety_for_auto_edit(token)
                if not safety.safe and safety.classifier_approvable is False:
                    from permission_types import PermissionDenyDecision, SafetyCheckReason
                    return PermissionDenyDecision(
                        message=safety.message,
                        decision_reason=SafetyCheckReason(
                            reason=safety.message, classifier_approvable=False,
                        ),
                    )
    except ImportError:
        pass
    return PermissionAllowDecision(updated_input=input.model_dump())


def _user_facing_name(inp: dict | None) -> str:
    if not inp or not inp.get("command"):
        return "Bash"
    cmd = inp["command"].strip()
    return f"Bash({cmd[:40]}{'…' if len(cmd) > 40 else ''})"


def _get_tool_use_summary(inp: dict | None) -> str | None:
    if not inp:
        return None
    return inp.get("description") or (
        (inp.get("command") or "")[:60] or None
    )


def _get_activity_description(inp: dict | None) -> str | None:
    s = _get_tool_use_summary(inp)
    return f"Running {s}" if s else "Running command"


def _is_search_or_read_command(inp: dict | None) -> dict:
    if not inp or not inp.get("command"):
        return {"isSearch": False, "isRead": False, "isList": False}
    cmd = inp["command"]
    subs = split_compound_command(cmd)
    if not subs:
        return {"isSearch": False, "isRead": False, "isList": False}
    bases = {_base_command(s) for s in subs}
    return {
        "isSearch": bool(bases & {"grep", "rg", "ag", "ack", "find"}),
        "isRead": bool(bases & {"cat", "head", "tail", "less", "more"}),
        "isList": bool(bases & {"ls", "tree", "du"}),
    }


# ═════════════════════════════════════════════════════════════════════════
# The tool
# ═════════════════════════════════════════════════════════════════════════

BashTool = build_tool(
    name=BASH_TOOL_NAME,
    search_hint="execute shell command run terminal",
    max_result_size_chars=MAX_RESULT_SIZE_CHARS,
    strict=True,
    description="Run a shell command",
    prompt=_prompt,
    input_schema=BashInput,
    call=_call,
    map_tool_result_to_tool_result_block_param=_map_result,
    render_tool_use_message=lambda inp, _opts: _user_facing_name(
        inp if isinstance(inp, dict) else None
    ),
    # Risk methods — argument-dependent, per source
    is_read_only=lambda inp: is_read_only_command(getattr(inp, "command", "")),
    is_concurrency_safe=lambda inp: is_read_only_command(getattr(inp, "command", "")),
    is_destructive=lambda inp: is_destructive_command(getattr(inp, "command", "")),
    to_auto_classifier_input=lambda inp: getattr(inp, "command", ""),
    get_path=lambda inp: os.getcwd(),
    validate_input=_validate_input,
    check_permissions=_check_permissions,
    is_search_or_read_command=lambda inp: _is_search_or_read_command(
        {"command": getattr(inp, "command", "")}
    ),
    user_facing_name=_user_facing_name,
    get_tool_use_summary=_get_tool_use_summary,
    get_activity_description=_get_activity_description,
    extract_search_text=lambda out: (
        (out.get("stdout", "") or "") + (
            "\n" + out.get("stderr", "") if out.get("stderr") else ""
        )
    ) if isinstance(out, dict) else "",
)


__all__ = [
    "BashTool", "BashInput", "BASH_TOOL_NAME",
    "DEFAULT_TIMEOUT_MS", "MAX_TIMEOUT_MS", "MAX_RESULT_SIZE_CHARS",
    "is_read_only_command", "is_destructive_command", "is_silent_command",
    "split_compound_command", "interpret_exit_code",
]