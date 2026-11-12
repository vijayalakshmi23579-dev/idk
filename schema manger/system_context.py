"""
system_context.py — system prompt injection (git, CLAUDE.md, date).

Extracted from src/context.ts. Cached for the session — the system prompt
prefix MUST stay byte-identical across turns or the prompt cache invalidates.

NOT the same as ToolUseContext (which lives in tool.py). This is about
what gets prepended to the SYSTEM PROMPT, not what a tool sees at call time.
"""
from __future__ import annotations

import os
import subprocess
from datetime import datetime
from functools import lru_cache
from pathlib import Path


MAX_STATUS_CHARS = 2000


@lru_cache(maxsize=1)
def get_system_context() -> dict[str, str]:
    """Git status, branch, recent commits. Cached for the session."""
    if not _is_git_repo():
        return {}
    try:
        branch = _git(["rev-parse", "--abbrev-ref", "HEAD"]).strip() or "unknown"
        main_branch = (
            _git(["rev-parse", "--abbrev-ref", "origin/HEAD"]).strip()
            or "main"
        )
        status = _git(["--no-optional-locks", "status", "--short"]).strip()
        if not status:
            status = "(clean)"
        elif len(status) > MAX_STATUS_CHARS:
            status = (
                status[:MAX_STATUS_CHARS]
                + '\n... (truncated; run "git status" via BashTool for more)'
            )
        log = _git(
            ["--no-optional-locks", "log", "--oneline", "-n", "5"]
        ).strip()
        user_name = _git(["config", "user.name"]).strip()

        parts = [
            "This is the git status at the start of the conversation. "
            "Note that this status is a snapshot in time, and will not "
            "update during the conversation.",
            f"Current branch: {branch}",
            f"Main branch (you will usually use this for PRs): {main_branch}",
        ]
        if user_name:
            parts.append(f"Git user: {user_name}")
        parts.append(f"Status:\n{status}")
        if log:
            parts.append(f"Recent commits:\n{log}")

        return {"gitStatus": "\n\n".join(parts)}
    except Exception:
        return {}


@lru_cache(maxsize=1)
def get_user_context() -> dict[str, str]:
    """CLAUDE.md contents + current date. Cached for the session."""
    date_str = f"Today's date is {datetime.now().strftime('%Y-%m-%d')}."

    if os.environ.get("CLAUDE_CODE_DISABLE_CLAUDE_MDS"):
        return {"currentDate": date_str}

    claude_md = _load_claude_md()
    result: dict[str, str] = {}
    if claude_md:
        result["claudeMd"] = claude_md
    result["currentDate"] = date_str
    return result


def format_system_context_for_prompt(context: dict[str, str]) -> str:
    """Format the context dict as a section to append to the system prompt."""
    if not context:
        return ""
    return "\n\n".join(f"# {k}\n{v}" for k, v in context.items())


def clear_cache() -> None:
    """Clear caches — call if you want to force a fresh computation."""
    get_system_context.cache_clear()
    get_user_context.cache_clear()


# ── Internals ─────────────────────────────────────────────────────────────

def _is_git_repo() -> bool:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--is-inside-work-tree"],
            capture_output=True, text=True, timeout=2,
        )
        return result.returncode == 0 and result.stdout.strip() == "true"
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def _git(args: list[str]) -> str:
    try:
        result = subprocess.run(
            ["git", *args], capture_output=True, text=True, timeout=2,
        )
        return result.stdout if result.returncode == 0 else ""
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return ""


def _load_claude_md() -> str:
    """Load CLAUDE.md from user → project → local, in that order."""
    paths = [
        Path.home() / ".claude" / "CLAUDE.md",
        Path.cwd() / "CLAUDE.md",
        Path.cwd() / "CLAUDE.local.md",
    ]
    parts = []
    for p in paths:
        if p.exists():
            try:
                parts.append(f"# From {p}\n\n{p.read_text()}")
            except OSError:
                continue
    return "\n\n---\n\n".join(parts)