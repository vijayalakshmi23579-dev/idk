#!/usr/bin/env python3
"""FastMCP server exposing file and shell tools to a Chainlit agent.

Security model:
  read_file / write_file / list_directory are restricted to $AGENT_WORKSPACE.
  execute_command runs on the HOST — not sandboxed. Use only with trusted models.
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

from fastmcp import FastMCP

WORKSPACE = Path(
    os.environ.get("AGENT_WORKSPACE", "/workspaces/idk/agent_workspace")
).resolve()
WORKSPACE.mkdir(parents=True, exist_ok=True)

mcp = FastMCP("file-tools")


def _safe_path(path: str) -> Path:
    p = (WORKSPACE / path) if not Path(path).is_absolute() else Path(path)
    p = p.resolve()
    try:
        p.relative_to(WORKSPACE)
    except ValueError:
        raise PermissionError(f"path outside workspace: {path}")
    return p


@mcp.tool
def read_file(path: str) -> str:
    """Read a text file from the workspace."""
    try:
        p = _safe_path(path)
    except PermissionError as e:
        return f"ERROR: {e}"
    if not p.is_file():
        return f"ERROR: not a file: {path}"
    try:
        return p.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return f"ERROR: {path} is not UTF-8 text"
    except Exception as e:
        return f"ERROR: {type(e).__name__}: {e}"


@mcp.tool
def write_file(path: str, content: str) -> str:
    """Write content to a file in the workspace."""
    try:
        p = _safe_path(path)
    except PermissionError as e:
        return f"ERROR: {e}"
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        return f"OK: wrote {len(content)} bytes to {path}"
    except Exception as e:
        return f"ERROR: {type(e).__name__}: {e}"


@mcp.tool
def list_directory(path: str = ".") -> str:
    """List files and folders under a workspace-relative path."""
    try:
        p = _safe_path(path)
    except PermissionError as e:
        return f"ERROR: {e}"
    if not p.is_dir():
        return f"ERROR: not a directory: {path}"
    entries = sorted(p.iterdir(), key=lambda x: (not x.is_dir(), x.name))
    if not entries:
        return "(empty)"
    lines = []
    for e in entries:
        rel = e.relative_to(WORKSPACE)
        if e.is_dir():
            lines.append(f"{rel}/")
        else:
            try:
                lines.append(f"{rel}  [{e.stat().st_size} bytes]")
            except OSError:
                lines.append(str(rel))
    return "\n".join(lines)


_DANGEROUS = (
    "rm -rf /",
    "rm -rf /*",
    "mkfs",
    "dd if=/dev/zero",
    "dd if=/dev/random",
    ":(){ :|:& };:",
)


@mcp.tool
def execute_command(command: str, timeout_seconds: int = 60) -> str:
    """Run a shell command inside the workspace and return stdout, stderr, exit code.

    WARNING: this runs on the host. Only use with trusted code.
    """
    for pat in _DANGEROUS:
        if pat in command:
            return f"ERROR: refused dangerous pattern: {pat!r}"
    try:
        result = subprocess.run(
            command,
            shell=True,
            cwd=str(WORKSPACE),
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
        parts = []
        if result.stdout:
            parts.append(f"STDOUT:\n{result.stdout}")
        if result.stderr:
            parts.append(f"STDERR:\n{result.stderr}")
        parts.append(f"EXIT: {result.returncode}")
        return "\n\n".join(parts)
    except subprocess.TimeoutExpired:
        return f"ERROR: timed out after {timeout_seconds}s"
    except Exception as e:
        return f"ERROR: {type(e).__name__}: {e}"


if __name__ == "__main__":
    mcp.run()