#!/usr/bin/env python3
"""E2B sandboxed terminal exposed as a FastMCP tool.

Each session_id gets its own persistent Firecracker microVM. The default E2B
template has Python; for Java/C++/Node you need a custom template (see
https://e2b.dev/docs/template/quickstart).

Install:
    pip install e2b-code-interpreter fastmcp

Run:
    python terminal_mcp_server.py    # stdio transport

Then add to .chainlit/config.toml:
    [[features.mcp.servers]]
        name = "terminal"
        type = "stdio"
        command = "python /workspaces/idk/terminal_mcp_server.py"
        env = { E2B_API_KEY = "e2b_..." }
"""
from __future__ import annotations

import os
from typing import Any

from e2b_code_interpreter import Sandbox
from fastmcp import Context, FastMCP

mcp = FastMCP("terminal")

# session_id -> Sandbox. Persists across calls for the same session.
_TERMINALS: dict[str, Sandbox] = {}


def _get_terminal(session_id: str) -> Sandbox:
    if session_id not in _TERMINALS:
        _TERMINALS[session_id] = Sandbox.create()   # NOT Sandbox()
    return _TERMINALS[session_id]


@mcp.tool
async def run_terminal_command(
    session_id: str,
    command: str,
    ctx: Context,
    timeout: int = 60,
) -> dict[str, Any]:
    """Run a shell command inside an isolated E2B Ubuntu microVM.

    The sandbox persists across calls for the same session_id, so files written
    in one call are visible in the next.

    Args:
        session_id: Arbitrary ID that identifies the persistent sandbox.
        command: Shell command to run (e.g. "ls -la", "pip install requests").
        timeout: Seconds before the command is killed.

    Returns:
        dict with exit_code, stdout, stderr.
    """
    await ctx.info(f"[{session_id}] {command}")                  # await
    await ctx.report_progress(30, 100, "Booting sandbox")        # await

    try:
        vm = _get_terminal(session_id)
        await ctx.report_progress(60, 100, "Executing")
        result = vm.commands.run(command, timeout=timeout)
        await ctx.report_progress(100, 100, "Done")
        return {
            "exit_code": result.exit_code,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }
    except Exception as exc:
        await ctx.error(f"terminal crash: {exc}")                # await
        return {"exit_code": -1, "stdout": "", "stderr": str(exc)}


@mcp.tool
async def kill_terminal(session_id: str, ctx: Context) -> str:
    """Kill the persistent sandbox for a session, freeing the microVM."""
    await ctx.info(f"Killing sandbox {session_id}")
    vm = _TERMINALS.pop(session_id, None)
    if vm is None:
        return f"No active sandbox for {session_id}"
    try:
        vm.kill()
        return f"Killed sandbox {session_id}"
    except Exception as exc:
        return f"Error killing {session_id}: {exc}"


if __name__ == "__main__":
    mcp.run()