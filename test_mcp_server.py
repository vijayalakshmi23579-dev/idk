#!/usr/bin/env python3
"""
Sample MCP server for testing Chainlit + Gemini/NVIDIA tool calling
====================================================================
Five tools that cover the important shapes:

  - add(a, b)                -> int, pure computation
  - echo(text)               -> str, passthrough
  - get_time(timezone)       -> str, side-effect (reads system clock)
  - reverse_list(items)      -> list, argument coercion test
  - divide(a, b)             -> float, raises on divide-by-zero (error path)

Run standalone to sanity-check:
    python test_mcp_server.py            # starts stdio transport
    # or introspect without spawning:
    python -c "import test_mcp_server as m; print([t.name for t in m.mcp._tool_manager.list_tools()])"
"""
from __future__ import annotations

import datetime
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from fastmcp import FastMCP

mcp = FastMCP("test")


@mcp.tool
def add(a: int, b: int) -> int:
    """Add two integers and return the sum."""
    return a + b


@mcp.tool
def echo(text: str) -> str:
    """Return the input text unchanged. Useful for verifying string round-trips."""
    return text


@mcp.tool
def get_time(timezone: str = "UTC") -> str:
    """Return the current time in the given IANA timezone (e.g. 'UTC', 'Asia/Kolkata').

    Returns an ISO 8601 timestamp with the timezone name appended.
    """
    try:
        tz = ZoneInfo(timezone)
    except ZoneInfoNotFoundError:
        # Fall back to UTC if the zone name is unknown, and say so in the output.
        return f"{datetime.datetime.now(datetime.timezone.utc).isoformat()} (unknown timezone '{timezone}', using UTC)"
    now = datetime.datetime.now(tz)
    return f"{now.isoformat()} ({timezone})"


@mcp.tool
def reverse_list(items: list[str]) -> list[str]:
    """Return the input list in reverse order.

    Exercises the list-argument path: some models send a bare string instead
    of a single-element list, and the app's coerce_args() should fix that.
    """
    return list(reversed(items))


@mcp.tool
def divide(a: float, b: float) -> float:
    """Divide a by b. Raises ValueError if b is zero (used to test the error path)."""
    if b == 0:
        raise ValueError("division by zero is not allowed")
    return a / b


if __name__ == "__main__":
    mcp.run()  # defaults to stdio transport