"""registry.py — the tool lookup table. A dict, not a filter function."""
from tool import Tool

TOOLS: dict[str, Tool] = {}


def register(tool: Tool) -> Tool:
    """Register a tool. Returns it so you can chain: bash = register(build_tool(...))"""
    TOOLS[tool.name] = tool
    return tool


def find(name: str) -> Tool | None:
    return TOOLS.get(name)


def all_tools() -> list[Tool]:
    return list(TOOLS.values())


def clear() -> None:
    """Clear the registry — useful for tests."""
    TOOLS.clear()