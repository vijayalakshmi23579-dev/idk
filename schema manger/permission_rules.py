"""
permission_rules.py — permission rule parsing and matching.

Extracted from src/utils/permissions/PermissionRule.ts and
permissionRuleParser.ts.

A rule is a string like:
  "Bash"              → the entire Bash tool (no content)
  "Bash(git *)"       → Bash commands starting with "git "
  "mcp__server1"      → ALL tools from MCP server1
  "mcp__server1__*"   → same (wildcard)
  "Edit(src/**)"      → Edit tool with paths under src/

parse_rule() converts to PermissionRuleValue{tool_name, rule_content}.
match_rule_value() checks if a tool call satisfies a rule value.
"""
from __future__ import annotations

import fnmatch
import re
from dataclasses import dataclass
from typing import Literal


# ── Types ─────────────────────────────────────────────────────────────────

PermissionBehavior = Literal["allow", "deny", "ask"]

PermissionRuleSource = Literal[
    "policySettings",   # enterprise MDM — highest authority
    "flagSettings",     # CLI flags
    "userSettings",     # ~/.claude/settings.json
    "projectSettings",  # .claude/settings.json
    "localSettings",    # .claude/settings.local.json
    "cliArg",           # --allowedTools flag
    "command",          # from a slash command
    "session",          # in-memory, session-only
]

RULE_SOURCES: tuple[PermissionRuleSource, ...] = (
    "policySettings",
    "flagSettings",
    "userSettings",
    "projectSettings",
    "localSettings",
    "cliArg",
    "command",
    "session",
)


@dataclass(frozen=True)
class PermissionRuleValue:
    """The parsed form of a rule string."""
    tool_name: str
    rule_content: str | None = None


@dataclass(frozen=True)
class PermissionRule:
    """A rule with its behavior and source."""
    source: PermissionRuleSource
    rule_behavior: PermissionBehavior
    rule_value: PermissionRuleValue


# ── Parsing ───────────────────────────────────────────────────────────────

# Tool name: starts with letter or underscore, allows letters/digits/_/*
# The '*' allows MCP wildcard form (mcp__server1__*).
# First char is still restricted to [A-Za-z_] — a rule cannot start with '*'.
_RULE_PATTERN = re.compile(r"^([A-Za-z_][A-Za-z0-9_*]*)(?:\((.*)\))?$", re.DOTALL)


class RuleParseError(ValueError):
    """Raised when a rule string is malformed."""


def parse_rule(rule_string: str) -> PermissionRuleValue:
    """
    Parse "Bash(git *)" → PermissionRuleValue("Bash", "git *").
    Parse "Bash"       → PermissionRuleValue("Bash", None).
    Parse "mcp__srv__*" → PermissionRuleValue("mcp__srv__*", None).

    Raises RuleParseError for malformed input.
    """
    s = rule_string.strip()
    if not s:
        raise RuleParseError("Empty rule string")

    m = _RULE_PATTERN.match(s)
    if not m:
        raise RuleParseError(
            f"Invalid rule format: {rule_string!r}. "
            f"Expected 'ToolName' or 'ToolName(content)'"
        )

    tool_name = m.group(1)
    raw_content = m.group(2)

    if raw_content is None or raw_content == "":
        return PermissionRuleValue(tool_name=tool_name, rule_content=None)

    return PermissionRuleValue(tool_name=tool_name, rule_content=raw_content)


def rule_value_to_string(value: PermissionRuleValue) -> str:
    """Serialize back to "ToolName" or "ToolName(content)"."""
    if value.rule_content is None:
        return value.tool_name
    return f"{value.tool_name}({value.rule_content})"


# ── MCP name handling ─────────────────────────────────────────────────────

def parse_mcp_name(name: str) -> tuple[str, str | None] | None:
    """
    Parse an MCP-qualified tool name.

    "mcp__server1__tool1" → ("server1", "tool1")
    "mcp__server1"        → ("server1", None)   ← server-level rule
    "mcp__server1__*"     → ("server1", "*")    ← wildcard rule
    "Bash"                → None                ← not MCP
    """
    if not name.startswith("mcp__"):
        return None
    parts = name.split("__")
    if len(parts) < 2 or parts[0] != "mcp" or not parts[1]:
        return None
    server_name = parts[1]
    if len(parts) == 2:
        return (server_name, None)
    tool_name = "__".join(parts[2:])
    return (server_name, tool_name)


# ── Matching ──────────────────────────────────────────────────────────────

def tool_matches_rule_value(
    tool_name: str,
    rule_value: PermissionRuleValue,
) -> bool:
    """
    Does this tool match the rule at the TOOL level?

    A tool-level match requires rule_content is None OR the rule is an
    MCP server-level rule (which applies to all tools from that server).

    Content-specific rules (e.g. "Bash(git *)") do NOT match here — they
    are delegated to the tool's checkPermissions() for content matching.
    """
    # Direct name match with no content
    if rule_value.rule_content is None and rule_value.tool_name == tool_name:
        return True

    # MCP server-level rule: "mcp__server1" or "mcp__server1__*"
    rule_mcp = parse_mcp_name(rule_value.tool_name)
    tool_mcp = parse_mcp_name(tool_name)
    if rule_mcp is not None and tool_mcp is not None:
        rule_server, rule_tool = rule_mcp
        tool_server, tool_tool = tool_mcp
        if rule_server != tool_server:
            return False
        if rule_tool is None or rule_tool == "*":
            return True
        return rule_tool == tool_tool

    return False


def content_matches_rule_value(
    content: str,
    rule_value: PermissionRuleValue,
) -> bool:
    """
    Does this content match the rule's content pattern?

    - rule_content is None → matches everything (whole-tool rule)
    - rule_content has no wildcards → exact match
    - rule_content has "*" or "?" → shell-style glob match
    """
    if rule_value.rule_content is None:
        return True
    return _glob_match(rule_value.rule_content, content)


def _glob_match(pattern: str, text: str) -> bool:
    return fnmatch.fnmatchcase(text, pattern)


def extract_rule_prefix(pattern: str) -> str | None:
    """
    For "git *" → returns "git" (the non-wildcard prefix).
    For "git"   → returns "git".
    For "*"     → returns None (no useful prefix).
    """
    star_idx = pattern.find("*")
    if star_idx == -1:
        return pattern
    prefix = pattern[:star_idx].rstrip()
    if not prefix:
        return None
    return prefix


# ── Rule list helpers ────────────────────────────────────────────────────

def find_matching_rule(
    tool_name: str,
    rules: list[PermissionRule],
) -> PermissionRule | None:
    """Find the first rule in the list that matches this tool at the tool level."""
    for rule in rules:
        if tool_matches_rule_value(tool_name, rule.rule_value):
            return rule
    return None


def filter_denied_agent_types(
    rules: list[PermissionRule],
    agent_tool_name: str,
) -> set[str]:
    """
    Collect the agent types denied by Agent(agentType) rules.

    Parses once, builds a set — O(agents × rules) becomes O(rules).
    """
    denied: set[str] = set()
    for rule in rules:
        if rule.rule_behavior != "deny":
            continue
        if rule.rule_value.tool_name != agent_tool_name:
            continue
        if rule.rule_value.rule_content is not None:
            denied.add(rule.rule_value.rule_content)
    return denied


# ── Validation ────────────────────────────────────────────────────────────

def validate_rule_string(rule_string: str) -> str | None:
    """
    Validate a rule string. Returns None if valid, or an error message.
    """
    try:
        parse_rule(rule_string)
    except RuleParseError as e:
        return str(e)
    return None