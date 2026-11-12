# test_permission_rules.py
from permission_rules import (
    parse_rule, rule_value_to_string, parse_mcp_name,
    tool_matches_rule_value, content_matches_rule_value,
    extract_rule_prefix, filter_denied_agent_types,
    PermissionRule, PermissionRuleValue,
    validate_rule_string,
)

# ── Parsing ──
assert parse_rule("Bash") == PermissionRuleValue("Bash", None)
assert parse_rule("Bash(git *)") == PermissionRuleValue("Bash", "git *")
assert parse_rule("Bash(echo 'hello (world)')") == PermissionRuleValue("Bash", "echo 'hello (world)'")
assert parse_rule("mcp__server1") == PermissionRuleValue("mcp__server1", None)
# NEW: wildcard in tool name
assert parse_rule("mcp__server1__*") == PermissionRuleValue("mcp__server1__*", None)
assert parse_rule("mcp__server1__tool1") == PermissionRuleValue("mcp__server1__tool1", None)
print("✅ Parsing OK")

# ── Serialization roundtrip ──
for s in ["Bash", "Bash(git *)", "Edit(src/**)", "mcp__server1__*"]:
    assert rule_value_to_string(parse_rule(s)) == s
print("✅ Roundtrip OK")

# ── MCP name parsing ──
assert parse_mcp_name("Bash") is None
assert parse_mcp_name("mcp__server1") == ("server1", None)
assert parse_mcp_name("mcp__server1__tool1") == ("server1", "tool1")
assert parse_mcp_name("mcp__server1__*") == ("server1", "*")
print("✅ MCP name parsing OK")

# ── Tool-level matching ──
v_bash = parse_rule("Bash")
v_git = parse_rule("Bash(git *)")
v_mcp_server = parse_rule("mcp__server1")
v_mcp_wild = parse_rule("mcp__server1__*")   # ← the case that broke
v_mcp_tool = parse_rule("mcp__server1__tool1")

assert tool_matches_rule_value("Bash", v_bash) is True
assert tool_matches_rule_value("Bash", v_git) is False
assert tool_matches_rule_value("Read", v_bash) is False

assert tool_matches_rule_value("mcp__server1__tool1", v_mcp_server) is True
assert tool_matches_rule_value("mcp__server1__tool2", v_mcp_server) is True
assert tool_matches_rule_value("mcp__server1__tool1", v_mcp_wild) is True
assert tool_matches_rule_value("mcp__server1__tool2", v_mcp_wild) is True
assert tool_matches_rule_value("mcp__server2__tool1", v_mcp_server) is False

assert tool_matches_rule_value("mcp__server1__tool1", v_mcp_tool) is True
assert tool_matches_rule_value("mcp__server1__tool2", v_mcp_tool) is False
print("✅ Tool-level matching OK")

# ── Content matching ──
assert content_matches_rule_value("git status", v_git) is True
assert content_matches_rule_value("npm install", v_git) is False
assert content_matches_rule_value("git push origin main", v_git) is True
print("✅ Content matching OK")

# ── Prefix extraction ──
assert extract_rule_prefix("git *") == "git"
assert extract_rule_prefix("git") == "git"
assert extract_rule_prefix("*") is None
assert extract_rule_prefix("git push*") == "git push"
print("✅ Prefix extraction OK")

# ── Denied agent types ──
rules = [
    PermissionRule("userSettings", "deny", PermissionRuleValue("Agent", "Explore")),
    PermissionRule("userSettings", "deny", PermissionRuleValue("Agent", "Plan")),
    PermissionRule("userSettings", "allow", PermissionRuleValue("Agent", "worker")),
    PermissionRule("userSettings", "deny", PermissionRuleValue("Bash", "rm -rf")),
]
denied = filter_denied_agent_types(rules, "Agent")
assert denied == {"Explore", "Plan"}
print("✅ Denied agent types OK")

# ── Validation ──
assert validate_rule_string("Bash(git *)") is None
assert validate_rule_string("mcp__server1__*") is None  # NEW
assert validate_rule_string("") is not None
assert validate_rule_string("123invalid") is not None
print("✅ Validation OK")

print("\n🎉 All permission_rules.py tests passed")