"""test_path_safety.py"""
import os
from path_safety import (
    DANGEROUS_FILES, DANGEROUS_DIRECTORIES,
    SafetyCheckResult, SAFE,
    normalize_case_for_comparison, expand_tilde,
    format_directory_list, get_glob_base_directory,
    is_dangerous_file_path_to_auto_edit,
    has_suspicious_windows_path_pattern,
    is_claude_config_file_path,
    is_dangerous_removal_path,
    check_path_safety_for_auto_edit,
)

# ── Constants match source ──
assert ".gitconfig" in DANGEROUS_FILES
assert ".bashrc" in DANGEROUS_FILES
assert ".claude.json" in DANGEROUS_FILES
assert ".git" in DANGEROUS_DIRECTORIES
assert ".claude" in DANGEROUS_DIRECTORIES
print("✅ Constants OK")

# ── Normalization ──
assert normalize_case_for_comparison(".ClAuDe") == ".claude"
assert expand_tilde("~/foo") == os.path.expanduser("~") + "/foo"
assert expand_tilde("~") == os.path.expanduser("~")
assert expand_tilde("/abs/path") == "/abs/path"
assert expand_tilde("~/foo").startswith("/")  # always absolute after expansion
print("✅ Normalization OK")

# ── Directory list formatting ──
assert format_directory_list(["a"]) == "'a'"
assert format_directory_list(["a", "b"]) == "'a', 'b'"
assert format_directory_list(["a", "b", "c", "d", "e", "f"]) == "'a', 'b', 'c', 'd', 'e', and 1 more"
print("✅ Directory list formatting OK")

# ── Glob base directory ──
assert get_glob_base_directory("/path/to/*.txt") == "/path/to"
assert get_glob_base_directory("*.txt") == "."
assert get_glob_base_directory("/path/to/file.txt") == "/path/to/file.txt"
assert get_glob_base_directory("/a/b?c") == "/a"
print("✅ Glob base directory OK")

# ── Dangerous file detection ──
assert is_dangerous_file_path_to_auto_edit("/home/user/.bashrc") is True
assert is_dangerous_file_path_to_auto_edit("/home/user/.gitconfig") is True
assert is_dangerous_file_path_to_auto_edit("/project/.git/hooks/pre-commit") is True
assert is_dangerous_file_path_to_auto_edit("/project/.vscode/settings.json") is True
assert is_dangerous_file_path_to_auto_edit("/project/.idea/workspace.xml") is True
assert is_dangerous_file_path_to_auto_edit("/home/user/.claude/settings.json") is True
# Not dangerous
assert is_dangerous_file_path_to_auto_edit("/home/user/README.md") is False
assert is_dangerous_file_path_to_auto_edit("/project/src/main.py") is False
# .claude/worktrees/ exception
assert is_dangerous_file_path_to_auto_edit("/project/.claude/worktrees/agent-1/file.py") is False
assert is_dangerous_file_path_to_auto_edit("/project/.claude/worktrees/agent-1/.claude/x") is True  # nested .claude not followed by worktrees
# Case-insensitive
assert is_dangerous_file_path_to_auto_edit("/home/user/.BASHRC") is True
assert is_dangerous_file_path_to_auto_edit("/project/.GIT/config") is True
print("✅ Dangerous file detection OK")

# ── Windows pattern detection ──
assert has_suspicious_windows_path_pattern("\\\\?\\C:\\foo") is True
assert has_suspicious_windows_path_pattern("//?/C:/foo") is True
assert has_suspicious_windows_path_pattern("GIT~1") is True
assert has_suspicious_windows_path_pattern("path/file.") is True
assert has_suspicious_windows_path_pattern("path/.../file") is True
assert has_suspicious_windows_path_pattern("CON") is True
assert has_suspicious_windows_path_pattern("settings.json.PRN") is True
# Normal paths
assert has_suspicious_windows_path_pattern("/home/user/README.md") is False
assert has_suspicious_windows_path_pattern("/project/src/main.py") is False
print("✅ Windows pattern detection OK")

# ── Claude config file detection ──
home = os.path.expanduser("~")
assert is_claude_config_file_path(f"{home}/.claude/settings.json") is True
assert is_claude_config_file_path(f"{home}/.claude/settings.local.json") is True
assert is_claude_config_file_path(f"{home}/.claude/commands/foo.md") is True
assert is_claude_config_file_path(f"{home}/.claude/agents/bar.md") is True
assert is_claude_config_file_path(f"{home}/.claude/mcp-servers.json") is True
assert is_claude_config_file_path("/project/.mcp.json") is True
# Not claude config
assert is_claude_config_file_path(f"{home}/.claude/projects/foo") is False
assert is_claude_config_file_path("/project/README.md") is False
print("✅ Claude config detection OK")

# ── Dangerous removal ──
assert is_dangerous_removal_path("/") is True
assert is_dangerous_removal_path("/etc") is True
assert is_dangerous_removal_path("/home") is True
assert is_dangerous_removal_path("/usr/local") is False
assert is_dangerous_removal_path("/tmp/foo") is False
print("✅ Dangerous removal OK")

# ── Main safety check ──
r = check_path_safety_for_auto_edit("/home/user/README.md")
assert r.safe is True
assert r is SAFE

# Dangerous file → safe=False, classifierApprovable=True
r = check_path_safety_for_auto_edit("/home/user/.bashrc")
assert r.safe is False
assert r.classifier_approvable is True
assert "sensitive file" in r.message

# Claude config → safe=False, classifierApprovable=True
r = check_path_safety_for_auto_edit(f"{home}/.claude/settings.json")
assert r.safe is False
assert r.classifier_approvable is True
assert "haven't granted" in r.message

# Windows pattern → safe=False, classifierApprovable=False (HARD DENY)
r = check_path_safety_for_auto_edit("\\\\?\\C:\\foo")
assert r.safe is False
assert r.classifier_approvable is False
assert "suspicious Windows path pattern" in r.message
print("✅ Main safety check OK")

# ── Order matters: Windows pattern checked before dangerous file ──
# A path that matches both should hit the Windows check first (harder deny)
r = check_path_safety_for_auto_edit("\\\\?\\C:\\.bashrc")
assert r.safe is False
assert r.classifier_approvable is False  # Windows wins, not the .bashrc rule
print("✅ Check ordering OK")

print("\n🎉 All path_safety.py tests passed")