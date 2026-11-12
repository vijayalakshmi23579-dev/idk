"""
path_safety.py — protected path detection for the permission gauntlet.

Consolidated from:
  - src/utils/permissions/filesystem.ts (DANGEROUS_FILES, checkPathSafetyForAutoEdit)
  - src/utils/permissions/pathValidation.ts (expandTilde, formatDirectoryList,
    getGlobBaseDirectory, isDangerousRemovalPath)

This is what stage 1g of the gauntlet calls. It answers: "would auto-editing
this path be safe?" — and if not, gives the human a clear reason.

The three safety checks (in order, matching Claude Code exactly):
  1. Suspicious Windows path patterns  → classifierApprovable=False (HARD DENY)
  2. Claude config files               → classifierApprovable=True
  3. Dangerous files/directories       → classifierApprovable=True

The classifierApprovable flag matters: a classifier-approvable safety check
can still be approved by the auto-mode classifier, but a non-approvable one
ALWAYS requires manual human approval even in bypass mode.

NOT in this file (deferred to tools/bash.py when built):
  - Bash command parsing (cd/rm/mv path extraction)
  - Shell redirect analysis
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import PurePosixPath, PureWindowsPath


# ── Dangerous paths (exact from filesystem.ts lines 34-38) ────────────────

DANGEROUS_FILES: tuple[str, ...] = (
    ".gitconfig",
    ".gitmodules",
    ".bashrc",
    ".bash_profile",
    ".zshrc",
    ".zprofile",
    ".profile",
    ".ripgreprc",
    ".mcp.json",
    ".claude.json",
)

DANGEROUS_DIRECTORIES: tuple[str, ...] = (
    ".git",
    ".vscode",
    ".idea",
    ".claude",
)

# Directories that, when removed, cause catastrophic data loss.
# Used by BashTool for `rm -rf /` detection, also useful here.
_DANGEROUS_REMOVAL_PATHS: frozenset[str] = frozenset({
    "/", "/usr", "/etc", "/bin", "/sbin", "/lib", "/lib64",
    "/boot", "/dev", "/proc", "/sys", "/var", "/root",
    "/home", "/opt", "/srv", "/tmp", "/var/tmp",
    "C:\\", "C:\\Windows", "C:\\Program Files",
})


# ── Result type ───────────────────────────────────────────────────────────

@dataclass(frozen=True)
class SafetyCheckResult:
    """Result of check_path_safety_for_auto_edit()."""
    safe: bool
    message: str = ""
    classifier_approvable: bool = True
    """False → HARD DENY, even the auto-mode classifier cannot approve.
    True → the classifier may approve, or the human may approve."""


SAFE = SafetyCheckResult(safe=True)


# ── Path normalization helpers ────────────────────────────────────────────

def normalize_case_for_comparison(path: str) -> str:
    """Lowercase for case-insensitive comparison on macOS/Windows.

    Matches normalizeCaseForComparison in filesystem.ts line 44-45.
    Always lowercase, even on Linux, for consistent security behavior.
    """
    return path.lower()


def expand_tilde(path: str) -> str:
    """Expand ~ at the start. Does NOT expand ~username (security).

    Matches expandTilde in pathValidation.ts.
    """
    if path == "~" or path.startswith("~/") or path.startswith("~\\"):
        return os.path.expanduser("~") + path[1:]
    return path


def format_directory_list(directories: list[str], max_display: int = 5) -> str:
    """Format a list of dirs as "'a', 'b', and N more".

    Matches formatDirectoryList in pathValidation.ts.
    """
    n = len(directories)
    if n <= max_display:
        return ", ".join(f"'{d}'" for d in directories)
    first = ", ".join(f"'{d}'" for d in directories[:max_display])
    return f"{first}, and {n - max_display} more"


_GLOB_CHARS = re.compile(r"[*?\[\]{}]")


def get_glob_base_directory(path: str) -> str:
    """Extract the non-glob base directory from a glob pattern.

    '/path/to/*.txt' → '/path/to'
    '*.txt'          → '.'

    Matches getGlobBaseDirectory in pathValidation.ts.
    """
    m = _GLOB_CHARS.search(path)
    if not m:
        return path
    before_glob = path[: m.start()]
    last_sep = max(before_glob.rfind("/"), before_glob.rfind("\\"))
    if last_sep == -1:
        return "."
    return before_glob[:last_sep] or "/"


# ── Dangerous path detection ──────────────────────────────────────────────

def is_dangerous_file_path_to_auto_edit(path: str) -> bool:
    """Should this path be blocked from auto-editing?

    Exact port of isDangerousFilePathToAutoEdit (filesystem.ts lines 217-240):

      - UNC paths (\\\\ or //)
      - Any path segment matching a DANGEROUS_DIRECTORY (case-insensitive)
        EXCEPT .claude/worktrees/ — structural, not user-created
      - Basename matching a DANGEROUS_FILE (case-insensitive)
    """
    absolute = expand_tilde(path)

    # UNC path defense-in-depth (matches the TS check)
    if absolute.startswith("\\\\") or absolute.startswith("//"):
        return True

    # Split on either separator — handles Windows and POSIX input
    segments = re.split(r"[/\\]", absolute)
    filename = segments[-1] if segments else ""

    # Check every segment for a dangerous directory
    for i, segment in enumerate(segments):
        normalized = normalize_case_for_comparison(segment)
        for dangerous_dir in DANGEROUS_DIRECTORIES:
            if normalized != normalize_case_for_comparison(dangerous_dir):
                continue
            # Special case: .claude/worktrees/ is structural — skip this .claude
            if dangerous_dir == ".claude":
                next_seg = segments[i + 1] if i + 1 < len(segments) else ""
                if normalize_case_for_comparison(next_seg) == "worktrees":
                    break  # skip this .claude, keep scanning other segments
            return True

    # Check filename against dangerous files
    if filename:
        normalized_file = normalize_case_for_comparison(filename)
        for dangerous_file in DANGEROUS_FILES:
            if normalize_case_for_comparison(dangerous_file) == normalized_file:
                return True

    return False


# ── Suspicious Windows patterns ───────────────────────────────────────────
# These are Windows-specific bypass techniques. We check on all platforms
# (defense-in-depth) because NTFS can be mounted on Linux/macOS. A hit here
# is NON-classifier-approvable — it always needs manual approval.

_WINDOWS_DOS_DEVICES = re.compile(
    r"^(CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])(\..*)?$", re.IGNORECASE
)


def has_suspicious_windows_path_pattern(path: str) -> bool:
    """Detect Windows path bypass patterns.

    Checks for:
      - NTFS Alternate Data Streams: 'file.txt:stream', 'file.txt::$DATA'
      - 8.3 short names: 'GIT~1', 'SETTIN~1.JSON'
      - Long-path / device prefixes: '\\\\?\\', '\\\\.\\', '//?/', '//./'
      - Trailing dots/spaces: '.git.', '.claude '
      - DOS device names as ANY dot-separated segment of the basename:
          'CON', 'CON.txt', 'settings.json.PRN'
      - Three+ consecutive dots: '...', 'path/.../file'
    """
    if not path:
        return False

    # 1. NTFS Alternate Data Streams (colon after drive letter position)
    if os.name == "nt" or os.environ.get("WSL_DISTRO_NAME"):
        colon_idx = path.find(":", 2)
        if colon_idx != -1:
            return True

    # 2. Long-path / device prefixes
    if path.startswith("\\\\?\\") or path.startswith("\\\\.\\"):
        return True
    if path.startswith("//?/") or path.startswith("//./"):
        return True

    # 3. 8.3 short names: a tilde followed by digits
    if re.search(r"~\d", path):
        return True

    # 4. Trailing dots or spaces on any segment
    for segment in re.split(r"[/\\]", path):
        if segment and (segment.endswith(".") or segment.endswith(" ")):
            return True

    # 5. DOS device names as ANY dot-separated segment of the basename.
    #    This catches:
    #      'CON'             → basename='CON', segments=['CON']
    #      'CON.txt'         → segments=['CON', 'txt']
    #      'settings.json.PRN' → segments=['settings', 'json', 'PRN']
    #    Anything matching CON/PRN/AUX/NUL/COM1-9/LPT1-9 in any position
    #    is flagged. Windows treats the FIRST segment as the device name
    #    when it matches, but hostile files can hide the device name in the
    #    trailing extension to slip past simpler checks.
    basename = re.split(r"[/\\]", path)[-1]
    for segment in basename.split("."):
        if segment and _WINDOWS_DOS_DEVICES.match(segment):
            return True

    # 6. Three or more consecutive dots
    if "..." in path:
        return True

    return False


# ── Claude config file detection ──────────────────────────────────────────

def is_claude_config_file_path(path: str) -> bool:
    """Is this a Claude Code config/state file that must not be auto-edited?

    Covers (approximation of isClaudeConfigFilePath in filesystem.ts):
      - <claude_home>/settings.json
      - <claude_home>/settings.local.json
      - <claude_home>/commands/...
      - <claude_home>/agents/...
      - <claude_home>/mcp* (MCP CLI state)
      - <project>/.mcp.json

    These return classifier_approvable=True — the classifier MAY approve,
    but they default to asking.
    """
    normalized = normalize_case_for_comparison(expand_tilde(path))

    claude_home = normalize_case_for_comparison(
        os.path.join(os.path.expanduser("~"), ".claude")
    )

    if normalized.startswith(claude_home + os.sep):
        rel = normalized[len(claude_home) + 1:]
        if rel in ("settings.json", "settings.local.json"):
            return True
        if rel.startswith("commands" + os.sep) or rel == "commands":
            return True
        if rel.startswith("agents" + os.sep) or rel == "agents":
            return True
        if rel.startswith("mcp"):
            return True

    # Project-level .mcp.json
    if normalized.endswith(os.sep + ".mcp.json"):
        return True

    return False


# ── Dangerous removal detection ───────────────────────────────────────────

def is_dangerous_removal_path(path: str) -> bool:
    """Is this path catastrophic to `rm -rf`?

    Used by BashTool's checkDangerousRemovalPaths to catch `rm -rf /`,
    `rm -rf /etc`, etc. Matches the intent of isDangerousRemovalPath in
    pathValidation.ts (which lives in the general module).
    """
    normalized = os.path.normpath(expand_tilde(path))
    # Normalize slashes for comparison
    candidates = {normalized, normalized.rstrip("/\\") or "/"}
    for candidate in candidates:
        if candidate in _DANGEROUS_REMOVAL_PATHS:
            return True
    return False


# ── The main safety check ────────────────────────────────────────────────

def check_path_safety_for_auto_edit(path: str) -> SafetyCheckResult:
    """Would auto-editing this path be safe?

    Exact port of checkPathSafetyForAutoEdit (filesystem.ts lines 314-332).

    Three checks, in order (first match wins):
      1. Suspicious Windows pattern  → safe=False, classifierApprovable=False
      2. Claude config file          → safe=False, classifierApprovable=True
      3. Dangerous file/directory    → safe=False, classifierApprovable=True

    If all pass → safe=True.

    Caller (permissions.py stage 1g) turns safe=False into an ASK decision
    with decisionReason={type: 'safetyCheck', classifierApprovable: ...}.
    The classifierApprovable flag gates whether auto-mode may approve.
    """
    # Check 1: Windows path bypass patterns (HARD DENY)
    if has_suspicious_windows_path_pattern(path):
        return SafetyCheckResult(
            safe=False,
            message=(
                f"Claude requested permissions to write to {path}, which "
                f"contains a suspicious Windows path pattern that requires "
                f"manual approval."
            ),
            classifier_approvable=False,
        )

    # Check 2: Claude config files
    if is_claude_config_file_path(path):
        return SafetyCheckResult(
            safe=False,
            message=(
                f"Claude requested permissions to write to {path}, but you "
                f"haven't granted it yet."
            ),
            classifier_approvable=True,
        )

    # Check 3: Dangerous files/directories
    if is_dangerous_file_path_to_auto_edit(path):
        return SafetyCheckResult(
            safe=False,
            message=(
                f"Claude requested permissions to edit {path} which is a "
                f"sensitive file."
            ),
            classifier_approvable=True,
        )

    return SAFE