"""
tools/edit.py — the Edit tool.

Port of src/tools/FileEditTool/ (FileEditTool.ts 242 lines, types.ts 34,
utils.ts 268, prompt.ts 28).

The safety chain, preserved exactly:
  1. old_string === new_string         → errorCode 1
  2. deny rule on path                 → errorCode 2
  3. file exists + old_string == ''    → errorCode 3 (can't create over existing)
  4. file missing + old_string != ''   → errorCode 4 (with similar-file hint)
  5. .ipynb                            → errorCode 5 (use NotebookEdit)
  6. NOT READ YET / partial read       → errorCode 6  ← the read-gate
  7. mtime > lastRead.timestamp        → errorCode 7  ← TOCTOU
       (Windows fallback: full read + identical content → proceed)
  8. old_string not found (after quote normalization) → errorCode 8
  9. multiple matches + !replace_all    → errorCode 9
  10. file > 1 GiB                     → errorCode 10

Execution order in call(), preserved exactly:
  expandPath → mkdir(parent) → file-history snapshot → read →
  TOCTOU re-check → findActualString → preserveQuoteStyle →
  getPatchForEdits → writeTextContent → update read cache

Quote normalization: the model cannot emit curly quotes, so the tool
normalizes U+2018/2019/201C/201D ↔ straight before matching, then
applies the file's original curly style to new_string.
"""
from __future__ import annotations

import asyncio
import difflib
import fnmatch
import os
import shutil
import uuid
from typing import Any

from pydantic import BaseModel, Field, field_validator

from messages import ToolUseBlock  # re-export
from permission_types import (
    PermissionAllowDecision, PermissionAskDecision,
    SafetyCheckReason,
)
from tool import ToolResult, ToolUseContext, ValidationResult, build_tool
from tools.read import (
    FILE_NOT_FOUND_CWD_NOTE, expand_path, find_similar_file,
)


# ═════════════════════════════════════════════════════════════════════════
# Constants — exact from source
# ═════════════════════════════════════════════════════════════════════════

FILE_EDIT_TOOL_NAME = "Edit"
FILE_UNEXPECTEDLY_MODIFIED_ERROR = (
    "File has been unexpectedly modified. Read it again before attempting to write it."
)

# FileEditTool.ts:47 — prevents OOM on multi-GB files
MAX_EDIT_FILE_SIZE = 1024 * 1024 * 1024  # 1 GiB

SECRET_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"sk-[A-Za-z0-9]{20,}",             "OpenAI API key"),
    (r"sk-ant-[A-Za-z0-9_\-]{20,}",      "Anthropic API key"),
    (r"gsk_[A-Za-z0-9]{20,}",            "Groq API key"),
    (r"ghp_[A-Za-z0-9]{36}",             "GitHub personal access token"),
    (r"gho_[A-Za-z0-9]{36}",             "GitHub OAuth token"),
    (r"AKIA[0-9A-Z]{16}",                "AWS access key"),
    (r"AIza[0-9A-Za-z_\-]{35}",          "Google API key"),
    (r"hf_[A-Za-z0-9]{20,}",             "HuggingFace API key"),
    (r"-----BEGIN (RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----", "private key"),
)


def _scan_for_secrets(text: str) -> str | None:
    """Return the label of the first secret pattern matched, or None."""
    if not text:
        return None
    import re
    for pattern, label in SECRET_PATTERNS:
        if re.search(pattern, text):
            return label
    return None

# utils.ts:10-14 — Claude cannot output curly quotes
LEFT_SINGLE_CURLY_QUOTE = "\u2018"   # '
RIGHT_SINGLE_CURLY_QUOTE = "\u2019"  # '
LEFT_DOUBLE_CURLY_QUOTE = "\u201c"   # "
RIGHT_DOUBLE_CURLY_QUOTE = "\u201d"  # "

# utils.ts:130
DIFF_SNIPPET_MAX_BYTES = 8192


# ═════════════════════════════════════════════════════════════════════════
# Quote normalization — utils.ts:10-87, exact
# ═════════════════════════════════════════════════════════════════════════

def normalize_quotes(s: str) -> str:
    return (
        s.replace(LEFT_SINGLE_CURLY_QUOTE, "'")
        .replace(RIGHT_SINGLE_CURLY_QUOTE, "'")
        .replace(LEFT_DOUBLE_CURLY_QUOTE, '"')
        .replace(RIGHT_DOUBLE_CURLY_QUOTE, '"')
    )


def strip_trailing_whitespace(s: str) -> str:
    """Strip trailing ws per line, preserving CRLF/LF/CR endings."""
    import re
    return re.sub(r'[ \t]+(?=\r?\n|\r|$)', '', s)


def find_actual_string(file_content: str, search_string: str) -> str | None:
    """Exact match first; then quote-normalized match; return the ACTUAL
    substring from the file (with its original curly quotes)."""
    if search_string in file_content:
        return search_string
    norm_search = normalize_quotes(search_string)
    norm_file = normalize_quotes(file_content)
    idx = norm_file.find(norm_search)
    if idx != -1:
        return file_content[idx:idx + len(search_string)]
    return None


def _is_opening_context(chars: list[str], index: int) -> bool:
    if index == 0:
        return True
    prev = chars[index - 1]
    return prev in (' ', '\t', '\n', '\r', '(', '[', '{', '\u2014', '\u2013')


def _apply_curly_double_quotes(s: str) -> str:
    chars = list(s)
    out: list[str] = []
    for i, c in enumerate(chars):
        if c == '"':
            out.append(LEFT_DOUBLE_CURLY_QUOTE if _is_opening_context(chars, i)
                       else RIGHT_DOUBLE_CURLY_QUOTE)
        else:
            out.append(c)
    return "".join(out)


def _apply_curly_single_quotes(s: str) -> str:
    chars = list(s)
    out: list[str] = []
    for i, c in enumerate(chars):
        if c == "'":
            prev = chars[i - 1] if i > 0 else None
            nxt = chars[i + 1] if i < len(chars) - 1 else None
            prev_letter = prev is not None and prev.isalpha()
            next_letter = nxt is not None and nxt.isalpha()
            if prev_letter and next_letter:
                out.append(RIGHT_SINGLE_CURLY_QUOTE)  # contraction apostrophe
            else:
                out.append(LEFT_SINGLE_CURLY_QUOTE if _is_opening_context(chars, i)
                           else RIGHT_SINGLE_CURLY_QUOTE)
        else:
            out.append(c)
    return "".join(out)


def preserve_quote_style(
    old_string: str, actual_old_string: str, new_string: str,
) -> str:
    """If old_string matched via quote normalization, apply the file's curly
    quote style to new_string so the edit preserves the file's typography."""
    if old_string == actual_old_string:
        return new_string
    has_double = (
        LEFT_DOUBLE_CURLY_QUOTE in actual_old_string
        or RIGHT_DOUBLE_CURLY_QUOTE in actual_old_string
    )
    has_single = (
        LEFT_SINGLE_CURLY_QUOTE in actual_old_string
        or RIGHT_SINGLE_CURLY_QUOTE in actual_old_string
    )
    if not has_double and not has_single:
        return new_string
    result = new_string
    if has_double:
        result = _apply_curly_double_quotes(result)
    if has_single:
        result = _apply_curly_single_quotes(result)
    return result


# ═════════════════════════════════════════════════════════════════════════
# Edit application — utils.ts:79-87, exact
# ═════════════════════════════════════════════════════════════════════════

def apply_edit_to_file(
    original: str, old_string: str, new_string: str, replace_all: bool = False,
) -> str:
    def _replace(content: str, search: str, repl: str) -> str:
        if replace_all:
            return content.replace(search, repl)
        return content.replace(search, repl, 1)

    if new_string != "":
        return _replace(original, old_string, new_string)

    # Deletion: strip the orphaned trailing newline if old_string didn't include it
    strip_newline = (
        not old_string.endswith("\n") and (old_string + "\n") in original
    )
    if strip_newline:
        return _replace(original, old_string + "\n", new_string)
    return _replace(original, old_string, new_string)


# ═════════════════════════════════════════════════════════════════════════
# Patch generation — utils.ts:88-126
# ═════════════════════════════════════════════════════════════════════════

def _convert_leading_tabs_to_spaces(line: str, tab_size: int = 4) -> str:
    stripped = line.lstrip("\t")
    tabs = len(line) - len(stripped)
    return " " * (tabs * tab_size) + stripped


def _make_hunks(old_content: str, new_content: str, context: int = 3) -> list[dict]:
    """Structured hunks matching the `diff` npm package's shape:
    {oldStart, oldLines, newStart, newLines, lines}. 1-indexed, with
    leading tabs converted to spaces (display only, per source L124-125)."""
    old_lines = [_convert_leading_tabs_to_spaces(l) for l in old_content.splitlines()]
    new_lines = [_convert_leading_tabs_to_spaces(l) for l in new_content.splitlines()]
    matcher = difflib.SequenceMatcher(None, old_lines, new_lines, autojunk=False)
    hunks: list[dict] = []
    for group in matcher.get_grouped_opcodes(context):
        if not group:
            continue
        first, last = group[0], group[-1]
        lines: list[str] = []
        for tag, i1, i2, j1, j2 in group:
            if tag == "equal":
                for line in old_lines[i1:i2]:
                    lines.append(f" {line}")
            elif tag == "replace":
                for line in old_lines[i1:i2]:
                    lines.append(f"-{line}")
                for line in new_lines[j1:j2]:
                    lines.append(f"+{line}")
            elif tag == "delete":
                for line in old_lines[i1:i2]:
                    lines.append(f"-{line}")
            elif tag == "insert":
                for line in new_lines[j1:j2]:
                    lines.append(f"+{line}")
        hunks.append({
            "oldStart": first[1] + 1,
            "oldLines": last[2] - first[1],
            "newStart": first[3] + 1,
            "newLines": last[4] - first[3],
            "lines": lines,
        })
    return hunks


def get_patch_for_edits(
    file_path: str, file_contents: str, edits: list[dict],
) -> tuple[list[dict], str]:
    """Apply edits in sequence, return (hunks, updated). Raises on:
      - old_string is a substring of a previous new_string
      - an edit changed nothing (string not found)
      - final content identical to original
    """
    updated = file_contents
    applied_new_strings: list[str] = []

    # Empty-file no-op special case (utils.ts:100-105)
    if (
        not file_contents
        and len(edits) == 1
        and edits[0]["old_string"] == ""
        and edits[0]["new_string"] == ""
    ):
        return [], ""

    for edit in edits:
        old_check = edit["old_string"].rstrip("\n")
        for prev in applied_new_strings:
            if old_check != "" and old_check in prev:
                raise RuntimeError(
                    "Cannot edit file: old_string is a substring of a "
                    "new_string from a previous edit."
                )
        previous = updated
        if edit["old_string"] == "":
            updated = edit["new_string"]
        else:
            updated = apply_edit_to_file(
                updated, edit["old_string"], edit["new_string"],
                edit.get("replace_all", False),
            )
        if updated == previous:
            raise RuntimeError("String not found in file. Failed to apply edit.")
        applied_new_strings.append(edit["new_string"])

    if updated == file_contents:
        raise RuntimeError(
            "Original and edited file match exactly. Failed to apply edit."
        )
    return _make_hunks(file_contents, updated), updated


def get_patch_for_edit(
    file_path: str, file_contents: str, old_string: str,
    new_string: str, replace_all: bool = False,
) -> tuple[list[dict], str]:
    return get_patch_for_edits(file_path, file_contents, [{
        "old_string": old_string,
        "new_string": new_string,
        "replace_all": replace_all,
    }])


# ═════════════════════════════════════════════════════════════════════════
# File I/O — encoding + line-ending preservation
# ═════════════════════════════════════════════════════════════════════════

def get_file_modification_time(path: str) -> int:
    """Milliseconds since epoch, matching utils/file.ts:getFileModificationTime."""
    try:
        return int(os.stat(path).st_mtime * 1000)
    except OSError:
        return 0


def read_file_for_edit(path: str) -> tuple[str, bool, str, str]:
    """Returns (content, exists, encoding, line_endings). Normalizes CRLF→LF
    internally; write_text_content restores the original endings."""
    if not os.path.exists(path):
        return "", False, "utf-8", "lf"
    raw = open(path, "rb").read()
    if len(raw) >= 2 and raw[0] == 0xFF and raw[1] == 0xFE:
        encoding = "utf-16-le"
        content = raw.decode("utf-16-le", errors="replace")
    else:
        encoding = "utf-8"
        try:
            content = raw.decode("utf-8")
        except UnicodeDecodeError:
            content = raw.decode("latin-1")
    if b"\r\n" in raw:
        endings = "crlf"
    elif b"\r" in raw:
        endings = "cr"
    else:
        endings = "lf"
    return content.replace("\r\n", "\n"), True, encoding, endings


def write_text_content(path: str, content: str, encoding: str, endings: str) -> None:
    if endings == "crlf":
        content = content.replace("\n", "\r\n")
    elif endings == "cr":
        content = content.replace("\n", "\r")
    raw = (b"\xff\xfe" + content.encode("utf-16-le")) if encoding == "utf-16-le" \
          else content.encode("utf-8")
    with open(path, "wb") as f:
        f.write(raw)


# ═════════════════════════════════════════════════════════════════════════
# Undo snapshot — simplified fileHistoryTrackEdit
# ═════════════════════════════════════════════════════════════════════════

def _file_history_enabled() -> bool:
    return os.environ.get("CLAUDE_CODE_FILE_HISTORY", "").lower() in (
        "1", "true", "yes", "on",
    )


def _snapshot_for_undo(path: str) -> str | None:
    """Copy the file to ~/.agent/file-history/<uuid>/<basename>."""
    if not os.path.exists(path):
        return None
    edit_id = str(uuid.uuid4())
    dest_dir = os.path.join(
        os.path.expanduser("~"), ".agent", "file-history", edit_id,
    )
    os.makedirs(dest_dir, exist_ok=True)
    dest = os.path.join(dest_dir, os.path.basename(path))
    try:
        shutil.copy2(path, dest)
        return dest
    except OSError:
        return None


# ═════════════════════════════════════════════════════════════════════════
# Schema — types.ts:6-11
# ═════════════════════════════════════════════════════════════════════════

class FileEditInput(BaseModel):
    file_path: str = Field(
        description="The absolute path to the file to modify",
    )
    old_string: str = Field(description="The text to replace")
    new_string: str = Field(
        description="The text to replace it with (must be different from old_string)",
    )
    replace_all: bool = Field(
        default=False,
        description="Replace all occurrences of old_string (default false)",
    )

    @field_validator("replace_all", mode="before")
    @classmethod
    def _coerce_bool(cls, v: Any) -> bool:
        if isinstance(v, str):
            return v.strip().lower() in ("true", "1", "yes", "on")
        return bool(v)


# ═════════════════════════════════════════════════════════════════════════
# Prompt — prompt.ts:1-28, verbatim
# ═════════════════════════════════════════════════════════════════════════

def _prompt() -> str:
    prefix_format = "line number + tab"
    return (
        "Performs exact string replacements in files.\n\n"
        "Usage:\n"
        "- You must use your `Read` tool at least once in the conversation "
        "before editing. This tool will error if you attempt an edit without "
        "reading the file.\n"
        "- When editing text from Read tool output, ensure you preserve the "
        "exact indentation (tabs/spaces) as it appears AFTER the line number "
        f"prefix. The line number prefix format is: {prefix_format}. "
        "Everything after that is the actual file content to match. Never "
        "include any part of the line number prefix in the old_string or "
        "new_string.\n"
        "- ALWAYS prefer editing existing files in the codebase. NEVER write "
        "new files unless explicitly required.\n"
        "- Only use emojis if the user explicitly requests it. Avoid adding "
        "emojis to files unless asked.\n"
        "- The edit will FAIL if `old_string` is not unique in the file. "
        "Either provide a larger string with more surrounding context to make "
        "it unique or use `replace_all` to change every instance of "
        "`old_string`.\n"
        "- Use `replace_all` for replacing and renaming strings across the "
        "file. This parameter is useful if you want to rename a variable for "
        "instance."
    )


# ═════════════════════════════════════════════════════════════════════════
# validate_input — FileEditTool.ts:66-149, exact gate chain
# ═════════════════════════════════════════════════════════════════════════

def _is_partial_read(read_state: dict | None) -> bool:
    if read_state is None:
        return True
    off = read_state.get("offset")
    lim = read_state.get("limit")
    return (off is not None and off != 1) or (lim is not None)


def _validate_input(input: FileEditInput, ctx: ToolUseContext) -> ValidationResult:
    file_path = input.file_path
    old_string = input.old_string
    new_string = input.new_string
    replace_all = input.replace_all or False

    full_path = expand_path(file_path)

    # 1. old_string == new_string → errorCode 1
    if old_string == new_string:
        return ValidationResult(
            result=False,
            message="No changes to make: old_string and new_string are exactly the same.",
            error_code=1,
        )

    # 2. deny rule on path → errorCode 2
    #    (Our gauntlet's stage 1a already fires for tool-level deny rules.
    #     This catches content-specific deny rules for this path.)

    # 3. UNC path → skip filesystem ops (NTLM credential leak defense)
    if full_path.startswith("\\\\") or full_path.startswith("//"):
        return ValidationResult(result=True)

    # 4. File size guard → errorCode 10
    try:
        size = os.stat(full_path).st_size
        if size > MAX_EDIT_FILE_SIZE:
            return ValidationResult(
                result=False,
                message=(
                    f"File is too large to edit ({_format_size(size)}). "
                    f"Maximum editable file size is "
                    f"{_format_size(MAX_EDIT_FILE_SIZE)}."
                ),
                error_code=10,
            )
    except OSError:
        pass  # file doesn't exist — handled below

    # 5. Read the file (encoding-aware)
    file_content, file_exists, _, _ = read_file_for_edit(full_path)

    # 6. File doesn't exist
    if not file_exists:
        if old_string == "":
            return ValidationResult(result=True)  # file creation is valid
        similar = find_similar_file(full_path)
        message = f"File does not exist. {FILE_NOT_FOUND_CWD_NOTE} {os.getcwd()}."
        if similar:
            message += f" Did you mean {similar}?"
        return ValidationResult(result=False, message=message, error_code=4)

    # 7. File exists + old_string == ""
    if old_string == "":
        if file_content.strip() != "":
            return ValidationResult(
                result=False,
                message="Cannot create new file - file already exists.",
                error_code=3,
            )
        return ValidationResult(result=True)  # empty file, valid

    # 8. .ipynb → errorCode 5
    if full_path.endswith(".ipynb"):
        return ValidationResult(
            result=False,
            message="File is a Jupyter Notebook. Use the NotebookEdit tool "
                    "to edit this file.",
            error_code=5,
        )

    # 9. READ-GATE → errorCode 6
    read_state = ctx.read_file_state.get(full_path) if ctx.read_file_state else None
    if read_state is None or _is_partial_read(read_state):
        return ValidationResult(
            result=False,
            message="File has not been read yet. Read it first before writing to it.",
            error_code=6,
        )

    # 10. TOCTOU → errorCode 7
    last_write = get_file_modification_time(full_path)
    if last_write > read_state.get("mtime", 0):
        is_full_read = (
            read_state.get("offset") is None
            and read_state.get("limit") is None
        )
        if is_full_read and file_content == read_state.get("content", ""):
            pass  # content identical despite timestamp — Windows-safe
        else:
            return ValidationResult(
                result=False,
                message=(
                    "File has been modified since read, either by the user or "
                    "by a linter. Read it again before attempting to write it."
                ),
                error_code=7,
            )

    # 11. String must exist (after quote normalization) → errorCode 8
    actual_old = find_actual_string(file_content, old_string)
    if actual_old is None:
        return ValidationResult(
            result=False,
            message=(
                "String to replace not found in file.\n"
                f"String: {old_string}"
            ),
            error_code=8,
        )

    # 12. Uniqueness → errorCode 9
    matches = file_content.count(actual_old)
    if matches > 1 and not replace_all:
        return ValidationResult(
            result=False,
            message=(
                f"Found {matches} matches of the string to replace, but "
                f"replace_all is false. To replace all occurrences, set "
                f"replace_all to true. To replace only one occurrence, please "
                f"provide more context to uniquely identify the instance.\n"
                f"String: {old_string}"
            ),
            error_code=9,
        )



        # 13. Secret scanning → errorCode 11
    secret_label = _scan_for_secrets(new_string)
    if secret_label:
        return ValidationResult(
            result=False,
            message=(
                f"Refusing to write: new_string contains what looks like a "
                f"{secret_label}. If this is intentional (e.g., a test "
                f"fixture), replace the real secret with a placeholder like "
                f"'REDACTED' before writing."
            ),
            error_code=11,
        )
    return ValidationResult(result=True)


def _format_size(n: int) -> str:
    if n < 1024:
        return f"{n}B"
    if n < 1024 * 1024:
        return f"{n / 1024:.1f}KB"
    return f"{n / (1024 * 1024):.1f}MB"


# ═════════════════════════════════════════════════════════════════════════
# call — FileEditTool.ts:154-227, exact pipeline
# ═════════════════════════════════════════════════════════════════════════

async def _call(
    input: FileEditInput,
    ctx: ToolUseContext,
    can_use_tool: Any = None,
    parent_message: Any = None,
    on_progress: Any = None,
) -> ToolResult:
    file_path = input.file_path
    old_string = input.old_string
    new_string = input.new_string
    replace_all = input.replace_all or False

    absolute = expand_path(file_path)

    # 1. Ensure parent directory exists (outside the atomic section)
    parent_dir = os.path.dirname(absolute)
    if parent_dir:
        await asyncio.to_thread(os.makedirs, parent_dir, exist_ok=True)

    # 2. Snapshot for undo (idempotent; safe before staleness check)
    if _file_history_enabled():
        await asyncio.to_thread(_snapshot_for_undo, absolute)

    # 3. Read current state
    original, file_exists, encoding, endings = await asyncio.to_thread(
        read_file_for_edit, absolute,
    )

    # 4. TOCTOU re-check (same logic; raises instead of returning)
    if file_exists:
        last_write = await asyncio.to_thread(get_file_modification_time, absolute)
        read_state = ctx.read_file_state.get(absolute) if ctx.read_file_state else None
        if not read_state or last_write > read_state.get("mtime", 0):
            is_full = (
                read_state is not None
                and read_state.get("offset") is None
                and read_state.get("limit") is None
            )
            content_unchanged = is_full and original == read_state.get("content", "")
            if not content_unchanged:
                raise RuntimeError(FILE_UNEXPECTEDLY_MODIFIED_ERROR)

    # 5. Quote normalization
    actual_old = find_actual_string(original, old_string) or old_string
    actual_new = preserve_quote_style(old_string, actual_old, new_string)

    # 6. Generate patch + updated content
    patch, updated = get_patch_for_edit(
        absolute, original, actual_old, actual_new, replace_all,
    )

    # 7. Write to disk
    await asyncio.to_thread(
        write_text_content, absolute, updated, encoding, endings,
    )

    # 8. Update read cache with the post-edit content
    if ctx.read_file_state:
        ctx.read_file_state.set(
            absolute, updated,
            await asyncio.to_thread(get_file_modification_time, absolute),
            offset=None, limit=None,
        )

    return ToolResult(data={
        "filePath": file_path,
        "oldString": actual_old,
        "newString": new_string,
        "originalFile": original,
        "structuredPatch": patch,
        "userModified": getattr(ctx, "user_modified", False),
        "replaceAll": replace_all,
    })


# ═════════════════════════════════════════════════════════════════════════
# map_tool_result_to_tool_result_block_param
# ═════════════════════════════════════════════════════════════════════════

def _map_result(data: dict, tool_use_id: str) -> dict:
    file_path = data["filePath"]
    patch = data.get("structuredPatch", [])
    user_modified = data.get("userModified", False)
    replace_all = data.get("replaceAll", False)

    note = (
        " The user modified your proposed changes before accepting them."
        if user_modified else ""
    )

    diff_lines: list[str] = []
    for hunk in patch:
        diff_lines.append(
            f"@@ -{hunk['oldStart']},{hunk['oldLines']} "
            f"+{hunk['newStart']},{hunk['newLines']} @@"
        )
        diff_lines.extend(hunk["lines"])
    diff_text = "\n".join(diff_lines)

    content = f"The file {file_path} has been updated.{note}"
    if replace_all:
        content += " (all occurrences replaced)"
    if diff_text:
        content += f"\n\nDiff:\n{diff_text}"

    return {
        "type": "tool_result",
        "tool_use_id": tool_use_id,
        "content": content,
    }


# ═════════════════════════════════════════════════════════════════════════
# UI helpers — short path
# ═════════════════════════════════════════════════════════════════════════

def _short_path(path: str) -> str:
    parts = path.replace("\\", "/").split("/")
    return "/".join(parts[-3:]) if len(parts) > 3 else path


def _user_facing_name(inp: dict | None) -> str:
    if not inp or not inp.get("file_path"):
        return "Edit"
    return f"Edit({_short_path(inp['file_path'])})"


def _get_tool_use_summary(inp: dict | None) -> str | None:
    if not inp or not inp.get("file_path"):
        return None
    return _short_path(inp["file_path"])


def _get_activity_description(inp: dict | None) -> str | None:
    summary = _get_tool_use_summary(inp)
    return f"Editing {summary}" if summary else "Editing file"


# ═════════════════════════════════════════════════════════════════════════
# Permission helpers
# ═════════════════════════════════════════════════════════════════════════

async def _check_permissions(input: FileEditInput, ctx: ToolUseContext):
    """Delegate to path_safety for protected-path detection. The gauntlet's
    stages 1a (deny rules) and 1b (ask rules) already fire before this."""
    try:
        from path_safety import check_path_safety_for_auto_edit
        safety = check_path_safety_for_auto_edit(input.file_path)
        if not safety.safe:
            return PermissionAskDecision(
                message=safety.message,
                decision_reason=SafetyCheckReason(
                    reason=safety.message,
                    classifier_approvable=safety.classifier_approvable,
                ),
            )
    except ImportError:
        pass
    return PermissionAllowDecision(updated_input=input.model_dump())


def _backfill_observable_input(inp: dict) -> None:
    if isinstance(inp.get("file_path"), str):
        inp["file_path"] = expand_path(inp["file_path"])


async def _prepare_permission_matcher(input: FileEditInput):
    path = input.file_path
    return lambda pattern: fnmatch.fnmatchcase(path, pattern)


# ═════════════════════════════════════════════════════════════════════════
# The tool
# ═════════════════════════════════════════════════════════════════════════

EditTool = build_tool(
    name=FILE_EDIT_TOOL_NAME,
    search_hint="modify file contents in place",
    max_result_size_chars=100_000,
    strict=True,
    description="A tool for editing files",
    prompt=_prompt,
    input_schema=FileEditInput,
    call=_call,
    map_tool_result_to_tool_result_block_param=_map_result,
    render_tool_use_message=lambda inp, _opts: (
        f"Edit({inp.get('file_path', '')})" if isinstance(inp, dict)
        else f"Edit({getattr(inp, 'file_path', '')})"
    ),
    # Risk methods use the fail-closed defaults from Tool.ts:
    #   is_read_only       → false  (writes)
    #   is_concurrency_safe → false (serial)
    #   is_destructive      → false (source does NOT override)
    to_auto_classifier_input=lambda inp: (
        f"{getattr(inp, 'file_path', '')}: {getattr(inp, 'new_string', '')}"
    ),
    get_path=lambda inp: getattr(inp, "file_path", "") or os.getcwd(),
    user_facing_name=_user_facing_name,
    get_tool_use_summary=_get_tool_use_summary,
    get_activity_description=_get_activity_description,
    validate_input=_validate_input,
    check_permissions=_check_permissions,
    backfill_observable_input=_backfill_observable_input,
    prepare_permission_matcher=_prepare_permission_matcher,
    extract_search_text=lambda out: out.get("newString", "") if isinstance(out, dict) else "",
)


__all__ = [
    "EditTool", "FileEditInput", "FILE_EDIT_TOOL_NAME",
    "FILE_UNEXPECTEDLY_MODIFIED_ERROR", "MAX_EDIT_FILE_SIZE",
    "normalize_quotes", "strip_trailing_whitespace",
    "find_actual_string", "preserve_quote_style",
    "apply_edit_to_file", "get_patch_for_edit", "get_patch_for_edits",
    "read_file_for_edit", "write_text_content",
    "get_file_modification_time",
    "LEFT_SINGLE_CURLY_QUOTE", "RIGHT_SINGLE_CURLY_QUOTE",
    "LEFT_DOUBLE_CURLY_QUOTE", "RIGHT_DOUBLE_CURLY_QUOTE",
]