"""
tools/read.py — the Read tool.

Port of src/tools/FileReadTool/FileReadTool.ts (387 lines) + prompt.ts +
limits.ts + utils/file.ts:addLineNumbers + utils/readFileInRange.ts.

Preserved exactly:
  • FILE_UNCHANGED_STUB, MAX_LINES_TO_READ=2000, DEFAULT_MAX_OUTPUT_TOKENS=25000,
    MAX_OUTPUT_SIZE=0.25MB, BLOCKED_DEVICE_PATHS (11 + /proc/*/fd/0-2)
  • addLineNumbers compact format "N\tline" (padded " N→line" behind a
    killswitch — source L106-121)
  • Read-dedup: same range + unchanged mtime → file_unchanged stub
    (source L223-247). Only fires for prior *Read* entries (offset set);
    Edit/Write entries store no offset, so they never dedup.
  • validateInput order: pages → deny rule → UNC → binary ext → device file
  • ENOENT → similar-file + cwd-relative suggestion (source L260-272)
  • maxResultSizeChars = Infinity (persisting a Read result creates a
    Read→file→Read loop — source L150-152)

Deferred (flagged):
  • Image resizing/compression (imageResizer.ts, ~300 lines). Raw images
    > 3.75 MB error instead of downsampling.
  • PDF page extraction, notebook cells (separate subsystems)
  • Filesystem deny-rule matching (permissions/filesystem.ts not ported)
"""
from __future__ import annotations

import asyncio
import fnmatch
import os
import stat as stat_mod
from typing import Any

from pydantic import BaseModel, Field

from messages import ToolUseBlock  # re-exported for callers that import from tools
from permission_types import PermissionAllowDecision
from tool import ToolResult, ToolUseContext, ValidationResult, build_tool


# ═════════════════════════════════════════════════════════════════════════
# Constants — exact from source
# ═════════════════════════════════════════════════════════════════════════

FILE_READ_TOOL_NAME = "Read"
DESCRIPTION = "Read a file from the local filesystem."

FILE_UNCHANGED_STUB = (
    "File unchanged since last read. The content from the earlier Read "
    "tool_result in this conversation is still current — refer to that "
    "instead of re-reading."
)

MAX_LINES_TO_READ = 2000
DEFAULT_MAX_OUTPUT_TOKENS = 25_000
MAX_OUTPUT_SIZE = int(0.25 * 1024 * 1024)          # 262,144 bytes
MAX_IMAGE_RAW_BYTES = 3_750_000                    # ~5MB base64 (API limit)

FILE_NOT_FOUND_CWD_NOTE = "Note: your current working directory is"

BLOCKED_DEVICE_PATHS = frozenset({
    "/dev/zero", "/dev/random", "/dev/urandom", "/dev/full",
    "/dev/stdin", "/dev/tty", "/dev/console",
    "/dev/stdout", "/dev/stderr",
    "/dev/fd/0", "/dev/fd/1", "/dev/fd/2",
})

IMAGE_EXTENSIONS = frozenset({"png", "jpg", "jpeg", "gif", "webp"})

# RECONSTRUCTED: constants/files.ts:hasBinaryExtension was not in the
# fetched slice. PDF, images, and SVG are deliberately excluded — Read
# renders those natively.
_BINARY_EXTENSIONS = frozenset({
    "exe", "dll", "so", "dylib", "bin", "o", "a", "lib", "obj", "class",
    "jar", "war", "pyc", "pyo", "pkl", "npy", "npz", "pt", "pth", "onnx",
    "h5", "parquet", "feather", "arrow", "db", "sqlite", "sqlite3",
    "zip", "tar", "gz", "bz2", "xz", "zst", "7z", "rar", "iso", "dmg",
    "mp3", "mp4", "avi", "mov", "mkv", "wav", "flac", "ogg", "webm",
    "woff", "woff2", "ttf", "otf", "eot", "ico",
    "doc", "docx", "xls", "xlsx", "ppt", "pptx",
})

# Source L63: narrow no-break space (U+202F) in macOS screenshot names
_THIN_SPACE = chr(8239)


# ═════════════════════════════════════════════════════════════════════════
# Path helpers
# ═════════════════════════════════════════════════════════════════════════

def expand_path(p: str) -> str:
    """Trim, expand ~, make absolute, normalize. Mirrors expandPath."""
    p = (p or "").strip()
    if p in ("~", "~/", "~\\") or p.startswith("~/") or p.startswith("~\\"):
        p = os.path.expanduser("~") + p[1:]
    if not os.path.isabs(p):
        p = os.path.join(os.getcwd(), p)
    return os.path.normpath(p)


def is_blocked_device_path(file_path: str) -> bool:
    if file_path in BLOCKED_DEVICE_PATHS:
        return True
    if file_path.startswith("/proc/") and file_path[-5:] in (
        "/fd/0", "/fd/1", "/fd/2",
    ):
        return True
    return False


def has_binary_extension(file_path: str) -> bool:
    ext = os.path.splitext(file_path)[1].lstrip(".").lower()
    return ext in _BINARY_EXTENSIONS


def get_alternate_screenshot_path(file_path: str) -> str | None:
    """macOS screenshot AM/PM spacing may be ' ' or U+202F (source L73-80)."""
    name = os.path.basename(file_path)
    import re
    m = re.match(r"^(.+)([ \u202F])(AM|PM)(\.png)$", name)
    if not m:
        return None
    alt = _THIN_SPACE if m.group(2) == " " else " "
    old = f"{m.group(2)}{m.group(3)}{m.group(4)}"
    new = f"{alt}{m.group(3)}{m.group(4)}"
    return file_path.replace(old, new)


def find_similar_file(file_path: str) -> str | None:
    """Same basename, different extension, same dir (source L69-77)."""
    directory = os.path.dirname(file_path)
    base = os.path.splitext(os.path.basename(file_path))[0]
    try:
        for entry in os.listdir(directory):
            if os.path.splitext(entry)[0] == base and os.path.join(
                directory, entry
            ) != file_path:
                return entry
    except OSError:
        return None
    return None


async def suggest_path_under_cwd(requested: str) -> str | None:
    """Detect the 'dropped repo folder' pattern (source L89-105)."""
    cwd = os.getcwd()
    parent = os.path.dirname(cwd)
    resolved = requested
    try:
        resolved = os.path.join(
            await asyncio.to_thread(os.path.realpath, os.path.dirname(requested)),
            os.path.basename(requested),
        )
    except OSError:
        pass
    prefix = os.sep if parent == os.sep else parent + os.sep
    if not resolved.startswith(prefix):
        return None
    if resolved.startswith(cwd + os.sep) or resolved == cwd:
        return None
    rel = os.path.relpath(resolved, parent)
    candidate = os.path.join(cwd, rel)
    try:
        await asyncio.to_thread(os.stat, candidate)
        return candidate
    except OSError:
        return None


# ═════════════════════════════════════════════════════════════════════════
# Line numbering — exact from utils/file.ts L106-121
# ═════════════════════════════════════════════════════════════════════════

def is_compact_line_prefix_enabled() -> bool:
    """3P default: compact enabled (killswitch off). Env override added
    so the padded format is reachable without GrowthBook."""
    return not (os.environ.get("CLAUDE_CODE_COMPACT_LINE_PREFIX", "").lower()
                in ("0", "false", "no", "off")
                or os.environ.get("CLAUDE_CODE_DISABLE_COMPACT_LINE_PREFIX", "")
                .lower() in ("1", "true", "yes", "on"))


def add_line_numbers(content: str, start_line: int) -> str:
    """cat -n style. Compact 'N\\tline' by default; ' N→line' when disabled."""
    if not content:
        return ""
    lines = content.split("\n")
    if is_compact_line_prefix_enabled():
        return "\n".join(f"{i + start_line}\t{line}" for i, line in enumerate(lines))
    out = []
    for i, line in enumerate(lines):
        n = str(i + start_line)
        out.append(f"{n}→{line}" if len(n) >= 6 else f"{n.rjust(6)}→{line}")
    return "\n".join(out)


# ═════════════════════════════════════════════════════════════════════════
# File reading — port of utils/readFileInRange.ts
# ═════════════════════════════════════════════════════════════════════════

class FileTooLargeError(Exception):
    def __init__(self, size: int, max_size: int):
        from sizes import format_file_size  # local helper below
        super().__init__(
            f"File content ({format_file_size(size)}) exceeds maximum allowed "
            f"size ({format_file_size(max_size)}). Use offset and limit "
            f"parameters to read specific portions of the file, or search for "
            f"specific content instead of reading the whole file."
        )
        self.size = size
        self.max_size = max_size


class MaxFileReadTokenExceededError(Exception):
    def __init__(self, token_count: int, max_tokens: int):
        super().__init__(
            f"File content ({token_count} tokens) exceeds maximum allowed "
            f"tokens ({max_tokens}). Use offset and limit parameters to read "
            f"specific portions of the file, or search for specific content "
            f"instead of reading the whole file."
        )
        self.token_count = token_count
        self.max_tokens = max_tokens


def format_file_size(n: int) -> str:
    if n < 1024:
        return f"{n}B"
    if n < 1024 * 1024:
        return f"{n / 1024:.1f}KB"
    return f"{n / (1024 * 1024):.1f}MB"


def _decode_text(raw: bytes) -> str:
    """UTF-8 first, then latin-1 fallback (never raises)."""
    for enc in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


async def read_file_in_range(
    path: str,
    offset: int = 1,
    limit: int | None = None,
    max_bytes: int | None = None,
) -> dict:
    """Return lines [offset, offset+limit). 1-indexed offset.

    Two paths from source: fast (read whole file, split) and streaming
    (count lines without accumulating). We use the fast path with a
    size guard — for a personal agent the 10MB threshold is academic,
    and max_bytes aborts before allocating the full file.
    """
    st = await asyncio.to_thread(os.stat, path)
    if stat_mod.S_ISDIR(st.st_mode):
        raise IsADirectoryError(path)
    if max_bytes is not None and st.st_size > max_bytes:
        raise FileTooLargeError(st.st_size, max_bytes)

    raw = await asyncio.to_thread(_read_bytes, path)
    text = _decode_text(raw)
    if text.startswith("\ufeff"):
        text = text[1:]
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    lines = text.split("\n")
    total_lines = len(lines)
    if total_lines and lines[-1] == "":
        total_lines -= 1
        lines = lines[:total_lines]      # trim trailing "" so slicing is correct

    start_index = max(0, offset - 1)
    effective_limit = limit if limit is not None else MAX_LINES_TO_READ
    window = lines[start_index:start_index + effective_limit]
    content = "\n".join(window)

    return {
        "content": content,
        "line_count": len(window),
        "total_lines": total_lines,
        "mtime_ms": int(st.st_mtime * 1000) // 1,   # floor, matches source
        "total_bytes": st.st_size,
    }


def _read_bytes(path: str) -> bytes:
    with open(path, "rb") as f:
        return f.read()


def _estimate_tokens(text: str) -> int:
    return (len(text) + 3) // 4


# ═════════════════════════════════════════════════════════════════════════
# Schemas
# ═════════════════════════════════════════════════════════════════════════

class ReadInput(BaseModel):
    file_path: str = Field(description="The absolute path to the file to read")
    offset: int | None = Field(
        default=None, ge=0,
        description="The line number to start reading from. Only provide if "
                    "the file is too large to read at once",
    )
    limit: int | None = Field(
        default=None, gt=0,
        description="The number of lines to read. Only provide if the file is "
                    "too large to read at once.",
    )
    pages: str | None = Field(
        default=None,
        description='Page range for PDF files (e.g., "1-5"). Maximum 20 pages '
                    "per request.",
    )


# ═════════════════════════════════════════════════════════════════════════
# Prompt — verbatim template from prompt.ts L17-39
# ═════════════════════════════════════════════════════════════════════════

_LINE_FORMAT = "- Results are returned using cat -n format, with line numbers starting at 1"
_OFFSET_DEFAULT = ("- You can optionally specify a line offset and limit "
                   "(especially handy for long files), but it's recommended to "
                   "read the whole file by not providing these parameters")


def _prompt() -> str:
    return (
        "Reads a file from the local filesystem. You can access any file "
        "directly by using this tool. Assume this tool is able to read all "
        "files on the machine. If the User provides a path to a file assume "
        "that path is valid. It is okay to read a file that does not exist; "
        "an error will be returned.\n\nUsage:\n"
        "- The file_path parameter must be an absolute path, not a relative path\n"
        f"- By default, it reads up to {MAX_LINES_TO_READ} lines starting from "
        "the beginning of the file\n"
        f"{_OFFSET_DEFAULT}\n{_LINE_FORMAT}\n"
        "- This tool allows Claude Code to read images (eg PNG, JPG, etc). When "
        "reading an image file the contents are presented visually as Claude "
        "Code is a multimodal LLM.\n"
        "- This tool can only read files, not directories. To read a directory, "
        "use an ls command via the Bash tool.\n"
        "- You will regularly be asked to read screenshots. If the user provides "
        "a path to a screenshot, ALWAYS use this tool to view the file at the "
        "path. This tool will work with all temporary file paths.\n"
        "- If you read a file that exists but has empty contents you will "
        "receive a system reminder warning in place of file contents."
    )


# ═════════════════════════════════════════════════════════════════════════
# validate_input — exact order from source L180-209
# ═════════════════════════════════════════════════════════════════════════

def _validate_input(input: ReadInput, ctx: ToolUseContext) -> ValidationResult:
    # 1. pages format (string parsing only, no I/O)
    if input.pages is not None:
        ok, msg = _validate_pages(input.pages)
        if not ok:
            return ValidationResult(result=False, message=msg, error_code=7)

    full_path = expand_path(input.file_path)

    # 2. Filesystem deny rules — not ported (permissions/filesystem.ts).
    #    Stage 1a of the gauntlet still runs before this, so deny rules
    #    configured by name still apply.

    # 3. UNC path → defer filesystem work until after permission (L196-197)
    if full_path.startswith("\\\\") or full_path.startswith("//"):
        return ValidationResult(result=True)

    ext = os.path.splitext(full_path)[1].lower()

    # 4. Binary extension (PDF/images/SVG render natively) → errorCode 4
    if (has_binary_extension(full_path)
            and ext != ".pdf"
            and ext.lstrip(".") not in IMAGE_EXTENSIONS):
        return ValidationResult(
            result=False,
            message=(
                "This tool cannot read binary files. The file appears to be a "
                f"binary {ext} file. Please use appropriate tools for binary "
                "file analysis."
            ),
            error_code=4,
        )

    # 5. Device files that would block or produce infinite output → errorCode 9
    if is_blocked_device_path(full_path):
        return ValidationResult(
            result=False,
            message=(
                f"Cannot read '{input.file_path}': this device file would "
                "block or produce infinite output."
            ),
            error_code=9,
        )

    return ValidationResult(result=True)


def _validate_pages(pages: str) -> tuple[bool, str]:
    import re
    p = (pages or "").strip()
    if not p:
        return False, f'Invalid pages parameter: "{pages}". Use formats like "1-5", "3", or "10-20".'
    if re.fullmatch(r"\d+", p):
        return True, ""
    m = re.fullmatch(r"(\d+)\s*-\s*(\d+)", p)
    if not m:
        return False, (f'Invalid pages parameter: "{pages}". Use formats like '
                       f'"1-5", "3", or "10-20". Pages are 1-indexed.')
    first, last = int(m.group(1)), int(m.group(2))
    if first < 1 or last < first:
        return False, f'Invalid pages parameter: "{pages}". Pages are 1-indexed.'
    if last - first + 1 > 20:
        return False, (f'Page range "{pages}" exceeds maximum of 20 pages per '
                       f"request. Please use a smaller range.")
    return True, ""


# ═════════════════════════════════════════════════════════════════════════
# call
# ═════════════════════════════════════════════════════════════════════════

async def _call(
    input: ReadInput,
    ctx: ToolUseContext,
    can_use_tool: Any = None,
    parent_message: Any = None,
    on_progress: Any = None,
) -> ToolResult:
    offset = input.offset if input.offset is not None else 1
    limit = input.limit
    full_path = expand_path(input.file_path)
    ext = os.path.splitext(full_path)[1].lower().lstrip(".")

    # ── Dedup (source L223-247) ──
    cache = getattr(ctx, "read_file_state", None)
    existing = cache.get(full_path) if cache is not None else None
    if (
        existing
        and not existing.get("is_partial_view")
        and existing.get("offset") is not None
        and existing.get("offset") == offset
        and existing.get("limit") == limit
    ):
        try:
            st = await asyncio.to_thread(os.stat, full_path)
            if int(st.st_mtime * 1000) // 1 == existing.get("mtime"):
                return ToolResult(data={
                    "type": "file_unchanged",
                    "file": {"filePath": input.file_path},
                })
        except OSError:
            pass  # fall through to full read

    # ── Image ──
    if ext in IMAGE_EXTENSIONS:
        return await _read_image(input, full_path, ctx, cache)

    # ── PDF / notebook: not ported ──
    if ext == "pdf":
        raise ValueError(
            "PDF reading is not available in this build. Use Bash with a PDF "
            "utility, or read the text source instead."
        )
    if ext == "ipynb":
        raise ValueError(
            "Notebook reading is not available in this build. Use Bash with jq "
            f'to inspect cells, e.g. `cat "{input.file_path}" | jq \'.cells[:20]\'`.'
        )

    # ── Text ──
    try:
        result = await read_file_in_range(full_path, offset, limit, MAX_OUTPUT_SIZE)
    except FileNotFoundError:
        alt = get_alternate_screenshot_path(full_path)
        if alt:
            try:
                result = await read_file_in_range(alt, offset, limit, MAX_OUTPUT_SIZE)
            except FileNotFoundError:
                result = None
        else:
            result = None
        if result is None:
            raise FileNotFoundError(_file_not_found_message(input.file_path, full_path))
    except IsADirectoryError:
        raise ValueError(
            f"'{input.file_path}' is a directory. Use an ls command via the "
            "Bash tool to list its contents."
        )

    content = result["content"]

    # Token guard (source L314-321) — local estimate only, no API round-trip
    tokens = _estimate_tokens(content)
    max_tokens = int(
        os.environ.get("CLAUDE_CODE_FILE_READ_MAX_OUTPUT_TOKENS", "")
        or DEFAULT_MAX_OUTPUT_TOKENS
    )
    if tokens > max_tokens:
        raise MaxFileReadTokenExceededError(tokens, max_tokens)

    # Record read state — this is the read-gate for Edit, and the dedup key
    if cache is not None:
        cache.set(
            full_path, content, result["mtime_ms"],
            offset=offset, limit=limit,
            is_partial_view=(limit is not None or (input.offset is not None and input.offset != 1)),
        )


    if not content:
        # Empty (or offset past EOF). Source emits a meta warning message;
        # we inline it so the model sees it as the tool result.
        warning = (
            "<system-reminder>Warning: the file exists but its contents are "
            "empty.</system-reminder>"
            if result["total_lines"] == 0
            else f"<system-reminder>Offset {offset} is past the end of the file "
                 f"({result['total_lines']} lines).</system-reminder>"
        )
        return ToolResult(data={
            "type": "text",
            "file": {
                "filePath": input.file_path,
                "content": warning,
                "numLines": 0,
                "startLine": offset,
                "totalLines": result["total_lines"],
            },
        })

    return ToolResult(data={
        "type": "text",
        "file": {
            "filePath": input.file_path,
            "content": add_line_numbers(content, offset),
            "numLines": result["line_count"],
            "startLine": offset,
            "totalLines": result["total_lines"],
        },
    })


def _file_not_found_message(display_path: str, full_path: str) -> str:
    parts = [f"File does not exist. {FILE_NOT_FOUND_CWD_NOTE} {os.getcwd()}."]
    similar = find_similar_file(full_path)
    if similar:
        parts.append(f"Did you mean {similar}?")
    else:
        suggestion = _suggest_under_cwd_sync(full_path)
        if suggestion:
            parts.append(f"Did you mean {suggestion}?")
    return " ".join(parts)


def _suggest_under_cwd_sync(requested: str) -> str | None:
    cwd = os.getcwd()
    parent = os.path.dirname(cwd)
    prefix = os.sep if parent == os.sep else parent + os.sep
    if not requested.startswith(prefix):
        return None
    if requested.startswith(cwd + os.sep) or requested == cwd:
        return None
    candidate = os.path.join(cwd, os.path.relpath(requested, parent))
    return candidate if os.path.exists(candidate) else None


def _detect_image_media_type(data: bytes) -> str | None:
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if data.startswith(b"GIF87a") or data.startswith(b"GIF89a"):
        return "image/gif"
    if len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    return None


async def _read_image(input: ReadInput, full_path: str, ctx: Any, cache: Any) -> ToolResult:
    """Base64 image read. ADAPTATION: no downscaling (imageResizer.ts not ported),
    so files above the API's raw-byte limit error instead of being resized."""
    raw = await asyncio.to_thread(_read_bytes, full_path)
    media = _detect_image_media_type(raw)
    if media is None:
        raise ValueError(f"'{input.file_path}' is not a supported image format.")
    if len(raw) > MAX_IMAGE_RAW_BYTES:
        raise ValueError(
            f"Image is {format_file_size(len(raw))}, above the "
            f"{format_file_size(MAX_IMAGE_RAW_BYTES)} raw limit. Resize it "
            "first, or read a smaller image."
        )

    if cache is not None:
        # Images are not stored for dedup (source: images aren't cached in
        # readFileState). Mark the path as read for the Edit read-gate only.
        cache.set(full_path, "", 0, offset=None, limit=None)

    return ToolResult(data={
        "type": "image",
        "file": {
            "filePath": input.file_path,
            "base64": _b64(raw),
            "mediaType": media,
            "originalSize": len(raw),
        },
    })


def _b64(raw: bytes) -> str:
    import base64
    return base64.b64encode(raw).decode("ascii")


# ═════════════════════════════════════════════════════════════════════════
# map_tool_result_to_tool_result_block_param — source L273-283
# ═════════════════════════════════════════════════════════════════════════

def _map_result(data: dict, tool_use_id: str) -> dict:
    dtype = data.get("type")
    if dtype == "image":
        f = data["file"]
        return {
            "type": "tool_result",
            "tool_use_id": tool_use_id,
            "content": [{
                "type": "image_url",
                "image_url": {"url": f"data:{f['mediaType']};base64,{f['base64']}"},
            }],
        }
    if dtype == "file_unchanged":
        return {
            "type": "tool_result",
            "tool_use_id": tool_use_id,
            "content": FILE_UNCHANGED_STUB,
        }
    f = data["file"]
    return {
        "type": "tool_result",
        "tool_use_id": tool_use_id,
        "content": f["content"],
    }


# ═════════════════════════════════════════════════════════════════════════
# Permissions / risk
# ═════════════════════════════════════════════════════════════════════════

async def _check_permissions(input: ReadInput, ctx: ToolUseContext):
    """Reads auto-allow (source delegates to checkReadPermissionForTool).

    Deny rules in stage 1a run BEFORE this, and ask rules in 1b still
    apply — so an explicit rule still wins.
    """
    return PermissionAllowDecision(updated_input=input.model_dump())


def _backfill_observable_input(input: dict) -> None:
    """hooks.mdx documents file_path as absolute (source L166-169)."""
    if isinstance(input.get("file_path"), str):
        input["file_path"] = expand_path(input["file_path"])


async def _prepare_permission_matcher(input: ReadInput):
    path = input.file_path
    return lambda pattern: fnmatch.fnmatchcase(path, pattern)


# ═════════════════════════════════════════════════════════════════════════
# The tool
# ═════════════════════════════════════════════════════════════════════════

ReadTool = build_tool(
    name=FILE_READ_TOOL_NAME,
    description=DESCRIPTION,
    prompt=_prompt,
    input_schema=ReadInput,
    call=_call,
    max_result_size_chars=float("inf"),      # source L151-152
    strict=True,
    map_tool_result_to_tool_result_block_param=_map_result,
    render_tool_use_message=lambda inp, _opts: (
        f"Read({inp.get('file_path', '')})" if isinstance(inp, dict)
        else f"Read({getattr(inp, 'file_path', '')})"
    ),
    is_read_only=lambda _inp: True,
    is_concurrency_safe=lambda _inp: True,
    is_search_or_read_command=lambda _inp: {
        "isSearch": False, "isRead": True, "isList": False,
    },
    to_auto_classifier_input=lambda inp: getattr(inp, "file_path", ""),
    get_path=lambda inp: getattr(inp, "file_path", None) or os.getcwd(),
    validate_input=_validate_input,
    check_permissions=_check_permissions,
    backfill_observable_input=_backfill_observable_input,
    prepare_permission_matcher=_prepare_permission_matcher,
    extract_search_text=lambda _out: "",     # source L179
    get_activity_description=lambda inp: (
        f"Reading {getattr(inp, 'file_path', '')}" if inp else "Reading file"
    ),
    search_hint="read file contents view source",
    aliases=["Read"],
)

__all__ = [
    "ReadTool", "ReadInput", "FILE_READ_TOOL_NAME", "FILE_UNCHANGED_STUB",
    "MAX_LINES_TO_READ", "DEFAULT_MAX_OUTPUT_TOKENS", "MAX_OUTPUT_SIZE",
    "BLOCKED_DEVICE_PATHS", "IMAGE_EXTENSIONS",
    "add_line_numbers", "expand_path", "read_file_in_range",
    "is_blocked_device_path", "find_similar_file",
    "FileTooLargeError", "MaxFileReadTokenExceededError",
    "ToolUseBlock",
]