"""
memdir.py — file-based memory system with meta-level selection.

Port of src/memdir/ (8 files, ~800 lines) consolidated into one module.

WHAT THIS DOES — four meta-level operations:

  1. SELECTION (meta): an LLM looks at the user's message and picks at most
     MAX_MEMORIES_PER_TURN=5 memories whose descriptions are relevant. This
     is context-by-relevance, not context-by-existence.

  2. BUDGETING (meta): the session accumulates at most MAX_SESSION_BYTES=60KB
     of injected memory. Once the cap is hit, no further injection happens —
     even if more memories would be relevant. The cap protects against drift.

  3. DEDUPLICATION (meta): three filters before selection — exclude memories
     already surfaced this session, exclude memories the model already read
     via FileRead, and suppress API-doc memories when that API isn't in the
     recently-used tool set.

  4. DEFERRAL (meta): MEMORY.md itself is excluded from selection because it
     is already in the system prompt. Injecting it again is pure waste.

VERIFIED BUDGETS (from source):
  MAX_ENTRYPOINT_LINES   = 200
  MAX_ENTRYPOINT_BYTES   = 25 KB
  MAX_MEMORY_FILES       = 200
  MAX_SESSION_BYTES      = 60 KB
  MAX_MEMORIES_PER_TURN  = 5
  FRONTMATTER_SCAN_LINES = 30

FOUR MEMORY TYPES (from source):
  user      — stable user preferences
  feedback  — corrections to agent behavior
  project   — durable project facts not obvious from the repo
  reference — pointers to external resources
"""
from __future__ import annotations

import json
import hashlib
import os
import re
import tempfile
import time
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import PureWindowsPath
from typing import Literal


# ═════════════════════════════════════════════════════════════════════════
# Constants — verified from source
# ═════════════════════════════════════════════════════════════════════════

MEMORY_DIR_NAME = "memory"
MEMORY_INDEX_FILE = "MEMORY.md"
MAX_ENTRYPOINT_LINES = 200
MAX_ENTRYPOINT_BYTES = 25 * 1024
MAX_MEMORY_FILES = 200
MAX_SESSION_BYTES = 60 * 1024
MAX_MEMORIES_PER_TURN = 5
FRONTMATTER_SCAN_LINES = 30
MEMORY_STALE_AFTER_SECONDS = 24 * 60 * 60

MemoryType = Literal["user", "feedback", "project", "reference"]
VALID_TYPES: frozenset[str] = frozenset({
    "user", "feedback", "project", "reference",
})


# ═════════════════════════════════════════════════════════════════════════
# Data model
# ═════════════════════════════════════════════════════════════════════════

@dataclass
class MemoryFile:
    path: str
    name: str
    description: str
    type: MemoryType
    mtime: float
    size_bytes: int = 0

    @property
    def filename(self) -> str:
        return os.path.basename(self.path)

    def age_label(self) -> str:
        s = max(0, time.time() - self.mtime)
        if s < 3600:
            return f"{int(s / 60)}m ago"
        if s < 86400:
            return f"{int(s / 3600)}h ago"
        return f"{int(s / 86400)}d ago"


@dataclass
class SelectionContext:
    """State carried across turns for the meta-level decisions."""
    user_message: str
    recently_used_tools: list[str] = field(default_factory=list)
    already_surfaced: set[str] = field(default_factory=set)   # filenames
    already_read: set[str] = field(default_factory=set)       # full paths
    session_bytes_used: int = 0


# ═════════════════════════════════════════════════════════════════════════
# Directory resolution
# ═════════════════════════════════════════════════════════════════════════

def validate_memory_path(path: str) -> str:
    """Require an absolute, non-root local path suitable for memory files."""
    if not isinstance(path, str) or not path:
        raise ValueError("Memory path must be a non-empty absolute path")
    if "\0" in path:
        raise ValueError("Memory path must not contain a null byte")
    if path.startswith(("\\\\", "//")):
        raise ValueError("UNC memory paths are not allowed")

    expanded = os.path.expanduser(path)
    windows_path = PureWindowsPath(expanded)
    if windows_path.drive and windows_path.root and len(windows_path.parts) <= 1:
        raise ValueError("A drive root cannot be used as the memory directory")
    if not os.path.isabs(expanded) and not windows_path.is_absolute():
        raise ValueError("Memory path must be absolute")
    if windows_path.is_absolute() and os.name != "nt":
        raise ValueError("Windows memory paths are not valid on this platform")

    normalized = os.path.normpath(expanded)
    if os.path.isabs(normalized):
        if normalized == os.path.abspath(os.sep) or len(normalized.strip(os.sep)) < 2:
            raise ValueError("Memory path is too close to the filesystem root")
    return normalized


@lru_cache(maxsize=32)
def _default_project_memory_dir(project_root: str) -> str:
    root = os.path.realpath(os.path.expanduser(project_root))
    candidate = os.path.realpath(os.path.join(root, ".claude", MEMORY_DIR_NAME))
    if os.path.commonpath((root, candidate)) != root:
        raise ValueError("Project memory directory resolves outside the project")
    return candidate


def get_memory_dir(
    project_root: str | None = None,
    *,
    user_setting: str | None = None,
    project_setting: str | None = None,
) -> str:
    """Resolve memory dir: env override → user setting → project default.

    Project settings cannot redirect memory writes. The default project path
    is memoized by its canonical project root and checked against symlink escape.
    """
    if project_setting is not None:
        raise ValueError("Project settings cannot override the memory directory")

    override = os.environ.get("CLAUDE_COWORK_MEMORY_PATH_OVERRIDE")
    legacy_override = os.environ.get("CLAUDE_MEMORY_DIR")
    remote_mode = os.environ.get("CLAUDE_CODE_REMOTE", "").lower() in (
        "1", "true", "yes", "on",
    )
    remote_dir = os.environ.get("REMOTE_MEMORY_DIR") if remote_mode else None
    if override:
        path = validate_memory_path(override)
    elif remote_dir:
        path = validate_memory_path(remote_dir)
    elif legacy_override:
        path = validate_memory_path(legacy_override)
    elif user_setting:
        path = validate_memory_path(user_setting)
    elif project_root:
        canonical_root = os.path.realpath(os.path.expanduser(project_root))
        path = validate_memory_path(_default_project_memory_dir(canonical_root))
    else:
        path = validate_memory_path(
            os.path.join(os.path.expanduser("~"), ".claude", MEMORY_DIR_NAME)
        )
    os.makedirs(path, exist_ok=True)
    return path


def is_llm_selection_enabled() -> bool:
    return os.environ.get("CLAUDE_MEMORY_LLM_SELECT", "1").strip().lower() in (
        "1", "true", "yes", "on",
    )


# ═════════════════════════════════════════════════════════════════════════
# Frontmatter parsing — first 30 lines, key: value only
# ═════════════════════════════════════════════════════════════════════════

_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)
_KV_RE = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)\s*:\s*(.+?)\s*$")


def parse_frontmatter(content: str) -> dict[str, str] | None:
    """Parse YAML-ish frontmatter from the first N lines."""
    head = "\n".join(content.split("\n")[:FRONTMATTER_SCAN_LINES])
    m = _FRONTMATTER_RE.match(head)
    if not m:
        return None
    result: dict[str, str] = {}
    for line in m.group(1).split("\n"):
        km = _KV_RE.match(line)
        if not km:
            continue
        key, value = km.group(1), km.group(2).strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]
        result[key] = value
    return result or None


# ═════════════════════════════════════════════════════════════════════════
# Scan — read frontmatter only, sort by mtime, cap at 200
# ═════════════════════════════════════════════════════════════════════════

def scan_memories(memory_dir: str) -> list[MemoryFile]:
    """Scan .md files (excludes MEMORY.md), read frontmatter only.
    Sorts by mtime desc, caps at MAX_MEMORY_FILES newest."""
    if not os.path.isdir(memory_dir):
        return []
    candidates: list[MemoryFile] = []
    for entry in os.scandir(memory_dir):
        if not entry.is_file() or not entry.name.endswith(".md"):
            continue
        if entry.name == MEMORY_INDEX_FILE:
            continue
        try:
            with open(entry.path, encoding="utf-8") as f:
                head = "".join(f.readline() for _ in range(FRONTMATTER_SCAN_LINES))
            fm = parse_frontmatter(head)
            if not fm:
                continue
            name = fm.get("name", "")
            description = fm.get("description", "")
            mtype = fm.get("type", "").strip().lower()
            if not name or not description or mtype not in VALID_TYPES:
                continue
            st = entry.stat()
            candidates.append(MemoryFile(
                path=entry.path, name=name, description=description,
                type=mtype, mtime=st.st_mtime, size_bytes=st.st_size,
            ))
        except OSError:
            continue
    candidates.sort(key=lambda m: m.mtime, reverse=True)
    return candidates[:MAX_MEMORY_FILES]


# ═════════════════════════════════════════════════════════════════════════
# Manifest for LLM selection
# ═════════════════════════════════════════════════════════════════════════

def build_manifest(memories: list[MemoryFile]) -> str:
    """Compact one-line-per-memory format for the selector."""
    return "\n".join(
        f"[{m.type}] {m.filename} ({m.age_label()}): {m.description}"
        for m in memories
    )


# ═════════════════════════════════════════════════════════════════════════
# Selection — LLM primary, heuristic fallback
# ═════════════════════════════════════════════════════════════════════════

_SELECTOR_PROMPT = (
    "USER MESSAGE:\n{message}\n\n"
    "AVAILABLE MEMORIES (format: [type] filename (age): description):\n"
    "{manifest}\n\n"
    "Select up to {n} memories most relevant to the user's message. "
    "Return ONLY a JSON object:\n"
    '{{"selected": ["filename1.md", "filename2.md"]}}\n'
    'If none are relevant, return {{"selected": []}}.'
)


async def select_memories_llm(
    memories: list[MemoryFile],
    ctx: SelectionContext,
) -> list[str] | None:
    """Ask the API to pick up to MAX_MEMORIES_PER_TURN filenames.
    Returns None on failure (caller falls back to heuristic)."""
    if not memories:
        return []
    try:
        from api import complete
        manifest = build_manifest(memories)
        prompt = _SELECTOR_PROMPT.format(
            message=ctx.user_message[:2_000],
            manifest=manifest[:8_000],
            n=MAX_MEMORIES_PER_TURN,
        )
        text = await complete(
            [{"role": "user", "content": prompt}],
            model=os.environ.get("CLAUDE_MEMORY_SELECT_MODEL") or None,
            max_tokens=200,
            response_format={"type": "json_object"},
        )
        data = json.loads(text or "{}")
        selected = data.get("selected", [])
        return [s for s in selected if isinstance(s, str)]
    except Exception:
        return None


_STOPWORDS = frozenset({
    "the", "a", "an", "and", "or", "but", "is", "are", "was", "were",
    "be", "to", "of", "in", "on", "at", "by", "for", "with", "as",
    "this", "that", "it", "i", "you", "we", "they", "my", "your",
    "do", "does", "did", "have", "has", "had", "can", "could",
})


def _words(text: str) -> set[str]:
    return {
        w for w in re.findall(r"[a-z0-9]+", text.lower())
        if w not in _STOPWORDS and len(w) > 2
    }


def select_memories_heuristic(
    memories: list[MemoryFile],
    ctx: SelectionContext,
) -> list[str]:
    """Keyword-overlap fallback. Sorted by (overlap desc, mtime desc)."""
    user_words = _words(ctx.user_message)
    if not user_words:
        return [m.filename for m in memories[:MAX_MEMORIES_PER_TURN]]
    scored: list[tuple[int, float, MemoryFile]] = []
    for m in memories:
        overlap = len(user_words & _words(m.name + " " + m.description))
        if overlap > 0:
            scored.append((overlap, m.mtime, m))
    scored.sort(key=lambda t: (-t[0], -t[1]))
    return [m.filename for _, _, m in scored[:MAX_MEMORIES_PER_TURN]]


# ═════════════════════════════════════════════════════════════════════════
# Deduplication filters — the meta-level skips
# ═════════════════════════════════════════════════════════════════════════

_API_KEYWORDS: dict[str, tuple[str, ...]] = {
    "openai": ("openai", "gpt"),
    "anthropic": ("anthropic", "claude"),
    "gemini": ("gemini", "google"),
    "groq": ("groq",),
}


def _filter_by_recent_tools(
    memories: list[MemoryFile],
    recently_used_tools: list[str],
) -> list[MemoryFile]:
    """Suppress API-doc memories when that API isn't recently used.
    Conservative: only suppress when we're confident about the API."""
    if not recently_used_tools:
        return memories
    used_lower = {t.lower() for t in recently_used_tools}
    out: list[MemoryFile] = []
    for m in memories:
        text = (m.name + " " + m.description).lower()
        suppressed = False
        for api, keywords in _API_KEYWORDS.items():
            if any(kw in text for kw in keywords):
                if api not in used_lower:
                    suppressed = True
                    break
        if not suppressed:
            out.append(m)
    return out


def _enforce_session_budget(
    selected: list[MemoryFile],
    ctx: SelectionContext,
) -> list[MemoryFile]:
    """Cap cumulative injection at MAX_SESSION_BYTES across the session."""
    out: list[MemoryFile] = []
    used = ctx.session_bytes_used
    for m in selected:
        if used + m.size_bytes > MAX_SESSION_BYTES:
            break
        out.append(m)
        used += m.size_bytes
    return out


# ═════════════════════════════════════════════════════════════════════════
# Orchestrator — the four meta-level operations in order
# ═════════════════════════════════════════════════════════════════════════

async def find_relevant_memories(
    memories: list[MemoryFile],
    ctx: SelectionContext,
    use_llm: bool | None = None,
) -> list[MemoryFile]:
    """Full pipeline: dedup → select → cap → budget.

    Returns MemoryFile objects, ready for content injection.
    """
    # Filter 1: already surfaced this session
    pool = [m for m in memories if m.filename not in ctx.already_surfaced]
    # Filter 2: already read via FileRead
    pool = [m for m in pool if m.path not in ctx.already_read]
    # Filter 3: API-doc suppression when API not in use
    pool = _filter_by_recent_tools(pool, ctx.recently_used_tools)
    if not pool:
        return []

    # Selection
    want_llm = use_llm if use_llm is not None else is_llm_selection_enabled()
    selected_names: list[str] | None = None
    if want_llm:
        selected_names = await select_memories_llm(pool, ctx)
    if selected_names is None:
        selected_names = select_memories_heuristic(pool, ctx)

    # Resolve to MemoryFile objects
    by_name = {m.filename: m for m in pool}
    selected = [by_name[n] for n in selected_names if n in by_name]

    # Cap to per-turn limit
    selected = selected[:MAX_MEMORIES_PER_TURN]

    # Session budget
    return _enforce_session_budget(selected, ctx)


# ═════════════════════════════════════════════════════════════════════════
# Content loading — strips frontmatter
# ═════════════════════════════════════════════════════════════════════════

def load_memory_content(memory: MemoryFile) -> str:
    try:
        with open(memory.path, encoding="utf-8") as f:
            content = f.read()
    except OSError:
        return ""
    m = _FRONTMATTER_RE.match(content)
    return content[m.end():] if m else content


def memory_freshness_note(
    memory: MemoryFile,
    now: float | None = None,
) -> str | None:
    """Warn the model when a selected memory has not been updated in a day."""
    age_seconds = (time.time() if now is None else now) - memory.mtime
    if age_seconds <= MEMORY_STALE_AFTER_SECONDS:
        return None
    age_days = max(1, int(age_seconds // (24 * 60 * 60)))
    return (
        f"Freshness warning: this memory was last modified {age_days} "
        "days ago and may be stale. Verify time-sensitive details."
    )


# ═════════════════════════════════════════════════════════════════════════
# Writing — frontmatter + body
# ═════════════════════════════════════════════════════════════════════════

def _slugify(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
    return slug or "memory"


def write_memory(
    memory_dir: str,
    name: str,
    description: str,
    mtype: MemoryType,
    body: str,
) -> str:
    if mtype not in VALID_TYPES:
        raise ValueError(f"Invalid memory type: {mtype!r}. "
                         f"Must be one of {sorted(VALID_TYPES)}")
    directory = validate_memory_path(memory_dir)
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, f"{_slugify(name)}.md")
    frontmatter = (
        f"---\n"
        f"name: {name}\n"
        f"description: {description}\n"
        f"type: {mtype}\n"
        f"---\n\n"
    )
    payload = (frontmatter + body).encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    try:
        with open(path, "rb") as current:
            if hashlib.sha256(current.read()).digest() == digest:
                return path
    except FileNotFoundError:
        pass

    fd, temporary_path = tempfile.mkstemp(prefix=".memory-", dir=directory)
    try:
        with os.fdopen(fd, "wb") as temporary:
            temporary.write(payload)
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_path, path)
    finally:
        try:
            os.unlink(temporary_path)
        except FileNotFoundError:
            pass
    return path


# ═════════════════════════════════════════════════════════════════════════
# MEMORY.md index — capped at 200 lines / 25KB
# ═════════════════════════════════════════════════════════════════════════

def update_index(memory_dir: str, memories: list[MemoryFile]) -> None:
    """Write MEMORY.md respecting MAX_ENTRYPOINT_LINES and MAX_ENTRYPOINT_BYTES."""
    memory_dir = validate_memory_path(memory_dir)
    os.makedirs(memory_dir, exist_ok=True)
    lines = ["# Memory Index", ""]
    for m in memories:
        lines.append(f"- [{m.type}] {m.filename}: {m.description}")
    text = "\n".join(lines) + "\n"
    # Byte cap — drop lines from the end until under
    while len(text.encode("utf-8")) > MAX_ENTRYPOINT_BYTES and len(lines) > 3:
        lines.pop()
        text = "\n".join(lines) + "\n"
    # Line cap
    line_list = text.split("\n")
    if len(line_list) > MAX_ENTRYPOINT_LINES:
        text = "\n".join(line_list[:MAX_ENTRYPOINT_LINES]) + "\n"
    with open(os.path.join(memory_dir, MEMORY_INDEX_FILE), "w",
              encoding="utf-8") as f:
        f.write(text)


def read_index(memory_dir: str) -> str:
    """Read MEMORY.md if it exists, else return empty."""
    memory_dir = validate_memory_path(memory_dir)
    try:
        with open(os.path.join(memory_dir, MEMORY_INDEX_FILE),
                  encoding="utf-8") as f:
            return f.read()
    except OSError:
        return ""


__all__ = [
    "MEMORY_DIR_NAME", "MEMORY_INDEX_FILE",
    "MAX_ENTRYPOINT_LINES", "MAX_ENTRYPOINT_BYTES",
    "MAX_MEMORY_FILES", "MAX_SESSION_BYTES", "MAX_MEMORIES_PER_TURN",
    "FRONTMATTER_SCAN_LINES", "MEMORY_STALE_AFTER_SECONDS",
    "VALID_TYPES", "MemoryType",
    "MemoryFile", "SelectionContext",
    "get_memory_dir", "validate_memory_path", "is_llm_selection_enabled",
    "parse_frontmatter", "scan_memories", "build_manifest",
    "select_memories_llm", "select_memories_heuristic",
    "find_relevant_memories", "load_memory_content", "memory_freshness_note",
    "write_memory", "update_index", "read_index",
]