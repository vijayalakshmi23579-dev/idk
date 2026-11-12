"""
plans.py — plan file persistence.

Port of src/utils/plans.ts (150 lines).

The plan is a MARKDOWN FILE on disk. Everything else — the tool that
displays it, the attachment that survives compaction, the resume recovery —
is a view over this file.

Preserved exactly:
  • Word slug, cached per session, retried up to MAX_SLUG_RETRIES=10
  • Directory: settings.plansDirectory (resolved against cwd) or
    <config_home>/plans
  • PATH TRAVERSAL DEFENSE: resolved dir must start with cwd+sep or == cwd,
    else fall back to config home and log
  • Sub-agent path: {slug}-agent-{agentId}.md
  • get_plan() returns None on ENOENT — a missing plan is not an error
  • recover_plan_from_messages() with the 3 source order:
      1. ExitPlanModeV2 tool_use input.plan
      2. UserMessage.planContent
      3. AttachmentMessage type='plan_file_reference' .planContent

Reconstructed (source file not in the leak):
  • _generate_word_slug — words.ts was not fetched. A 3-part
    adjective-noun-verb slug, deterministic per session via the cache.
"""
from __future__ import annotations

import os
import random
import threading

from tool import ToolUseContext


# ── Constants (source: L16) ──────────────────────────────────────────────

MAX_SLUG_RETRIES = 10

EXIT_PLAN_MODE_V2_TOOL_NAME = "ExitPlanMode"


# ── Word slug — RECONSTRUCTED (words.ts not fetched) ────────────────────

_ADJECTIVES = (
    "brave", "calm", "eager", "fierce", "gentle", "happy", "keen",
    "lively", "mighty", "noble", "proud", "quick", "rapid", "sharp",
    "swift", "tidy", "vivid", "warm", "wise", "bold",
)
_NOUNS = (
    "tiger", "river", "mountain", "forest", "harbor", "canyon", "meadow",
    "glacier", "island", "desert", "valley", "summit", "prairie", "lagoon",
    "ridge", "cliff", "grove", "dune", "bay", "trail",
)
_VERBS = (
    "dancing", "singing", "running", "flying", "gliding", "leaping",
    "sailing", "soaring", "drifting", "racing", "climbing", "swimming",
    "wandering", "exploring", "roaming", "spinning", "flowing", "rising",
    "shining", "dreaming",
)


def _generate_word_slug() -> str:
    return "-".join((
        random.choice(_ADJECTIVES),
        random.choice(_NOUNS),
        random.choice(_VERBS),
    ))


# ── Session slug cache ──────────────────────────────────────────────────

_slug_cache: dict[str, str] = {}
_slug_lock = threading.Lock()


def get_plan_slug(session_id: str) -> str:
    """Get or generate a word slug for a session's plan.

    Source: plans.ts:19-26. Retries up to MAX_SLUG_RETRIES if the
    generated slug collides with an existing file."""
    with _slug_lock:
        if session_id in _slug_cache:
            return _slug_cache[session_id]

        plans_dir = get_plans_directory()
        slug: str | None = None
        for _ in range(MAX_SLUG_RETRIES):
            candidate = _generate_word_slug()
            if not os.path.exists(os.path.join(plans_dir, f"{candidate}.md")):
                slug = candidate
                break
        if slug is None:
            # Exhausted retries — accept the last candidate
            slug = candidate
        _slug_cache[session_id] = slug
        return slug


def set_plan_slug(session_id: str, slug: str) -> None:
    """Source: plans.ts:27-28. Used on resume."""
    with _slug_lock:
        _slug_cache[session_id] = slug


def clear_plan_slug(session_id: str) -> None:
    """Source: plans.ts:30-31. Called on /clear."""
    with _slug_lock:
        _slug_cache.pop(session_id, None)


def clear_all_plan_slugs() -> None:
    """Source: plans.ts:33-34."""
    with _slug_lock:
        _slug_cache.clear()


# ── Plans directory — source: plans.ts:39-50 ───────────────────────────

_plans_dir_cache: str | None = None


def get_plans_directory() -> str:
    """Resolve the plans directory once, memoized (source: L38-50).

    Path-traversal defense: if settings.plansDirectory resolves outside
    the project root, fall back to the config home and log.
    """
    global _plans_dir_cache
    if _plans_dir_cache is not None:
        return _plans_dir_cache

    cwd = os.getcwd()
    configured = _get_configured_plans_directory()

    if configured:
        resolved = os.path.normpath(os.path.join(cwd, configured))
        inside_project = (
            resolved.startswith(cwd + os.sep) or resolved == cwd
        )
        if not inside_project:
            # Source: plans.ts:44-46 — logError + fall back
            plans_path = os.path.join(_claude_config_home(), "plans")
        else:
            plans_path = resolved
    else:
        plans_path = os.path.join(_claude_config_home(), "plans")

    try:
        os.makedirs(plans_path, exist_ok=True)
    except OSError:
        pass

    _plans_dir_cache = plans_path
    return plans_path


def _get_configured_plans_directory() -> str | None:
    """Read settings.plansDirectory. Source uses getInitialSettings()."""
    try:
        from settings import get_initial_settings
        settings = get_initial_settings()
        return getattr(settings, "plans_directory", None)
    except (ImportError, AttributeError):
        return None


def _claude_config_home() -> str:
    env = os.environ.get("CLAUDE_CONFIG_DIR")
    if env:
        return os.path.expanduser(env)
    return os.path.join(os.path.expanduser("~"), ".claude")


# ── Plan file paths — source: plans.ts:54-58 ───────────────────────────

def get_plan_file_path(session_id: str, agent_id: str | None = None) -> str:
    """Main: {plansDir}/{slug}.md. Sub-agent: {plansDir}/{slug}-agent-{id}.md."""
    slug = get_plan_slug(session_id)
    plans_dir = get_plans_directory()
    if agent_id is None:
        return os.path.join(plans_dir, f"{slug}.md")
    return os.path.join(plans_dir, f"{slug}-agent-{agent_id}.md")


# ── Read / write — source: plans.ts:60-63 ─────────────────────────────

def get_plan(session_id: str, agent_id: str | None = None) -> str | None:
    """Read the plan file. ENOENT → None (not an error)."""
    path = get_plan_file_path(session_id, agent_id)
    try:
        with open(path, encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return None
    except OSError:
        return None


def write_plan(session_id: str, content: str, agent_id: str | None = None) -> str:
    """Write the plan file atomically. Returns the path written."""
    path = get_plan_file_path(session_id, agent_id)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(content)
    os.replace(tmp, path)
    return path


def plan_exists(session_id: str, agent_id: str | None = None) -> bool:
    return os.path.exists(get_plan_file_path(session_id, agent_id))


# ── Recovery — source: plans.ts:115-129 ────────────────────────────────

def recover_plan_from_messages(messages: list) -> str | None:
    """Recover plan content from message history, scanning backward.

    Source: plans.ts:109-129. Three sources in this order per message:
      1. assistant tool_use with name == 'ExitPlanMode' and input['plan']
      2. UserMessage.planContent
      3. AttachmentMessage.attachment['type'] == 'plan_file_reference'
         and attachment['planContent']
    """
    for msg in reversed(messages):
        mtype = getattr(msg, "type", None)

        if mtype == "assistant":
            for tc in (getattr(msg, "tool_calls", None) or []):
                if getattr(tc, "name", None) != EXIT_PLAN_MODE_V2_TOOL_NAME:
                    continue
                inp = getattr(tc, "input", None) or {}
                plan = inp.get("plan") if isinstance(inp, dict) else None
                if isinstance(plan, str) and plan:
                    return plan

        if mtype == "user":
            plan = getattr(msg, "plan_content", None)
            if isinstance(plan, str) and plan:
                return plan

        if mtype == "attachment":
            att = getattr(msg, "attachment", None) or {}
            if isinstance(att, dict) and att.get("type") == "plan_file_reference":
                plan = att.get("planContent")
                if isinstance(plan, str) and plan:
                    return plan

    return None


def copy_plan_for_resume(session_id: str, messages: list) -> bool:
    """Source: plans.ts:73-96. Restore plan on resume.

    If the file is missing, recover from message history and write it back.
    Returns True if a plan file exists (or was recovered)."""
    if plan_exists(session_id):
        return True
    recovered = recover_plan_from_messages(messages)
    if recovered is None:
        return False
    try:
        write_plan(session_id, recovered)
        return True
    except OSError:
        return False


# ── Plan mode display — for the /plan command ─────────────────────────

def get_plan_display(session_id: str) -> str:
    """Source: plan.tsx:34-36 — what /plan shows when already in plan mode."""
    plan = get_plan(session_id)
    if not plan:
        return "Already in plan mode. No plan written yet."
    path = get_plan_file_path(session_id)
    return f"Current Plan\n\n{plan}\n\n/plan open to edit this plan in your editor. ({path})"


__all__ = [
    "MAX_SLUG_RETRIES", "EXIT_PLAN_MODE_V2_TOOL_NAME",
    "get_plan_slug", "set_plan_slug", "clear_plan_slug",
    "clear_all_plan_slugs", "get_plans_directory",
    "get_plan_file_path", "get_plan", "write_plan", "plan_exists",
    "recover_plan_from_messages", "copy_plan_for_resume", "get_plan_display",
]