"""
compact.py — the compaction pipeline (single-file port).

Port of src/services/compact/ (15 files, ~3,960 lines) into one file.

EXACT ports (verified against raw source):
  • autoCompact.ts   — 6 constants, threshold math, warning state,
                       is_auto_compact_enabled, should_auto_compact,
                       circuit breaker (3 consecutive failures)
  • microCompact.ts  — TIME_BASED_MC_CLEARED_MESSAGE, IMAGE_MAX_TOKEN_SIZE,
                       COMPACTABLE_TOOLS whitelist, token estimators,
                       collectCompactableToolIds
  • compact.ts       — POST_COMPACT_* constants, CompactionResult,
                       build_post_compact_messages (exact field order),
                       strip_images_from_messages, truncate_head_for_ptl_retry
  • prompt.ts        — NO_TOOLS_PREAMBLE, 9-section template, NO_TOOLS_TRAILER,
                       format_compact_summary, get_compact_user_summary_message
  • grouping.ts      — group_messages_by_api_round

RECONSTRUCTED (source truncated at fetch — algorithm per documented behavior):
  • microcompact_messages — clear old compactable tool results, protect
                            recent tail. Constants and whitelist are exact.

ADAPTATIONS (flagged in code):
  • Summary generation uses api.complete() instead of a forked agent.
  • group_messages_by_api_round uses a round_id attribute when present,
    else falls back to assistant-adjacency (our messages have no API id).

STUBBED (feature-gated in source; exact signatures so query.py works):
  • try_reactive_compact, recover_from_overflow, snip_compact_if_needed,
    apply_collapses_if_needed, session-memory compaction, cached-MC funcs
"""
from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass
from typing import Any

from messages import (
    AttachmentMessage, AssistantMessage, ToolResultMessage, UserMessage,
    create_compact_boundary_message, create_user_message,
)


logger = logging.getLogger(__name__)


# ═════════════════════════════════════════════════════════════════════════
# 1. CONSTANTS — exact from source
# ═════════════════════════════════════════════════════════════════════════

# autoCompact.ts:22-41
MAX_OUTPUT_TOKENS_FOR_SUMMARY = 20_000   # p99.99 of summary output = 17,387
AUTOCOMPACT_BUFFER_TOKENS = 13_000
WARNING_THRESHOLD_BUFFER_TOKENS = 20_000
ERROR_THRESHOLD_BUFFER_TOKENS = 20_000
MANUAL_COMPACT_BUFFER_TOKENS = 3_000
# BQ 2026-03-10: 1,279 sessions had 50+ consecutive failures (up to 3,272),
# wasting ~250K API calls/day globally.
MAX_CONSECUTIVE_AUTOCOMPACT_FAILURES = 3

# compact.ts:63-70
POST_COMPACT_MAX_FILES_TO_RESTORE = 5
POST_COMPACT_TOKEN_BUDGET = 50_000
POST_COMPACT_MAX_TOKENS_PER_FILE = 5_000
POST_COMPACT_MAX_TOKENS_PER_SKILL = 5_000
POST_COMPACT_SKILLS_TOKEN_BUDGET = 25_000
MAX_COMPACT_STREAMING_RETRIES = 2

# microCompact.ts:27-31
TIME_BASED_MC_CLEARED_MESSAGE = "[Old tool result content cleared]"
IMAGE_MAX_TOKEN_SIZE = 2000
COMPACTABLE_TOOLS: frozenset[str] = frozenset({
    "Read",
    "Bash",
    "PowerShell",
    "Grep",
    "Glob",
    "WebSearch",
    "WebFetch",
    "Edit",
    "Write",
})

# compact.ts:104-139
ERROR_MESSAGE_NOT_ENOUGH_MESSAGES = "Not enough messages to compact."
ERROR_MESSAGE_PROMPT_TOO_LONG = (
    "Conversation too long. Press esc twice to go up a few messages and try again."
)
ERROR_MESSAGE_USER_ABORT = "API Error: Request was aborted."
ERROR_MESSAGE_INCOMPLETE_RESPONSE = (
    "Compaction interrupted \u00b7 This may be due to network issues \u2014 "
    "please try again."
)
MAX_PTL_RETRIES = 3
PTL_RETRY_MARKER = "[earlier conversation truncated for compaction retry]"

# Reconstructed: errors.ts text not in the fetched slice. Used only to
# detect a prompt-too-long response from the summary call.
PROMPT_TOO_LONG_ERROR_MESSAGE = "Prompt is too long"

# Reconstructed: microCompact's time-based config was truncated at fetch.
# Behavioral equivalent: never clear tool results from the most recent
# N assistant turns that carried tool calls.
PROTECTED_RECENT_ASSISTANT_TURNS = 3


# ═════════════════════════════════════════════════════════════════════════
# 2. MODEL INFO — lazy delegate to query.py with local fallback
#    (avoids importing the query loop unless a threshold is computed)
# ═════════════════════════════════════════════════════════════════════════

_FAMILY_WINDOWS = {
    "claude": 200_000,
    "gemini": 1_048_576,
    "llama": 131_072,
    "deepseek": 131_072,
    "qwen": 131_072,
    "mistral": 131_072,
}


def _context_window_for_model(model: str) -> int:
    try:
        from query import get_context_window_for_model
        return get_context_window_for_model(model)
    except Exception:
        low = model.lower()
        for family, n in _FAMILY_WINDOWS.items():
            if family in low:
                return n
        return 200_000


def _max_output_tokens_for_model(model: str) -> int:
    try:
        from query import get_max_output_tokens_for_model
        return get_max_output_tokens_for_model(model)
    except Exception:
        env = os.environ.get("CLAUDE_CODE_MAX_OUTPUT_TOKENS")
        if env:
            try:
                return int(env)
            except ValueError:
                pass
        return 8_192


def _current_model(context: Any = None) -> str:
    model_override = getattr(context, "model_override", None)
    if model_override:
        return model_override
    try:
        from api import get_active_model
        return get_active_model()
    except Exception:
        return "claude-sonnet-4-5"


def _env_truthy(name: str) -> bool:
    return os.environ.get(name, "").lower() in ("1", "true", "yes", "on")


# ═════════════════════════════════════════════════════════════════════════
# 3. TOKEN ESTIMATION — exact from microCompact.ts:65-96
# ═════════════════════════════════════════════════════════════════════════

def rough_token_count(text: str) -> int:
    """~1 token per 4 chars. Matches roughTokenCountEstimation."""
    return (len(text) + 3) // 4 if text else 0


def calculate_tool_result_tokens(content: Any) -> int:
    """microCompact.ts:65-73. Images/documents = 2000 tokens flat."""
    if not content:
        return 0
    if isinstance(content, str):
        return rough_token_count(content)
    total = 0
    for item in content:
        t = item.get("type") if isinstance(item, dict) else None
        if t == "text":
            total += rough_token_count(item.get("text", ""))
        elif t in ("image", "document"):
            total += IMAGE_MAX_TOKEN_SIZE
    return total


def estimate_message_tokens(messages: list) -> int:
    """microCompact.ts:73-96. Pads by 4/3 (conservative).

    ADAPTATION: our AssistantMessage carries text in .content (str),
    tool calls in .tool_calls, thinking in .thinking_blocks — the source
    reads a unified content-block array. Field names differ; the rule
    (count text + name/input per tool_use, skip JSON wrappers and ids)
    is identical.
    """
    total = 0
    for m in messages:
        mtype = getattr(m, "type", None)
        if mtype == "user":
            content = getattr(m, "content", "")
            if isinstance(content, str):
                total += rough_token_count(content)
            elif isinstance(content, list):
                for block in content:
                    if not isinstance(block, dict):
                        continue
                    t = block.get("type")
                    if t == "text":
                        total += rough_token_count(block.get("text", ""))
                    elif t == "tool_result":
                        total += calculate_tool_result_tokens(block.get("content"))
                    elif t in ("image", "document"):
                        total += IMAGE_MAX_TOKEN_SIZE
        elif mtype == "assistant":
            total += rough_token_count(getattr(m, "content", "") or "")
            for tc in getattr(m, "tool_calls", None) or []:
                total += rough_token_count(
                    f"{getattr(tc, 'name', '')}"
                    f"{json.dumps(getattr(tc, 'input', None) or {}, default=str)}"
                )
            for tb in getattr(m, "thinking_blocks", None) or []:
                total += rough_token_count(getattr(tb, "thinking", "") if not isinstance(tb, dict) else tb.get("thinking", ""))
        elif mtype == "tool_result":
            total += calculate_tool_result_tokens(getattr(m, "content", ""))
    return -(-total * 4 // 3)  # ceil(total * 4/3)


def token_count_with_estimation(messages: list) -> int:
    """Prefer the last API-reported usage; fall back to estimation.

    ADAPTATION of utils/tokens.ts:tokenCountWithEstimation — the source
    sums input_tokens + cache_read + cache_creation from the last response.
    """
    for m in reversed(messages):
        usage = getattr(m, "usage", None)
        if isinstance(usage, dict) and usage:
            return int(
                usage.get("input_tokens", 0)
                + usage.get("cache_read_input_tokens", 0)
                + usage.get("cache_creation_input_tokens", 0)
            )
    return estimate_message_tokens(messages)


# ═════════════════════════════════════════════════════════════════════════
# 4. GROUPING — exact logic from grouping.ts, adapted for our message model
# ═════════════════════════════════════════════════════════════════════════

def group_messages_by_api_round(messages: list) -> list[list]:
    """One group per API round-trip.

    Source (grouping.ts:16-37): a boundary fires when a NEW assistant
    response begins (different message.id from the prior assistant).

    ADAPTATION: our AssistantMessage has no API message.id. Two paths:
      1. If messages carry `round_id` (set by the streaming adapter per
         API call), use it — EXACT equivalence to the source.
      2. Else fall back to adjacency: a new group starts at each assistant
         message that does not immediately follow another assistant.
         Consecutive assistant blocks (multi-block stream) stay together.
    """
    groups: list[list] = []
    current: list = []
    last_round_id: str | None = None
    prev_type: str | None = None

    for msg in messages:
        mtype = getattr(msg, "type", None)
        round_id = getattr(msg, "round_id", None)

        if mtype == "assistant":
            if round_id is not None:
                is_new_round = round_id != last_round_id
                last_round_id = round_id
            else:
                is_new_round = prev_type != "assistant"
            if is_new_round and current:
                groups.append(current)
                current = [msg]
            else:
                current.append(msg)
        else:
            current.append(msg)

        prev_type = mtype

    if current:
        groups.append(current)
    return groups


# ═════════════════════════════════════════════════════════════════════════
# 5. MICROCOMPACT
# ═════════════════════════════════════════════════════════════════════════

def collect_compactable_tool_ids(messages: list) -> list[str]:
    """microCompact.ts:101-108 (exact rule: assistant tool_use in whitelist).

    ADAPTATION: tool calls live on AssistantMessage.tool_calls, not in a
    unified content array.
    """
    ids: list[str] = []
    for m in messages:
        if getattr(m, "type", None) != "assistant":
            continue
        for tc in getattr(m, "tool_calls", None) or []:
            if getattr(tc, "name", None) in COMPACTABLE_TOOLS:
                ids.append(tc.id)
    return ids


def _find_microcompact_cutoff(messages: list) -> int:
    """Index at/after which tool results are PROTECTED from clearing.

    RECONSTRUCTED: the time-based config (getTimeBasedMCConfig) was
    truncated at fetch. Behavioral equivalent: protect tool results from
    the most recent PROTECTED_RECENT_ASSISTANT_TURNS assistant turns that
    carried tool calls. Returns 0 when nothing is old enough to clear.
    """
    with_tools = [
        i for i, m in enumerate(messages)
        if getattr(m, "type", None) == "assistant"
        and getattr(m, "tool_calls", None)
    ]
    if len(with_tools) <= PROTECTED_RECENT_ASSISTANT_TURNS:
        return 0
    return with_tools[-PROTECTED_RECENT_ASSISTANT_TURNS]


async def microcompact(
    messages: list,
    tool_use_context: Any = None,
    query_source: str = "",
) -> dict:
    """Clear old compactable tool results in place.

    Returns {"messages": [...], "compaction_info": None} — the shape
    query_deps.py expects.

    RECONSTRUCTED body (source truncated at fetch). Uses the EXACT
    whitelist, cleared-message constant, and token estimators from source.
    The cached-MC path (ant-only, CACHED_MICROCOMPACT) is not implemented.
    """
    compactable_ids = collect_compactable_tool_ids(messages)
    if not compactable_ids:
        return {"messages": messages, "compaction_info": None}

    cutoff = _find_microcompact_cutoff(messages)
    if cutoff <= 0:
        return {"messages": messages, "compaction_info": None}

    id_set = set(compactable_ids)
    store = getattr(tool_use_context, "_store", None)
    session_id = getattr(tool_use_context, "session_id", None)
    out: list = []
    cleared = 0
    for i, msg in enumerate(messages):
        if (
            i < cutoff
            and getattr(msg, "type", None) == "tool_result"
            and getattr(msg, "tool_use_id", None) in id_set
            and getattr(msg, "content", "") != TIME_BASED_MC_CLEARED_MESSAGE
        ):
            message_id = None
            if store is not None and session_id:
                try:
                    message_id = store.get_message_id_by_tool_use_id(
                        session_id, msg.tool_use_id,
                    )
                except Exception:
                    message_id = None
            replacement = (
                f"[Tool result stored in memory as message {message_id}. "
                f"Use MemoryExpand(node_id={message_id}, kind='message') to retrieve.]"
                if message_id is not None
                else TIME_BASED_MC_CLEARED_MESSAGE
            )
            out.append(ToolResultMessage(
                tool_use_id=msg.tool_use_id,
                content=replacement,
                is_error=getattr(msg, "is_error", False),
                uuid=getattr(msg, "uuid", None) or ToolResultMessage(
                    tool_use_id=msg.tool_use_id, content=""
                ).uuid,
                timestamp=getattr(msg, "timestamp", None),
            ))
            cleared += 1
        else:
            out.append(msg)

    if cleared:
        logger.debug("[microcompact] cleared %d tool results", cleared)
    return {"messages": out, "compaction_info": None}


def reset_microcompact_state() -> None:
    """microCompact.ts:63-65 — cached-MC state reset. No-op (cached MC unused)."""
    return None


def consume_pending_cache_edits() -> None:
    """microCompact.ts:48-50 — no cached MC; always None."""
    return None


def get_pinned_cache_edits() -> list:
    """microCompact.ts:52-54 — no cached MC; always empty."""
    return []


def pin_cache_edits(_user_message_index: int, _block: Any) -> None:
    """microCompact.ts:56-59 — no cached MC; no-op."""
    return None


def mark_tools_sent_to_api_state() -> None:
    """microCompact.ts:59-62 — no cached MC; no-op."""
    return None


# ═════════════════════════════════════════════════════════════════════════
# 6. AUTO-COMPACT THRESHOLDS — exact from autoCompact.ts:24-75
# ═════════════════════════════════════════════════════════════════════════

def get_effective_context_window_size(model: str) -> int:
    """autoCompact.ts:24-31."""
    reserved = min(_max_output_tokens_for_model(model), MAX_OUTPUT_TOKENS_FOR_SUMMARY)
    window = _context_window_for_model(model)
    env = os.environ.get("CLAUDE_CODE_AUTO_COMPACT_WINDOW")
    if env:
        try:
            parsed = int(env)
            if parsed > 0:
                window = min(window, parsed)
        except ValueError:
            pass
    return window - reserved


def get_auto_compact_threshold(model: str) -> int:
    """autoCompact.ts:42-50."""
    effective = get_effective_context_window_size(model)
    threshold = effective - AUTOCOMPACT_BUFFER_TOKENS
    env_pct = os.environ.get("CLAUDE_AUTOCOMPACT_PCT_OVERRIDE")
    if env_pct:
        try:
            pct = float(env_pct)
            if 0 < pct <= 100:
                pct_threshold = int(effective * (pct / 100))
                return min(pct_threshold, threshold)
        except ValueError:
            pass
    return threshold


def calculate_token_warning_state(token_usage: int, model: str) -> dict:
    """autoCompact.ts:50-70 — exact, including blocking-limit override."""
    auto_threshold = get_auto_compact_threshold(model)
    threshold = auto_threshold if is_auto_compact_enabled() else get_effective_context_window_size(model)

    percent_left = max(0, round(((threshold - token_usage) / threshold) * 100))
    warning_threshold = threshold - WARNING_THRESHOLD_BUFFER_TOKENS
    error_threshold = threshold - ERROR_THRESHOLD_BUFFER_TOKENS

    actual_window = get_effective_context_window_size(model)
    default_blocking = actual_window - MANUAL_COMPACT_BUFFER_TOKENS
    blocking_limit = default_blocking
    env = os.environ.get("CLAUDE_CODE_BLOCKING_LIMIT_OVERRIDE")
    if env:
        try:
            parsed = int(env)
            if parsed > 0:
                blocking_limit = parsed
        except ValueError:
            pass

    return {
        "percent_left": percent_left,
        "is_above_warning_threshold": token_usage >= warning_threshold,
        "is_above_error_threshold": token_usage >= error_threshold,
        "is_above_auto_compact_threshold": is_auto_compact_enabled() and token_usage >= auto_threshold,
        "is_at_blocking_limit": token_usage >= blocking_limit,
    }


def is_auto_compact_enabled() -> bool:
    """autoCompact.ts:71-76. No global config file → env only, default True."""
    if _env_truthy("DISABLE_COMPACT"):
        return False
    if _env_truthy("DISABLE_AUTO_COMPACT"):
        return False
    return True


def should_auto_compact(
    messages: list,
    model: str,
    query_source: str | None = None,
    snip_tokens_freed: int = 0,
) -> bool:
    """autoCompact.ts:76-121 — including both recursion guards."""
    # session_memory / compact are forked agents that would deadlock
    if query_source in ("session_memory", "compact"):
        return False
    if not is_auto_compact_enabled():
        return False

    token_count = token_count_with_estimation(messages) - snip_tokens_freed
    state = calculate_token_warning_state(token_count, model)
    return state["is_above_auto_compact_threshold"]


# ═════════════════════════════════════════════════════════════════════════
# 7. PROMPTS — exact from prompt.ts (XML tags restored; fetcher stripped them)
# ═════════════════════════════════════════════════════════════════════════

NO_TOOLS_PREAMBLE = """CRITICAL: Respond with TEXT ONLY. Do NOT call any tools.

- Do NOT use Read, Bash, Grep, Glob, Edit, Write, or ANY other tool.
- You already have all the context you need in the conversation above.
- Tool calls will be REJECTED and will waste your only turn — you will fail the task.
- Your entire response must be plain text: an <analysis> block followed by a <summary> block.
"""

NO_TOOLS_TRAILER = (
    "\n\nREMINDER: Do NOT call any tools. Respond with plain text only — "
    "an <analysis> block followed by a <summary> block. "
    "Tool calls will be rejected and you will fail the task."
)

DETAILED_ANALYSIS_INSTRUCTION_BASE = """Before providing your final summary, wrap your analysis in <analysis> tags to organize your thoughts and ensure you've covered all necessary points. In your analysis process:

1. Chronologically analyze each message and section of the conversation. For each section thoroughly identify:
   - The user's explicit requests and intents
   - Your approach to addressing the user's requests
   - Key decisions, technical concepts and code patterns
   - Specific details like:
     - file names
     - full code snippets
     - function signatures
     - file edits
   - Errors that you ran into and how you fixed them
   - Pay special attention to specific user feedback that you received, especially if the user told you to do something differently.
2. Double-check for technical accuracy and completeness, addressing each required element thoroughly."""

DETAILED_ANALYSIS_INSTRUCTION_PARTIAL = """Before providing your final summary, wrap your analysis in <analysis> tags to organize your thoughts and ensure you've covered all necessary points. In your analysis process:

1. Analyze the recent messages chronologically. For each section thoroughly identify:
   - The user's explicit requests and intents
   - Your approach to addressing the user's requests
   - Key decisions, technical concepts and code patterns
   - Specific details like:
     - file names
     - full code snippets
     - function signatures
     - file edits
   - Errors that you ran into and how you fixed them
   - Pay special attention to specific user feedback that you received, especially if the user told you to do something differently.
2. Double-check for technical accuracy and completeness, addressing each required element thoroughly."""

_EXAMPLE_STRUCTURE = """Here's an example of how your output should be structured:

<example>
<analysis>
[Your thought process, ensuring all points are covered thoroughly and accurately]
</analysis>

<summary>
1. Primary Request and Intent:
   [Detailed description]
2. Key Technical Concepts:
   - [Concept 1]
   - [Concept 2]
   - [...]
3. Files and Code Sections:
   - [File Name 1]
     - [Summary of why this file is important]
     - [Summary of the changes made to this file, if any]
     - [Important Code Snippet]
   - [File Name 2]
     - [Important Code Snippet]
   - [...]
4. Errors and fixes:
   - [Detailed description of error 1]:
     - [How you fixed the error]
     - [User feedback on the error if any]
   - [...]
5. Problem Solving:
   [Description of solved problems and ongoing troubleshooting]
6. All user messages:
   - [Detailed non tool use user message]
   - [...]
7. Pending Tasks:
   - [Task 1]
   - [Task 2]
   - [...]
8. Current Work:
   [Precise description of current work]
9. Optional Next Step:
   [Optional Next step to take]
</summary>
</example>"""

# (Consolidated: source repeats this example in all three templates
#  verbatim — interpolating one constant is behaviorally identical.)

BASE_COMPACT_PROMPT = f"""Your task is to create a detailed summary of the conversation so far, paying close attention to the user's explicit requests and your previous actions. This summary should be thorough in capturing technical details, code patterns, and architectural decisions that would be essential for continuing development work without losing context.

{DETAILED_ANALYSIS_INSTRUCTION_BASE}

Your summary should include the following sections:

1. Primary Request and Intent: Capture all of the user's explicit requests and intents in detail
2. Key Technical Concepts: List all important technical concepts, technologies, and frameworks discussed.
3. Files and Code Sections: Enumerate specific files and code sections examined, modified, or created. Pay special attention to the most recent messages and include full code snippets where applicable and include a summary of why this file read or edit is important.
4. Errors and fixes: List all errors that you ran into, and how you fixed them. Pay special attention to specific user feedback that you received, especially if the user told you to do something differently.
5. Problem Solving: Document problems solved and any ongoing troubleshooting efforts.
6. All user messages: List ALL user messages that are not tool results. These are critical for understanding the users' feedback and changing intent.
7. Pending Tasks: Outline any pending tasks that you have explicitly been asked to work on.
8. Current Work: Describe in detail precisely what was being worked on immediately before this summary request, paying special attention to the most recent messages from both user and assistant. Include file names and code snippets where applicable.
9. Optional Next Step: List the next step that you will take that is related to the most recent work you were doing. IMPORTANT: ensure that this step is DIRECTLY in line with the user's most recent explicit requests, and the task you were working on immediately before this summary request. If your last task was concluded, then only list next steps if they are explicitly in line with the users request. Do not start on tangential requests or really old requests that were already completed without confirming with the user first. If there is a next step, include direct quotes from the most recent conversation showing exactly what task you were working on and where you left off. This should be verbatim to ensure there's no drift in task interpretation.

{_EXAMPLE_STRUCTURE}

Please provide your summary based on the conversation so far, following this structure and ensuring precision and thoroughness in your response. There may be additional summarization instructions provided in the included context. If so, remember to follow these instructions when creating the above summary. Examples of instructions include:

## Compact Instructions
When summarizing the conversation focus on typescript code changes and also remember the mistakes you made and how you fixed them.

# Summary instructions
When you are using compact - please focus on test output and code changes. Include file reads verbatim.
"""

PARTIAL_COMPACT_PROMPT = f"""Your task is to create a detailed summary of the RECENT portion of the conversation — the messages that follow earlier retained context. The earlier messages are being kept intact and do NOT need to be summarized. Focus your summary on what was discussed, learned, and accomplished in the recent messages only.

{DETAILED_ANALYSIS_INSTRUCTION_PARTIAL}

Your summary should include the following sections:

1. Primary Request and Intent: Capture the user's explicit requests and intents from the recent messages
2. Key Technical Concepts: List important technical concepts, technologies, and frameworks discussed recently.
3. Files and Code Sections: Enumerate specific files and code sections examined, modified, or created. Include full code snippets where applicable and include a summary of why this file read or edit is important.
4. Errors and fixes: List errors encountered and how they were fixed.
5. Problem Solving: Document problems solved and any ongoing troubleshooting efforts.
6. All user messages: List ALL user messages from the recent portion that are not tool results.
7. Pending Tasks: Outline any pending tasks from the recent messages.
8. Current Work: Describe precisely what was being worked on immediately before this summary request.
9. Optional Next Step: List the next step related to the most recent work. Include direct quotes from the most recent conversation.

{_EXAMPLE_STRUCTURE}

Please provide your summary based on the RECENT messages only (after the retained earlier context), following this structure and ensuring precision and thoroughness in your response."""

PARTIAL_COMPACT_UP_TO_PROMPT = f"""Your task is to create a detailed summary of this conversation. This summary will be placed at the start of a continuing session; newer messages that build on this context will follow after your summary (you do not see them here). Summarize thoroughly so that someone reading only your summary and then the newer messages can fully understand what happened and continue the work.

{DETAILED_ANALYSIS_INSTRUCTION_BASE}

Your summary should include the following sections:

1. Primary Request and Intent: Capture the user's explicit requests and intents in detail
2. Key Technical Concepts: List important technical concepts, technologies, and frameworks discussed.
3. Files and Code Sections: Enumerate specific files and code sections examined, modified, or created. Include full code snippets where applicable and include a summary of why this file read or edit is important.
4. Errors and fixes: List errors encountered and how they were fixed.
5. Problem Solving: Document problems solved and any ongoing troubleshooting efforts.
6. All user messages: List ALL user messages that are not tool results.
7. Pending Tasks: Outline any pending tasks.
8. Work Completed: Describe what was accomplished by the end of this portion.
9. Context for Continuing Work: Summarize any context, decisions, or state that would be needed to understand and continue the work in subsequent messages.

{_EXAMPLE_STRUCTURE}

Please provide your summary following this structure, ensuring precision and thoroughness in your response."""


def get_compact_prompt(custom_instructions: str | None = None) -> str:
    """prompt.ts:165-169 (exact composition)."""
    prompt = NO_TOOLS_PREAMBLE + BASE_COMPACT_PROMPT
    if custom_instructions and custom_instructions.strip():
        prompt += f"\n\nAdditional Instructions:\n{custom_instructions}"
    prompt += NO_TOOLS_TRAILER
    return prompt


def get_partial_compact_prompt(
    custom_instructions: str | None = None,
    direction: str = "from",
) -> str:
    """prompt.ts:159-164 (exact composition)."""
    template = PARTIAL_COMPACT_UP_TO_PROMPT if direction == "up_to" else PARTIAL_COMPACT_PROMPT
    prompt = NO_TOOLS_PREAMBLE + template
    if custom_instructions and custom_instructions.strip():
        prompt += f"\n\nAdditional Instructions:\n{custom_instructions}"
    prompt += NO_TOOLS_TRAILER
    return prompt


_ANALYSIS_STRIP = re.compile(r"[\s\S]*?</analysis>")
_SUMMARY_UNWRAP = re.compile(r"<summary>([\s\S]*?)</summary>")


def format_compact_summary(summary: str) -> str:
    """prompt.ts:174-183 — exact regexes."""
    formatted = _ANALYSIS_STRIP.sub("", summary)
    m = _SUMMARY_UNWRAP.search(formatted)
    if m:
        content = (m.group(1) or "").strip()
        formatted = _SUMMARY_UNWRAP.sub(f"Summary:\n{content}", formatted)
    formatted = re.sub(r"\n\n+", "\n\n", formatted)
    return formatted.strip()


def get_compact_user_summary_message(
    summary: str,
    suppress_follow_up_questions: bool = False,
    transcript_path: str | None = None,
    recent_messages_preserved: bool = False,
) -> str:
    """prompt.ts:184-203 (exact; tail reconstructed — fetch ended mid-function)."""
    formatted = format_compact_summary(summary)
    base = (
        "This session is being continued from a previous conversation that "
        "ran out of context. The summary below covers the earlier portion "
        f"of the conversation.\n\n{formatted}"
    )
    if transcript_path:
        base += (
            "\n\nIf you need specific details from before compaction (like "
            "exact code snippets, error messages, or content you generated), "
            f"read the full transcript at: {transcript_path}"
        )
    if recent_messages_preserved:
        base += "\n\nRecent messages are preserved verbatim."
    if suppress_follow_up_questions:
        base += (
            "\n\nContinue the conversation from where it left off without "
            "asking the user any further questions. Resume directly — do not "
            "acknowledge the summary, do not recap what was happening, do not "
            'preface with "I\'ll continue" or similar. Pick up the last task '
            "as if the break never happened."
        )
    return base


# ═════════════════════════════════════════════════════════════════════════
# 8. COMPACTION RESULT — exact shape from compact.ts:140-144
# ═════════════════════════════════════════════════════════════════════════

@dataclass
class CompactionResult:
    boundary_marker: Any
    summary_messages: list
    attachments: list
    hook_results: list
    messages_to_keep: list | None = None
    user_display_message: str | None = None
    pre_compact_token_count: int | None = None
    post_compact_token_count: int | None = None
    true_post_compact_token_count: int | None = None
    compaction_usage: dict | None = None


def build_post_compact_messages(result: CompactionResult) -> list:
    """compact.ts:154-157 — EXACT field order."""
    return [
        result.boundary_marker,
        *result.summary_messages,
        *(result.messages_to_keep or []),
        *result.attachments,
        *result.hook_results,
    ]


# ═════════════════════════════════════════════════════════════════════════
# 9. STRIPPING — exact from compact.ts:78-103
# ═════════════════════════════════════════════════════════════════════════

def strip_images_from_messages(messages: list) -> list:
    """compact.ts:78-94 — images/documents (incl. inside tool_result) →
    text markers. Images are useless for a summary and can make the compact
    call itself hit prompt-too-long."""
    out = []
    for message in messages:
        if getattr(message, "type", None) != "user":
            out.append(message)
            continue
        content = getattr(message, "content", None)
        if not isinstance(content, list):
            out.append(message)
            continue
        has_media = False
        new_content = []
        for block in content:
            if not isinstance(block, dict):
                new_content.append(block)
                continue
            if block.get("type") == "image":
                has_media = True
                new_content.append({"type": "text", "text": "[image]"})
            elif block.get("type") == "document":
                has_media = True
                new_content.append({"type": "text", "text": "[document]"})
            elif block.get("type") == "tool_result" and isinstance(block.get("content"), list):
                tool_has_media = False
                new_tool = []
                for item in block["content"]:
                    if isinstance(item, dict) and item.get("type") == "image":
                        tool_has_media = True
                        new_tool.append({"type": "text", "text": "[image]"})
                    elif isinstance(item, dict) and item.get("type") == "document":
                        tool_has_media = True
                        new_tool.append({"type": "text", "text": "[document]"})
                    else:
                        new_tool.append(item)
                if tool_has_media:
                    has_media = True
                    new_content.append({**block, "content": new_tool})
                else:
                    new_content.append(block)
            else:
                new_content.append(block)
        if not has_media:
            out.append(message)
        else:
            message.content = new_content
            out.append(message)
    return out


def strip_reinjected_attachments(messages: list) -> list:
    """compact.ts:100-103 — skill_discovery/skill_listing are re-surfaced
    post-compact; feeding them to the summarizer wastes tokens."""
    out = []
    for m in messages:
        if getattr(m, "type", None) == "attachment":
            att = getattr(m, "attachment", None) or {}
            if isinstance(att, dict) and att.get("type") in ("skill_discovery", "skill_listing"):
                continue
        out.append(m)
    return out


# ═════════════════════════════════════════════════════════════════════════
# 10. PTL TRUNCATION — exact from compact.ts:106-135
# ═════════════════════════════════════════════════════════════════════════

_TOKEN_GAP_RE = re.compile(r"(\d+)\s*tokens?\s*[>»]\s*(\d+)")


def _get_prompt_too_long_token_gap(text: str) -> int | None:
    """RECONSTRUCTED: errors.ts:getPromptTooLongTokenGap not in the fetched
    slice. Parses "N tokens > M" → N - M. Returns None when unparseable
    (source falls back to dropping 20% of groups)."""
    m = _TOKEN_GAP_RE.search(text or "")
    if not m:
        return None
    gap = int(m.group(1)) - int(m.group(2))
    return gap if gap > 0 else None


def truncate_head_for_ptl_retry(messages: list, ptl_response_text: str) -> list | None:
    """compact.ts:115-135 — exact: drop oldest API-round groups until the
    token gap is covered (or 20% when unparseable). Keeps ≥1 group.
    Prepends a synthetic user marker when the slice starts with assistant."""
    # Strip our own marker from a previous retry before grouping.
    if (
        messages
        and getattr(messages[0], "type", None) == "user"
        and getattr(messages[0], "is_meta", False)
        and getattr(messages[0], "content", None) == PTL_RETRY_MARKER
    ):
        input_messages = messages[1:]
    else:
        input_messages = messages

    groups = group_messages_by_api_round(input_messages)
    if len(groups) < 2:
        return None

    gap = _get_prompt_too_long_token_gap(ptl_response_text)
    if gap is not None:
        acc = 0
        drop_count = 0
        for g in groups:
            acc += estimate_message_tokens(g)
            drop_count += 1
            if acc >= gap:
                break
    else:
        drop_count = max(1, int(len(groups) * 0.2))

    drop_count = min(drop_count, len(groups) - 1)
    if drop_count < 1:
        return None

    sliced = [m for g in groups[drop_count:] for m in g]

    # Dropping group 0 leaves an assistant-first sequence the API rejects.
    if sliced and getattr(sliced[0], "type", None) == "assistant":
        return [create_user_message(PTL_RETRY_MARKER, is_meta=True), *sliced]
    return sliced


# ═════════════════════════════════════════════════════════════════════════
# 11. POST-COMPACT FILE RESTORATION
#     (createPostCompactFileAttachments — source body truncated at fetch;
#      ported from the documented constants + shape)
# ═════════════════════════════════════════════════════════════════════════

def create_post_compact_file_attachments(
    pre_compact_read_file_state: dict,
    max_files: int = POST_COMPACT_MAX_FILES_TO_RESTORE,
) -> list:
    """Re-attach up to 5 recently-read files, 5K tokens each, 50K total.

    Files are the most expensive thing to lose: the model would otherwise
    have to re-Read them. Most-recently-read first.
    """
    if not pre_compact_read_file_state:
        return []
    items = sorted(
        pre_compact_read_file_state.items(),
        key=lambda kv: (kv[1] or {}).get("mtime", 0),
        reverse=True,
    )[:max_files]

    max_total_chars = POST_COMPACT_TOKEN_BUDGET * 4       # 50K tok ≈ 200K chars
    max_chars_per_file = POST_COMPACT_MAX_TOKENS_PER_FILE * 4  # 5K tok ≈ 20K chars
    total_chars = 0
    attachments: list = []

    for path, state in items:
        content = (state or {}).get("content", "")
        if not content:
            continue
        truncated = False
        if len(content) > max_chars_per_file:
            content = content[:max_chars_per_file]
            truncated = True
        remaining = max_total_chars - total_chars
        if remaining <= 0:
            break
        if len(content) > remaining:
            content = content[:remaining]
            truncated = True
        total_chars += len(content)
        attachments.append(AttachmentMessage(attachment={
            "type": "file",
            "filename": path,
            "content": content,
            "truncated": truncated,
        }))
    return attachments


# ═════════════════════════════════════════════════════════════════════════
# 12. SUMMARY GENERATION — ADAPTATION: api.complete() instead of forked agent
# ═════════════════════════════════════════════════════════════════════════

async def _stream_compact_summary(
    messages_to_summarize: list,
    summary_request: UserMessage,
    pre_compact_token_count: int,
    model: str,
) -> str:
    """compact.ts:streamCompactSummary — ADAPTATION.

    Source forks an agent over cacheSafeParams.forkContextMessages so the
    summary call shares the main conversation's prompt cache. We have no
    forked agent; api.complete() sends the same prompt as a one-shot call.

    Input is identical (strip media → strip re-injected attachments →
    append the summary request). Output is the raw summary text.
    """
    from api import complete

    compact_model = os.environ.get("COMPACTION_MODEL", model)
    cleaned = strip_images_from_messages(list(messages_to_summarize))
    cleaned = strip_reinjected_attachments(cleaned)
    call_messages = [*cleaned, summary_request]

    max_tokens = min(
        MAX_OUTPUT_TOKENS_FOR_SUMMARY,
        _max_output_tokens_for_model(compact_model),
    )
    return await complete(
        call_messages,
        system_prompt="",
        model=compact_model,
        max_tokens=max_tokens,
    )


def _is_prompt_too_long_error(error: Exception) -> bool:
    name = type(error).__name__
    if "ContextWindow" in name or "ContextLength" in name:
        return True
    return "too long" in str(error).lower() or "context length" in str(error).lower()


def _starts_with_api_error_prefix(text: str) -> bool:
    return bool(text) and text.lstrip().startswith("API Error")


# ═════════════════════════════════════════════════════════════════════════
# 13. COMPACT CONVERSATION — exact flow from compact.ts:176-351
# ═════════════════════════════════════════════════════════════════════════

async def compact_conversation(
    messages: list,
    context: Any,
    cache_safe_params: dict | None = None,
    suppress_follow_up_questions: bool = False,
    custom_instructions: str | None = None,
    is_auto_compact: bool = False,
    recompaction_info: dict | None = None,
) -> CompactionResult:
    """compact.ts:176-351.

    Flow: pre-compact hooks → stream summary (with PTL retry loop) →
    snapshot+clear read state → restore up to 5 files → boundary + summary
    → post-compact hooks. Hook calls are no-ops (no hook system) but the
    progress events are emitted exactly.
    """
    try:
        if not messages:
            raise ValueError(ERROR_MESSAGE_NOT_ENOUGH_MESSAGES)

        model = _current_model(context)
        pre_compact_token_count = token_count_with_estimation(messages)

        _progress(context, {"type": "hooks_start", "hook_type": "pre_compact"})
        _set_sdk_status(context, "compacting")
        # PreCompact hooks: no-op (no hook system). customInstructions unchanged.

        _set_stream_mode(context, "requesting")
        _set_response_length(context, lambda: 0)
        _progress(context, {"type": "compact_start"})

        compact_prompt = get_compact_prompt(custom_instructions)
        summary_request = create_user_message(compact_prompt)

        messages_to_summarize = messages
        summary: str | None = None
        ptl_attempts = 0

        while True:
            try:
                summary = await _stream_compact_summary(
                    messages_to_summarize, summary_request,
                    pre_compact_token_count, model,
                )
            except Exception as e:
                if not _is_prompt_too_long_error(e):
                    raise
                summary = PROMPT_TOO_LONG_ERROR_MESSAGE

            is_ptl = bool(summary) and summary.startswith(PROMPT_TOO_LONG_ERROR_MESSAGE)
            if not is_ptl:
                break

            ptl_attempts += 1
            truncated = (
                truncate_head_for_ptl_retry(messages_to_summarize, summary)
                if ptl_attempts <= MAX_PTL_RETRIES else None
            )
            if truncated is None:
                logger.error("Compaction failed: prompt too long after %d retries", ptl_attempts)
                raise RuntimeError(ERROR_MESSAGE_PROMPT_TOO_LONG)
            messages_to_summarize = truncated

        if not summary:
            raise RuntimeError(
                "Failed to generate conversation summary - response did not "
                "contain valid text content"
            )
        if _starts_with_api_error_prefix(summary):
            raise RuntimeError(summary)

        # Snapshot the read-file cache BEFORE clearing it
        pre_compact_read_file_state = {}
        read_cache = getattr(context, "read_file_state", None)
        if read_cache is not None and getattr(read_cache, "files", None):
            pre_compact_read_file_state = {
                path: dict(state) for path, state in read_cache.files.items()
            }
            read_cache.files.clear()

        file_attachments = create_post_compact_file_attachments(
            pre_compact_read_file_state,
        )

        boundary_marker = create_compact_boundary_message(
            "auto" if is_auto_compact else "manual",
            pre_compact_token_count,
        )
        summary_messages = [create_user_message(
            get_compact_user_summary_message(
                summary,
                suppress_follow_up_questions,
                transcript_path=None,
            ),
            is_compact_summary=True,
        )]

        true_post_count = estimate_message_tokens(
            [boundary_marker, *summary_messages, *file_attachments]
        )
        _progress(context, {"type": "hooks_start", "hook_type": "post_compact"})

        logger.info(
            "[compact] %s compaction: %d → %d tokens (%d files restored)",
            "auto" if is_auto_compact else "manual",
            pre_compact_token_count, true_post_count, len(file_attachments),
        )

        return CompactionResult(
            boundary_marker=boundary_marker,
            summary_messages=summary_messages,
            attachments=file_attachments,
            hook_results=[],
            messages_to_keep=None,
            user_display_message=None,
            pre_compact_token_count=pre_compact_token_count,
            # Kept for field continuity: the compact API call's total usage
            post_compact_token_count=pre_compact_token_count,
            true_post_compact_token_count=true_post_count,
            compaction_usage=None,
        )
    finally:
        _set_stream_mode(context, "requesting")
        _set_response_length(context, lambda: 0)
        _progress(context, {"type": "compact_end"})
        _set_sdk_status(context, None)


# small context-callback helpers — tolerate None / missing attrs
def _progress(ctx: Any, event: dict) -> None:
    fn = getattr(ctx, "on_compact_progress", None)
    if fn:
        try:
            fn(event)
        except Exception:
            pass


def _set_sdk_status(ctx: Any, status: Any) -> None:
    fn = getattr(ctx, "set_sdk_status", None)
    if fn:
        try:
            fn(status)
        except Exception:
            pass


def _set_stream_mode(ctx: Any, mode: Any) -> None:
    fn = getattr(ctx, "set_stream_mode", None)
    if fn:
        try:
            fn(mode)
        except Exception:
            pass


def _set_response_length(ctx: Any, fn_factory: Any) -> None:
    fn = getattr(ctx, "set_response_length", None)
    if fn:
        try:
            fn(fn_factory)
        except Exception:
            pass


# ═════════════════════════════════════════════════════════════════════════
# 14. AUTOCCOMPACT — exact flow from autoCompact.ts:121-168
# ═════════════════════════════════════════════════════════════════════════

async def autocompact(
    messages: list,
    tool_use_context: Any,
    cache_safe_params: dict | None,
    query_source: str | None,
    tracking: dict | None,
    snip_tokens_freed: int,
) -> dict:
    """autoCompact.ts:121-168. Returns the shape query_deps.py expects:
        {"compaction_result": CompactionResult | None,
         "consecutive_failures": int | None}
    """
    if _env_truthy("DISABLE_COMPACT"):
        return {"compaction_result": None, "consecutive_failures": None}

    # Circuit breaker: stop retrying after N consecutive failures.
    if (
        tracking
        and tracking.get("consecutiveFailures", 0) >= MAX_CONSECUTIVE_AUTOCOMPACT_FAILURES
    ):
        return {"compaction_result": None, "consecutive_failures": None}

    model = _current_model(tool_use_context)
    if not should_auto_compact(messages, model, query_source, snip_tokens_freed):
        return {"compaction_result": None, "consecutive_failures": None}

    recompaction_info = {
        "isRecompactionInChain": bool(tracking and tracking.get("compacted")),
        "turnsSincePreviousCompact": (tracking or {}).get("turnCounter", -1),
        "previousCompactTurnId": (tracking or {}).get("turnId"),
        "autoCompactThreshold": get_auto_compact_threshold(model),
        "querySource": query_source,
    }

    try:
        # Source tries session-memory compaction first (cheaper). Stubbed.
        result = await compact_conversation(
            messages,
            tool_use_context,
            cache_safe_params,
            suppress_follow_up_questions=True,
            custom_instructions=None,
            is_auto_compact=True,
            recompaction_info=recompaction_info,
        )
        return {"compaction_result": result, "consecutive_failures": 0}
    except Exception as e:
        logger.warning("Autocompact failed: %s", e)
        failures = (tracking or {}).get("consecutiveFailures", 0) + 1
        return {"compaction_result": None, "consecutive_failures": failures}


# ═════════════════════════════════════════════════════════════════════════
# 15. FEATURE-GATED STUBS — signatures match the call sites in query.py
# ═════════════════════════════════════════════════════════════════════════

async def try_reactive_compact(
    has_attempted: bool,
    query_source: str | None,
    aborted: bool,
    messages: list,
    cache_safe_params: dict | None,
) -> Any | None:
    """reactiveCompact.ts (REACTIVE_COMPACT gate). Not implemented → None.
    query.py's reactive_compact_retry transition simply doesn't fire."""
    return None


def recover_from_overflow(messages: list, query_source: str | None) -> dict | None:
    """contextCollapse (CONTEXT_COLLAPSE gate). Not implemented → None."""
    return None


def snip_compact_if_needed(messages: list) -> dict:
    """snipCompact.ts (HISTORY_SNIP gate). Not implemented → no-op shape."""
    return {"messages": messages, "tokens_freed": 0, "boundary_message": None}


def apply_collapses_if_needed(
    messages: list, tool_use_context: Any = None, query_source: str | None = None,
) -> list:
    """contextCollapse (CONTEXT_COLLAPSE gate). Not implemented → passthrough."""
    return messages


async def try_session_memory_compaction(
    messages: list, agent_id: Any, threshold: int,
) -> Any | None:
    """sessionMemoryCompact.ts. Not implemented → None (autocompact falls
    through to full summarization, which is the correct behavior)."""
    return None


# ═════════════════════════════════════════════════════════════════════════
# 16. EXPORTS
# ═════════════════════════════════════════════════════════════════════════

__all__ = [
    # constants
    "MAX_OUTPUT_TOKENS_FOR_SUMMARY", "AUTOCOMPACT_BUFFER_TOKENS",
    "WARNING_THRESHOLD_BUFFER_TOKENS", "ERROR_THRESHOLD_BUFFER_TOKENS",
    "MANUAL_COMPACT_BUFFER_TOKENS", "MAX_CONSECUTIVE_AUTOCOMPACT_FAILURES",
    "POST_COMPACT_MAX_FILES_TO_RESTORE", "POST_COMPACT_TOKEN_BUDGET",
    "POST_COMPACT_MAX_TOKENS_PER_FILE", "TIME_BASED_MC_CLEARED_MESSAGE",
    "IMAGE_MAX_TOKEN_SIZE", "COMPACTABLE_TOOLS",
    "ERROR_MESSAGE_NOT_ENOUGH_MESSAGES", "ERROR_MESSAGE_PROMPT_TOO_LONG",
    # estimation
    "rough_token_count", "calculate_tool_result_tokens",
    "estimate_message_tokens", "token_count_with_estimation",
    # grouping
    "group_messages_by_api_round",
    # microcompact
    "microcompact", "collect_compactable_tool_ids",
    "reset_microcompact_state", "consume_pending_cache_edits",
    "get_pinned_cache_edits", "pin_cache_edits", "mark_tools_sent_to_api_state",
    # thresholds
    "get_effective_context_window_size", "get_auto_compact_threshold",
    "calculate_token_warning_state", "is_auto_compact_enabled",
    "should_auto_compact",
    # prompts
    "get_compact_prompt", "get_partial_compact_prompt",
    "format_compact_summary", "get_compact_user_summary_message",
    # results
    "CompactionResult", "build_post_compact_messages",
    # stripping
    "strip_images_from_messages", "strip_reinjected_attachments",
    "truncate_head_for_ptl_retry",
    # pipeline
    "compact_conversation", "autocompact",
    # stubs
    "try_reactive_compact", "recover_from_overflow", "snip_compact_if_needed",
    "apply_collapses_if_needed", "try_session_memory_compaction",
]