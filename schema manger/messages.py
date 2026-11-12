"""
messages.py — the shapes that flow through the agent.

FULL message hierarchy extracted from src/utils/messages.ts:

  UserMessage        — human input, tool results, attachment content
  AssistantMessage   — model output (text, thinking, tool_use blocks)
  ProgressMessage    — streaming progress events
  SystemMessage      — status (compact boundaries, API errors, etc.)
  AttachmentMessage  — contextual injections (file changes, reminders)
  ToolUseSummaryMessage — SDK-only progress summaries

Everything converts to OpenAI format via to_openai() — LiteLLM handles
the final provider translation.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal
from uuid import uuid4


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _uuid() -> str:
    return str(uuid4())


# ── Content blocks (protocol representation) ──────────────────────────────

@dataclass
class TextBlock:
    text: str
    type: Literal["text"] = "text"


@dataclass
class ToolUseBlock:
    id: str
    name: str
    input: dict
    type: Literal["tool_use"] = "tool_use"


@dataclass
class ToolResultBlock:
    tool_use_id: str
    content: str | list
    is_error: bool = False
    type: Literal["tool_result"] = "tool_result"


@dataclass
class ThinkingBlock:
    thinking: str
    type: Literal["thinking"] = "thinking"


# ── Message types ──────────────────────────────────────────────────────────

@dataclass
class UserMessage:
    content: str | list  # string OR list of ContentBlockParam
    type: Literal["user"] = "user"
    uuid: str = field(default_factory=_uuid)
    timestamp: str = field(default_factory=_now)
    is_meta: bool = False
    is_virtual: bool = False
    is_compact_summary: bool = False
    summarize_metadata: dict | None = None
    tool_use_result: Any = None
    mcp_meta: dict | None = None
    image_paste_ids: list[int] | None = None
    source_tool_assistant_uuid: str | None = None
    permission_mode: str | None = None
    origin: dict | None = None



@dataclass
class AssistantMessage:
    content: str | None = None
    tool_calls: list[ToolUseBlock] = field(default_factory=list)
    thinking_blocks: list[ThinkingBlock] = field(default_factory=list)
    stop_reason: str | None = None
    usage: dict | None = None
    type: Literal["assistant"] = "assistant"
    uuid: str = field(default_factory=_uuid)
    timestamp: str = field(default_factory=_now)
    is_virtual: bool = False
    is_api_error: bool = False          # ← NEW
    api_error: str | None = None        # ← NEW (e.g. "max_output_tokens", "invalid_request")


@dataclass
class ToolResultMessage:
    tool_use_id: str
    content: str
    is_error: bool = False
    type: Literal["tool_result"] = "tool_result"
    uuid: str = field(default_factory=_uuid)
    timestamp: str = field(default_factory=_now)


@dataclass
class ProgressMessage:
    tool_use_id: str
    parent_tool_use_id: str
    data: dict
    type: Literal["progress"] = "progress"
    uuid: str = field(default_factory=_uuid)
    timestamp: str = field(default_factory=_now)


@dataclass
class SystemMessage:
    subtype: str
    content: str = ""
    level: str = "info"
    metadata: dict | None = None
    type: Literal["system"] = "system"
    uuid: str = field(default_factory=_uuid)
    timestamp: str = field(default_factory=_now)
    is_meta: bool = False


@dataclass
class AttachmentMessage:
    attachment: dict  # {"type": "edited_text_file", "filename": ..., ...}
    type: Literal["attachment"] = "attachment"
    uuid: str = field(default_factory=_uuid)
    timestamp: str = field(default_factory=_now)


@dataclass
class ToolUseSummaryMessage:
    summary: str
    preceding_tool_use_ids: list[str]
    type: Literal["tool_use_summary"] = "tool_use_summary"
    uuid: str = field(default_factory=_uuid)
    timestamp: str = field(default_factory=_now)


# ── Factories ──────────────────────────────────────────────────────────────

def create_user_message(
    content: str | list,
    is_meta: bool = False,
    is_virtual: bool = False,
    is_compact_summary: bool = False,
    summarize_metadata: dict | None = None,
    tool_use_result: Any = None,
    source_tool_assistant_uuid: str | None = None,
    origin: dict | None = None,
) -> UserMessage:
    return UserMessage(
        content=content or "(empty)",
        is_meta=is_meta,
        is_virtual=is_virtual,
        is_compact_summary=is_compact_summary,
        summarize_metadata=summarize_metadata,
        tool_use_result=tool_use_result,
        source_tool_assistant_uuid=source_tool_assistant_uuid,
        origin=origin,
    )

def create_assistant_message(
    content: str | None = None,
    tool_calls: list[ToolUseBlock] | None = None,
    stop_reason: str | None = None,
    usage: dict | None = None,
) -> AssistantMessage:
    return AssistantMessage(
        content=content,
        tool_calls=tool_calls or [],
        stop_reason=stop_reason,
        usage=usage,
    )


def create_tool_result(
    tool_use_id: str,
    content: str,
    is_error: bool = False,
) -> ToolResultMessage:
    return ToolResultMessage(
        tool_use_id=tool_use_id,
        content=content,
        is_error=is_error,
    )


def create_progress_message(
    tool_use_id: str,
    parent_tool_use_id: str,
    data: dict,
) -> ProgressMessage:
    return ProgressMessage(
        tool_use_id=tool_use_id,
        parent_tool_use_id=parent_tool_use_id,
        data=data,
    )


def create_compact_boundary_message(
    trigger: str,
    pre_tokens: int,
    messages_summarized: int | None = None,
) -> SystemMessage:
    return SystemMessage(
        subtype="compact_boundary",
        content="Conversation compacted",
        metadata={
            "trigger": trigger,
            "pre_tokens": pre_tokens,
            "messages_summarized": messages_summarized,
        },
    )


def create_api_error_message(error: str, retry_attempt: int = 0) -> SystemMessage:
    return SystemMessage(
        subtype="api_error",
        content=error,
        level="error",
        metadata={"retry_attempt": retry_attempt},
    )


# ── OpenAI format conversion (what LiteLLM expects) ──────────────────────

def to_openai(messages: list) -> list[dict]:
    """
    Convert our message hierarchy to OpenAI chat-completions format.

    Handles:
      - UserMessage → {"role": "user", "content": "..."}
      - Meta UserMessages → {"role": "system", "content": "<system-reminder>...</system-reminder>"}
      - AssistantMessage → {"role": "assistant", "content": ..., "tool_calls": [...]}
      - ToolResultMessage → {"role": "tool", "tool_call_id": "...", "content": "..."}
      - ProgressMessage → skipped (not API-bound)
      - SystemMessage → skipped except compact_boundary (inlined as system msg)
      - AttachmentMessage → inlined as system-reminder wrapped user message

    Merges adjacent same-role messages (some providers require strict
    alternation; OpenAI allows same-role but merging prevents double-token).
    """
    out: list[dict] = []

    for m in messages:
        if m.type == "user":
            if m.is_virtual:
                continue
            if m.is_meta:
                # Meta messages become system messages
                text = m.content if isinstance(m.content, str) else _blocks_to_text(m.content)
                out.append({"role": "system", "content": text})
            else:
                if isinstance(m.content, str):
                    out.append({"role": "user", "content": m.content})
                else:
                    out.append({"role": "user", "content": _blocks_to_openai(m.content)})

        elif m.type == "assistant":
            if m.is_virtual:
                continue
            msg: dict = {"role": "assistant"}
            msg["content"] = m.content if m.content else None
            if m.tool_calls:
                msg["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": json.dumps(tc.input),
                        },
                    }
                    for tc in m.tool_calls
                ]
            out.append(msg)

        elif m.type == "tool_result":
            content = m.content
            if m.is_error:
                content = f"ERROR: {content}"
            out.append({
                "role": "tool",
                "tool_call_id": m.tool_use_id,
                "content": content,
            })

        elif m.type == "system":
            if m.subtype == "compact_boundary":
                out.append({
                    "role": "system",
                    "content": f"[Previous conversation compacted: {m.content}]",
                })
            elif m.subtype == "api_error":
                out.append({
                    "role": "system",
                    "content": f"[API error: {m.content}]",
                })
            # Other subtypes skipped — not relevant to model context

        elif m.type == "attachment":
            # Attachments become system-reminder user messages
            att = m.attachment
            text = _format_attachment(att)
            if text:
                out.append({
                    "role": "system",
                    "content": f"<system-reminder>\n{text}\n</system-reminder>",
                })

        # ProgressMessage and ToolUseSummaryMessage are not API-bound

    return _merge_adjacent_same_role(out)


def _blocks_to_text(blocks: list) -> str:
    parts = []
    for b in blocks:
        if isinstance(b, dict):
            if b.get("type") == "text":
                parts.append(b.get("text", ""))
        elif hasattr(b, "text"):
            parts.append(b.text)
    return "\n".join(parts)


def _blocks_to_openai(blocks: list) -> list[dict]:
    """Convert content blocks to OpenAI content array."""
    result = []
    for b in blocks:
        if isinstance(b, dict):
            if b.get("type") == "text":
                result.append({"type": "text", "text": b.get("text", "")})
            elif b.get("type") == "image":
                result.append({"type": "image_url", "image_url": b.get("image_url", {})})
        elif hasattr(b, "type") and b.type == "text":
            result.append({"type": "text", "text": b.text})
    return result


def _format_attachment(att: dict) -> str:
    att_type = att.get("type", "")
    if att_type == "edited_text_file":
        return (
            f"Note: {att.get('filename')} was modified, either by the user or "
            f"by a linter. This change was intentional, so make sure to take "
            f"it into account as you proceed.\n{att.get('snippet', '')}"
        )
    if att_type == "directory":
        return f"Contents of {att.get('path')}:\n{att.get('content', '')}"
    if att_type == "file":
        return f"Contents of {att.get('filename')}:\n{att.get('content', '')}"
    if att_type == "nested_memory":
        c = att.get("content", {})
        return f"Contents of {c.get('path')}:\n\n{c.get('content', '')}"
    if att_type == "relevant_memories":
        parts = []
        for m in att.get("memories", []):
            parts.append(m.get("content", ""))
        return "\n\n".join(parts)
    if att_type == "task_status":
        return (
            f"Task {att.get('taskId')} ({att.get('status')}): "
            f"{att.get('description', '')}"
        )
    return ""


def _merge_adjacent_same_role(msgs: list[dict]) -> list[dict]:
    """Merge consecutive same-role messages. User/system: concatenate text.
    Assistant: concatenate content + tool_calls arrays (needed because our
    streaming adapter yields one assistant message per content block)."""
    if not msgs:
        return msgs
    out = [msgs[0]]
    for m in msgs[1:]:
        prev = out[-1]
        if prev["role"] != m["role"]:
            out.append(m)
            continue
        if m["role"] in ("user", "system"):
            pc = prev.get("content", "")
            nc = m.get("content", "")
            if isinstance(pc, str) and isinstance(nc, str):
                out[-1] = {**prev, "content": f"{pc}\n\n{nc}"}
            else:
                out.append(m)
        elif m["role"] == "assistant":
            merged = dict(prev)
            if m.get("content"):
                merged["content"] = (merged.get("content") or "") + (m["content"] or "")
            if m.get("tool_calls"):
                merged["tool_calls"] = list(merged.get("tool_calls") or []) + list(m["tool_calls"])
            out[-1] = merged
        else:
            out.append(m)
    return out


# ── Message accessors ──────────────────────────────────────────────────────

def get_text(message: Any) -> str | None:
    """Extract text from any message type."""
    if message.type == "user":
        if isinstance(message.content, str):
            return message.content
        return _blocks_to_text(message.content) or None
    if message.type == "assistant":
        return message.content
    if message.type == "tool_result":
        return message.content
    if message.type == "system":
        return message.content
    return None


def get_tool_use_ids(messages: list) -> set[str]:
    """All tool_use ids present in assistant messages."""
    ids: set[str] = set()
    for m in messages:
        if m.type == "assistant" and m.tool_calls:
            for tc in m.tool_calls:
                ids.add(tc.id)
    return ids


def get_tool_result_ids(messages: list) -> set[str]:
    """All tool_result ids present."""
    return {m.tool_use_id for m in messages if m.type == "tool_result"}


def ensure_tool_result_pairing(messages: list) -> list:
    """
    Every tool_use must have a matching tool_result. Inserts a synthetic
    error result for missing pairs. Extracted from ensureToolResultPairing
    in messages.ts.
    """
    tool_use_ids = get_tool_use_ids(messages)
    tool_result_ids = get_tool_result_ids(messages)

    missing = tool_use_ids - tool_result_ids
    if not missing:
        return messages

    result = list(messages)
    for missing_id in missing:
        result.append(
            create_tool_result(
                tool_use_id=missing_id,
                content="[Tool result missing due to internal error]",
                is_error=True,
            )
        )
    return result


def is_synthetic_message(message: Any) -> bool:
    """Detect our synthesized messages (interrupt, cancel, etc.)."""
    synthetic_texts = {
        "[Request interrupted by user]",
        "[Request interrupted by user for tool use]",
        "No response requested.",
    }
    text = get_text(message)
    return text in synthetic_texts if text else False