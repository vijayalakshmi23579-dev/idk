"""
classifier.py — auto-mode two-stage LLM classifier.

Extracted from src/utils/permissions/yoloClassifier.ts (simplified from
the two-stage XML-response architecture).

Auto mode uses this to decide whether an action is safe. It's NOT a regex
matcher — it's an LLM call that sees:
  - The last 20 conversation messages
  - A short description of the proposed action
  - The available tools
  - (In production, also: permission context, CLAUDE.md)

Two stages:
  Stage 1 (fast): max_tokens=64, return <decision>allow/deny</decision>
  Stage 2 (thinking): only if Stage 1 denied, use CoT

Fail-closed contract: on any API error, unavailable=True and should_block=False.
The CALLER (permissions.py) decides whether to fail open (ask human) or
fail closed (deny). Permissions.py fails closed by default.

Uses LiteLLM so it shares the same provider set as the main model.
"""
from __future__ import annotations

import json
import logging
import os
import re
import time
from typing import Any

import litellm


logger = logging.getLogger(__name__)


# ── Configuration ─────────────────────────────────────────────────────────

CLASSIFIER_MODEL = os.environ.get(
    "CLASSIFIER_MODEL",
    "groq/llama-3.1-8b-instant",   # cheap + fast by default
)

STAGE_1_MAX_TOKENS = 64
STAGE_2_MAX_TOKENS = 2048
MAX_TRANSCRIPT_CHARS = 8_000
MAX_MESSAGES_FOR_TRANSCRIPT = 20


# ── System prompts ────────────────────────────────────────────────────────

STAGE_1_SYSTEM = """You are a safety classifier for an AI coding agent. Decide if the proposed tool action is SAFE or UNSAFE given the conversation context.

SAFE if:
- Read-only (reading files, searching, checking status)
- Modifies files ONLY inside the working directory
- Runs common dev commands (npm, pip, make, git, pytest, etc.)
- Well-known, non-destructive operation

UNSAFE if:
- Deletes/overwrites files outside working directory
- Runs `rm -rf` on system paths
- Exposes secrets, credentials, or .env files externally
- Makes network calls with sensitive data to unknown hosts
- Modifies shell configs, .git internals, or .claude/
- Uses obfuscation, encoding tricks, or suspicious path patterns
- Outside the intent of the user's original request

Respond with EXACTLY one of these tags and nothing else:
<decision>allow</decision>
<decision>deny</decision>
"""

STAGE_2_SYSTEM = """You are a safety classifier for an AI coding agent. Stage 1 marked a proposed action as UNSAFE. Reconsider carefully with full reasoning.

An action is SAFE if:
- Read-only OR well-contained within the working directory
- Common development operation
- Intent matches the user's request

An action is UNSAFE if:
- Irreversible harm outside the working directory
- Improper access to secrets/credentials
- Outside scope of the user's request
- Any bypass technique (encoding, obfuscation, path tricks)

Think step by step:
1. What does the action actually do?
2. What's the user's intent (from conversation)?
3. Could it cause harm the user wouldn't want?
4. Any red flags?

Then respond EXACTLY:
<reasoning>Short reasoning here</reasoning>
<decision>allow</decision>
or
<decision>deny</decision>
"""


# ── Action formatting ────────────────────────────────────────────────────

def format_action_for_classifier(tool_name: str, tool_input: dict) -> str:
    """Turn a tool call into a human-readable action description.

    Truncated to ~1000 chars to keep the classifier prompt bounded.
    """
    try:
        input_str = json.dumps(tool_input, default=str)
    except Exception:
        input_str = str(tool_input)
    if len(input_str) > 1000:
        input_str = input_str[:1000] + "..."
    return f"{tool_name}: {input_str}"


# ── Transcript formatting ────────────────────────────────────────────────

def _format_transcript(messages: list, max_messages: int = MAX_MESSAGES_FOR_TRANSCRIPT) -> tuple[str, bool]:
    """Format recent messages as transcript for the classifier.

    Returns (transcript_text, transcript_too_long). If the formatted text
    exceeds MAX_TRANSCRIPT_CHARS, transcript_too_long is True.
    """
    recent = messages[-max_messages:] if len(messages) > max_messages else messages

    parts: list[str] = []
    for m in recent:
        mtype = getattr(m, "type", None)
        if mtype == "user":
            content = getattr(m, "content", "")
            if isinstance(content, str):
                parts.append(f"USER: {content[:500]}")
        elif mtype == "assistant":
            content = getattr(m, "content", "") or ""
            if content:
                parts.append(f"ASSISTANT: {content[:500]}")
            for tc in (getattr(m, "tool_calls", None) or []):
                parts.append(f"  → CALLED: {tc.name}")
        elif mtype == "tool_result":
            content = getattr(m, "content", "")[:200]
            parts.append(f"  RESULT: {content}")

    text = "\n".join(parts)
    too_long = len(text) > MAX_TRANSCRIPT_CHARS
    if too_long:
        text = text[:MAX_TRANSCRIPT_CHARS] + "\n... (truncated)"
    return text, too_long


# ── Response parsing ─────────────────────────────────────────────────────

_DECISION_RE = re.compile(r"<decision>\s*(allow|deny)\s*</decision>", re.IGNORECASE)
_REASONING_RE = re.compile(r"<reasoning>\s*(.+?)\s*</reasoning>", re.DOTALL | re.IGNORECASE)


def _parse_decision(text: str) -> tuple[bool, str]:
    """Parse <decision> from classifier response. Fail-closed on parse error.

    Returns (should_block, reason). should_block=True means DENY.
    """
    m = _DECISION_RE.search(text)
    if not m:
        return True, "Classifier response missing decision tag (defaulting to deny)"

    decision = m.group(1).lower()
    should_block = decision == "deny"

    rm = _REASONING_RE.search(text)
    reason = rm.group(1).strip() if rm else ""

    if not reason:
        reason = ("Action deemed unsafe by classifier" if should_block
                  else "Action deemed safe")

    return should_block, reason


# ── Usage extraction ─────────────────────────────────────────────────────

def _extract_usage(response: Any) -> dict | None:
    """Pull token usage from the LiteLLM response."""
    usage = getattr(response, "usage", None)
    if usage is None:
        return None
    return {
        "input_tokens": getattr(usage, "prompt_tokens", 0) or 0,
        "output_tokens": getattr(usage, "completion_tokens", 0) or 0,
        "cache_read_input_tokens": getattr(usage, "cache_read_input_tokens", 0) or 0,
        "cache_creation_input_tokens": getattr(usage, "cache_creation_input_tokens", 0) or 0,
    }


def _empty_result(start: float, reason: str, **overrides) -> dict:
    """Build a fail-safe result dict."""
    result = {
        "should_block": False,
        "reason": reason,
        "unavailable": True,
        "transcript_too_long": False,
        "model": CLASSIFIER_MODEL,
        "stage": "unavailable",
        "usage": None,
        "stage1_usage": None,
        "stage2_usage": None,
        "duration_ms": (time.time() - start) * 1000,
    }
    result.update(overrides)
    return result


# ── The two-stage classifier ─────────────────────────────────────────────

async def classify_yolo_action(
    messages: list,
    action: str,
    tools: list,
    context: Any,
    signal: Any = None,
) -> dict:
    """Two-stage classifier. Returns a YoloClassifierResult-shaped dict.

    Stage 1: fast binary decision (64 tokens).
    Stage 2: only if Stage 1 denied, run CoT reasoning.

    Fail-closed contract: on any API error, unavailable=True and
    should_block=False — the CALLER decides whether to fail open or closed.
    """
    start = time.time()

    transcript, transcript_too_long = _format_transcript(messages)
    if transcript_too_long:
        return _empty_result(
            start,
            "Transcript too long for classifier context window",
            unavailable=False,
            transcript_too_long=True,
        )

    classifier_user = (
        f"CONVERSATION TRANSCRIPT:\n{transcript}\n\n"
        f"PROPOSED ACTION:\n{action}\n\n"
        f"AVAILABLE TOOLS: {', '.join(t.name for t in tools[:20])}\n\n"
        f"Decide: is this action SAFE or UNSAFE?"
    )

    # ── STAGE 1: FAST ──
    try:
        from api import (
            _estimate_message_tokens,
            _usage_token_total,
            provider_request_slot,
        )

        stage1_messages = [
            {"role": "system", "content": STAGE_1_SYSTEM},
            {"role": "user", "content": classifier_user},
        ]
        async with provider_request_slot(
            CLASSIFIER_MODEL,
            estimated_tokens=(
                _estimate_message_tokens(stage1_messages) + STAGE_1_MAX_TOKENS
            ),
        ) as stage1_reservation:
            stage1_response = await litellm.acompletion(
                model=CLASSIFIER_MODEL,
                max_tokens=STAGE_1_MAX_TOKENS,
                temperature=0,
                messages=stage1_messages,
                timeout=10,
            )
    except Exception as e:
        logger.warning("Classifier stage 1 failed: %s", e)
        return _empty_result(start, f"Classifier stage 1 failed: {e}")

    stage1_text = stage1_response.choices[0].message.content or ""
    stage1_usage = _extract_usage(stage1_response)
    actual_tokens = _usage_token_total(stage1_usage)
    if actual_tokens is not None:
        await stage1_reservation.settle(actual_tokens)

    should_block, reason = _parse_decision(stage1_text)

    # If Stage 1 allowed, return immediately
    if not should_block:
        return {
            "should_block": False,
            "reason": reason,
            "unavailable": False,
            "transcript_too_long": False,
            "model": CLASSIFIER_MODEL,
            "stage": "fast",
            "usage": stage1_usage,
            "stage1_usage": stage1_usage,
            "stage2_usage": None,
            "duration_ms": (time.time() - start) * 1000,
        }

    # ── STAGE 2: THINKING (only if Stage 1 denied) ──
    try:
        stage2_messages = [
            {"role": "system", "content": STAGE_2_SYSTEM},
            {"role": "user", "content": classifier_user},
        ]
        async with provider_request_slot(
            CLASSIFIER_MODEL,
            estimated_tokens=(
                _estimate_message_tokens(stage2_messages) + STAGE_2_MAX_TOKENS
            ),
        ) as stage2_reservation:
            stage2_response = await litellm.acompletion(
                model=CLASSIFIER_MODEL,
                max_tokens=STAGE_2_MAX_TOKENS,
                temperature=0,
                messages=stage2_messages,
                timeout=20,
            )
    except Exception as e:
        # Stage 1 denied, Stage 2 failed → trust Stage 1 (fail-closed)
        logger.warning("Classifier stage 2 failed: %s", e)
        return {
            "should_block": True,
            "reason": f"{reason} (Stage 2 unavailable: {e})",
            "unavailable": False,
            "transcript_too_long": False,
            "model": CLASSIFIER_MODEL,
            "stage": "fast",
            "usage": stage1_usage,
            "stage1_usage": stage1_usage,
            "stage2_usage": None,
            "duration_ms": (time.time() - start) * 1000,
        }

    stage2_text = stage2_response.choices[0].message.content or ""
    stage2_usage = _extract_usage(stage2_response)
    actual_tokens = _usage_token_total(stage2_usage)
    if actual_tokens is not None:
        await stage2_reservation.settle(actual_tokens)
    stage2_should_block, stage2_reason = _parse_decision(stage2_text)

    # Combine usage
    combined_usage = {
        k: (stage1_usage or {}).get(k, 0) + (stage2_usage or {}).get(k, 0)
        for k in ("input_tokens", "output_tokens",
                  "cache_read_input_tokens", "cache_creation_input_tokens")
    }

    return {
        "should_block": stage2_should_block,
        "reason": stage2_reason,
        "unavailable": False,
        "transcript_too_long": False,
        "model": CLASSIFIER_MODEL,
        "stage": "thinking",
        "usage": combined_usage,
        "stage1_usage": stage1_usage,
        "stage2_usage": stage2_usage,
        "duration_ms": (time.time() - start) * 1000,
    }