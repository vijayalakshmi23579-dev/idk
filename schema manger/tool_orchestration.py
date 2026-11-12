"""
tool_orchestration.py — serial fallback tool orchestration.

Exact port of src/services/tools/toolOrchestration.ts (64 minified lines).

This is the FALLBACK path, used when config.gates.streamingToolExecution is
False. The primary path is StreamingToolExecutor (streaming_executor.py),
which starts tools during the model stream.

Source semantics preserved exactly:
  - partition_tool_calls uses reduce: merge into last batch ONLY if both
    sides are concurrency-safe; otherwise new batch. [toolOrchestration.ts:41-43]
  - Three-layer fail-closed: parse failure → unsafe; isConcurrencySafe
    throws → unsafe; falsey → unsafe. [toolOrchestration.ts:36-40]
  - Concurrent batch: modifiers QUEUED by toolUseID, applied after the
    batch in BLOCK ORDER. Siblings see the batch's ORIGINAL context. [L16-26]
  - Serial batch: modifiers applied IMMEDIATELY per tool. [L46-52]
  - Max concurrency: env CLAUDE_CODE_MAX_TOOL_USE_CONCURRENCY, default 10.
  - setInProgressToolUseIDs add before, remove after. [L47, L57, L61-64]
"""
from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator

from execute import execute_tool
from registry import find
from tool import ToolUseContext


logger = logging.getLogger(__name__)


# ── Types ────────────────────────────────────────────────────────────────

@dataclass
class MessageUpdate:
    """Matches MessageUpdate from toolOrchestration.ts:10.

    message: the tool result message (None for the batch-boundary
             context-only yield).
    new_context: the context after this update.
    context_modifier: optional {"toolUseID": str, "modifyContext": Callable}
                      — populated only by the concurrent path so the caller
                      can queue it.
    """
    message: Any = None
    new_context: ToolUseContext | None = None
    context_modifier: dict | None = None


@dataclass
class Batch:
    is_concurrency_safe: bool
    blocks: list[dict] = field(default_factory=list)


# ── Concurrency limit (exact: toolOrchestration.ts:8-9) ─────────────────

def get_max_tool_use_concurrency() -> int:
    """parseInt(env || '', 10) || 10 — invalid/absent falls back to 10."""
    try:
        return int(os.environ.get("CLAUDE_CODE_MAX_TOOL_USE_CONCURRENCY", "")) or 10
    except (ValueError, TypeError):
        return 10


# ── Partition (exact: toolOrchestration.ts:33-43) ───────────────────────

def partition_tool_calls(
    tool_use_messages: list[dict],
    tool_use_context: ToolUseContext,
) -> list[Batch]:
    """Greedy, order-preserving, fail-closed partitioner.

    Two consecutive concurrency-safe tools merge into one batch.
    Two consecutive serial tools do NOT merge — each gets its own batch
    (reduce only merges when BOTH sides are safe).
    """
    batches: list[Batch] = []

    for tool_use in tool_use_messages:
        tool = find(tool_use.get("name", ""))
        is_safe = False

        # Layer 1: schema parse
        parsed = None
        if tool is not None:
            try:
                raw = tool_use.get("input", {})
                parsed = tool.input_schema(**raw) if raw else tool.input_schema()
            except Exception:
                parsed = None  # parse failure → unsafe

        # Layer 2+3: is_concurrency_safe may throw → unsafe; falsey → unsafe
        if parsed is not None:
            try:
                is_safe = bool(tool.is_concurrency_safe(parsed))
            except Exception:
                is_safe = False

        # Merge with last batch only if BOTH are safe
        if is_safe and batches and batches[-1].is_concurrency_safe:
            batches[-1].blocks.append(tool_use)
        else:
            batches.append(Batch(is_concurrency_safe=is_safe, blocks=[tool_use]))

    return batches


# ── The generator (exact: toolOrchestration.ts:11-30) ───────────────────

async def run_tools(
    tool_use_messages: list[dict],
    assistant_messages: list,
    can_use_tool: Any,
    tool_use_context: ToolUseContext,
) -> AsyncGenerator[MessageUpdate, None]:
    """Run tool calls in batches. Yields MessageUpdate per tool + per batch.

    Concurrent batches yield each tool's message as it completes, then a
    final context-only yield with the batch's updated context.
    Serial batches yield each tool's message with the immediately-updated
    context.
    """
    current_context = tool_use_context

    for batch in partition_tool_calls(tool_use_messages, current_context):
        if batch.is_concurrency_safe:
            queued_modifiers: dict[str, list] = {}

            async for update in run_tools_concurrently(
                batch.blocks, assistant_messages, can_use_tool, current_context,
            ):
                if update.context_modifier is not None:
                    tid = update.context_modifier["toolUseID"]
                    queued_modifiers.setdefault(tid, []).append(
                        update.context_modifier["modifyContext"]
                    )
                # NOTE: siblings see the batch's ORIGINAL context — we do not
                # update current_context mid-batch.
                yield MessageUpdate(
                    message=update.message,
                    new_context=current_context,
                )

            # Apply modifiers AFTER the batch, in BLOCK order (not completion
            # order). Exact match for toolOrchestration.ts:23-26.
            for block in batch.blocks:
                for modifier in queued_modifiers.get(block.get("id", ""), []):
                    try:
                        current_context = modifier(current_context)
                    except Exception as e:
                        logger.error("Context modifier failed: %s", e)

            yield MessageUpdate(new_context=current_context)

        else:
            async for update in run_tools_serially(
                batch.blocks, assistant_messages, can_use_tool, current_context,
            ):
                if update.new_context is not None:
                    current_context = update.new_context
                yield MessageUpdate(
                    message=update.message,
                    new_context=current_context,
                )


# ── Serial path (exact: toolOrchestration.ts:43-53) ─────────────────────

async def run_tools_serially(
    tool_use_messages: list[dict],
    assistant_messages: list,
    can_use_tool: Any,
    tool_use_context: ToolUseContext,
) -> AsyncGenerator[MessageUpdate, None]:
    current_context = tool_use_context

    for tool_use in tool_use_messages:
        _add_in_progress(current_context, tool_use.get("id", ""))
        try:
            parent = _find_parent_message(assistant_messages, tool_use.get("id", ""))
            result = await execute_tool(
                tool_use, current_context, can_use_tool, parent,
            )
            # Apply context modifier IMMEDIATELY (serial = exclusive).
            if result.context_modifier is not None:
                try:
                    current_context = result.context_modifier(current_context)
                except Exception as e:
                    logger.error("Context modifier failed: %s", e)
            yield MessageUpdate(
                message=result.result_message, new_context=current_context,
            )
        finally:
            _mark_complete(current_context, tool_use.get("id", ""))


# ── Concurrent path (exact: toolOrchestration.ts:53-61) ─────────────────

async def run_tools_concurrently(
    tool_use_messages: list[dict],
    assistant_messages: list,
    can_use_tool: Any,
    tool_use_context: ToolUseContext,
) -> AsyncGenerator[MessageUpdate, None]:
    """Bounded-parallel. Yields updates in COMPLETION order.

    The caller (run_tools) handles ordering: it queues modifiers and applies
    them in block order after the batch finishes.
    """
    limit = get_max_tool_use_concurrency()
    sem = asyncio.Semaphore(limit)

    async def run_one(tool_use: dict):
        async with sem:
            _add_in_progress(tool_use_context, tool_use.get("id", ""))
            try:
                parent = _find_parent_message(
                    assistant_messages, tool_use.get("id", ""),
                )
                result = await execute_tool(
                    tool_use, tool_use_context, can_use_tool, parent,
                )
                return result
            finally:
                _mark_complete(tool_use_context, tool_use.get("id", ""))

    tasks = [asyncio.create_task(run_one(tu)) for tu in tool_use_messages]
    try:
        for coro in asyncio.as_completed(tasks):
            result = await coro
            modifier_payload = None
            if result.context_modifier is not None:
                modifier_payload = {
                    "toolUseID": result.tool_use_id,
                    "modifyContext": result.context_modifier,
                }
            yield MessageUpdate(
                message=result.result_message,
                new_context=tool_use_context,   # batch's original context
                context_modifier=modifier_payload,
            )
    finally:
        for t in tasks:
            if not t.done():
                t.cancel()


# ── Helpers ─────────────────────────────────────────────────────────────

def _find_parent_message(assistant_messages: list, tool_use_id: str) -> Any:
    """Find the assistant message containing the tool_use block with this id.
    Exact match for assistantMessages.find(...) at toolOrchestration.ts:48-50.
    """
    for m in assistant_messages or []:
        if getattr(m, "type", None) != "assistant":
            continue
        for tc in (getattr(m, "tool_calls", None) or []):
            if getattr(tc, "id", None) == tool_use_id:
                return m
    return None


def _add_in_progress(ctx: ToolUseContext, tool_use_id: str) -> None:
    setter = getattr(ctx, "set_in_progress_tool_use_ids", None)
    if setter is not None:
        try:
            setter(lambda prev: {*prev, tool_use_id})
        except Exception:
            pass


def _mark_complete(ctx: ToolUseContext, tool_use_id: str) -> None:
    setter = getattr(ctx, "set_in_progress_tool_use_ids", None)
    if setter is not None:
        try:
            setter(lambda prev: {x for x in prev if x != tool_use_id})
        except Exception:
            pass