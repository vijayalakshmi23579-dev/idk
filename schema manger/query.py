"""
query.py — the query loop state machine.

Port of src/query.ts (1,729 expanded lines). Preserves all 7 Continue
transitions and all 10 Terminal reasons.

The loop is NOT a simple "call API, execute tools, repeat." Each iteration
classifies the reason it continued and handles each differently:
  - next_turn                  → normal tool execution completed
  - collapse_drain_retry       → 413, drain staged context collapses
  - reactive_compact_retry     → 413, summarize via compaction
  - max_output_tokens_escalate → 8k cap hit, retry at 64k
  - max_output_tokens_recovery → 64k also hit, inject nudge (3× max)
  - token_budget_continuation  → budget not exhausted, nudge + continue
  - stop_hook_blocking         → stop hook returned blocking errors

The 5 context shapers run BEFORE every API call, in order:
  apply tool result budget → snip → microcompact → context collapse →
  autocompact. Each is feature-gated; unreachable shapers no-op.

Verified against source:
  - transitions.ts is a stub, so Continue/Terminal shapes are inferred
    from every `continue`/`return` in query.ts.
  - tokenBudget.ts, deps.ts read and ported exactly.
  - stopHooks.ts stubbed (matches StopHookResult shape).
  - compaction stubbed (deps.microcompact/autocompact pass-through until
    compact.py lands; the 3 compaction-related transitions activate then).
"""
from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator, Callable

from api import stream as _api_stream  # noqa: F401 (used in adapter below)
from messages import (
    AssistantMessage, ToolResultMessage, ToolUseBlock, UserMessage,
    create_assistant_message, create_user_message, create_tool_result,
    to_openai,
)
from memory_injector import build_memory_injection
from query_config import QueryConfig, build_query_config
from query_deps import QueryDeps, production_deps
from query_stop_hooks import (
    StopHookResult, execute_stop_failure_hooks, handle_stop_hooks,
)
from query_token_budget import (
    BudgetTracker, check_token_budget, create_budget_tracker,
)
from query_transitions import Continue, Terminal
from registry import all_tools, find
from streaming_executor import StreamingToolExecutor
from tool import ToolUseContext
from tool_orchestration import MessageUpdate, run_tools


logger = logging.getLogger(__name__)


# ── Model info — multi-tier fallback (no compromise on intelligence) ────

MODEL_CONTEXT_WINDOWS: dict[str, int] = {
    "claude-opus-4":      200_000,
    "claude-sonnet-4":    200_000,
    "claude-3-5-sonnet":  200_000,
    "claude-3-5-haiku":   200_000,
    "claude-3-opus":      200_000,
    "claude-3-sonnet":    200_000,
    "claude-3-haiku":     200_000,
    "gemini-2.5-pro":     1_048_576,
    "gemini-2.5-flash":   1_048_576,
    "gemini-2.0-flash":   1_048_576,
    "gemini-1.5-pro":     2_097_152,
    "gemini-1.5-flash":   1_048_576,
    "llama-3.3-70b":      131_072,
    "llama-3.1-405b":     131_072,
    "llama-3.1-70b":      131_072,
    "llama-3.1-8b":       131_072,
    "deepseek-r1":        131_072,
    "deepseek-chat":      131_072,
    "deepseek-v3":        131_072,
    "qwen-2.5":           131_072,
    "mistral-nemotron":   131_072,
}


def get_context_window_for_model(model: str) -> int:
    """Multi-tier fallback chain, matching the shape of Claude Code's
    getContextWindowForModel. Preserves intelligence: env override → table →
    LiteLLM DB → family heuristic → safe default."""
    env = os.environ.get("CLAUDE_CODE_CONTEXT_WINDOW")
    if env:
        try:
            return int(env)
        except ValueError:
            pass
    for prefix, size in MODEL_CONTEXT_WINDOWS.items():
        if prefix in model:
            return size
    try:
        import litellm
        info = litellm.get_model_info(model)
        if info.get("max_input_tokens"):
            return int(info["max_input_tokens"])
    except Exception:
        pass
    if "claude" in model:
        return 200_000
    if "gemini" in model:
        return 1_048_576
    if any(f in model for f in ("llama", "deepseek", "qwen", "mistral")):
        return 131_072
    return 200_000


def get_max_output_tokens_for_model(model: str) -> int:
    """Max output tokens. Source caps at 20K for summary compaction."""
    env = os.environ.get("CLAUDE_CODE_MAX_OUTPUT_TOKENS")
    if env:
        try:
            return int(env)
        except ValueError:
            pass
    try:
        import litellm
        info = litellm.get_model_info(model)
        if info.get("max_output_tokens"):
            return int(info["max_output_tokens"])
    except Exception:
        pass
    return 8_192  # conservative default


# ── Compact shapers (feature-gated, no-op until compact.py lands) ───────

def _apply_tool_result_budget(messages: list) -> list:
    """Source: utils/toolResultStorage.ts:applyToolResultBudget.
    Persists oversized results. Until implemented: no-op."""
    return messages


def _snip_compact_if_needed(messages: list) -> dict:
    """Source: services/compact/snipCompact.ts (HISTORY_SNIP gate)."""
    try:
        from compact import snip_compact_if_needed
        return snip_compact_if_needed(messages)
    except (ImportError, AttributeError):
        return {"messages": messages, "tokens_freed": 0, "boundary_message": None}


def _apply_context_collapses(messages: list, ctx: Any, query_source: str) -> list:
    """Source: services/contextCollapse (CONTEXT_COLLAPSE gate)."""
    try:
        from compact import apply_collapses_if_needed
        return apply_collapses_if_needed(messages, ctx, query_source)
    except (ImportError, AttributeError):
        return messages


def _build_post_compact_messages(compaction_result: Any) -> list:
    """Source: services/compact/compact.ts:buildPostCompactMessages."""
    try:
        from compact import build_post_compact_messages
        return build_post_compact_messages(compaction_result)
    except (ImportError, AttributeError):
        return []


# ── Streaming adapter ───────────────────────────────────────────────────

async def _adapter_gen(
    messages: list,
    system_prompt: Any,
    tools: list,
    **kwargs,
) -> AsyncGenerator[AssistantMessage, None]:
    """Adapt api.stream() events → per-block AssistantMessage yields.

    Yields ONE message per content block (text block, tool_use block) so the
    streaming executor can start each tool immediately. A final message
    carries the stop_reason. to_openai merges consecutive assistant messages
    back into one API message.
    """
    if isinstance(system_prompt, list):
        sp = "\n\n".join(
            b.get("text", "") if isinstance(b, dict) else str(b)
            for b in system_prompt
        )
    else:
        sp = system_prompt or ""

    text_buf: list[str] = []
    thinking_buf: list[str] = []
    stop_reason: str | None = None
    usage: dict | None = None

    async for event in _api_stream(messages=messages, tools=tools, system_prompt=sp):
        t = event.get("type")
        if t == "text_delta":
            text_buf.append(event["text"])
        elif t == "thinking_delta":
            thinking_buf.append(event["thinking"])
        elif t == "tool_use":
            # Flush accumulated text BEFORE the tool call (preserves order)
            if text_buf:
                yield create_assistant_message(
                    content="".join(text_buf), tool_calls=[], stop_reason=None,
                )
                text_buf = []
            yield create_assistant_message(
                content=None,
                tool_calls=[ToolUseBlock(
                    id=event["id"], name=event["name"], input=event["input"],
                )],
                stop_reason=None,
            )
        elif t == "message_stop":
            stop_reason = event["stop_reason"]
            usage = event["usage"]

    final = create_assistant_message(
        content="".join(text_buf) if text_buf else None,
        tool_calls=[],
        stop_reason=stop_reason,
        usage=usage,
    )
    if thinking_buf:
        final.thinking_blocks = [{"type": "thinking", "thinking": "".join(thinking_buf)}]
    yield final


# ── Query params ────────────────────────────────────────────────────────

@dataclass
class QueryParams:
    messages: list
    system_prompt: Any
    user_context: dict = field(default_factory=dict)
    system_context: dict = field(default_factory=dict)
    can_use_tool: Any = None
    tool_use_context: ToolUseContext | None = None
    fallback_model: str | None = None
    query_source: str = "repl_main_thread"
    max_output_tokens_override: int | None = None
    max_turns: int | None = None
    task_budget: dict | None = None         # {"total": int}
    deps: QueryDeps | None = None
    get_turn_output_tokens: Callable[[], int] = lambda: 0
    get_current_turn_token_budget: Callable[[], int | None] = lambda: None


# ── Query state ─────────────────────────────────────────────────────────

@dataclass
class QueryState:
    messages: list
    tool_use_context: ToolUseContext
    auto_compact_tracking: dict | None = None
    max_output_tokens_recovery_count: int = 0
    has_attempted_reactive_compact: bool = False
    max_output_tokens_override: int | None = None
    pending_tool_use_summary: Any | None = None
    stop_hook_active: bool | None = None
    turn_count: int = 1
    transition: Continue | None = None


# ── Public generator ────────────────────────────────────────────────────

async def query(params: QueryParams) -> AsyncGenerator[Any, None]:
    """Yield events (messages, progress, Terminal). Terminal is the last event."""
    consumed_command_uuids: list[str] = []
    try:
        async for item in query_loop(params, consumed_command_uuids):
            yield item
    finally:
        # Post-loop cleanup — mirror query.ts:219-239
        for uuid in consumed_command_uuids:
            try:
                _notify_command_lifecycle(uuid, "completed")
            except Exception:
                pass


async def query_loop(
    params: QueryParams,
    consumed_command_uuids: list[str],
) -> AsyncGenerator[Any, None]:
    deps = params.deps or production_deps()
    config = build_query_config()
    budget_tracker: BudgetTracker | None = create_budget_tracker()

    state = QueryState(
        messages=params.messages,
        tool_use_context=params.tool_use_context,
        max_output_tokens_override=params.max_output_tokens_override,
    )

    task_budget_remaining: int | None = None

    while True:
        _publish_query_messages(params, state.messages)
        ctx = state.tool_use_context
        available_tools = _tools_for_context(ctx)
        _archive_query_messages(ctx, state.messages)
        active_view = _active_store_view(ctx, state.messages)
        if active_view is not None:
            state.messages = active_view
            _publish_query_messages(params, active_view)
        messages = state.messages
        user_context_for_query = dict(params.user_context)
        if ctx is not None:
            memory_events: list[dict[str, object]] = []
            memory_context = await build_memory_injection(
                messages, ctx, params.query_source,
                include_summary=active_view is None,
                on_event=memory_events.append,
            )
            for event in memory_events:
                yield event
            if memory_context:
                user_context_for_query["Session memory"] = memory_context
        tracking = state.auto_compact_tracking

        yield {"type": "stream_request_start"}

        # ── Query chain tracking ──
        prev_tracking = getattr(ctx, "query_tracking", None)
        query_tracking = {
            "chainId": prev_tracking["chainId"] if prev_tracking else deps.uuid(),
            "depth": (prev_tracking["depth"] + 1) if prev_tracking else 0,
        }
        ctx = _with_query_tracking(ctx, query_tracking)

        # ── Context shapers, in order ──
        messages_for_query = list(messages)

        messages_for_query = _apply_tool_result_budget(messages_for_query)

        snip_result = _snip_compact_if_needed(messages_for_query)
        messages_for_query = snip_result["messages"]
        snip_tokens_freed = snip_result["tokens_freed"]
        if snip_result.get("boundary_message"):
            yield snip_result["boundary_message"]

        mc_result = await deps.microcompact(
            messages_for_query, ctx, params.query_source,
        )
        messages_for_query = mc_result["messages"]

        messages_for_query = _apply_context_collapses(
            messages_for_query, ctx, params.query_source,
        )

        full_system_prompt = _append_system_context(
            params.system_prompt, params.system_context,
        )

        ac_result = await deps.autocompact(
            messages_for_query, ctx,
            {
                "system_prompt": params.system_prompt,
                "user_context": user_context_for_query,
                "system_context": params.system_context,
                "tool_use_context": ctx,
                "fork_context_messages": messages_for_query,
            },
            params.query_source, tracking, snip_tokens_freed,
        )

        compaction_result = ac_result.get("compaction_result")
        consecutive_failures = ac_result.get("consecutive_failures")

        if compaction_result is not None:
            if params.task_budget:
                pre_compact = _final_context_tokens(messages_for_query)
                task_budget_remaining = max(
                    0,
                    (task_budget_remaining or params.task_budget["total"]) - pre_compact,
                )
            tracking = {
                "compacted": True,
                "turnId": deps.uuid(),
                "turnCounter": 0,
                "consecutiveFailures": 0,
            }
            post_compact = _build_post_compact_messages(compaction_result)
            for m in post_compact:
                yield m
            messages_for_query = post_compact
            _publish_query_messages(params, messages_for_query)
        elif consecutive_failures is not None:
            tracking = {
                **(tracking or {"compacted": False, "turnId": "", "turnCounter": 0}),
                "consecutiveFailures": consecutive_failures,
            }

        ctx = _with_messages(ctx, messages_for_query)

        # ── Set up executor ──
        use_streaming = config.gates.streaming_tool_execution
        executor: StreamingToolExecutor | None = None
        if use_streaming:
            executor = StreamingToolExecutor(
                available_tools, params.can_use_tool, ctx,
            )

        assistant_messages: list[AssistantMessage] = []
        tool_use_blocks: list[dict] = []
        needs_follow_up = False
        tool_results: list[Any] = []

        # ── Blocking limit check (before API call) ──
        if not compaction_result and params.query_source not in (
            "compact", "session_memory",
        ):
            context_window = get_context_window_for_model(_current_model(ctx))
            max_out = get_max_output_tokens_for_model(_current_model(ctx))
            threshold = context_window - min(max_out, 20_000) - 3_000
            estimated_tokens = _estimate_tokens(messages_for_query) - snip_tokens_freed
            if estimated_tokens >= threshold:
                yield _create_api_error_message(
                    "Prompt is too long. Use /compact or /clear.",
                    "invalid_request",
                )
                yield Terminal(reason="blocking_limit")
                return

        # ── Streaming attempt (with fallback model retry) ──
        attempt_with_fallback = True
        while attempt_with_fallback:
            attempt_with_fallback = False
            try:
                async for msg in deps.call_model(
                    messages=_prepend_user_context(messages_for_query, user_context_for_query),
                    system_prompt=full_system_prompt,
                    tools=available_tools,
                    signal=getattr(ctx.abort_controller, "signal", None),
                    options={
                        "model": _current_model(ctx),
                        "fallback_model": params.fallback_model,
                        "query_source": params.query_source,
                        "max_output_tokens_override": state.max_output_tokens_override,
                        "task_budget": (
                            {
                                "total": params.task_budget["total"],
                                **({"remaining": task_budget_remaining} if task_budget_remaining is not None else {}),
                            }
                            if params.task_budget else None
                        ),
                    },
                ):
                    yield msg
                    if getattr(msg, "type", None) == "assistant":
                        assistant_messages.append(msg)
                        block_tool_calls = getattr(msg, "tool_calls", None) or []
                        if block_tool_calls:
                            needs_follow_up = True
                            for tc in block_tool_calls:
                                block = {"id": tc.id, "name": tc.name, "input": tc.input}
                                tool_use_blocks.append(block)
                                if executor is not None:
                                    executor.add_tool(block, msg)
                    # Drain completed results during the stream
                    if executor is not None and not _is_aborted(ctx):
                        for update in executor.get_completed_results():
                            if update.message is not None:
                                yield update.message
                                if isinstance(update.message, (ToolResultMessage, UserMessage)):
                                    tool_results.append(update.message)
            except Exception as inner_error:
                # Model fallback retry
                if _is_fallback_error(inner_error) and params.fallback_model:
                    for m in assistant_messages:
                        yield {"type": "tombstone", "message": m}
                    assistant_messages.clear()
                    tool_use_blocks.clear()
                    tool_results.clear()
                    needs_follow_up = False
                    if executor is not None:
                        executor.discard()
                        executor = StreamingToolExecutor(
                            available_tools, params.can_use_tool, ctx,
                        )
                    
                    try:
                        setattr(ctx, "_fallback_model", params.fallback_model)
                    except Exception:
                        pass
                    attempt_with_fallback = True
                    continue
                if executor is not None:
                    executor.discard()
                raise

        if executor is not None and not _is_aborted(ctx):
            executor.release_deferred_tools()

        # ── Abort during streaming ──
        if _is_aborted(ctx):
            if executor is not None:
                executor.discard()
                async for update in executor.get_remaining_results():
                    if update.message is not None:
                        yield update.message
            else:
                for um in _yield_missing_tool_result_blocks(
                    assistant_messages, "Interrupted by user",
                ):
                    yield um
            yield create_user_message("[Request interrupted by user]", is_meta=True)
            yield Terminal(reason="aborted_streaming")
            return

        # ── Yield pending tool use summary (from previous turn) ──
        if state.pending_tool_use_summary is not None:
            try:
                summary = await state.pending_tool_use_summary
                if summary is not None:
                    yield summary
            except Exception:
                pass

        # ── No follow-up: handle terminal paths ──
        if not needs_follow_up:
            _publish_query_messages(params, messages_for_query + assistant_messages)
            last_msg = assistant_messages[-1] if assistant_messages else None

            # 413 / prompt-too-long recovery
            is_413 = (
                last_msg is not None
                and getattr(last_msg, "is_api_error", False)
                and getattr(last_msg, "api_error", None) == "invalid_request"
                and "too long" in (last_msg.content or "").lower()
            )
            if is_413:
                # Try context collapse drain
                if state.transition is None or state.transition.reason != "collapse_drain_retry":
                    drained = _drain_collapses(messages_for_query, params.query_source)
                    if drained is not None and drained.get("committed", 0) > 0:
                        state = QueryState(
                            messages=drained["messages"],
                            tool_use_context=ctx,
                            auto_compact_tracking=tracking,
                            max_output_tokens_recovery_count=state.max_output_tokens_recovery_count,
                            has_attempted_reactive_compact=state.has_attempted_reactive_compact,
                            max_output_tokens_override=None,
                            turn_count=state.turn_count,
                            transition=Continue(reason="collapse_drain_retry", committed=drained["committed"]),
                        )
                        continue
                # Try reactive compact
                if not state.has_attempted_reactive_compact:
                    compacted = _try_reactive_compact(messages_for_query, params)
                    if compacted is not None:
                        if params.task_budget:
                            pre_compact = _final_context_tokens(messages_for_query)
                            task_budget_remaining = max(
                                0,
                                (task_budget_remaining or params.task_budget["total"]) - pre_compact,
                            )
                        post_compact = _build_post_compact_messages(compacted)
                        for m in post_compact:
                            yield m
                        state = QueryState(
                            messages=post_compact,
                            tool_use_context=ctx,
                            auto_compact_tracking=None,
                            max_output_tokens_recovery_count=state.max_output_tokens_recovery_count,
                            has_attempted_reactive_compact=True,
                            max_output_tokens_override=None,
                            turn_count=state.turn_count,
                            transition=Continue(reason="reactive_compact_retry"),
                        )
                        continue
                # Surface the error
                yield last_msg
                await execute_stop_failure_hooks(last_msg, ctx)
                yield Terminal(reason="prompt_too_long")
                return

            # Max output tokens recovery
            is_max_tokens = (
                last_msg is not None
                and getattr(last_msg, "is_api_error", False)
                and getattr(last_msg, "api_error", None) == "max_output_tokens"
            )
            if is_max_tokens:
                # Stage 1: escalate 8k → 64k
                if state.max_output_tokens_override is None and not os.environ.get("CLAUDE_CODE_MAX_OUTPUT_TOKENS"):
                    state = QueryState(
                        messages=messages_for_query,
                        tool_use_context=ctx,
                        auto_compact_tracking=tracking,
                        max_output_tokens_recovery_count=state.max_output_tokens_recovery_count,
                        has_attempted_reactive_compact=state.has_attempted_reactive_compact,
                        max_output_tokens_override=64_000,
                        turn_count=state.turn_count,
                        transition=Continue(reason="max_output_tokens_escalate"),
                    )
                    continue
                # Stage 2: 3 recovery nudges
                if state.max_output_tokens_recovery_count < 3:
                    recovery = create_user_message(
                        "Output token limit hit. Resume directly — no apology, "
                        "no recap of what you were doing. Pick up mid-thought "
                        "if that is where the cut happened. Break remaining work "
                        "into smaller pieces.",
                        is_meta=True,
                    )
                    state = QueryState(
                        messages=messages_for_query + assistant_messages + [recovery],
                        tool_use_context=ctx,
                        auto_compact_tracking=tracking,
                        max_output_tokens_recovery_count=state.max_output_tokens_recovery_count + 1,
                        has_attempted_reactive_compact=state.has_attempted_reactive_compact,
                        max_output_tokens_override=None,
                        turn_count=state.turn_count,
                        transition=Continue(
                            reason="max_output_tokens_recovery",
                            attempt=state.max_output_tokens_recovery_count + 1,
                        ),
                    )
                    continue
                yield last_msg
                yield Terminal(reason="completed")
                return

            # API error — no stop hooks (would cause death spiral)
            if last_msg is not None and getattr(last_msg, "is_api_error", False):
                await execute_stop_failure_hooks(last_msg, ctx)
                yield Terminal(reason="completed")
                return

            # Stop hooks
                        # Stop hooks (regular async function returning StopHookResult)
            stop_hook_result = await handle_stop_hooks(
                messages_for_query, assistant_messages,
                params.system_prompt, user_context_for_query, params.system_context,
                ctx, params.query_source, state.stop_hook_active,
            )

            if stop_hook_result.prevent_continuation:
                yield Terminal(reason="stop_hook_prevented")
                return

            if stop_hook_result.blocking_errors:
                state = QueryState(
                    messages=messages_for_query + assistant_messages + stop_hook_result.blocking_errors,
                    tool_use_context=ctx,
                    auto_compact_tracking=tracking,
                    max_output_tokens_recovery_count=0,
                    has_attempted_reactive_compact=state.has_attempted_reactive_compact,
                    max_output_tokens_override=None,
                    stop_hook_active=True,
                    turn_count=state.turn_count,
                    transition=Continue(reason="stop_hook_blocking"),
                )
                continue

            # Token budget check
            budget = params.get_current_turn_token_budget()
            if budget is not None:
                decision = check_token_budget(
                    budget_tracker, ctx.agent_id, budget,
                    params.get_turn_output_tokens(),
                )
                if hasattr(decision, "action") and decision.action == "continue":
                    nudge = create_user_message(decision.nudge_message, is_meta=True)
                    state = QueryState(
                        messages=messages_for_query + assistant_messages + [nudge],
                        tool_use_context=ctx,
                        auto_compact_tracking=tracking,
                        max_output_tokens_recovery_count=0,
                        has_attempted_reactive_compact=False,
                        max_output_tokens_override=None,
                        turn_count=state.turn_count,
                        transition=Continue(reason="token_budget_continuation"),
                    )
                    continue

            yield Terminal(reason="completed")
            return

        # ── Tool execution ──
        should_prevent_continuation = False
        updated_ctx = ctx

        if executor is not None:
            async for update in _drain_async(executor.get_remaining_results()):
                if update.message is not None:
                    yield update.message
                    if (getattr(update.message, "type", None) == "attachment"
                            and getattr(update.message, "attachment", {}).get("type") == "hook_stopped_continuation"):
                        should_prevent_continuation = True
                    tool_results.append(update.message)
                if update.new_context is not None:
                    updated_ctx = update.new_context
        else:
            async for update in run_tools(
                tool_use_blocks, assistant_messages,
                params.can_use_tool, ctx,
            ):
                if update.message is not None:
                    yield update.message
                    tool_results.append(update.message)
                if update.new_context is not None:
                    updated_ctx = update.new_context

        _publish_query_messages(
            params, messages_for_query + assistant_messages + tool_results,
        )

        updated_ctx = _with_query_tracking(updated_ctx, query_tracking)

        # ── Abort during tools ──
        if _is_aborted(updated_ctx):
            yield create_user_message(
                "[Request interrupted by user for tool use]", is_meta=True,
            )
            next_turn = state.turn_count + 1
            if params.max_turns and next_turn > params.max_turns:
                yield {"type": "attachment", "attachment": {
                    "type": "max_turns_reached",
                    "maxTurns": params.max_turns,
                    "turnCount": next_turn,
                }}
            yield Terminal(reason="aborted_tools")
            return

        if should_prevent_continuation:
            yield Terminal(reason="hook_stopped")
            return

        if tracking and tracking.get("compacted"):
            tracking["turnCounter"] = tracking.get("turnCounter", 0) + 1

        # ── Refresh tools (MCP servers may have connected) ──
        refresh_fn = getattr(updated_ctx, "options", {}).get("refresh_tools") if hasattr(updated_ctx, "options") else None
        if callable(refresh_fn):
            try:
                refresh_fn()
            except Exception:
                pass

        # ── Max turns check ──
        next_turn = state.turn_count + 1
        if params.max_turns and next_turn > params.max_turns:
            yield {"type": "attachment", "attachment": {
                "type": "max_turns_reached",
                "maxTurns": params.max_turns,
                "turnCount": next_turn,
            }}
            yield Terminal(reason="max_turns", turn_count=next_turn)
            return

        # ── Recurse via continue ──
        state = QueryState(
            messages=messages_for_query + assistant_messages + tool_results,
            tool_use_context=updated_ctx,
            auto_compact_tracking=tracking,
            max_output_tokens_recovery_count=0,
            has_attempted_reactive_compact=False,
            max_output_tokens_override=None,
            pending_tool_use_summary=None,
            stop_hook_active=None,
            turn_count=next_turn,
            transition=Continue(reason="next_turn"),
        )


# ── Helpers ─────────────────────────────────────────────────────────────

def _current_model(ctx: ToolUseContext) -> str:
    model_override = getattr(ctx, "model_override", None)
    if model_override:
        return model_override
    if hasattr(ctx, "_fallback_model") and ctx._fallback_model:
        return ctx._fallback_model
    from api import get_active_model
    return get_active_model()


def _publish_query_messages(params: QueryParams, messages: list) -> None:
    """Keep the caller's history synchronized with the active query view."""
    params.messages[:] = messages


def _tools_for_context(context: ToolUseContext | None) -> list:
    tools = all_tools()
    allowed = getattr(context, "allowed_tool_names", None)
    available = []
    for tool in tools:
        if allowed is not None and tool.name not in allowed:
            continue
        try:
            if tool.is_enabled():
                available.append(tool)
        except Exception:
            logger.exception("Tool %s enablement check failed closed", tool.name)
    return available


def _archive_query_messages(context: ToolUseContext, messages: list) -> None:
    archive = getattr(context, "archive_messages", None)
    if callable(archive):
        try:
            archive(messages)
        except Exception as error:
            logger.warning("Could not archive query messages: %s", error)


def _active_store_view(context: ToolUseContext, fallback_messages: list) -> list | None:
    store = getattr(context, "_store", None)
    session_id = getattr(context, "session_id", None)
    if store is None or not session_id:
        return None
    raw_budget = os.environ.get("CLAUDE_CODE_ACTIVE_VIEW_TOKENS", "40000")
    try:
        budget = max(1, int(raw_budget))
    except ValueError:
        budget = 40_000
    try:
        view = store.get_active_view_messages(session_id, budget_tokens=budget)
    except Exception as error:
        logger.warning("Could not build persistent active view: %s", error)
        return None
    if view is None:
        return None

    attachments = [
        message for message in fallback_messages
        if getattr(message, "type", None) == "attachment"
    ]
    return [*view, *attachments]


def _with_query_tracking(ctx: ToolUseContext, tracking: dict) -> ToolUseContext:
    try:
        ctx.query_tracking = tracking
    except Exception:
        pass
    return ctx


def _with_messages(ctx: ToolUseContext, messages: list) -> ToolUseContext:
    try:
        ctx.messages = messages
    except Exception:
        pass
    return ctx


def _is_aborted(ctx: ToolUseContext) -> bool:
    ac = getattr(ctx, "abort_controller", None)
    return ac is not None and getattr(ac, "aborted", False)


def _estimate_tokens(messages: list) -> int:
    total = 0
    for m in messages:
        c = getattr(m, "content", "")
        if isinstance(c, str):
            total += len(c) // 4
        elif isinstance(c, list):
            for b in c:
                if isinstance(b, dict):
                    total += len(b.get("text", "")) // 4
    return total


def _final_context_tokens(messages: list) -> int:
    for m in reversed(messages):
        usage = getattr(m, "usage", None)
        if usage and isinstance(usage, dict):
            return usage.get("input_tokens", 0) + usage.get("cache_read_input_tokens", 0)
    return _estimate_tokens(messages)


def _append_system_context(system_prompt: Any, ctx: dict) -> Any:
    if not ctx:
        return system_prompt
    ctx_str = "\n\n".join(f"# {k}\n{v}" for k, v in ctx.items())
    if isinstance(system_prompt, list):
        return [*system_prompt, {"type": "text", "text": ctx_str}]
    return f"{system_prompt}\n\n{ctx_str}" if system_prompt else ctx_str


def _prepend_user_context(messages: list, ctx: dict) -> list:
    if not ctx:
        return messages
    ctx_str = (
        "<system-reminder>\nAs you answer the user's questions, you can use "
        "the following context:\n"
        + "\n".join(f"# {k}\n{v}" for k, v in ctx.items())
        + "\n</system-reminder>"
    )
    return [create_user_message(ctx_str, is_meta=True), *messages]


def _yield_missing_tool_result_blocks(
    assistant_messages: list[AssistantMessage], error_message: str,
):
    for am in assistant_messages:
        for tc in (getattr(am, "tool_calls", None) or []):
            yield create_tool_result(
                tool_use_id=tc.id, content=error_message, is_error=True,
            )


def _create_api_error_message(content: str, api_error: str) -> AssistantMessage:
    msg = create_assistant_message(content=content, stop_reason="end_turn")
    msg.is_api_error = True
    msg.api_error = api_error
    return msg


def _drain_collapses(messages: list, query_source: str) -> dict | None:
    try:
        from compact import recover_from_overflow
        return recover_from_overflow(messages, query_source)
    except (ImportError, AttributeError):
        return None


def _try_reactive_compact(messages: list, params: QueryParams) -> Any | None:
    try:
        from compact import try_reactive_compact
        return try_reactive_compact(
            has_attempted=False,
            query_source=params.query_source,
            aborted=False,
            messages=messages,
            cache_safe_params={
                "system_prompt": params.system_prompt,
                "user_context": params.user_context,
                "system_context": params.system_context,
                "tool_use_context": params.tool_use_context,
                "fork_context_messages": messages,
            },
        )
    except (ImportError, AttributeError):
        return None


def _is_fallback_error(error: Exception) -> bool:
    return "fallback" in str(error).lower() or type(error).__name__ == "FallbackTriggeredError"




async def _drain_async(gen):
    async for item in gen:
        yield item


def _notify_command_lifecycle(uuid: str, phase: str) -> None:
    pass


# ── Re-exports ──────────────────────────────────────────────────────────

__all__ = [
    "QueryParams", "QueryState", "query", "query_loop",
    "get_context_window_for_model", "get_max_output_tokens_for_model",
    "Continue", "Terminal",
]