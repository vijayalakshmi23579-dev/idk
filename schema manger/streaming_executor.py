"""
streaming_executor.py — exact port of StreamingToolExecutor.ts.

Executes tools as they stream in with concurrency control:
  - Concurrent-safe tools run in parallel with other concurrent-safe tools
  - Non-concurrent tools run alone (exclusive access)
  - Results buffered and emitted in request order

Three-level abort chain:
  query_controller (parent, from ToolUseContext)
    └── sibling_abort_controller (fires on Bash errors only)
          └── tool_abort_controller (per-tool, bubbles non-sibling aborts up)

Aborting sibling_abort_controller kills sibling subprocesses but does NOT
abort the parent query — the turn continues.
"""
from __future__ import annotations

import asyncio
import copy
import logging
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator, Generator

from execute import execute_tool, ExecutionResult, _error_result
from messages import (
    ToolResultMessage, create_progress_message,
)
from registry import find
from tool import Tool, ToolUseContext
from loop_guard import LoopGuardState


logger = logging.getLogger(__name__)


BASH_TOOL_NAME = "Bash"


# ── MessageUpdate — what generators yield ────────────────────────────────

@dataclass
class MessageUpdate:
    message: Any = None
    new_context: Any = None


# ── Tracked tool ─────────────────────────────────────────────────────────

@dataclass
class _TrackedTool:
    id: str
    block: dict
    assistant_message: Any
    status: str = "queued"          # queued | executing | completed | yielded
    is_concurrency_safe: bool = False
    is_speculative_safe: bool = False
    promise: asyncio.Task | None = None
    results: list = field(default_factory=list)
    pending_progress: list = field(default_factory=list)
    context_modifiers: list = field(default_factory=list)
    abort_controller: Any = None


# ── The executor ─────────────────────────────────────────────────────────

class StreamingToolExecutor:
    def __init__(
        self,
        tool_definitions: list[Tool],
        can_use_tool: Any,
        tool_use_context: ToolUseContext,
    ):
        self.tool_definitions = tool_definitions
        self.can_use_tool = can_use_tool
        self.tool_use_context = tool_use_context

        self.tools: list[_TrackedTool] = []
        self.has_errored = False
        self.errored_tool_description = ""

        # Child of tool_use_context.abort_controller. Bash errors only.
        # Does NOT abort the parent query.
        self.sibling_abort_controller = _ChildAbortController(
            tool_use_context.abort_controller,
        )

        self.discarded = False
        self.deferred_tools_released = False
        self.progress_available_resolve = None

    # ── Lifecycle ─────────────────────────────────────────────────────────

    def discard(self) -> None:
        """Cancel speculative work and abandon results from this attempt."""
        if self.discarded:
            return
        self.discarded = True
        for tool in self.tools:
            if tool.abort_controller is not None:
                tool.abort_controller.abort("streaming_fallback")
            if tool.promise is not None and not tool.promise.done():
                tool.promise.cancel()

    def release_deferred_tools(self) -> None:
        """Allow non-concurrency-safe tools after the model stream succeeds."""
        if self.discarded:
            return
        self.deferred_tools_released = True
        self.process_queue()

    def add_tool(self, block: dict, assistant_message: Any) -> None:
        """Queue a tool. Starts immediately if conditions allow."""
        tool_definition = find(block.get("name", ""))

        if tool_definition is None:
            self.tools.append(_TrackedTool(
                id=block.get("id", ""),
                block=block,
                assistant_message=assistant_message,
                status="completed",
                is_concurrency_safe=True,
                results=[_synthetic_no_such_tool(block)],
            ))
            return

        parsed = None
        try:
            parsed = tool_definition.input_schema(**block.get("input", {}))
        except Exception:
            parsed = None

        is_speculative_safe = False
        if parsed is not None:
            try:
                is_safe = bool(tool_definition.is_concurrency_safe(parsed))
            except Exception:
                is_safe = False
            if tool_definition.name != BASH_TOOL_NAME:
                try:
                    is_speculative_safe = bool(tool_definition.is_read_only(parsed))
                except Exception:
                    is_speculative_safe = False
        else:
            is_safe = False

        self.tools.append(_TrackedTool(
            id=block.get("id", ""),
            block=block,
            assistant_message=assistant_message,
            status="queued",
            is_concurrency_safe=is_safe,
            is_speculative_safe=is_speculative_safe,
        ))
        self.process_queue()

    # ── Queue processing ─────────────────────────────────────────────────

    def can_execute_tool(self, is_concurrency_safe: bool) -> bool:
        executing = [t for t in self.tools if t.status == "executing"]
        return (
            len(executing) == 0
            or (is_concurrency_safe and all(t.is_concurrency_safe for t in executing))
        )

    def process_queue(self) -> None:
        """Iterate in order. Start queued tools when safe. Break on serial."""
        if self.discarded:
            return
        for tool in self.tools:
            if tool.status != "queued":
                continue
            if not tool.is_speculative_safe and not self.deferred_tools_released:
                break
            if self.can_execute_tool(tool.is_concurrency_safe):
                self._start_tool(tool)
            else:
                # Non-concurrent tool can't start — break to preserve order
                if not tool.is_concurrency_safe:
                    break

    def _start_tool(self, tool: _TrackedTool) -> None:
        """Fire-and-forget execution. Returns immediately."""
        tool.status = "executing"
        self._add_in_progress(tool.id)
        self._update_interruptible_state()

        task = asyncio.create_task(self._collect_results(tool))
        tool.promise = task
        task.add_done_callback(lambda _t: self.process_queue())

    # ── Execution ────────────────────────────────────────────────────────

    async def _collect_results(self, tool: _TrackedTool) -> None:
        messages: list = []
        context_modifiers: list = []

        # Aborted before starting → synthesize error
        initial_reason = self._get_abort_reason(tool)
        if initial_reason is not None:
            messages.append(self._create_synthetic_error(tool.id, initial_reason))
            tool.results = messages
            tool.status = "completed"
            self._update_interruptible_state()
            return

        # Per-tool child of sibling_abort_controller
        tool_abort = _ChildAbortController(self.sibling_abort_controller)
        tool.abort_controller = tool_abort
        _install_bubble_up(
            tool_abort, self.tool_use_context.abort_controller, self,
        )

        this_tool_errored = False
        loop = asyncio.get_running_loop()

        def on_progress(data: dict) -> None:
            loop.call_soon_threadsafe(self._enqueue_progress, tool, data)

        try:
            result: ExecutionResult = await execute_tool(
                tool.block,
                _override_abort_controller(self.tool_use_context, tool_abort),
                self.can_use_tool,
                tool.assistant_message,
                on_progress,
            )
        except asyncio.CancelledError:
            messages.append(self._create_synthetic_error(
                tool.id, "streaming_fallback",
            ))
            tool.results = messages
            tool.status = "completed"
            self._update_interruptible_state()
            return

        # Aborted mid-execution?
        abort_reason = self._get_abort_reason(tool)
        if abort_reason is not None and not this_tool_errored:
            messages.append(self._create_synthetic_error(tool.id, abort_reason))
            tool.results = messages
            tool.status = "completed"
            self._update_interruptible_state()
            return

        # Append result
        if result.result_message is not None:
            messages.append(result.result_message)

        # Bash error cascade
        if result.is_error:
            this_tool_errored = True
            if tool.block.get("name") == BASH_TOOL_NAME:
                self.has_errored = True
                self.errored_tool_description = self._get_tool_description(tool)
                if hasattr(self.sibling_abort_controller, "abort"):
                    self.sibling_abort_controller.abort("sibling_error")

        if result.context_modifier is not None:
            context_modifiers.append(result.context_modifier)

        tool.results = messages
        tool.context_modifiers = context_modifiers
        tool.status = "completed"
        self._update_interruptible_state()

        # Apply context modifiers in-place ONLY for serial tools.
        # Concurrent tools must not mutate shared context (race).
        if not tool.is_concurrency_safe and context_modifiers:
            for modifier in context_modifiers:
                try:
                    self.tool_use_context = modifier(self.tool_use_context)
                except Exception as e:
                    logger.error("Context modifier failed: %s", e)

    def _enqueue_progress(self, tool: _TrackedTool, data: dict) -> None:
        tool.pending_progress.append(create_progress_message(
            tool.id,
            self.tool_use_context.tool_use_id or "",
            data,
        ))
        resolve = self.progress_available_resolve
        self.progress_available_resolve = None
        if resolve is not None:
            resolve()

    # ── Abort handling ───────────────────────────────────────────────────

    def _get_abort_reason(self, tool: _TrackedTool) -> str | None:
        if self.discarded:
            return "streaming_fallback"
        if self.has_errored:
            return "sibling_error"
        ac = self.tool_use_context.abort_controller
        if ac is not None and getattr(ac, "aborted", False):
            reason = getattr(ac, "reason", None)
            if reason == "interrupt":
                return ("user_interrupted"
                        if self._get_tool_interrupt_behavior(tool) == "cancel"
                        else None)
            return "user_interrupted"
        return None

    def _get_tool_interrupt_behavior(self, tool: _TrackedTool) -> str:
        definition = find(tool.block.get("name", ""))
        if definition is None or definition.interrupt_behavior is None:
            return "block"
        try:
            return definition.interrupt_behavior()
        except Exception:
            return "block"

    def _get_tool_description(self, tool: _TrackedTool) -> str:
        inp = tool.block.get("input") or {}
        summary = inp.get("command") or inp.get("file_path") or inp.get("pattern") or ""
        if isinstance(summary, str) and summary:
            return summary[:50]
        return tool.block.get("name", "")

    def _create_synthetic_error(
        self, tool_use_id: str, reason: str,
    ) -> ToolResultMessage:
        if reason == "user_interrupted":
            return ToolResultMessage(
                tool_use_id=tool_use_id,
                content=(
                    "The user doesn't want to take this action right now. "
                    "STOP what you are doing and wait for the user to tell you "
                    "how to proceed."
                ),
                is_error=True,
            )
        if reason == "streaming_fallback":
            return ToolResultMessage(
                tool_use_id=tool_use_id,
                content="Error: Streaming fallback - tool execution discarded",
                is_error=True,
            )
        desc = self.errored_tool_description
        msg = (f"Cancelled: parallel tool call {desc} errored"
               if desc else "Cancelled: parallel tool call errored")
        return ToolResultMessage(
            tool_use_id=tool_use_id, content=msg, is_error=True,
        )

    # ── Result generation ───────────────────────────────────────────────

    def get_completed_results(self) -> Generator[MessageUpdate, None, None]:
        """Non-blocking. Yields completed results in request order.

        Progress messages are yielded immediately (order-bypassing).
        Stops at the first executing serial tool.
        """
        if self.discarded:
            return
        for tool in self.tools:
            while tool.pending_progress:
                progress = tool.pending_progress.pop(0)
                yield MessageUpdate(
                    message=progress, new_context=self.tool_use_context,
                )

            if tool.status == "yielded":
                continue

            if tool.status == "completed" and tool.results:
                tool.status = "yielded"
                for message in tool.results:
                    yield MessageUpdate(
                        message=message, new_context=self.tool_use_context,
                    )
                self._mark_tool_use_complete(tool.id)

            elif tool.status == "executing" and not tool.is_concurrency_safe:
                break

    async def get_remaining_results(self) -> AsyncGenerator[MessageUpdate, None]:
        """Wait for remaining tools, yielding results as they complete."""
        if self.discarded:
            return
        while self._has_unfinished_tools():
            self.process_queue()
            for update in self.get_completed_results():
                yield update

            if (self._has_executing_tools()
                    and not self._has_completed_results()
                    and not self._has_pending_progress()):
                executing_promises = [
                    t.promise for t in self.tools
                    if t.status == "executing" and t.promise is not None
                ]
                progress_promise = self._make_progress_promise()
                if executing_promises:
                    await _race_first(
                        [*executing_promises, progress_promise],
                    )
        for update in self.get_completed_results():
            yield update

    def _make_progress_promise(self) -> asyncio.Future:
        loop = asyncio.get_running_loop()
        fut = loop.create_future()
        def _resolve():
            if not fut.done():
                fut.set_result(None)
        self.progress_available_resolve = _resolve
        return fut

    def _has_pending_progress(self) -> bool:
        return any(t.pending_progress for t in self.tools)

    def _has_completed_results(self) -> bool:
        return any(t.status == "completed" for t in self.tools)

    def _has_executing_tools(self) -> bool:
        return any(t.status == "executing" for t in self.tools)

    def _has_unfinished_tools(self) -> bool:
        return any(t.status != "yielded" for t in self.tools)

    def get_updated_context(self) -> ToolUseContext:
        return self.tool_use_context

    # ── In-progress tracking ────────────────────────────────────────────

    def _add_in_progress(self, tool_use_id: str) -> None:
        setter = self.tool_use_context.set_in_progress_tool_use_ids
        if setter is not None:
            try:
                setter(lambda prev: {*prev, tool_use_id})
            except Exception:
                pass

    def _mark_tool_use_complete(self, tool_use_id: str) -> None:
        setter = self.tool_use_context.set_in_progress_tool_use_ids
        if setter is not None:
            try:
                setter(lambda prev: {x for x in prev if x != tool_use_id})
            except Exception:
                pass

    def _update_interruptible_state(self) -> None:
        setter = self.tool_use_context.set_has_interruptible_tool_in_progress
        if setter is None:
            return
        executing = [t for t in self.tools if t.status == "executing"]
        all_cancel = bool(executing) and all(
            self._get_tool_interrupt_behavior(t) == "cancel"
            for t in executing
        )
        try:
            setter(all_cancel)
        except Exception:
            pass


# ── Abort controller chain ──────────────────────────────────────────────

class _ChildAbortController:
    """Minimal AbortController with parent linkage."""

    def __init__(self, parent: Any = None):
        self._aborted = False
        self._reason: Any = None
        self._parent = parent
        self._listeners: list = []
        self._event = asyncio.Event()

    @property
    def aborted(self) -> bool:
        if self._aborted:
            return True
        if self._parent is not None and getattr(self._parent, "aborted", False):
            return True
        return False

    @property
    def reason(self) -> Any:
        if self._aborted:
            return self._reason
        if self._parent is not None and getattr(self._parent, "aborted", False):
            return getattr(self._parent, "reason", None)
        return None

    def abort(self, reason: Any = None) -> None:
        if self._aborted:
            return
        self._aborted = True
        self._reason = reason
        self._event.set()
        for fn in list(self._listeners):
            try:
                fn()
            except Exception:
                pass

    def add_listener(self, fn) -> None:
        self._listeners.append(fn)

    async def wait(self) -> None:
        """Wait for this controller or any parent controller to abort."""
        if self.aborted:
            return
        waiters = [asyncio.create_task(self._event.wait())]
        if self._parent is not None:
            waiters.append(asyncio.create_task(_wait_for_abort_signal(self._parent)))
        try:
            await asyncio.wait(waiters, return_when=asyncio.FIRST_COMPLETED)
        finally:
            for waiter in waiters:
                if not waiter.done():
                    waiter.cancel()
            await asyncio.gather(*waiters, return_exceptions=True)


async def _wait_for_abort_signal(controller: Any) -> None:
    if getattr(controller, "aborted", False):
        return
    wait = getattr(controller, "wait", None)
    if callable(wait):
        await wait()
        return
    while not getattr(controller, "aborted", False):
        await asyncio.sleep(0.05)


def _install_bubble_up(
    tool_controller: _ChildAbortController,
    query_controller: Any,
    executor: StreamingToolExecutor,
) -> None:
    """Bubble non-sibling aborts up to the query controller."""
    def _on_abort():
        reason = tool_controller.reason
        if (
            reason != "sibling_error"
            and not getattr(query_controller, "aborted", False)
            and not executor.discarded
        ):
            if hasattr(query_controller, "abort"):
                query_controller.abort(reason)

    tool_controller.add_listener(_on_abort)


def _override_abort_controller(
    ctx: ToolUseContext, controller: Any,
) -> ToolUseContext:
    """Shallow-copy context with a different abort controller."""
    loop_guard = getattr(ctx, "_loop_guard", None)
    if loop_guard is None:
        loop_guard = LoopGuardState()
        ctx._loop_guard = loop_guard
    overridden = copy.copy(ctx)
    overridden.abort_controller = controller
    return overridden


def _synthetic_no_such_tool(block: dict) -> ToolResultMessage:
    return ToolResultMessage(
        tool_use_id=block.get("id", ""),
        content=f"Error: No such tool available: {block.get('name', '')}",
        is_error=True,
    )


async def _race_first(awaitables: list) -> None:
    """Wait for any awaitable to complete without cancelling pending work."""
    futures = [
        a if isinstance(a, asyncio.Future) else asyncio.ensure_future(a)
        for a in awaitables
    ]
    await asyncio.wait(
        futures, return_when=asyncio.FIRST_COMPLETED,
    )