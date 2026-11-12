"""
query_stop_hooks.py — simplified stop hook handler.

Source: src/query/stopHooks.ts (483 lines) is an async GENERATOR that yields
progress messages and returns StopHookResult. Python async generators CANNOT
return values, so we adapt: this is a regular async function returning
StopHookResult directly.

For a personal agent with no hooks system there is nothing to yield mid-run,
so the adaptation is lossless today. When hooks arrive, add an `on_progress:
Callable[[Any], None]` parameter — do not turn this back into a generator.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class StopHookResult:
    blocking_errors: list = field(default_factory=list)
    prevent_continuation: bool = False


async def handle_stop_hooks(
    messages_for_query: list,
    assistant_messages: list,
    system_prompt: Any,
    user_context: dict,
    system_context: dict,
    tool_use_context: Any,
    query_source: str,
    stop_hook_active: bool | None = None,
) -> StopHookResult:
    """No-op until hooks exist. Returns a clean StopHookResult.

    Replace this body with real hook execution when you build a hook system.
    The signature matches the call site in query.py — no changes there.
    """
    return StopHookResult(
        blocking_errors=[],
        prevent_continuation=False,
    )


async def execute_stop_failure_hooks(
    last_message: Any,
    tool_use_context: Any,
) -> None:
    """Called on API errors before returning. No-op until hooks exist."""
    return None