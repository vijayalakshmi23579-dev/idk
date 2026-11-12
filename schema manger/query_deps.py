"""
query_deps.py — dependency injection for the query loop.

Exact port of src/query/deps.ts (21 lines):

    export type QueryDeps = {
        callModel: typeof queryModelWithStreaming
        microcompact: typeof microcompactMessages
        autocompact: typeof autoCompactIfNeeded
        uuid: () => string
    }
    export function productionDeps(): QueryDeps { ... }

Source comment: "Scope is intentionally narrow (4 deps) to prove the pattern."

NoopDeps: yields ONE scripted response per call_model invocation, so the
loop actually terminates. (Yielding all responses in one call would cause
each iteration to see every response → infinite loop.)
"""
from __future__ import annotations

import uuid as _uuid
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator, Callable


# ── Types (deps.ts:16-19) ────────────────────────────────────────────────

CallModelFn = Callable[..., AsyncGenerator[Any, None]]
MicrocompactFn = Callable[..., Any]
AutocompactFn = Callable[..., Any]
UuidFn = Callable[[], str]


@dataclass
class QueryDeps:
    call_model: CallModelFn
    microcompact: MicrocompactFn
    autocompact: AutocompactFn
    uuid: UuidFn


# ── Production implementations ──────────────────────────────────────────

async def _production_microcompact(messages, tool_use_context, query_source):
    """Pass-through until compact.py lands."""
    try:
        from compact import microcompact
        return await microcompact(messages, tool_use_context, query_source)
    except (ImportError, AttributeError):
        return {"messages": messages, "compaction_info": None}


async def _production_autocompact(
    messages, tool_use_context, cache_safe_params,
    query_source, tracking, snip_tokens_freed,
):
    """Pass-through until compact.py lands."""
    try:
        from compact import autocompact
        return await autocompact(
            messages, tool_use_context, cache_safe_params,
            query_source, tracking, snip_tokens_freed,
        )
    except (ImportError, AttributeError):
        return {"compaction_result": None, "consecutive_failures": None}


def _production_call_model(
    messages, system_prompt, thinking_config=None,
    tools=None, signal=None, options=None, **kwargs,
):
    """Adapter: api.stream() → Message-shaped generator.

    Yields one AssistantMessage per content block (text or tool_use),
    plus a final message carrying stop_reason + usage.
    """
    options = options or {}
    return _adapter_gen(
        messages=messages, system_prompt=system_prompt, tools=tools or [],
        signal=signal,
        model=options.get("model"),
        fallback_model=options.get("fallback_model"),
        query_source=options.get("query_source"),
        max_tokens=options.get("max_output_tokens_override"),
    )


async def _adapter_gen(
    messages, system_prompt, tools, signal=None, model=None,
    fallback_model=None, query_source=None, max_tokens=None,
):
    from api import ProviderRequestError, stream
    from messages import ToolUseBlock, create_assistant_message

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

    try:
        async for event in stream(
            messages=messages,
            tools=tools,
            system_prompt=sp,
            signal=signal,
            model=model,
            fallback_model=fallback_model,
            query_source=query_source,
            max_tokens=max_tokens or 8192,
        ):
            t = event.get("type")
            if t == "runtime_activity":
                yield event
            elif t == "text_delta":
                text_buf.append(event["text"])
            elif t == "thinking_delta":
                thinking_buf.append(event["thinking"])
            elif t == "tool_use":
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
    except ProviderRequestError as error:
        if error.category == "context_overflow":
            content = "Prompt is too long. Use /compact or /clear."
            api_error = "invalid_request"
        else:
            content = f"Provider request failed: {error}"
            api_error = error.category
        failure = create_assistant_message(content=content, stop_reason="end_turn")
        failure.is_api_error = True
        failure.api_error = api_error
        yield failure
        return

    final = create_assistant_message(
        content="".join(text_buf) if text_buf else None,
        tool_calls=[], stop_reason=stop_reason, usage=usage,
    )
    if thinking_buf:
        final.thinking_blocks = [
            {"type": "thinking", "thinking": "".join(thinking_buf)}
        ]
    yield final


def production_deps() -> QueryDeps:
    return QueryDeps(
        call_model=_production_call_model,
        microcompact=_production_microcompact,
        autocompact=_production_autocompact,
        uuid=lambda: str(_uuid.uuid4()),
    )


# ── Test double ─────────────────────────────────────────────────────────

@dataclass
class NoopDeps:
    """Scripted responses for testing query.py without an API.

    CRITICAL: yields ONE response per call_model call (via cursor).
    Yielding all responses at once would cause the loop to see them all
    in every iteration → infinite loop or max_turns exhaustion.
    """
    responses: list = field(default_factory=list)
    _cursor: int = 0
    _uuid_counter: int = 0
    microcompact_calls: int = 0
    autocompact_calls: int = 0

    def call_model(self, **_kwargs) -> AsyncGenerator[Any, None]:
        async def _gen():
            if self._cursor < len(self.responses):
                yield self.responses[self._cursor]
                self._cursor += 1
        return _gen()

    async def microcompact(self, messages, ctx, source):
        self.microcompact_calls += 1
        return {"messages": messages, "compaction_info": None}

    async def autocompact(self, messages, ctx, params, source, tracking, snip):
        self.autocompact_calls += 1
        return {"compaction_result": None, "consecutive_failures": None}

    def uuid(self) -> str:
        self._uuid_counter += 1
        return f"uuid-{self._uuid_counter}"

    def as_deps(self) -> QueryDeps:
        return QueryDeps(
            call_model=self.call_model,
            microcompact=self.microcompact,
            autocompact=self.autocompact,
            uuid=self.uuid,
        )