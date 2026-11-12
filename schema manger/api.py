"""
api.py — provider-agnostic streaming client via LiteLLM.

LiteLLM normalizes every provider to the OpenAI format. We normalize
LiteLLM's output to OUR event stream. query.py sees only our events.

Compared to claude.ts (~2000 lines, Anthropic-native):
  ✅ Multi-provider (Groq, Gemini, NVIDIA, OpenRouter, Anthropic)
  ✅ Prompt caching via cache_control markers (LiteLLM forwards)
  ✅ Cache markers on system + last user + last assistant (matches claude.ts)
  ✅ Media stripping (matches stripExcessMediaItems)
  ✅ Thinking via keyword detection + model capability check
  ✅ Fallback on rate limit + non-streaming fallback on stream failure
  ✅ Usage accumulation across chunks

  ❌ Fine-grained tool streaming (eager_input_streaming) — LiteLLM limitation
  ❌ Beta headers — LiteLLM handles its own; use extra_headers per-call if needed
  ❌ 1h cache TTL — LiteLLM only forwards base ephemeral
  ❌ task_budget — Anthropic-native only
  ❌ Request ID chaining — provider-inconsistent
"""
from __future__ import annotations

import asyncio
import inspect
import json
import logging
import os
import random
import re
import threading
import time
import weakref
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import AsyncGenerator, Callable

from dotenv import load_dotenv

load_dotenv()

import litellm

from messages import to_openai
from tool import to_openai_schema

logger = logging.getLogger(__name__)

litellm.suppress_debug_info = True
litellm.drop_params = True


def _sync_provider_credentials() -> None:
    """Bridge the app's legacy NVIDIA key name to LiteLLM's NIM name."""
    if not os.environ.get("NVIDIA_NIM_API_KEY") and os.environ.get("NVIDIA_API_KEY"):
        os.environ["NVIDIA_NIM_API_KEY"] = os.environ["NVIDIA_API_KEY"]


_sync_provider_credentials()


# ── Provider catalog (same as before) ────────────────────────────────────

PROVIDERS: dict[str, dict] = {
    "groq": {
        "prefix": "groq/",
        "env_key": "GROQ_API_KEY",
        "models": [
            "groq/openai/gpt-oss-120b",
            "groq/openai/gpt-oss-20b",
        ],
    },
    "gemini": {
        "prefix": "gemini/",
        "env_key": "GEMINI_API_KEY",
        "models": ["gemini/gemini-3.8-flash"],
    },
    "nvidia": {
        "prefix": "nvidia_nim/",
        "env_key": "NVIDIA_NIM_API_KEY",
        "legacy_env_key": "NVIDIA_API_KEY",
        "models": [
            "nvidia_nim/nvidia/nemotron-3.5-bolt-e4b",
        ],
    },
    "openrouter": {
        "prefix": "openrouter/",
        "env_key": "OPENROUTER_API_KEY",
        "models": [
            "openrouter/deepseek/deepseek-r1:free",
            "openrouter/qwen/qwen-2.5-coder-32b-instruct:free",
        ],
    },
}

def _normalize_model_name(model: str) -> str:
    if model.startswith("nvidia/"):
        return f"nvidia_nim/{model}"
    return model


ACTIVE_MODEL = _normalize_model_name(
    os.environ.get("ACTIVE_MODEL", "groq/openai/gpt-oss-120b")
)


def _configured_fallback_models() -> list[str]:
    configured = os.environ.get("LITELLM_FALLBACK_MODELS", "")
    return [
        _normalize_model_name(model.strip())
        for model in configured.split(",")
        if model.strip()
    ]


FALLBACK_MODELS: list[str] = _configured_fallback_models()


# ── Constants ────────────────────────────────────────────────────────────

MAX_MEDIA_PER_REQUEST = 100
MAX_RETRIES_PER_MODEL = 2
MAX_RETRY_BACKOFF_SECONDS = 8.0
MAX_RETRY_AFTER_SECONDS = 120.0
RETRY_BACKOFF_BASE_SECONDS = 1.0
DEFAULT_PROVIDER_MAX_CONCURRENCY = 4

_PROVIDER_SEMAPHORES: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()
_PROVIDER_SEMAPHORES_LOCK = threading.Lock()
_PROVIDER_RATE_LIMITERS: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()
_PROVIDER_RATE_LIMITERS_LOCK = threading.Lock()


def _provider_concurrency_limit(provider: str) -> int:
    provider_key = re.sub(r"[^A-Z0-9]+", "_", provider.upper())
    raw = os.environ.get(
        f"AGENT_PROVIDER_MAX_CONCURRENCY_{provider_key}",
        os.environ.get("AGENT_PROVIDER_MAX_CONCURRENCY", ""),
    )
    try:
        return max(1, int(raw)) if raw else DEFAULT_PROVIDER_MAX_CONCURRENCY
    except ValueError:
        return DEFAULT_PROVIDER_MAX_CONCURRENCY


def _provider_semaphore(model: str) -> asyncio.Semaphore:
    provider = model.split("/", 1)[0].lower()
    limit = _provider_concurrency_limit(provider)
    loop = asyncio.get_running_loop()
    with _PROVIDER_SEMAPHORES_LOCK:
        per_loop = _PROVIDER_SEMAPHORES.setdefault(loop, {})
        configured = per_loop.get(provider)
        if configured is None or configured[0] != limit:
            per_loop[provider] = (limit, asyncio.Semaphore(limit))
        return per_loop[provider][1]


def _configured_provider_rate_limit(name: str, provider: str) -> int:
    provider_key = re.sub(r"[^A-Z0-9]+", "_", provider.upper())
    raw = os.environ.get(
        f"AGENT_PROVIDER_MAX_{name}_{provider_key}",
        os.environ.get(f"AGENT_PROVIDER_MAX_{name}", "0"),
    )
    try:
        return max(0, int(raw))
    except ValueError:
        logger.warning("Ignoring invalid AGENT_PROVIDER_MAX_%s value for %s", name, provider)
        return 0


def _provider_rate_limits(provider: str) -> tuple[int, int]:
    return (
        _configured_provider_rate_limit("RPM", provider),
        _configured_provider_rate_limit("TPM", provider),
    )


@dataclass(frozen=True)
class _ProviderReservation:
    limiter: "_ProviderRateLimiter"
    tokens: int

    async def settle(self, actual_tokens: int) -> None:
        await self.limiter.settle(self.tokens, actual_tokens)


class _ProviderRateLimiter:
    """Per-loop token bucket for configured provider request/token rates."""

    def __init__(self, provider: str, rpm: int, tpm: int) -> None:
        self.provider = provider
        self.rpm = rpm
        self.tpm = tpm
        self._requests = float(rpm)
        self._tokens = float(tpm)
        self._updated_at = time.monotonic()
        self._lock = asyncio.Lock()

    def _refill(self) -> None:
        now = time.monotonic()
        elapsed = max(0.0, now - self._updated_at)
        self._updated_at = now
        if self.rpm:
            self._requests = min(float(self.rpm), self._requests + elapsed * self.rpm / 60)
        if self.tpm:
            self._tokens = min(float(self.tpm), self._tokens + elapsed * self.tpm / 60)

    async def acquire(self, tokens: int) -> _ProviderReservation:
        if self.tpm and tokens > self.tpm:
            raise ProviderRequestError(
                f"Estimated request exceeds configured TPM capacity for {self.provider}.",
                category="rate_limited",
                status_code=429,
            )
        while True:
            async with self._lock:
                self._refill()
                request_wait = (
                    max(0.0, 1.0 - self._requests) * 60 / self.rpm
                    if self.rpm else 0.0
                )
                token_wait = (
                    max(0.0, tokens - self._tokens) * 60 / self.tpm
                    if self.tpm else 0.0
                )
                wait_seconds = max(request_wait, token_wait)
                if wait_seconds <= 0:
                    if self.rpm:
                        self._requests -= 1
                    if self.tpm:
                        self._tokens -= tokens
                    return _ProviderReservation(self, tokens if self.tpm else 0)
            await asyncio.sleep(wait_seconds)

    async def settle(self, reserved_tokens: int, actual_tokens: int) -> None:
        if not self.tpm:
            return
        async with self._lock:
            self._refill()
            self._tokens = min(
                float(self.tpm),
                self._tokens + reserved_tokens - max(0, actual_tokens),
            )


def _provider_rate_limiter(model: str) -> _ProviderRateLimiter:
    provider = model.split("/", 1)[0].lower()
    rpm, tpm = _provider_rate_limits(provider)
    loop = asyncio.get_running_loop()
    with _PROVIDER_RATE_LIMITERS_LOCK:
        per_loop = _PROVIDER_RATE_LIMITERS.setdefault(loop, {})
        configured = per_loop.get(provider)
        if configured is None or configured[:2] != (rpm, tpm):
            limiter = _ProviderRateLimiter(provider, rpm, tpm)
            per_loop[provider] = (rpm, tpm, limiter)
        return per_loop[provider][2]


def _estimate_message_tokens(messages: list) -> int:
    try:
        content = json.dumps(messages, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        content = str(messages)
    return max(1, (len(content) + 3) // 4)


def _usage_token_total(usage: Any) -> int | None:
    if usage is None:
        return None
    if isinstance(usage, dict):
        input_tokens = usage.get("input_tokens", usage.get("prompt_tokens", 0))
        output_tokens = usage.get("output_tokens", usage.get("completion_tokens", 0))
    else:
        input_tokens = getattr(usage, "prompt_tokens", 0)
        output_tokens = getattr(usage, "completion_tokens", 0)
    return max(0, int(input_tokens or 0)) + max(0, int(output_tokens or 0))


@asynccontextmanager
async def provider_request_slot(model: str, estimated_tokens: int = 0):
    """Apply configured RPM/TPM budgets and a shared in-flight request cap."""
    reservation = await _provider_rate_limiter(model).acquire(max(0, estimated_tokens))
    async with _provider_semaphore(model):
        yield reservation


class ProviderRequestError(RuntimeError):
    """Safe user-facing summary for a failed provider request."""

    def __init__(
        self,
        message: str,
        *,
        category: str = "provider_error",
        status_code: int | None = None,
        model: str | None = None,
        request_id: str | None = None,
    ) -> None:
        super().__init__(message)
        self.category = category
        self.status_code = status_code
        self.model = model
        self.request_id = request_id


def _status_code(error: Exception) -> int | None:
    response = getattr(error, "response", None)
    value = getattr(error, "status_code", None) or getattr(response, "status_code", None)
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _request_id(error: Exception) -> str | None:
    value = getattr(error, "request_id", None)
    if value:
        return str(value)[:128]
    response = getattr(error, "response", None)
    headers = getattr(error, "headers", None) or getattr(response, "headers", None) or {}
    for key in ("x-request-id", "request-id", "x-nvidia-request-id"):
        value = headers.get(key) if hasattr(headers, "get") else None
        if value:
            return str(value)[:128]
    return None


def _failure_category(error: Exception) -> str:
    status = _status_code(error)
    name = type(error).__name__.lower()
    if name == "contextwindowexceedederror":
        return "context_overflow"
    if status == 404 or "notfounderror" in name:
        return "model_or_route_not_found"
    if status == 429 or "ratelimiterror" in name:
        return "rate_limited"
    if status == 529 or (status is not None and 500 <= status <= 599):
        return "provider_overloaded"
    if "apiconnectionerror" in name or "timeout" in name:
        return "connection_failure"
    if status in (401, 403):
        return "authentication_or_permission"
    return "provider_error"


def _is_hard_rate_limit(error: Exception) -> bool:
    category = str(getattr(error, "category", "")).lower()
    rate_limit_type = str(getattr(error, "rate_limit_type", "")).lower()
    detail = " ".join((
        str(error),
        str(getattr(error, "message", "")),
        str(getattr(error, "detail", "")),
        str(getattr(error, "body", "")),
    )).lower()
    markers = ("spend", "billing", "insufficient_quota", "usage limit", "credit")
    return any(marker in value for value in (category, rate_limit_type, detail)
               for marker in markers)


def _retry_after_seconds(error: Exception) -> float | None:
    response = getattr(error, "response", None)
    headers = getattr(error, "headers", None) or getattr(response, "headers", None) or {}
    value = None
    if hasattr(headers, "items"):
        value = next((v for k, v in headers.items() if str(k).lower() == "retry-after"), None)
    if value is None:
        return None
    try:
        delay = float(value)
    except (TypeError, ValueError):
        try:
            retry_at = parsedate_to_datetime(str(value))
            if retry_at.tzinfo is None:
                retry_at = retry_at.replace(tzinfo=timezone.utc)
            delay = (retry_at - datetime.now(timezone.utc)).total_seconds()
        except (TypeError, ValueError, OverflowError):
            return None
    if delay < 0 or delay > MAX_RETRY_AFTER_SECONDS:
        return None
    return delay


def _retry_delay(error: Exception, retry_index: int) -> float | None:
    response = getattr(error, "response", None)
    headers = getattr(error, "headers", None) or getattr(response, "headers", None) or {}
    if hasattr(headers, "items") and any(
        str(key).lower() == "retry-after" for key, _value in headers.items()
    ):
        return _retry_after_seconds(error)
    retry_after = _retry_after_seconds(error)
    if retry_after is not None:
        return retry_after
    delay = min(
        RETRY_BACKOFF_BASE_SECONDS * (2 ** retry_index),
        MAX_RETRY_BACKOFF_SECONDS,
    )
    return delay * random.uniform(0.75, 1.25)


async def _retry_sleep(delay: float) -> None:
    await asyncio.sleep(delay)


def _log_provider_failure(
    error: Exception, model: str, query_source: str | None, attempt: int,
) -> str:
    category = _failure_category(error)
    status = _status_code(error)
    request_id = _request_id(error)
    provider = model.split("/", 1)[0] if "/" in model else "unknown"
    logger.warning(
        "LLM request failed provider=%s model=%s category=%s status=%s "
        "request_id=%s query_source=%s attempt=%s",
        provider, model, category, status or "unknown",
        request_id or "unknown", query_source or "unknown", attempt,
    )
    details = f"provider={provider} model={model} category={category}"
    if status is not None:
        details += f" status={status}"
    if request_id:
        details += f" request_id={request_id}"
    return details


def _provider_request_error(
    error: Exception, details: str, model: str,
) -> ProviderRequestError:
    return ProviderRequestError(
        details,
        category=_failure_category(error),
        status_code=_status_code(error),
        model=model,
        request_id=_request_id(error),
    )


# ── Mode control ─────────────────────────────────────────────────────────

def set_active_model(model: str) -> None:
    global ACTIVE_MODEL
    ACTIVE_MODEL = _normalize_model_name(model)


def get_active_model() -> str:
    return ACTIVE_MODEL


def list_models() -> list[str]:
    _sync_provider_credentials()
    return [
        m for provider in PROVIDERS.values()
        if (
            os.environ.get(provider["env_key"])
            or os.environ.get(provider.get("legacy_env_key", ""))
        )
        for m in provider["models"]
    ]


# ── Thinking detection ──────────────────────────────────────────────────

_THINKING_HARD_KEYWORDS = {"ultrathink", "think harder", "think deeply"}
_THINKING_KEYWORDS = {"think", "plan", "analyze", "why", "reason", "explain"}


def _model_supports_thinking(model: str) -> bool:
    """Check if the model supports thinking/reasoning.

    Check our known-prefix list FIRST (strong signal, we control it).
    Then fall back to LiteLLM's database for newer models. This is
    defense in depth — LiteLLM's model registry can lag new releases,
    and we don't want that to silently disable thinking.
    """
    lower = model.lower()
    # Known thinking-capable model families
    known_thinking_prefixes = (
        "claude", "sonnet", "opus", "haiku",
        "deepseek-r1", "deepseek-reasoner",
        "o1", "o3", "o4",
        "gemini-2.5", "gemini-2.0-flash-thinking",
        "qwen", "qwq",
        "grok-3",
    )
    if any(prefix in lower for prefix in known_thinking_prefixes):
        return True
    # Fall back to LiteLLM's database
    try:
        return bool(litellm.supports_reasoning(model=model))
    except Exception:
        return False


def should_enable_thinking(messages: list, model: str | None = None) -> bool:
    """Enable thinking for hard tasks + on models that support it."""
    if model is None:
        model = ACTIVE_MODEL

    if not _model_supports_thinking(model):
        return False
    if not messages:
        return False
    last_user = next(
        (m for m in reversed(messages)
         if getattr(m, "type", None) == "user" and not getattr(m, "is_meta", False)),
        None,
    )
    if last_user is None:
        return False
    content = getattr(last_user, "content", "")
    text = content if isinstance(content, str) else ""
    lower = text.lower()
    if any(kw in lower for kw in _THINKING_HARD_KEYWORDS):
        return True
    if any(kw in lower for kw in _THINKING_KEYWORDS):
        return True
    if len(text) > 500:
        return True
    return False


# ── System prompt freezing ──────────────────────────────────────────────

def _freeze_system_prompt(system_prompt: str) -> list[dict]:
    if not system_prompt:
        return []
    return [{
        "type": "text",
        "text": system_prompt,
        "cache_control": {"type": "ephemeral"},
    }]


# ── Cache control on user + assistant messages ──────────────────────────

def _apply_cache_control_to_messages(oai_messages: list[dict]) -> list[dict]:
    """Tag last user + last assistant blocks with cache_control.

    Matches userMessageToMessageParam / assistantMessageToMessageParam
    from claude.ts. LiteLLM forwards cache_control to providers that
    support it.
    """
    if not oai_messages:
        return oai_messages

    last_user_idx = None
    last_asst_idx = None
    for i in range(len(oai_messages) - 1, -1, -1):
        role = oai_messages[i].get("role")
        if role == "user" and last_user_idx is None:
            last_user_idx = i
        elif role == "assistant" and last_asst_idx is None:
            last_asst_idx = i
        if last_user_idx is not None and last_asst_idx is not None:
            break

    def _tag(idx):
        if idx is None:
            return
        msg = oai_messages[idx]
        content = msg.get("content")
        if isinstance(content, list) and content:
            last = content[-1]
            if last.get("type") not in ("thinking", "redacted_thinking"):
                content[-1] = {**last, "cache_control": {"type": "ephemeral"}}

    _tag(last_user_idx)
    _tag(last_asst_idx)
    return oai_messages


# ── Media stripping ──────────────────────────────────────────────────────

def _strip_excess_media(oai_messages: list[dict], limit: int = MAX_MEDIA_PER_REQUEST) -> list[dict]:
    """Cap total media blocks per request. Strips oldest first.

    Port of stripExcessMediaItems from claude.ts.
    """
    positions: list[tuple[int, int]] = []
    for i, msg in enumerate(oai_messages):
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for j, block in enumerate(content):
            if block.get("type") in ("image_url", "image", "document"):
                positions.append((i, j))

    excess = len(positions) - limit
    if excess <= 0:
        return oai_messages

    to_remove = set(positions[:excess])
    for i, msg in enumerate(oai_messages):
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        oai_messages[i] = {
            **msg,
            "content": [
                {"type": "text", "text": "[media removed: request limit]"}
                if (i, j) in to_remove else block
                for j, block in enumerate(content)
            ],
        }
    return oai_messages


# ── Tool call accumulator ───────────────────────────────────────────────

class ToolCallAccumulator:
    def __init__(self):
        self._calls: dict[int, dict] = {}

    def feed(self, delta_tool_calls: list) -> None:
        for tc in delta_tool_calls:
            idx = getattr(tc, "index", 0)
            if idx not in self._calls:
                self._calls[idx] = {"id": "", "name": "", "arguments": ""}
            if tc.id:
                self._calls[idx]["id"] = tc.id
            fn = getattr(tc, "function", None)
            if fn is not None:
                if getattr(fn, "name", None):
                    self._calls[idx]["name"] += fn.name
                if getattr(fn, "arguments", None):
                    self._calls[idx]["arguments"] += fn.arguments

    def assemble(self) -> list[dict]:
        result = []
        for idx in sorted(self._calls):
            tc = self._calls[idx]
            if not tc["name"]:
                continue
            try:
                parsed = json.loads(tc["arguments"]) if tc["arguments"] else {}
            except json.JSONDecodeError:
                logger.warning("Malformed tool args for %s", tc["name"])
                parsed = {}
            result.append({
                "id": tc["id"] or f"call_{idx}",
                "name": tc["name"],
                "input": parsed,
            })
        return result


# ── Usage extraction ────────────────────────────────────────────────────

def _extract_usage(chunk) -> dict | None:
    usage = getattr(chunk, "usage", None)
    if usage is None:
        return None
    return {
        "input_tokens": getattr(usage, "prompt_tokens", 0) or 0,
        "output_tokens": getattr(usage, "completion_tokens", 0) or 0,
        "cache_read_input_tokens": getattr(usage, "cache_read_input_tokens", 0) or 0,
        "cache_creation_input_tokens": getattr(usage, "cache_creation_input_tokens", 0) or 0,
    }


# ── The stream function ─────────────────────────────────────────────────

async def stream(
    messages: list,
    tools: list | None = None,
    system_prompt: str = "",
    model: str | None = None,
    max_tokens: int = 8192,
    enable_thinking: bool | None = None,
    signal: Any = None,
    fallback_model: str | None = None,
    query_source: str | None = None,
) -> AsyncGenerator[dict, None]:
    """Stream a model response with normalized events.

    Events:
      {"type": "message_start", "model": str}
      {"type": "text_delta", "text": str}
      {"type": "thinking_delta", "thinking": str}
      {"type": "tool_use", "id": str, "name": str, "input": dict}
      {"type": "message_stop", "stop_reason": str, "usage": dict, "model": str}
    """
    target_model = model or ACTIVE_MODEL
    if _signal_aborted(signal):
        yield {"type": "message_start", "model": target_model}
        yield {
            "type": "message_stop", "stop_reason": "interrupted",
            "usage": _empty_usage(), "model": target_model,
        }
        return
    oai_messages = to_openai(messages)

    # Apply cache control + media stripping
    oai_messages = _apply_cache_control_to_messages(oai_messages)
    oai_messages = _strip_excess_media(oai_messages)

    system_blocks = _freeze_system_prompt(system_prompt)
    if system_blocks:
        oai_messages = [{"role": "system", "content": system_blocks}, *oai_messages]

    kwargs: dict = {
        "model": target_model,
        "messages": oai_messages,
        "max_tokens": max_tokens,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if tools:
        kwargs["tools"] = [to_openai_schema(t) for t in tools]
        kwargs["tool_choice"] = "auto"

    if enable_thinking is None:
        enable_thinking = should_enable_thinking(messages, target_model)

    activity_events: list[dict[str, object]] = []
    yield {
        "type": "runtime_activity", "area": "model", "status": "started",
        "model": target_model, "query_source": query_source or "unknown",
    }
    try:
        response, target_model, reservation = await _call_with_fallback(
            kwargs, enable_thinking, fallback_model=fallback_model,
            query_source=query_source, on_event=activity_events.append,
        )
    except ProviderRequestError as error:
        for event in activity_events:
            yield event
        yield {
            "type": "runtime_activity", "area": "model", "status": "failed",
            "model": error.model or target_model, "category": error.category,
            "status_code": error.status_code, "request_id": error.request_id,
            "query_source": query_source or "unknown",
        }
        raise

    for event in activity_events:
        yield event
    yield {
        "type": "runtime_activity", "area": "model", "status": "streaming",
        "model": target_model, "query_source": query_source or "unknown",
    }

    yield {"type": "message_start", "model": target_model}

    text_buf: list[str] = []
    thinking_buf: list[str] = []
    accumulator = ToolCallAccumulator()
    stop_reason: str | None = None
    final_usage: dict | None = None

    try:
        async for chunk in _iterate_response(response, signal):
            usage = _extract_usage(chunk)
            if usage is not None:
                final_usage = usage

            if not getattr(chunk, "choices", None):
                continue

            choice = chunk.choices[0]
            delta = choice.delta

            if getattr(choice, "finish_reason", None):
                stop_reason = choice.finish_reason

            content = getattr(delta, "content", None)
            if content:
                text_buf.append(content)
                yield {"type": "text_delta", "text": content}

            reasoning = getattr(delta, "reasoning_content", None)
            if reasoning:
                thinking_buf.append(reasoning)
                yield {"type": "thinking_delta", "thinking": reasoning}

            tool_call_deltas = getattr(delta, "tool_calls", None)
            if tool_call_deltas:
                accumulator.feed(tool_call_deltas)
    except litellm.APIConnectionError as e:
        # Stream failed mid-response — retry non-streaming
        logger.warning("Stream failed, retrying non-streaming: %s", e)
        try:
            text = await complete(
                messages, system_prompt=system_prompt,
                model=target_model, max_tokens=max_tokens,
                fallback_model=fallback_model, query_source=query_source,
            )
            yield {"type": "text_delta", "text": text}
            yield {
                "type": "message_stop", "stop_reason": "end_turn",
                "usage": final_usage or {"input_tokens": 0, "output_tokens": 0,
                                         "cache_read_input_tokens": 0,
                                         "cache_creation_input_tokens": 0},
                "model": target_model, "text": text, "thinking": None,
            }
            yield {
                "type": "runtime_activity", "area": "model",
                "status": "completed", "model": target_model,
                "query_source": query_source or "unknown",
            }
            return
        except Exception as e2:
            logger.error("Non-streaming fallback failed: %s", e2)
            raise e2

    if _signal_aborted(signal):
        yield {
            "type": "runtime_activity", "area": "model", "status": "interrupted",
            "model": target_model, "query_source": query_source or "unknown",
        }
        yield {
            "type": "message_stop", "stop_reason": "interrupted",
            "usage": final_usage or _empty_usage(), "model": target_model,
        }
        return

    # Emit tool calls after the stream (fragments are fully assembled)
    for tc in accumulator.assemble():
        yield {"type": "tool_use", "id": tc["id"], "name": tc["name"], "input": tc["input"]}

    if stop_reason == "tool_calls":
        stop_reason = "tool_use"
    elif stop_reason == "stop":
        stop_reason = "end_turn"
    elif stop_reason is None:
        stop_reason = "end_turn"

    actual_tokens = _usage_token_total(final_usage)
    if actual_tokens is not None:
        await reservation.settle(actual_tokens)

    yield {
        "type": "message_stop",
        "stop_reason": stop_reason,
        "usage": final_usage or {
            "input_tokens": 0, "output_tokens": 0,
            "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0,
        },
        "model": target_model,
        "text": "".join(text_buf) if text_buf else None,
        "thinking": "".join(thinking_buf) if thinking_buf else None,
    }
    yield {
        "type": "runtime_activity", "area": "model", "status": "completed",
        "model": target_model, "query_source": query_source or "unknown",
    }


def _empty_usage() -> dict[str, int]:
    return {
        "input_tokens": 0,
        "output_tokens": 0,
        "cache_read_input_tokens": 0,
        "cache_creation_input_tokens": 0,
    }


def _signal_aborted(signal: Any) -> bool:
    return signal is not None and bool(getattr(signal, "aborted", False))


async def _wait_for_abort(signal: Any) -> None:
    if _signal_aborted(signal):
        return
    wait = getattr(signal, "wait", None)
    if callable(wait):
        await wait()
        return
    while not _signal_aborted(signal):
        await asyncio.sleep(0.05)


async def _iterate_response(response: Any, signal: Any):
    iterator = response.__aiter__()
    while True:
        if _signal_aborted(signal):
            break
        next_task = asyncio.create_task(iterator.__anext__())
        abort_task = (
            asyncio.create_task(_wait_for_abort(signal))
            if signal is not None else None
        )
        if abort_task is None:
            try:
                yield await next_task
            except StopAsyncIteration:
                return
            continue

        done, _ = await asyncio.wait(
            (next_task, abort_task), return_when=asyncio.FIRST_COMPLETED,
        )
        if abort_task in done:
            next_task.cancel()
            await asyncio.gather(next_task, return_exceptions=True)
            break
        abort_task.cancel()
        await asyncio.gather(abort_task, return_exceptions=True)
        try:
            yield next_task.result()
        except StopAsyncIteration:
            return

    close = getattr(iterator, "aclose", None)
    if callable(close):
        result = close()
        if inspect.isawaitable(result):
            await result


# ── Fallback on rate limit ──────────────────────────────────────────────

async def _call_with_fallback(
    kwargs: dict,
    enable_thinking: bool,
    *,
    fallback_model: str | None = None,
    query_source: str | None = None,
    on_event: Callable[[dict[str, object]], None] | None = None,
):
    original_model = kwargs["model"]
    fallback_models = [
        model for model in _configured_fallback_models()
        if model != original_model
    ]
    if fallback_model and fallback_model != original_model:
        fallback_models.insert(0, _normalize_model_name(fallback_model))
    fallback_models = list(dict.fromkeys(fallback_models))
    attempts = [original_model, *fallback_models]

    def notify(event: dict[str, object]) -> None:
        if on_event is None:
            return
        try:
            on_event({"type": "runtime_activity", "area": "provider", **event})
        except Exception:
            pass

    for model_index, model in enumerate(attempts):
        retry_index = 0
        while True:
            call_kwargs = {**kwargs, "model": model, "num_retries": 0}
            if enable_thinking and _model_supports_thinking(model):
                call_kwargs["thinking"] = {"type": "enabled", "budget_tokens": 4096}
            try:
                estimated_tokens = (
                    _estimate_message_tokens(call_kwargs.get("messages", []))
                    + max(0, int(call_kwargs.get("max_tokens", 0) or 0))
                )
                async with provider_request_slot(
                    model, estimated_tokens=estimated_tokens,
                ) as reservation:
                    response = await litellm.acompletion(**call_kwargs)
                if model != original_model:
                    logger.info("Fell back from %s to %s", original_model, model)
                return response, model, reservation
            except ProviderRequestError:
                raise
            except Exception as error:
                category = _failure_category(error)
                details = _log_provider_failure(
                    error, model, query_source, retry_index + 1,
                )

                if category == "rate_limited":
                    if not _is_hard_rate_limit(error) and retry_index < MAX_RETRIES_PER_MODEL:
                        delay = _retry_delay(error, retry_index)
                        if delay is not None:
                            notify({
                                "status": "retrying", "model": model,
                                "category": category, "attempt": retry_index + 1,
                                "delay_seconds": round(delay, 2),
                            })
                            await _retry_sleep(delay)
                            retry_index += 1
                            continue
                    raise _provider_request_error(error, details, model) from error

                if category in ("provider_overloaded", "connection_failure"):
                    if retry_index < MAX_RETRIES_PER_MODEL:
                        delay = _retry_delay(error, retry_index)
                        if delay is not None:
                            notify({
                                "status": "retrying", "model": model,
                                "category": category, "attempt": retry_index + 1,
                                "delay_seconds": round(delay, 2),
                            })
                            await _retry_sleep(delay)
                            retry_index += 1
                            continue
                    if model_index + 1 < len(attempts):
                        notify({
                            "status": "fallback", "model": model,
                            "alternate_model": attempts[model_index + 1],
                            "category": category,
                        })
                        logger.info(
                            "Trying configured alternate after %s model=%s query_source=%s",
                            category, model, query_source or "unknown",
                        )
                        break

                elif category == "model_or_route_not_found" and model_index + 1 < len(attempts):
                    notify({
                        "status": "fallback", "model": model,
                        "alternate_model": attempts[model_index + 1],
                        "category": category,
                    })
                    logger.info(
                        "Trying configured alternate after model/route not found "
                        "model=%s query_source=%s",
                        model, query_source or "unknown",
                    )
                    break

                raise _provider_request_error(error, details, model) from error

    raise ProviderRequestError(
        f"No configured model could complete the request (primary={original_model})"
    )


# ── Non-streaming complete ──────────────────────────────────────────────

async def complete(
    messages: list,
    system_prompt: str = "",
    model: str | None = None,
    max_tokens: int = 512,
    response_format: dict | None = None,
    fallback_model: str | None = None,
    query_source: str | None = None,
) -> str:
    """Non-streaming one-shot completion. Used by compaction and classifier."""
    target_model = model or ACTIVE_MODEL
    oai_messages = to_openai(messages)

    system_blocks = _freeze_system_prompt(system_prompt)
    if system_blocks:
        oai_messages = [{"role": "system", "content": system_blocks}, *oai_messages]

    kwargs: dict = {
        "model": target_model,
        "messages": oai_messages,
        "max_tokens": max_tokens,
        "stream": False,
    }
    if response_format:
        kwargs["response_format"] = response_format

    response, _response_model, reservation = await _call_with_fallback(
        kwargs, enable_thinking=False, fallback_model=fallback_model,
        query_source=query_source,
    )
    actual_tokens = _usage_token_total(getattr(response, "usage", None))
    if actual_tokens is not None:
        await reservation.settle(actual_tokens)

    return response.choices[0].message.content or ""