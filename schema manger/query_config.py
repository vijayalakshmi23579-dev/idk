"""
query_config.py — session-stable config for the query loop.

Exact port of src/query/config.ts (25 minified lines).

QueryConfig is built ONCE per query() call and never changes mid-turn.
It captures the session ID and the feature gates that were latched at
entry — GrowthBook flips mid-session would otherwise churn the loop's
behavior between iterations.

Source gates (from buildQueryConfig):
  - streamingToolExecution  → CLAUDE_CODE_DISABLE_STREAMING_TOOL_EXECUTION
                              (inverted: default ON)
  - emitToolUseSummaries    → CLAUDE_CODE_EMIT_TOOL_USE_SUMMARIES
                              (default OFF)
  - isAnt                   → USER_TYPE === 'ant'
  - fastModeEnabled         → !CLAUDE_CODE_DISABLE_FAST_MODE
"""
from __future__ import annotations

import os
import uuid
from dataclasses import dataclass, field


@dataclass(frozen=True)
class QueryGates:
    streaming_tool_execution: bool
    emit_tool_use_summaries: bool
    is_ant: bool
    fast_mode_enabled: bool


@dataclass(frozen=True)
class QueryConfig:
    session_id: str
    gates: QueryGates


def _env_truthy(name: str) -> bool:
    return os.environ.get(name, "").lower() in ("1", "true", "yes", "on")


def build_query_config(session_id: str | None = None) -> QueryConfig:
    """Snapshot gates once. Never call mid-loop."""
    return QueryConfig(
        session_id=session_id or str(uuid.uuid4()),
        gates=QueryGates(
            streaming_tool_execution=not _env_truthy(
                "CLAUDE_CODE_DISABLE_STREAMING_TOOL_EXECUTION"
            ),
            emit_tool_use_summaries=_env_truthy(
                "CLAUDE_CODE_EMIT_TOOL_USE_SUMMARIES"
            ),
            is_ant=os.environ.get("USER_TYPE") == "ant",
            fast_mode_enabled=not _env_truthy(
                "CLAUDE_CODE_DISABLE_FAST_MODE"
            ),
        ),
    )