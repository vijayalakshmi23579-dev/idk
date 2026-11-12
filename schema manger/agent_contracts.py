"""Typed contracts shared by the planner, task workers, and verifier."""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class TaskBudget(BaseModel):
    max_tokens: int = Field(default=8_000, ge=1, description="Maximum cumulative output tokens.")
    max_calls: int = Field(default=8, ge=1, le=32, description="Maximum model/tool turns.")
    max_seconds: int = Field(default=120, ge=1, le=1_800, description="Wall-clock deadline.")
    max_tool_errors: int = Field(default=3, ge=1, le=20, description="Maximum tool errors before stopping.")
    max_verifier_tokens: int = Field(default=1_000, ge=1, description="Verifier output-token cap.")
    max_verifier_calls: int = Field(default=2, ge=1, le=4, description="Verifier model/tool-turn cap.")
    max_verifier_seconds: int = Field(default=60, ge=1, le=300, description="Verifier wall-clock deadline.")


class TaskRunBudget(BaseModel):
    schema_version: Literal["1.0"] = "1.0"
    max_output_tokens: int = Field(default=72_000, ge=1)
    max_model_calls: int = Field(default=80, ge=1)
    max_worker_seconds: float = Field(default=1_800, ge=1)


class TaskLifecycleEvent(BaseModel):
    model_config = ConfigDict(extra="allow")

    schema_version: Literal["1.0"]
    type: Literal["task_lifecycle"]
    event_id: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    attempt_id: str | None = None
    task_id: str = Field(min_length=1)
    status: Literal["started", "succeeded", "failed", "blocked"]
    needs: list[str] = Field(default_factory=list)
    occurred_at: float = Field(ge=0)
    model_id: str | None = None


class AcceptanceCheck(BaseModel):
    kind: Literal["exists", "assert", "schema"]
    spec: str = Field(
        min_length=1,
        description=(
            "exists: relative path. assert/schema: JSON spec with a confined "
            "file path, or target='output' for a worker's structured output."
        ),
    )


class ArtifactRef(BaseModel):
    schema_version: Literal["1.0"] = "1.0"
    path: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    size_bytes: int = Field(ge=0)
    storage_key: str | None = None


class WorkerUsage(BaseModel):
    output_tokens: int = Field(default=0, ge=0)
    calls: int = Field(default=0, ge=0)
    seconds: float = Field(default=0, ge=0)


class VerificationResult(BaseModel):
    schema_version: Literal["1.0"] = "1.0"
    passed: bool
    confidence: float = Field(ge=0, le=1)
    findings: list[str] = Field(default_factory=list, max_length=10)
    model_id: str = Field(min_length=1)
    usage: WorkerUsage = Field(default_factory=WorkerUsage)


class TaskResultEnvelope(BaseModel):
    schema_version: Literal["1.0"] = "1.0"
    run_id: str
    node_id: str
    status: Literal["ok", "failed", "needs_escalation", "contract_change_request"]
    model_id: str | None = None
    summary: str = Field(max_length=1_200)
    output: dict[str, Any] = Field(default_factory=dict)
    artifacts: list[ArtifactRef] = Field(default_factory=list)
    usage: WorkerUsage = Field(default_factory=WorkerUsage)
    verification: VerificationResult | None = None
    diagnostics: str | None = None


__all__ = [
    "AcceptanceCheck", "ArtifactRef", "TaskBudget", "TaskLifecycleEvent",
    "TaskRunBudget", "TaskResultEnvelope",
    "VerificationResult", "WorkerUsage",
]