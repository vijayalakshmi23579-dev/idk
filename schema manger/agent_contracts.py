"""Typed contracts shared by the planner, task workers, and verifier."""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from split_gate import SplitEstimate


class TaskBudget(BaseModel):
    max_tokens: int = Field(default=8_000, ge=1, description="Maximum cumulative output tokens.")
    max_input_tokens: int = Field(default=100_000, ge=1, description="Maximum cumulative input tokens.")
    max_calls: int = Field(default=8, ge=1, le=32, description="Maximum model/tool turns.")
    max_provider_attempts: int = Field(default=32, ge=1, description="Maximum provider requests including retries.")
    max_seconds: int = Field(default=120, ge=1, le=1_800, description="Wall-clock deadline.")
    max_tool_errors: int = Field(default=3, ge=1, le=20, description="Maximum tool errors before stopping.")
    max_verifier_tokens: int = Field(default=1_000, ge=1, description="Verifier output-token cap.")
    max_verifier_calls: int = Field(default=2, ge=1, le=4, description="Verifier model/tool-turn cap.")
    max_verifier_seconds: int = Field(default=60, ge=1, le=300, description="Verifier wall-clock deadline.")


class TaskRunBudget(BaseModel):
    schema_version: Literal["1.0"] = "1.0"
    max_output_tokens: int = Field(default=72_000, ge=1)
    max_input_tokens: int = Field(default=1_000_000, ge=1)
    max_model_calls: int = Field(default=80, ge=1)
    max_provider_attempts: int = Field(default=320, ge=1)
    max_worker_seconds: float = Field(default=1_800, ge=1)


class TaskLifecycleEvent(BaseModel):
    model_config = ConfigDict(extra="allow")

    schema_version: Literal["1.0"]
    type: Literal["task_lifecycle"]
    event_id: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    attempt_id: str | None = None
    task_id: str = Field(min_length=1)
    status: Literal[
        "started", "succeeded", "failed", "blocked",
        "needs_escalation", "contract_change_request",
    ]
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


class JoinValueRef(BaseModel):
    dependency_id: str = Field(min_length=1)
    json_path: str = ""


class JoinCheck(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal["equals", "schema"]
    left: JoinValueRef | None = None
    right: JoinValueRef | None = None
    schema_definition: dict[str, Any] | None = None

    @model_validator(mode="after")
    def _validate_kind_fields(self) -> "JoinCheck":
        if self.kind == "equals" and (self.left is None or self.right is None):
            raise ValueError("equals join checks require left and right references")
        if self.kind == "schema" and self.schema_definition is None:
            raise ValueError("schema join checks require a JSON Schema")
        return self


class TaskSpec(BaseModel):
    schema_version: Literal["1.0"] = "1.0"
    id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
    role: Literal["worker", "integrator"] = Field(
        default="worker",
        description=(
            "Mark the single final integration node when a code DAG has multiple "
            "writers. It must depend on every writer and own disjoint output paths."
        ),
    )
    instruction: str = Field(min_length=1, max_length=20_000)
    recovery_of: str | None = Field(
        default=None,
        description="Failed or blocked node ID this bounded recovery task replaces.",
    )
    needs: list[str] = Field(default_factory=list)
    read_files: list[str] = Field(default_factory=list)
    write_files: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(
        default_factory=list,
        max_length=20,
        description="Context assumptions the worker should preserve or report as invalid.",
    )
    join_criteria: list[str] = Field(
        default_factory=list,
        max_length=20,
        description="Conditions the parent should check when integrating this result.",
    )
    join_checks: list[JoinCheck] = Field(
        default_factory=list,
        description="Deterministic assertions over outputs from declared dependencies.",
    )
    non_goals: list[str] = Field(
        default_factory=list,
        description="Explicitly excluded work for this node.",
    )
    output_schema: dict[str, Any] | None = Field(
        default=None,
        description="JSON Schema required for the worker's structured output.",
    )
    dependency_output_schemas: dict[str, dict[str, Any]] = Field(
        default_factory=dict,
        description=(
            "Optional JSON Schemas keyed by prerequisite ID, checked before "
            "this task receives dependency outputs."
        ),
    )
    acceptance: list[AcceptanceCheck] = Field(
        default_factory=list,
        description="Deterministic checks; exists checks must target write_files.",
    )
    budget: TaskBudget = Field(
        default_factory=TaskBudget,
        description="Hard per-node token, call, time, and tool-error limits.",
    )
    complexity: int = Field(
        default=3, ge=1, le=5,
        description="Planner estimate from routine (1) to strong reasoning (5).",
    )
    tier_hint: Literal["T1", "T2", "T3"] | None = Field(
        default=None,
        description="Optional override of the configured complexity-to-tier mapping.",
    )
    verification_mode: Literal["none", "independent"] = Field(
        default="none",
        description=(
            "Request a fresh-context independent verifier after deterministic checks. "
            "Requires a verifier model distinct from the worker model."
        ),
    )
    max_escalations: int = Field(
        default=1,
        ge=0,
        le=1,
        description="Maximum evidence-triggered stronger-tier retries within this task budget.",
    )

    @field_validator("instruction")
    @classmethod
    def _trim_instruction(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("instruction must not be empty")
        return value

    @field_validator("needs", "read_files", "write_files")
    @classmethod
    def _unique_values(cls, values: list[str]) -> list[str]:
        if len(values) != len(set(values)):
            raise ValueError("list values must be unique")
        return values


class TaskExpansionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.0"] = "1.0"
    reason: str = Field(min_length=1, max_length=1_200)
    estimate: SplitEstimate
    tasks: list[TaskSpec] = Field(min_length=1, max_length=8)

    @model_validator(mode="after")
    def _validate_proposal(self) -> "TaskExpansionRequest":
        task_ids = [task.id for task in self.tasks]
        if len(task_ids) != len(set(task_ids)):
            raise ValueError("Expansion task IDs must be unique")
        if any(task.recovery_of is not None for task in self.tasks):
            raise ValueError("Expansion proposals cannot contain recovery tasks")
        return self


class ArtifactRef(BaseModel):
    schema_version: Literal["1.0"] = "1.0"
    path: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    size_bytes: int = Field(ge=0)
    storage_key: str | None = None


class WorkerUsage(BaseModel):
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    calls: int = Field(default=0, ge=0)
    provider_attempts: int = Field(default=0, ge=0)
    seconds: float = Field(default=0, ge=0)
    tool_errors: int = Field(default=0, ge=0)


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
    expansion_request: TaskExpansionRequest | None = None
    diagnostics: str | None = None
    models_used: list[str] = Field(default_factory=list)
    escalations: int = Field(default=0, ge=0)


class TaskWorkerRequest(BaseModel):
    status: Literal["failed", "needs_escalation", "contract_change_request"]
    summary: str = Field(max_length=1_200)
    output: dict[str, Any] = Field(default_factory=dict)
    expansion_request: TaskExpansionRequest | None = None
    diagnostics: str | None = None

    @model_validator(mode="after")
    def _validate_expansion_status(self) -> "TaskWorkerRequest":
        if self.expansion_request is not None and self.status != "contract_change_request":
            raise ValueError(
                "Expansion proposals require contract_change_request status"
            )
        return self


__all__ = [
    "AcceptanceCheck", "ArtifactRef", "JoinCheck", "JoinValueRef",
    "TaskBudget", "TaskLifecycleEvent",
    "TaskExpansionRequest", "TaskSpec", "TaskRunBudget",
    "TaskResultEnvelope", "TaskWorkerRequest",
    "VerificationResult", "WorkerUsage",
]