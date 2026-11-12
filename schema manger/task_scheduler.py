"""Validated task DAG scheduling with dependency payloads and file leases."""
from __future__ import annotations

import asyncio
import os
import threading
import time
import uuid
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Awaitable, Callable, Literal

from pydantic import BaseModel, Field, field_validator
from agent_contracts import AcceptanceCheck, TaskBudget
from task_checks import validate_acceptance_check


class TaskSpec(BaseModel):
    schema_version: Literal["1.0"] = "1.0"
    id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
    instruction: str = Field(min_length=1, max_length=20_000)
    recovery_of: str | None = Field(
        default=None,
        description="Failed or blocked node ID this bounded recovery task replaces.",
    )
    needs: list[str] = Field(default_factory=list)
    read_files: list[str] = Field(default_factory=list)
    write_files: list[str] = Field(default_factory=list)
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
        description="Optional override for the configured complexity-to-tier mapping.",
    )
    verification_mode: Literal["none", "independent"] = Field(
        default="none",
        description=(
            "Request a fresh-context independent verifier after deterministic checks. "
            "Requires a verifier model distinct from the worker model."
        ),
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


class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    BLOCKED = "blocked"


@dataclass(frozen=True)
class ValidatedTask:
    spec: TaskSpec
    read_paths: frozenset[str]
    write_paths: frozenset[str]


@dataclass
class TaskOutcome:
    task_id: str
    status: TaskStatus = TaskStatus.PENDING
    output: Any = None
    error: str = ""


@dataclass(frozen=True)
class TaskFileScope:
    task_id: str
    project_root: str
    read_paths: frozenset[str]
    write_paths: frozenset[str]
    leases: "FileLeaseRegistry"

    def check_access(self, tool_name: str, file_path: str) -> str | None:
        try:
            canonical = canonical_task_path(file_path, self.project_root)
        except ValueError as error:
            return str(error)

        if tool_name in ("Edit", "Write"):
            if canonical not in self.write_paths:
                return f"Task {self.task_id} is not permitted to write {canonical}."
            if not self.leases.owns(self.task_id, canonical):
                return f"Task {self.task_id} does not hold the write lease for {canonical}."
            return None

        if tool_name == "Read" and canonical not in self.read_paths | self.write_paths:
            return f"Task {self.task_id} is not permitted to read {canonical}."
        return None


class FileLeaseConflict(ValueError):
    pass


class FileLeaseRegistry:
    """Atomic, process-local write leases for canonical project file paths."""

    def __init__(self, project_root: str):
        self.project_root = os.path.realpath(os.path.abspath(project_root))
        self._owners: dict[str, str] = {}
        self._lock = threading.Lock()

    def resolve(self, path: str) -> str:
        return canonical_task_path(path, self.project_root)

    def acquire(self, task_id: str, paths: frozenset[str]) -> None:
        with self._lock:
            conflicts = [
                path for path in paths
                if any(
                    owner != task_id and _paths_overlap(path, leased)
                    for leased, owner in self._owners.items()
                )
            ]
            if conflicts:
                raise FileLeaseConflict(
                    f"Write lease conflict for task {task_id}: "
                    + ", ".join(sorted(conflicts))
                )
            for path in paths:
                self._owners[path] = task_id

    def release(self, task_id: str) -> None:
        with self._lock:
            self._owners = {
                path: owner for path, owner in self._owners.items()
                if owner != task_id
            }

    def owns(self, task_id: str, path: str) -> bool:
        with self._lock:
            return self._owners.get(path) == task_id


def canonical_task_path(path: str, project_root: str) -> str:
    if not isinstance(path, str) or not path.strip():
        raise ValueError("Task file paths must be non-empty strings")
    expanded = os.path.expanduser(path.strip())
    if not os.path.isabs(expanded):
        expanded = os.path.join(project_root, expanded)
    canonical = os.path.realpath(os.path.abspath(expanded))
    root = os.path.realpath(os.path.abspath(project_root))
    try:
        if os.path.commonpath((root, canonical)) != root:
            raise ValueError(f"Task file path is outside the project: {path}")
    except ValueError as error:
        if str(error).startswith("Task file path is outside"):
            raise
        raise ValueError(f"Task file path is outside the project: {path}") from error
    if canonical == root:
        raise ValueError("A project directory cannot be declared as a file")
    return canonical


def _paths_overlap(left: str, right: str) -> bool:
    try:
        return os.path.commonpath((left, right)) in (left, right)
    except ValueError:
        return False


def validate_task_graph(
    tasks: list[TaskSpec], project_root: str,
) -> list[ValidatedTask]:
    if not tasks:
        raise ValueError("A task graph must contain at least one task")
    if len({task.id for task in tasks}) != len(tasks):
        raise ValueError("Task IDs must be unique")

    by_id = {task.id: task for task in tasks}
    for task in tasks:
        if task.id in task.needs:
            raise ValueError(f"Task {task.id} cannot depend on itself")
        missing = set(task.needs) - by_id.keys()
        if missing:
            raise ValueError(
                f"Task {task.id} has unknown dependencies: {', '.join(sorted(missing))}"
            )

    visit_state: dict[str, int] = {}

    def visit(task_id: str) -> None:
        state = visit_state.get(task_id, 0)
        if state == 1:
            raise ValueError(f"Task graph contains a dependency cycle at {task_id}")
        if state == 2:
            return
        visit_state[task_id] = 1
        for dependency in by_id[task_id].needs:
            visit(dependency)
        visit_state[task_id] = 2

    for task in tasks:
        visit(task.id)

    def depends_transitively(task_id: str, dependency_id: str) -> bool:
        pending = list(by_id[task_id].needs)
        seen: set[str] = set()
        while pending:
            current = pending.pop()
            if current == dependency_id:
                return True
            if current not in seen:
                seen.add(current)
                pending.extend(by_id[current].needs)
        return False

    normalized: list[ValidatedTask] = []
    for task in tasks:
        reads = frozenset(canonical_task_path(path, project_root) for path in task.read_files)
        writes = frozenset(canonical_task_path(path, project_root) for path in task.write_files)
        for check in task.acceptance:
            declared_path = validate_acceptance_check(check)
            if declared_path is None:
                continue
            check_path = canonical_task_path(declared_path, project_root)
            if check_path not in writes:
                raise ValueError(
                    f"Task {task.id} acceptance path must be in write_files: {declared_path}"
                )
        if task.output_schema is not None:
            from jsonschema import Draft202012Validator
            from jsonschema.exceptions import SchemaError

            try:
                Draft202012Validator.check_schema(task.output_schema)
            except SchemaError as error:
                raise ValueError(
                    f"Task {task.id} has an invalid output schema"
                ) from error
        unknown_dependency_schemas = (
            set(task.dependency_output_schemas) - set(task.needs)
        )
        if unknown_dependency_schemas:
            raise ValueError(
                f"Task {task.id} has dependency schemas for undeclared needs: "
                + ", ".join(sorted(unknown_dependency_schemas))
            )
        if task.dependency_output_schemas:
            from jsonschema import Draft202012Validator
            from jsonschema.exceptions import SchemaError

            for dependency_id, schema in task.dependency_output_schemas.items():
                try:
                    Draft202012Validator.check_schema(schema)
                except SchemaError as error:
                    raise ValueError(
                        f"Task {task.id} has an invalid output schema for "
                        f"dependency {dependency_id}"
                    ) from error
        normalized.append(ValidatedTask(task, reads, writes))

    for index, left in enumerate(normalized):
        for right in normalized[index + 1:]:
            ordered = (
                depends_transitively(left.spec.id, right.spec.id)
                or depends_transitively(right.spec.id, left.spec.id)
            )
            write_overlaps = any(
                _paths_overlap(path_a, path_b)
                for path_a in left.write_paths
                for path_b in right.write_paths
            )
            if write_overlaps and not ordered:
                raise ValueError(
                    f"Tasks {left.spec.id} and {right.spec.id} have overlapping "
                    "write scopes without an explicit dependency path"
                )
            read_write_overlaps = any(
                _paths_overlap(write_path, read_path)
                for write_paths, read_paths in (
                    (left.write_paths, right.read_paths),
                    (right.write_paths, left.read_paths),
                )
                for write_path in write_paths
                for read_path in read_paths
            )
            if read_write_overlaps and not ordered:
                raise ValueError(
                    f"Tasks {left.spec.id} and {right.spec.id} have overlapping "
                    "read/write scopes without an explicit dependency path"
                )
    return normalized


TaskRunner = Callable[
    [TaskSpec, dict[str, Any], TaskFileScope], Awaitable[Any],
]


async def run_task_graph(
    tasks: list[TaskSpec],
    runner: TaskRunner,
    *,
    project_root: str,
    run_id: str | None = None,
    max_parallel: int = 4,
    on_event: Callable[[dict[str, object]], None] | None = None,
) -> list[TaskOutcome]:
    """Run conflict-free ready tasks concurrently within the configured limit.

    A dependent receives a mapping of its declared prerequisite IDs to their
    successful output. Write conflicts are serialized by dependencies and
    process-local leases; independent readers and disjoint writers may overlap.
    Failed prerequisites block dependents; independent branches continue.
    """
    if max_parallel < 1:
        raise ValueError("max_parallel must be at least one")
    graph = validate_task_graph(tasks, project_root)
    run_id = run_id or uuid.uuid4().hex
    leases = FileLeaseRegistry(project_root)
    outcomes = {task.spec.id: TaskOutcome(task.spec.id) for task in graph}
    nodes = {task.spec.id: task for task in graph}
    active: dict[asyncio.Task, str] = {}
    started_at: dict[str, float] = {}
    attempt_ids: dict[str, str] = {}

    def emit(task_id: str, status: str, **details: object) -> None:
        if on_event is None:
            return
        node = nodes[task_id]
        event: dict[str, object] = {
            "schema_version": "1.0",
            "type": "task_lifecycle",
            "event_id": uuid.uuid4().hex,
            "run_id": run_id,
            "attempt_id": attempt_ids.get(task_id),
            "task_id": task_id,
            "status": status,
            "needs": list(node.spec.needs),
            "occurred_at": time.time(),
        }
        event.update(details)
        try:
            on_event(event)
        except Exception:
            pass

    async def execute(node: ValidatedTask) -> Any:
        scope = TaskFileScope(
            task_id=node.spec.id,
            project_root=leases.project_root,
            read_paths=node.read_paths,
            write_paths=node.write_paths,
            leases=leases,
        )
        if node.write_paths:
            leases.acquire(node.spec.id, node.write_paths)
        try:
            dependency_outputs: dict[str, Any] = {
                dependency: outcomes[dependency].output
                for dependency in node.spec.needs
            }
            for dependency_id, schema in node.spec.dependency_output_schemas.items():
                result = dependency_outputs[dependency_id]
                if hasattr(result, "output"):
                    instance = result.output
                elif isinstance(result, dict) and "output" in result:
                    instance = result["output"]
                else:
                    instance = result
                from jsonschema import Draft202012Validator

                errors = list(Draft202012Validator(schema).iter_errors(instance))
                if errors:
                    path = ".".join(str(part) for part in errors[0].absolute_path)
                    location = f" at {path}" if path else ""
                    raise ValueError(
                        f"Dependency {dependency_id} output failed its schema{location}"
                    )
            return await runner(node.spec, dependency_outputs, scope)
        finally:
            if node.write_paths:
                leases.release(node.spec.id)

    while True:
        pending = [
            node for node in graph
            if outcomes[node.spec.id].status == TaskStatus.PENDING
        ]
        if not pending and not active:
            break

        for node in pending:
            dependency_states = [
                outcomes[dependency].status for dependency in node.spec.needs
            ]
            if any(state in (TaskStatus.FAILED, TaskStatus.BLOCKED) for state in dependency_states):
                outcomes[node.spec.id].status = TaskStatus.BLOCKED
                outcomes[node.spec.id].error = "A required dependency did not succeed."
                emit(node.spec.id, "blocked")

        ready = [
            node for node in pending
            if outcomes[node.spec.id].status == TaskStatus.PENDING
            and all(outcomes[dep].status == TaskStatus.SUCCEEDED for dep in node.spec.needs)
        ]
        available_slots = max_parallel - len(active)
        launch = ready[:available_slots] if available_slots > 0 else []

        for node in launch:
            outcome = outcomes[node.spec.id]
            outcome.status = TaskStatus.RUNNING
            started_at[node.spec.id] = time.perf_counter()
            attempt_ids[node.spec.id] = uuid.uuid4().hex
            emit(node.spec.id, "started")
            task = asyncio.create_task(execute(node))
            active[task] = node.spec.id

        if active and (not launch or len(launch) < len(ready)):
            done, _ = await asyncio.wait(active, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                task_id = active.pop(task)
                outcome = outcomes[task_id]
                try:
                    outcome.output = task.result()
                    outcome.status = TaskStatus.SUCCEEDED
                    emit(
                        task_id,
                        "succeeded",
                        duration_ms=round((time.perf_counter() - started_at[task_id]) * 1000),
                        model_id=getattr(outcome.output, "model_id", None),
                    )
                except asyncio.CancelledError:
                    outcome.status = TaskStatus.FAILED
                    outcome.error = "Task runner was cancelled."
                    emit(
                        task_id,
                        "failed",
                        error_type="CancelledError",
                        duration_ms=round((time.perf_counter() - started_at[task_id]) * 1000),
                    )
                except Exception as error:
                    outcome.status = TaskStatus.FAILED
                    outcome.error = f"{type(error).__name__}: {error}"
                    emit(
                        task_id,
                        "failed",
                        error_type=type(error).__name__,
                        duration_ms=round((time.perf_counter() - started_at[task_id]) * 1000),
                    )
        elif not active and not launch:
            unresolved = [
                task_id for task_id, outcome in outcomes.items()
                if outcome.status == TaskStatus.PENDING
            ]
            if unresolved:
                raise RuntimeError(f"Scheduler stalled with pending tasks: {unresolved}")

    return [outcomes[node.spec.id] for node in graph]


__all__ = [
    "TaskSpec", "TaskStatus", "TaskOutcome", "TaskFileScope",
    "FileLeaseConflict", "FileLeaseRegistry", "ValidatedTask",
    "canonical_task_path", "validate_task_graph", "run_task_graph",
]