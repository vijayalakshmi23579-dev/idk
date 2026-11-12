"""Validated task DAG scheduling with dependency payloads and file leases."""
from __future__ import annotations

import asyncio
import hashlib
import os
import shutil
import subprocess
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass, field, replace
from enum import Enum
from pathlib import Path
from typing import Any, Awaitable, Callable

from agent_contracts import (
    AcceptanceCheck, JoinCheck, TaskBudget, TaskResultEnvelope, TaskSpec,
)
from task_checks import evaluate_join_check, validate_acceptance_check, validate_join_check


class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    BLOCKED = "blocked"
    NEEDS_ESCALATION = "needs_escalation"
    CONTRACT_CHANGE_REQUEST = "contract_change_request"


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
    staging_root: str | None = None
    baseline_write_hashes: dict[str, str | None] = field(default_factory=dict)
    staging_kind: str | None = None

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

    def execution_path(self, tool_name: str, file_path: str) -> str:
        error = self.check_access(tool_name, file_path)
        if error:
            raise ValueError(error)
        canonical = canonical_task_path(file_path, self.project_root)
        if self.staging_root is None:
            return canonical
        return os.path.join(
            self.staging_root,
            os.path.relpath(canonical, self.project_root),
        )

    def externalize_result(self, value: Any) -> Any:
        if self.staging_root is None:
            return value
        if isinstance(value, list):
            return [self.externalize_result(item) for item in value]
        if not isinstance(value, dict):
            return value
        external = {}
        path_fields = {"file_path", "filePath", "path"}
        for key, item in value.items():
            if key in path_fields and isinstance(item, str):
                staged_prefix = self.staging_root + os.sep
                if item.startswith(staged_prefix):
                    relative = os.path.relpath(item, self.staging_root)
                    external[key] = os.path.join(self.project_root, relative)
                    continue
            external[key] = self.externalize_result(item)
        return external


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


def _file_sha256(path: str) -> str | None:
    if not os.path.isfile(path):
        return None
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stage_task_scope(scope: TaskFileScope) -> TaskFileScope:
    if not scope.write_paths:
        return scope
    staging_root, staging_kind = _create_staging_area(scope.project_root)
    baseline: dict[str, str | None] = {}
    staged_scope = replace(
        scope,
        staging_root=staging_root,
        staging_kind=staging_kind,
        baseline_write_hashes=baseline,
    )
    try:
        for path in sorted(scope.read_paths | scope.write_paths):
            staged_path = os.path.join(
                staging_root,
                os.path.relpath(path, scope.project_root),
            )
            os.makedirs(os.path.dirname(staged_path), exist_ok=True)
            if os.path.isfile(path):
                shutil.copy2(path, staged_path)
            elif os.path.exists(path):
                raise ValueError(f"Task-scoped path is not a regular file: {path}")
            if path in scope.write_paths:
                baseline[path] = _file_sha256(path)
        return staged_scope
    except Exception:
        _remove_staging_area(staged_scope)
        raise


def _create_staging_area(project_root: str) -> tuple[str, str]:
    """Use a detached worktree for Git roots; otherwise use an isolated directory."""
    root = os.path.realpath(os.path.abspath(project_root))
    try:
        git_root = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        git_root = None

    if git_root is not None and git_root.returncode == 0:
        canonical_git_root = os.path.realpath(git_root.stdout.strip())
        if canonical_git_root == root:
            container = tempfile.mkdtemp(prefix="agent-task-worktree-")
            worktree = os.path.join(container, "workspace")
            try:
                result = subprocess.run(
                    ["git", "worktree", "add", "--detach", worktree, "HEAD"],
                    cwd=root,
                    capture_output=True,
                    text=True,
                    timeout=30,
                    check=False,
                )
            except (OSError, subprocess.TimeoutExpired):
                result = None
            if result is not None and result.returncode == 0:
                return worktree, "git_worktree"
            shutil.rmtree(container, ignore_errors=True)
            try:
                subprocess.run(
                    ["git", "worktree", "prune"],
                    cwd=root,
                    capture_output=True,
                    text=True,
                    timeout=30,
                    check=False,
                )
            except (OSError, subprocess.TimeoutExpired):
                pass

    return tempfile.mkdtemp(prefix="agent-task-stage-"), "directory"


def _remove_staging_area(scope: TaskFileScope) -> None:
    if scope.staging_root is None:
        return
    if scope.staging_kind != "git_worktree":
        shutil.rmtree(scope.staging_root, ignore_errors=True)
        return

    try:
        result = subprocess.run(
            ["git", "worktree", "remove", "--force", scope.staging_root],
            cwd=scope.project_root,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        if result.returncode != 0:
            shutil.rmtree(scope.staging_root, ignore_errors=True)
            subprocess.run(
                ["git", "worktree", "prune"],
                cwd=scope.project_root,
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
    except (OSError, subprocess.TimeoutExpired):
        shutil.rmtree(scope.staging_root, ignore_errors=True)
    shutil.rmtree(os.path.dirname(scope.staging_root), ignore_errors=True)


def _commit_task_staging(scope: TaskFileScope) -> None:
    if scope.staging_root is None:
        return

    changes: list[tuple[str, str]] = []
    for path in sorted(scope.write_paths):
        if not scope.leases.owns(scope.task_id, path):
            raise FileLeaseConflict(
                f"Task {scope.task_id} lost its write lease for {path}."
            )
        staged_path = scope.execution_path("Write", path)
        current_hash = _file_sha256(path)
        staged_hash = _file_sha256(staged_path)
        if current_hash == staged_hash:
            continue
        if current_hash != scope.baseline_write_hashes.get(path):
            raise FileLeaseConflict(
                f"Workspace file changed while Task {scope.task_id} was staged: {path}"
            )
        if not os.path.isfile(staged_path):
            if scope.baseline_write_hashes.get(path) is not None:
                raise ValueError(f"Staged worker output is missing: {path}")
            continue
        if staged_hash == scope.baseline_write_hashes.get(path):
            continue
        changes.append((path, staged_path))

    prepared: list[tuple[str, str, str | None]] = []
    applied: list[tuple[str, str | None]] = []
    try:
        for destination, staged_path in changes:
            os.makedirs(os.path.dirname(destination), exist_ok=True)
            fd, temporary = tempfile.mkstemp(
                prefix=".agent-task-output-",
                dir=os.path.dirname(destination),
            )
            os.close(fd)
            shutil.copy2(staged_path, temporary)
            backup = None
            if os.path.isfile(destination):
                backup_fd, backup = tempfile.mkstemp(
                    prefix=".agent-task-backup-",
                    dir=os.path.dirname(destination),
                )
                os.close(backup_fd)
                shutil.copy2(destination, backup)
            prepared.append((destination, temporary, backup))

        for destination, temporary, backup in prepared:
            os.replace(temporary, destination)
            applied.append((destination, backup))
    except Exception:
        for destination, backup in reversed(applied):
            try:
                if backup is None:
                    os.unlink(destination)
                else:
                    os.replace(backup, destination)
            except OSError:
                pass
        raise
    finally:
        for _destination, temporary, backup in prepared:
            for path in (temporary, backup):
                if path is not None:
                    try:
                        os.unlink(path)
                    except FileNotFoundError:
                        pass


def _cleanup_task_staging(scope: TaskFileScope) -> None:
    _remove_staging_area(scope)


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
        for check in task.join_checks:
            validate_join_check(check)
            if check.kind == "equals":
                assert check.left is not None and check.right is not None
                referenced = {check.left.dependency_id, check.right.dependency_id}
                unknown = referenced - set(task.needs)
                if unknown:
                    raise ValueError(
                        f"Task {task.id} join check references undeclared dependencies: "
                        + ", ".join(sorted(unknown))
                    )
        normalized.append(ValidatedTask(task, reads, writes))

    integrators = [node for node in normalized if node.spec.role == "integrator"]
    if integrators:
        if len(integrators) != 1:
            raise ValueError("A Task DAG may declare only one integrator node")
        integrator = integrators[0]
        if not integrator.write_paths:
            raise ValueError("The integrator node must declare its output write scope")
        for node in normalized:
            if node.spec.id == integrator.spec.id:
                continue
            if node.write_paths and not depends_transitively(
                integrator.spec.id, node.spec.id,
            ):
                raise ValueError(
                    f"Integrator {integrator.spec.id} must depend on writer {node.spec.id}."
                )
            if any(
                _paths_overlap(integrator_path, worker_path)
                for integrator_path in integrator.write_paths
                for worker_path in node.write_paths
            ):
                raise ValueError(
                    f"Integrator {integrator.spec.id} must not share write paths "
                    f"with Task {node.spec.id}."
                )

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
NodeCommitCallback = Callable[
    [TaskSpec, TaskResultEnvelope, str, list[dict[str, Any]]], None,
]


def _staged_write_manifest(scope: TaskFileScope) -> list[dict[str, Any]]:
    return [
        {
            "path": os.path.relpath(path, scope.project_root),
            "before_sha256": scope.baseline_write_hashes.get(path),
            "after_sha256": _file_sha256(scope.execution_path("Write", path)),
        }
        for path in sorted(scope.write_paths)
    ]


async def run_task_graph(
    tasks: list[TaskSpec],
    runner: TaskRunner,
    *,
    project_root: str,
    run_id: str | None = None,
    max_parallel: int = 4,
    on_event: Callable[[dict[str, object]], None] | None = None,
    on_node_commit: NodeCommitCallback | None = None,
    cancel_event: asyncio.Event | None = None,
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
            scope = _stage_task_scope(scope)
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
            for check in node.spec.join_checks:
                if not evaluate_join_check(check, dependency_outputs):
                    raise ValueError(
                        f"Task {node.spec.id} join check failed: {check.kind}"
                    )
            result = await runner(node.spec, dependency_outputs, scope)
            if isinstance(result, TaskResultEnvelope) and result.status == "ok":
                if on_node_commit is not None and node.write_paths:
                    manifest = _staged_write_manifest(scope)
                    on_node_commit(node.spec, result, "prepared", manifest)
                _commit_task_staging(scope)
                if on_node_commit is not None and node.write_paths:
                    on_node_commit(node.spec, result, "committed", manifest)
            return result
        finally:
            _cleanup_task_staging(scope)
            if node.write_paths:
                leases.release(node.spec.id)

    while True:
        pending = [
            node for node in graph
            if outcomes[node.spec.id].status == TaskStatus.PENDING
        ]
        if not pending and not active:
            break

        if cancel_event is not None and cancel_event.is_set():
            for node in pending:
                outcome = outcomes[node.spec.id]
                outcome.status = TaskStatus.FAILED
                outcome.error = "Task run was cancelled before this node started."
                emit(node.spec.id, "failed", error_type="RunCancelled")
            for task in active:
                task.cancel()
            if not active:
                break
            ready = []
            launch = []
        else:
            ready = None
            launch = None

        for node in pending:
            if cancel_event is not None and cancel_event.is_set():
                break
            dependency_states = [
                outcomes[dependency].status for dependency in node.spec.needs
            ]
            if any(
                state in {
                    TaskStatus.FAILED,
                    TaskStatus.BLOCKED,
                    TaskStatus.NEEDS_ESCALATION,
                    TaskStatus.CONTRACT_CHANGE_REQUEST,
                }
                for state in dependency_states
            ):
                outcomes[node.spec.id].status = TaskStatus.BLOCKED
                outcomes[node.spec.id].error = "A required dependency did not succeed."
                emit(node.spec.id, "blocked")

        if ready is None:
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
            waiters: set[asyncio.Future] = set(active)
            cancel_waiter = None
            if cancel_event is not None and not cancel_event.is_set():
                cancel_waiter = asyncio.create_task(cancel_event.wait())
                waiters.add(cancel_waiter)
            done, _ = await asyncio.wait(waiters, return_when=asyncio.FIRST_COMPLETED)
            if cancel_waiter is not None:
                if cancel_waiter in done:
                    continue
                cancel_waiter.cancel()
                await asyncio.gather(cancel_waiter, return_exceptions=True)
            for task in done:
                if task not in active:
                    continue
                task_id = active.pop(task)
                outcome = outcomes[task_id]
                try:
                    outcome.output = task.result()
                    if (
                        isinstance(outcome.output, TaskResultEnvelope)
                        and outcome.output.status != "ok"
                    ):
                        outcome.status = TaskStatus(outcome.output.status)
                        outcome.error = (
                            outcome.output.diagnostics or outcome.output.summary
                        )
                    else:
                        outcome.status = TaskStatus.SUCCEEDED
                    emit(
                        task_id,
                        outcome.status.value,
                        duration_ms=round((time.perf_counter() - started_at[task_id]) * 1000),
                        model_id=getattr(outcome.output, "model_id", None),
                        model_ids=getattr(outcome.output, "models_used", []),
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