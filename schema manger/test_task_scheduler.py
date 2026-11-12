"""Offline tests for DAG validation, dependency payloads, and file leases."""
import asyncio
import hashlib
import os
import subprocess
import tempfile

from agent_contracts import TaskResultEnvelope, WorkerUsage
from task_scheduler import (
    FileLeaseConflict,
    FileLeaseRegistry,
    TaskFileScope,
    TaskSpec,
    TaskStatus,
    _commit_task_staging,
    run_task_graph,
    validate_task_graph,
)


async def main():
    with tempfile.TemporaryDirectory() as root:
        os.makedirs(os.path.join(root, "src"))
        alpha_path = os.path.join(root, "src", "alpha.py")
        beta_path = os.path.join(root, "src", "beta.py")
        tasks = [
            TaskSpec(id="alpha", instruction="Inspect alpha", read_files=[alpha_path]),
            TaskSpec(id="beta", instruction="Inspect beta", read_files=[beta_path]),
            TaskSpec(
                id="summary", instruction="Summarize findings",
                needs=["alpha", "beta"],
            ),
        ]
        active = 0
        max_active = 0
        received = {}
        events = []

        async def runner(spec, dependencies, scope):
            nonlocal active, max_active
            received[spec.id] = dependencies
            if spec.id in ("alpha", "beta"):
                active += 1
                max_active = max(max_active, active)
                await asyncio.sleep(0.01)
                active -= 1
                return f"{spec.id} findings"
            assert dependencies == {
                "alpha": "alpha findings",
                "beta": "beta findings",
            }
            return "combined"

        results = await run_task_graph(
            tasks, runner, project_root=root, max_parallel=2,
            on_event=events.append,
        )
        assert [result.status for result in results] == [
            TaskStatus.SUCCEEDED, TaskStatus.SUCCEEDED, TaskStatus.SUCCEEDED,
        ]
        assert max_active == 2
        assert received["summary"] == {
            "alpha": "alpha findings",
            "beta": "beta findings",
        }
        lifecycle = [(event["task_id"], event["status"]) for event in events]
        lifecycle_events = [event for event in events if event["type"] == "task_lifecycle"]
        assert all(event["schema_version"] == "1.0" for event in lifecycle_events)
        assert all(event["run_id"] for event in lifecycle_events)
        assert all(event["event_id"] and event["occurred_at"] > 0 for event in lifecycle_events)
        for task_id in ("alpha", "beta", "summary"):
            attempts = {
                event["attempt_id"] for event in lifecycle_events
                if event["task_id"] == task_id and event["attempt_id"] is not None
            }
            assert len(attempts) == 1
        assert lifecycle[:2] == [("alpha", "started"), ("beta", "started")]
        assert set(lifecycle[2:4]) == {
            ("alpha", "succeeded"), ("beta", "succeeded"),
        }
        assert lifecycle[4:] == [
            ("summary", "started"), ("summary", "succeeded"),
        ]
        assert all("instruction" not in event and "write_files" not in event for event in events)
        assert all(
            "duration_ms" in event
            for event in events if event["status"] in ("succeeded", "failed")
        )

        joined_dependencies = {}
        typed_calls = []

        async def typed_runner(spec, dependencies, scope):
            typed_calls.append(spec.id)
            if spec.id == "producer":
                return {"answer": "42"}
            joined_dependencies.update(dependencies)
            return "joined"

        dependency_schema = {
            "type": "object",
            "properties": {"answer": {"type": "string"}},
            "required": ["answer"],
            "additionalProperties": False,
        }
        typed_results = await run_task_graph([
            TaskSpec(id="producer", instruction="Return a typed answer"),
            TaskSpec(
                id="typed-join",
                instruction="Consume the typed answer",
                needs=["producer"],
                dependency_output_schemas={"producer": dependency_schema},
            ),
        ], typed_runner, project_root=root)
        assert all(result.status == TaskStatus.SUCCEEDED for result in typed_results)
        assert joined_dependencies == {"producer": {"answer": "42"}}

        join_runner_calls = []

        async def join_runner(spec, dependencies, scope):
            join_runner_calls.append(spec.id)
            if spec.id == "left":
                return {"total": 12}
            if spec.id == "right":
                return {"total": 12}
            assert dependencies == {"left": {"total": 12}, "right": {"total": 12}}
            return "joined consistent outputs"

        equal_check = {
            "kind": "equals",
            "left": {"dependency_id": "left", "json_path": "total"},
            "right": {"dependency_id": "right", "json_path": "total"},
        }
        consistent_join = await run_task_graph([
            TaskSpec(id="left", instruction="Produce left total"),
            TaskSpec(id="right", instruction="Produce right total"),
            TaskSpec(
                id="join",
                instruction="Join matching totals",
                needs=["left", "right"],
                join_checks=[equal_check],
            ),
        ], join_runner, project_root=root)
        assert all(result.status == TaskStatus.SUCCEEDED for result in consistent_join)
        assert join_runner_calls == ["left", "right", "join"]

        dependency_map_schema = {
            "type": "object",
            "properties": {
                "left": {
                    "type": "object",
                    "properties": {"total": {"type": "integer"}},
                    "required": ["total"],
                },
                "right": {
                    "type": "object",
                    "properties": {"total": {"type": "integer"}},
                    "required": ["total"],
                },
            },
            "required": ["left", "right"],
        }
        schema_join = await run_task_graph([
            TaskSpec(id="left", instruction="Produce left total"),
            TaskSpec(id="right", instruction="Produce right total"),
            TaskSpec(
                id="join",
                instruction="Validate the combined dependency shape",
                needs=["left", "right"],
                join_checks=[{
                    "kind": "schema",
                    "schema_definition": dependency_map_schema,
                }],
            ),
        ], join_runner, project_root=root)
        assert all(result.status == TaskStatus.SUCCEEDED for result in schema_join)

        join_runner_calls.clear()

        async def inconsistent_join_runner(spec, dependencies, scope):
            join_runner_calls.append(spec.id)
            if spec.id == "left":
                return {"total": 12}
            if spec.id == "right":
                return {"total": 13}
            return "must not run"

        inconsistent_join = await run_task_graph([
            TaskSpec(id="left", instruction="Produce left total"),
            TaskSpec(id="right", instruction="Produce right total"),
            TaskSpec(
                id="join",
                instruction="Reject inconsistent totals",
                needs=["left", "right"],
                join_checks=[equal_check],
            ),
        ], inconsistent_join_runner, project_root=root)
        assert inconsistent_join[0].status == TaskStatus.SUCCEEDED
        assert inconsistent_join[1].status == TaskStatus.SUCCEEDED
        assert inconsistent_join[2].status == TaskStatus.FAILED
        assert "join check failed" in inconsistent_join[2].error
        assert join_runner_calls == ["left", "right"]

        invalid_calls = []

        async def invalid_join_runner(spec, dependencies, scope):
            invalid_calls.append(spec.id)
            return {"answer": "not an integer"}

        invalid_join = await run_task_graph([
            TaskSpec(id="producer", instruction="Return a mismatched answer"),
            TaskSpec(
                id="typed-join",
                instruction="Reject the mismatched answer",
                needs=["producer"],
                dependency_output_schemas={
                    "producer": {
                        "type": "object",
                        "properties": {"answer": {"type": "integer"}},
                        "required": ["answer"],
                    },
                },
            ),
        ], invalid_join_runner, project_root=root)
        assert invalid_join[0].status == TaskStatus.SUCCEEDED
        assert invalid_join[1].status == TaskStatus.FAILED
        assert "Dependency producer output failed its schema" in invalid_join[1].error
        assert invalid_calls == ["producer"]

        disjoint_write_a = os.path.join(root, "src", "writer-a.py")
        disjoint_write_b = os.path.join(root, "src", "writer-b.py")
        active_writers = 0
        max_active_writers = 0
        writers_started = asyncio.Event()

        async def disjoint_writer(spec, dependencies, scope):
            nonlocal active_writers, max_active_writers
            assert scope.check_access("Write", spec.write_files[0]) is None
            active_writers += 1
            max_active_writers = max(max_active_writers, active_writers)
            if active_writers == 2:
                writers_started.set()
            await asyncio.wait_for(writers_started.wait(), timeout=1)
            active_writers -= 1
            return spec.id

        disjoint_results = await run_task_graph([
            TaskSpec(
                id="write-a", instruction="Write A", write_files=[disjoint_write_a],
            ),
            TaskSpec(
                id="write-b", instruction="Write B", write_files=[disjoint_write_b],
            ),
        ], disjoint_writer, project_root=root, max_parallel=2)
        assert all(result.status == TaskStatus.SUCCEEDED for result in disjoint_results)
        assert max_active_writers == 2

        staged_path = os.path.join(root, "src", "staged.py")
        with open(staged_path, "w", encoding="utf-8") as staged_file:
            staged_file.write("before")

        async def staged_runner(spec, dependencies, scope):
            with open(
                scope.execution_path("Write", staged_path),
                "w",
                encoding="utf-8",
            ) as staged_file:
                staged_file.write(spec.id)
            if spec.id == "discard-staged":
                return TaskResultEnvelope(
                    run_id="staging-test",
                    node_id=spec.id,
                    status="failed",
                    summary="Discard staged output",
                )
            if spec.id == "workspace-conflict":
                with open(staged_path, "w", encoding="utf-8") as workspace_file:
                    workspace_file.write("external edit")
            return TaskResultEnvelope(
                run_id="staging-test",
                node_id=spec.id,
                status="ok",
                summary="Staged output ready",
            )

        discarded = await run_task_graph([
            TaskSpec(
                id="discard-staged",
                instruction="Fail after writing in staging",
                write_files=[staged_path],
            ),
        ], staged_runner, project_root=root)
        assert discarded[0].status == TaskStatus.FAILED
        assert open(staged_path, encoding="utf-8").read() == "before"

        commit_events = []
        committed = await run_task_graph([
            TaskSpec(
                id="commit-staged",
                instruction="Commit successful staged output",
                write_files=[staged_path],
            ),
        ], staged_runner, project_root=root, on_node_commit=(
            lambda spec, envelope, status, writes: commit_events.append(
                (spec.id, envelope.status, status, writes),
            )
        ))
        assert committed[0].status == TaskStatus.SUCCEEDED
        assert open(staged_path, encoding="utf-8").read() == "commit-staged"
        assert [event[2] for event in commit_events] == ["prepared", "committed"]
        assert all(event[0] == "commit-staged" and event[1] == "ok" for event in commit_events)
        assert commit_events[0][3] == commit_events[1][3]
        assert commit_events[0][3][0]["before_sha256"] == hashlib.sha256(b"before").hexdigest()
        assert commit_events[0][3][0]["after_sha256"] == hashlib.sha256(b"commit-staged").hexdigest()

        retry_leases = FileLeaseRegistry(root)
        retry_path = os.path.realpath(staged_path)
        retry_leases.acquire("commit-retry", frozenset({retry_path}))
        with tempfile.TemporaryDirectory(prefix="task-commit-retry-") as retry_staging:
            retry_output = os.path.join(retry_staging, "src", "staged.py")
            os.makedirs(os.path.dirname(retry_output), exist_ok=True)
            with open(retry_output, "w", encoding="utf-8") as target:
                target.write("commit-staged")
            _commit_task_staging(TaskFileScope(
                task_id="commit-retry",
                project_root=root,
                read_paths=frozenset(),
                write_paths=frozenset({retry_path}),
                leases=retry_leases,
                staging_root=retry_staging,
                baseline_write_hashes={
                    retry_path: hashlib.sha256(b"before").hexdigest(),
                },
            ))
        assert open(staged_path, encoding="utf-8").read() == "commit-staged"
        retry_leases.release("commit-retry")

        conflicted = await run_task_graph([
            TaskSpec(
                id="workspace-conflict",
                instruction="Do not overwrite a concurrent workspace edit",
                write_files=[staged_path],
            ),
        ], staged_runner, project_root=root)
        assert conflicted[0].status == TaskStatus.FAILED
        assert "Workspace file changed" in conflicted[0].error
        assert open(staged_path, encoding="utf-8").read() == "external edit"

        git_root = os.path.join(root, "git-project")
        os.makedirs(git_root)
        subprocess.run(
            ["git", "init", git_root],
            check=True,
            capture_output=True,
            text=True,
        )
        tracked_path = os.path.join(git_root, "tracked.py")
        with open(tracked_path, "w", encoding="utf-8") as source:
            source.write("before")
        subprocess.run(
            ["git", "-C", git_root, "add", "tracked.py"],
            check=True,
            capture_output=True,
            text=True,
        )
        subprocess.run(
            [
                "git", "-C", git_root,
                "-c", "user.name=Task Test",
                "-c", "user.email=task-test@example.invalid",
                "commit", "-m", "initial",
            ],
            check=True,
            capture_output=True,
            text=True,
        )

        async def worktree_runner(spec, dependencies, scope):
            assert scope.staging_kind == "git_worktree"
            with open(
                scope.execution_path("Write", tracked_path),
                "w",
                encoding="utf-8",
            ) as output:
                output.write("after")
                return TaskResultEnvelope(
                run_id="worktree-test",
                node_id=spec.id,
                status="ok",
                summary="Updated tracked file",
            )

        worktree_result = await run_task_graph([
            TaskSpec(
                id="git-worktree", instruction="Update a tracked file",
                write_files=[tracked_path],
            ),
        ], worktree_runner, project_root=git_root)
        assert worktree_result[0].status == TaskStatus.SUCCEEDED
        assert open(tracked_path, encoding="utf-8").read() == "after"
        worktrees = subprocess.run(
            ["git", "worktree", "list", "--porcelain"],
            cwd=git_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.splitlines()
        assert [line for line in worktrees if line.startswith("worktree ")] == [
            f"worktree {git_root}",
        ]

        module_a = os.path.join(root, "src", "module_a.py")
        module_b = os.path.join(root, "src", "module_b.py")
        integrated_output = os.path.join(root, "src", "integrated.py")

        async def code_integrator_runner(spec, dependencies, scope):
            if spec.role == "integrator":
                assert set(dependencies) == {"worker-a", "worker-b"}
                module_contents = [
                    open(scope.execution_path("Read", path), encoding="utf-8").read()
                    for path in (module_a, module_b)
                ]
                with open(
                    scope.execution_path("Write", integrated_output),
                    "w",
                    encoding="utf-8",
                ) as output:
                    output.write("\n".join(module_contents))
            else:
                output_path = module_a if spec.id == "worker-a" else module_b
                with open(
                    scope.execution_path("Write", output_path),
                    "w",
                    encoding="utf-8",
                ) as output:
                    output.write(f"# {spec.id}")
            return TaskResultEnvelope(
                run_id="integrator-test",
                node_id=spec.id,
                status="ok",
                summary=f"{spec.id} complete",
            )

        integration_tasks = [
            TaskSpec(
                id="worker-a", instruction="Write module A",
                write_files=[module_a],
            ),
            TaskSpec(
                id="worker-b", instruction="Write module B",
                write_files=[module_b],
            ),
            TaskSpec(
                id="integrate", role="integrator",
                instruction="Integrate both modules",
                needs=["worker-a", "worker-b"],
                read_files=[module_a, module_b],
                write_files=[integrated_output],
            ),
        ]
        integrated = await run_task_graph(
            integration_tasks,
            code_integrator_runner,
            project_root=root,
            max_parallel=2,
        )
        assert all(result.status == TaskStatus.SUCCEEDED for result in integrated)
        assert open(integrated_output, encoding="utf-8").read() == (
            "# worker-a\n# worker-b"
        )

        try:
            validate_task_graph([
                integration_tasks[0],
                integration_tasks[1],
                integration_tasks[2].model_copy(update={"needs": ["worker-a"]}),
            ], root)
            raise AssertionError("integrator must depend on every writer")
        except ValueError as error:
            assert "must depend on writer worker-b" in str(error)

        try:
            validate_task_graph([
                TaskSpec(
                    id="source-writer", instruction="Update shared source",
                    write_files=[alpha_path],
                ),
                TaskSpec(
                    id="source-reader", instruction="Read shared source",
                    read_files=[alpha_path],
                ),
            ], root)
            raise AssertionError("unordered read/write overlap should be rejected")
        except ValueError as error:
            assert "overlapping read/write scopes" in str(error)

        async def failing_runner(spec, dependencies, scope):
            if spec.id == "broken":
                raise RuntimeError("expected failure")
            return "ok"

        failed = await run_task_graph([
            TaskSpec(id="broken", instruction="Fail"),
            TaskSpec(id="blocked", instruction="Needs broken", needs=["broken"]),
            TaskSpec(id="independent", instruction="Continue independently"),
        ], failing_runner, project_root=root, on_event=events.append)
        assert [result.status for result in failed] == [
            TaskStatus.FAILED, TaskStatus.BLOCKED, TaskStatus.SUCCEEDED,
        ]
        assert any(
            event["task_id"] == "blocked" and event["status"] == "blocked"
            for event in events
        )

        typed_events = []
        typed_calls = []

        async def typed_request_runner(spec, dependencies, scope):
            typed_calls.append(spec.id)
            if spec.id == "request":
                return TaskResultEnvelope(
                    run_id="typed-request-run",
                    node_id=spec.id,
                    status="needs_escalation",
                    summary="The task needs a stronger model.",
                    output={"reason": "ambiguous interface"},
                    usage=WorkerUsage(output_tokens=7, calls=1, seconds=0.1),
                )
            return spec.id

        typed_request_outcomes = await run_task_graph([
            TaskSpec(id="request", instruction="Request escalation"),
            TaskSpec(
                id="dependent-request",
                instruction="Wait for the requested result",
                needs=["request"],
            ),
            TaskSpec(id="independent-request", instruction="Continue independently"),
        ], typed_request_runner, project_root=root, on_event=typed_events.append)
        assert [outcome.status for outcome in typed_request_outcomes] == [
            TaskStatus.NEEDS_ESCALATION,
            TaskStatus.BLOCKED,
            TaskStatus.SUCCEEDED,
        ]
        assert "dependent-request" not in typed_calls
        assert any(
            event["task_id"] == "request"
            and event["status"] == "needs_escalation"
            for event in typed_events
        )

        joined_dependencies = {}
        typed_calls = []

        async def typed_runner(spec, dependencies, scope):
            typed_calls.append(spec.id)
            if spec.id == "producer":
                return {"answer": "42"}
            joined_dependencies.update(dependencies)
            return "joined"

        dependency_schema = {
            "type": "object",
            "properties": {"answer": {"type": "string"}},
            "required": ["answer"],
            "additionalProperties": False,
        }
        typed_results = await run_task_graph([
            TaskSpec(id="producer", instruction="Return a typed answer"),
            TaskSpec(
                id="typed-join",
                instruction="Consume the typed answer",
                needs=["producer"],
                dependency_output_schemas={"producer": dependency_schema},
            ),
        ], typed_runner, project_root=root)
        assert all(result.status == TaskStatus.SUCCEEDED for result in typed_results)
        assert joined_dependencies == {"producer": {"answer": "42"}}
        assert typed_calls == ["producer", "typed-join"]

        invalid_calls = []

        async def invalid_join_runner(spec, dependencies, scope):
            invalid_calls.append(spec.id)
            return {"answer": "not an integer"}

        invalid_join = await run_task_graph([
            TaskSpec(id="producer", instruction="Return a mismatched answer"),
            TaskSpec(
                id="typed-join",
                instruction="Reject the mismatched answer",
                needs=["producer"],
                dependency_output_schemas={
                    "producer": {
                        "type": "object",
                        "properties": {"answer": {"type": "integer"}},
                        "required": ["answer"],
                    },
                },
            ),
        ], invalid_join_runner, project_root=root)
        assert invalid_join[0].status == TaskStatus.SUCCEEDED
        assert invalid_join[1].status == TaskStatus.FAILED
        assert "Dependency producer output failed its schema" in invalid_join[1].error
        assert invalid_calls == ["producer"]

        writers = [
            TaskSpec(id="writer-a", instruction="write A", write_files=[alpha_path]),
            TaskSpec(id="writer-b", instruction="write B", write_files=[alpha_path]),
        ]
        try:
            validate_task_graph(writers, root)
            raise AssertionError("unordered overlapping writes should be rejected")
        except ValueError as error:
            assert "overlapping write scopes" in str(error)

        ordered_writers = [
            writers[0],
            TaskSpec(
                id="writer-b", instruction="write B", needs=["writer-a"],
                write_files=[alpha_path],
            ),
        ]
        validated = validate_task_graph(ordered_writers, root)
        leases = FileLeaseRegistry(root)
        leases.acquire("writer-a", validated[0].write_paths)
        scope = TaskFileScope(
            "writer-a", root, validated[0].read_paths, validated[0].write_paths, leases,
        )
        assert scope.check_access("Write", alpha_path) is None
        assert scope.check_access("Write", beta_path) is not None
        assert scope.check_access("Read", beta_path) is not None
        try:
            leases.acquire("writer-b", validated[1].write_paths)
            raise AssertionError("an active lease conflict should fail")
        except FileLeaseConflict:
            pass
        leases.release("writer-a")

        invalid_graphs = [
            [TaskSpec(id="one", instruction="one", needs=["missing"])],
            [
                TaskSpec(id="one", instruction="one", needs=["two"]),
                TaskSpec(id="two", instruction="two", needs=["one"]),
            ],
        ]
        for invalid in invalid_graphs:
            try:
                validate_task_graph(invalid, root)
                raise AssertionError("invalid task graph should be rejected")
            except ValueError:
                pass

        try:
            validate_task_graph([
                TaskSpec(id="escape", instruction="outside", read_files=["../outside.py"]),
            ], root)
            raise AssertionError("out-of-project paths should be rejected")
        except ValueError as error:
            assert "outside the project" in str(error)

        try:
            validate_task_graph([
                TaskSpec(
                    id="bad-check", instruction="check undeclared file",
                    write_files=[alpha_path],
                    acceptance=[{"kind": "exists", "spec": beta_path}],
                ),
            ], root)
            raise AssertionError("acceptance checks outside write scope should fail")
        except ValueError as error:
            assert "acceptance path must be in write_files" in str(error)

        try:
            validate_task_graph([
                TaskSpec(
                    id="bad-schema", instruction="invalid schema",
                    output_schema={"type": "not-a-json-schema-type"},
                ),
            ], root)
            raise AssertionError("invalid output schemas should fail")
        except ValueError as error:
            assert "invalid output schema" in str(error)

        cancel_path = os.path.join(root, "src", "cancelled.py")
        with open(cancel_path, "w", encoding="utf-8") as target:
            target.write("before")
        cancel_event = asyncio.Event()
        worker_started = asyncio.Event()
        cancel_calls = []

        async def cancellable_runner(spec, dependencies, scope):
            cancel_calls.append(spec.id)
            with open(
                scope.execution_path("Write", cancel_path),
                "w",
                encoding="utf-8",
            ) as target:
                target.write("uncommitted")
            worker_started.set()
            await asyncio.Event().wait()
            return TaskResultEnvelope(
                run_id="cancel-test",
                node_id=spec.id,
                status="ok",
                summary="Should not complete",
            )

        cancellation = asyncio.create_task(run_task_graph(
            [
                TaskSpec(
                    id="cancel-active", instruction="Wait for cancellation",
                    write_files=[cancel_path],
                ),
                TaskSpec(
                    id="cancel-pending", instruction="Remain pending",
                    write_files=[os.path.join(root, "src", "pending.py")],
                ),
            ],
            cancellable_runner,
            project_root=root,
            max_parallel=1,
            cancel_event=cancel_event,
        ))
        await asyncio.wait_for(worker_started.wait(), timeout=1)
        cancel_event.set()
        cancelled = await asyncio.wait_for(cancellation, timeout=1)
        assert [outcome.status for outcome in cancelled] == [
            TaskStatus.FAILED, TaskStatus.FAILED,
        ]
        assert cancel_calls == ["cancel-active"]
        assert open(cancel_path, encoding="utf-8").read() == "before"

    print("Task DAG scheduler and file lease tests passed")


if __name__ == "__main__":
    asyncio.run(main())