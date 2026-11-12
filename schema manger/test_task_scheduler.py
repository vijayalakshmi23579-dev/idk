"""Offline tests for DAG validation, dependency payloads, and file leases."""
import asyncio
import os
import tempfile

from task_scheduler import (
    FileLeaseConflict,
    FileLeaseRegistry,
    TaskFileScope,
    TaskSpec,
    TaskStatus,
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

    print("Task DAG scheduler and file lease tests passed")


if __name__ == "__main__":
    asyncio.run(main())