"""Offline tests for quarantined read-only workflows."""
import asyncio
import os
import subprocess
import tempfile
from unittest.mock import patch

from execute import execute_tool
from registry import clear, register
from state import AppState, FileStateCache, ToolPermissionContext
from store import Store
from tool import ToolUseContext
from os_sandbox import (
    MAX_SANDBOX_CPU_SECONDS,
    MAX_SANDBOX_FILE_BYTES,
    MAX_SANDBOX_OPEN_FILES,
    MAX_SANDBOX_PROCESSES,
    build_readonly_sandbox_command,
)
from tools.bash import BashTool
from tools.read import ReadTool
from workflows import (
    WorkflowDefinition,
    WorkflowListTool,
    WorkflowPromoteInput,
    WorkflowPromoteTool,
    WorkflowProposeInput,
    WorkflowProposeTool,
    WorkflowRunInput,
    WorkflowRunTool,
)


def make_context(root: str, store: Store, session_id: str, query_id: str) -> ToolUseContext:
    state = AppState(
        messages=[],
        tool_permission_context=ToolPermissionContext(mode="bypassPermissions"),
        read_file_cache=FileStateCache(),
    )
    context = ToolUseContext(
        get_app_state=lambda: state,
        set_app_state=lambda _update: None,
        read_file_state=state.read_file_cache,
    )
    context.project_root = root
    context.session_id = session_id
    context.query_tracking = {"chainId": query_id}
    context._store = store
    return context


async def approve(_decision):
    return True


async def main() -> None:
    clear()
    register(ReadTool)
    register(BashTool)
    register(WorkflowProposeTool)
    register(WorkflowRunTool)
    register(WorkflowListTool)
    register(WorkflowPromoteTool)
    with tempfile.TemporaryDirectory() as root:
        store = Store(os.path.join(root, "session.db"))
        first_path = os.path.join(root, "alpha.txt")
        second_path = os.path.join(root, "beta.txt")
        with open(first_path, "w", encoding="utf-8") as file:
            file.write("alpha facts")
        with open(second_path, "w", encoding="utf-8") as file:
            file.write("beta facts")

        definition = WorkflowDefinition.model_validate({
            "id": "read-project-facts",
            "version": 1,
            "description": "Read two declared project files",
            "input_schema": {
                "type": "object",
                "properties": {
                    "alpha": {"type": "string"},
                    "beta": {"type": "string"},
                },
                "required": ["alpha", "beta"],
                "additionalProperties": False,
            },
            "output_schema": {
                "type": "object",
                "properties": {
                    "steps": {
                        "type": "object",
                        "properties": {
                            "read-alpha": {"type": "string"},
                            "read-beta": {"type": "string"},
                        },
                        "required": ["read-alpha", "read-beta"],
                        "additionalProperties": False,
                    },
                },
                "required": ["steps"],
            },
            "steps": [
                {
                    "id": "read-alpha",
                    "tool": "Read",
                    "arguments": {"file_path": {"$input": "alpha"}},
                },
                {
                    "id": "read-beta",
                    "tool": "Read",
                    "arguments": {"file_path": {"$input": "beta"}},
                },
            ],
            "tests": [{
                "inputs": {"alpha": first_path, "beta": second_path},
                "expected_contains": {
                    "read-alpha": "alpha facts",
                    "read-beta": "beta facts",
                },
            }],
        })
        try:
            WorkflowDefinition.model_validate({
                "id": "unsafe-shell-workflow",
                "version": 1,
                "description": "This must be rejected",
                "input_schema": {"type": "object"},
                "steps": [{
                    "id": "shell",
                        "tool": "Bash",
                        "arguments": {"command": "echo unsafe"},
                }],
                "tests": [{
                    "inputs": {},
                    "expected_contains": {"shell": "unsafe"},
                }],
            })
            raise AssertionError("workflow definitions must reject Bash steps")
        except Exception as error:
            assert "read_files manifest" in str(error)

        try:
            WorkflowDefinition.model_validate({
                "id": "credential-file-workflow",
                "version": 1,
                "description": "This must be rejected",
                "input_schema": {"type": "object"},
                "read_files": [".env"],
                "steps": [{
                    "id": "shell",
                    "tool": "Bash",
                    "arguments": {"command": "cat $WORKFLOW_ROOT/.env"},
                }],
                "tests": [{
                    "inputs": {},
                    "expected_contains": {"shell": "secret"},
                }],
            })
            raise AssertionError("credential-like files must be rejected")
        except ValueError as error:
            assert "credential-like" in str(error)

        with patch("os_sandbox.shutil.which", return_value=None):
            try:
                build_readonly_sandbox_command(root, "cat input.txt")
                raise AssertionError("sandbox absence must fail closed")
            except RuntimeError as error:
                assert "bubblewrap" in str(error)
        limits = subprocess.run(
            build_readonly_sandbox_command(
                root, "prlimit --pid $$ --nproc --nofile --fsize --cpu",
            ),
            capture_output=True,
            text=True,
            check=True,
        )
        observed_limits = {
            fields[0]: (int(fields[-3]), int(fields[-2]))
            for fields in (
                line.split() for line in limits.stdout.splitlines()[1:]
            )
        }
        assert observed_limits["NPROC"] == (
            MAX_SANDBOX_PROCESSES, MAX_SANDBOX_PROCESSES,
        )
        assert observed_limits["NOFILE"] == (
            MAX_SANDBOX_OPEN_FILES, MAX_SANDBOX_OPEN_FILES,
        )
        assert observed_limits["FSIZE"] == (
            MAX_SANDBOX_FILE_BYTES, MAX_SANDBOX_FILE_BYTES,
        )
        assert observed_limits["CPU"] == (
            MAX_SANDBOX_CPU_SECONDS, MAX_SANDBOX_CPU_SECONDS,
        )

        owner_context = make_context(root, store, "owner-session", "query-1")
        proposal_permission = await WorkflowProposeTool.check_permissions(
            {"definition": definition.model_dump(mode="json")}, owner_context,
        )
        assert proposal_permission.behavior == "ask"
        proposal = await WorkflowProposeTool.call(
            WorkflowProposeInput(definition=definition), owner_context,
        )
        assert proposal.data["status"] == "quarantined"

        other_context = make_context(root, store, "other-session", "query-other")
        assert (await WorkflowListTool.call({}, other_context)).data["workflows"] == []

        workflow_input = {
            "workflow_id": definition.id,
            "version": definition.version,
            "inputs": {"alpha": first_path, "beta": second_path},
        }
        first_run = await WorkflowRunTool.call(
            WorkflowRunInput(**workflow_input), owner_context,
        )
        assert first_run.data["status"] == "succeeded"
        assert "alpha facts" in first_run.data["output"]["steps"]["read-alpha"]
        readiness = (await WorkflowListTool.call({}, owner_context)).data["workflows"][0]
        assert readiness["successful_runs"] == 1
        assert readiness["successful_query_runs"] == 1
        assert readiness["promotion_ready"] is False

        owner_context.query_tracking = {"chainId": "query-2"}
        second_run = await WorkflowRunTool.call(
            WorkflowRunInput(**workflow_input), owner_context,
        )
        assert second_run.data["status"] == "succeeded"
        readiness = (await WorkflowListTool.call({}, owner_context)).data["workflows"][0]
        assert readiness["successful_runs"] == 2
        assert readiness["promotion_ready"] is False
        try:
            await WorkflowPromoteTool.call(
                WorkflowPromoteInput(
                    workflow_id=definition.id,
                    version=definition.version,
                ),
                owner_context,
            )
            raise AssertionError("promotion must require three successful uses")
        except ValueError as error:
            assert "requires 3 successful runs across 2 query runs" in str(error)

        owner_context.query_tracking = {"chainId": "query-3"}
        third_run = await WorkflowRunTool.call(
            WorkflowRunInput(**workflow_input), owner_context,
        )
        assert third_run.data["status"] == "succeeded"
        readiness = (await WorkflowListTool.call({}, owner_context)).data["workflows"][0]
        assert readiness["successful_runs"] == 3
        assert readiness["successful_query_runs"] == 3
        assert readiness["promotion_ready"] is True
        promoted = await execute_tool(
            {
                "id": "workflow-promote",
                "name": "WorkflowPromote",
                "input": {
                    "workflow_id": definition.id,
                    "version": definition.version,
                },
            },
            owner_context,
            can_use_tool=approve,
        )
        assert not promoted.is_error, promoted.result_message.content
        assert "promoted" in promoted.result_message.content
        assert len((await WorkflowListTool.call({}, other_context)).data["workflows"]) == 1

        promoted_run = await WorkflowRunTool.call(
            WorkflowRunInput(**workflow_input), other_context,
        )
        assert promoted_run.data["status"] == "succeeded"

        shell_definition = WorkflowDefinition.model_validate({
            "id": "sandbox-read-facts",
            "version": 1,
            "description": "Inspect a declared file in an offline sandbox",
            "input_schema": {"type": "object", "additionalProperties": False},
            "read_files": ["alpha.txt"],
            "steps": [{
                "id": "show-alpha",
                "tool": "Bash",
                "arguments": {"command": "cat \"$WORKFLOW_ROOT/alpha.txt\""},
            }],
            "tests": [{
                "inputs": {},
                "expected_contains": {"show-alpha": "alpha facts"},
            }],
        })
        await WorkflowProposeTool.call(
            WorkflowProposeInput(definition=shell_definition), owner_context,
        )
        shell_run = await WorkflowRunTool.call(
            WorkflowRunInput(
                workflow_id=shell_definition.id,
                version=shell_definition.version,
            ),
            owner_context,
        )
        assert shell_run.data["status"] == "succeeded", shell_run.data
        assert "alpha facts" in shell_run.data["output"]["steps"]["show-alpha"]

        outside_read = WorkflowDefinition.model_validate({
            "id": "sandbox-no-host-files",
            "version": 1,
            "description": "The sandbox cannot read unmounted host files",
            "input_schema": {"type": "object", "additionalProperties": False},
            "read_files": ["alpha.txt"],
            "steps": [{
                "id": "read-host-passwd",
                "tool": "Bash",
                "arguments": {"command": "cat /etc/passwd"},
            }],
            "tests": [{
                "inputs": {},
                "expected_contains": {"read-host-passwd": "root:"},
            }],
        })
        await WorkflowProposeTool.call(
            WorkflowProposeInput(definition=outside_read), owner_context,
        )
        denied_host_read = await WorkflowRunTool.call(
            WorkflowRunInput(workflow_id=outside_read.id, version=1),
            owner_context,
        )
        assert denied_host_read.data["status"] == "failed"
        assert "root:" not in denied_host_read.data["results"][0]["result"]

        write_workflow = WorkflowDefinition.model_validate({
            "id": "sandbox-no-writes",
            "version": 1,
            "description": "The sandbox rejects write commands",
            "input_schema": {"type": "object", "additionalProperties": False},
            "read_files": ["alpha.txt"],
            "steps": [{
                "id": "try-write",
                "tool": "Bash",
                "arguments": {
                    "command": "printf changed > \"$WORKFLOW_ROOT/alpha.txt\"",
                },
            }],
            "tests": [{
                "inputs": {},
                "expected_contains": {"try-write": "changed"},
            }],
        })
        await WorkflowProposeTool.call(
            WorkflowProposeInput(definition=write_workflow), owner_context,
        )
        try:
            await WorkflowRunTool.call(
                WorkflowRunInput(workflow_id=write_workflow.id, version=1),
                owner_context,
            )
            raise AssertionError("write-capable Bash steps must be rejected")
        except ValueError as error:
            assert "not read-only" in str(error)
        assert open(first_path, encoding="utf-8").read() == "alpha facts"
        store.close()
    clear()
    print("Workflow quarantine, read-only execution, and promotion tests passed")


if __name__ == "__main__":
    asyncio.run(main())