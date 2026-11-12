"""Offline tests for quarantined read-only workflows."""
import asyncio
import os
import tempfile

from execute import execute_tool
from registry import clear, register
from state import AppState, FileStateCache, ToolPermissionContext
from store import Store
from tool import ToolUseContext
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
            assert "Read" in str(error)

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

        owner_context.query_tracking = {"chainId": "query-2"}
        second_run = await WorkflowRunTool.call(
            WorkflowRunInput(**workflow_input), owner_context,
        )
        assert second_run.data["status"] == "succeeded"
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
        store.close()
    clear()
    print("Workflow quarantine, read-only execution, and promotion tests passed")


if __name__ == "__main__":
    asyncio.run(main())