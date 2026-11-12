"""Quarantined, declarative read-only workflows built from registered tools."""
from __future__ import annotations

import json
import re
import uuid
from typing import Any, Literal

from jsonschema import Draft202012Validator
from pydantic import BaseModel, ConfigDict, Field, model_validator

from execute import execute_tool
from permission_types import PermissionAskDecision, PermissionAllowDecision
from registry import find
from tool import ToolResult, ToolUseContext, ValidationResult, build_tool
from tools.read import expand_path


MAX_WORKFLOW_STEPS = 8
MAX_WORKFLOW_RESULT_CHARS = 24_000
WORKFLOW_READ_TOOLS = frozenset({"Read"})


class WorkflowStep(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
    tool: Literal["Read"]
    arguments: dict[str, Any]


class WorkflowTestCase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    inputs: dict[str, Any] = Field(default_factory=dict)
    expected_contains: dict[str, str] = Field(min_length=1)


class WorkflowDefinition(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.0"] = "1.0"
    id: str = Field(pattern=r"^[a-z][a-z0-9-]{2,63}$")
    version: int = Field(ge=1)
    description: str = Field(min_length=1, max_length=500)
    input_schema: dict[str, Any]
    output_schema: dict[str, Any] | None = None
    steps: list[WorkflowStep] = Field(min_length=1, max_length=MAX_WORKFLOW_STEPS)
    tests: list[WorkflowTestCase] = Field(min_length=1, max_length=20)

    @model_validator(mode="after")
    def _validate_definition(self) -> "WorkflowDefinition":
        step_ids = [step.id for step in self.steps]
        if len(step_ids) != len(set(step_ids)):
            raise ValueError("Workflow step IDs must be unique")
        known_steps = set(step_ids)
        for case in self.tests:
            unknown = set(case.expected_contains) - known_steps
            if unknown:
                raise ValueError(
                    "Workflow tests reference unknown steps: "
                    + ", ".join(sorted(unknown))
                )
        Draft202012Validator.check_schema(self.input_schema)
        if self.output_schema is not None:
            Draft202012Validator.check_schema(self.output_schema)
        return self


class WorkflowProposeInput(BaseModel):
    definition: WorkflowDefinition


class WorkflowRunInput(BaseModel):
    workflow_id: str
    version: int = Field(ge=1)
    inputs: dict[str, Any] = Field(default_factory=dict)


class WorkflowPromoteInput(BaseModel):
    workflow_id: str
    version: int = Field(ge=1)


class WorkflowListInput(BaseModel):
    pass


def _session_context(context: ToolUseContext) -> tuple[Any, str, str]:
    store = getattr(context, "_store", None)
    session_id = getattr(context, "session_id", None)
    project_root = getattr(context, "project_root", None)
    if store is None or not session_id or not project_root:
        raise ValueError("Workflows require a project-scoped session Store")
    return store, session_id, project_root


def _resolve_input_reference(value: Any, inputs: dict[str, Any]) -> Any:
    if isinstance(value, dict):
        if set(value) == {"$input"}:
            path = value["$input"]
            if not isinstance(path, str) or not path:
                raise ValueError("Workflow input reference must be a non-empty path")
            current: Any = inputs
            for part in path.split("."):
                if isinstance(current, dict) and part in current:
                    current = current[part]
                elif isinstance(current, list) and part.isdigit() and int(part) < len(current):
                    current = current[int(part)]
                else:
                    raise ValueError(f"Workflow input reference is missing: {path}")
            return current
        return {
            key: _resolve_input_reference(item, inputs)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_resolve_input_reference(item, inputs) for item in value]
    return value


def _query_id(context: ToolUseContext) -> str:
    tracking = getattr(context, "query_tracking", None) or {}
    return str(tracking.get("chainId") or uuid.uuid4().hex)


async def _execute_definition(
    definition: WorkflowDefinition,
    inputs: dict[str, Any],
    context: ToolUseContext,
    can_use_tool: Any,
) -> tuple[dict[str, Any], bool]:
    input_errors = list(Draft202012Validator(definition.input_schema).iter_errors(inputs))
    if input_errors:
        path = ".".join(str(part) for part in input_errors[0].absolute_path)
        raise ValueError(f"Workflow input failed its schema at {path or '<root>'}")

    step_outputs: dict[str, str] = {}
    results: list[dict[str, Any]] = []
    succeeded = True
    for step in definition.steps:
        tool = find(step.tool)
        if tool is None or tool.name not in WORKFLOW_READ_TOOLS:
            raise ValueError(f"Workflow tool is unavailable or not permitted: {step.tool}")
        arguments = _resolve_input_reference(step.arguments, inputs)
        parsed = tool.input_schema.model_validate(arguments)
        if not tool.is_read_only(parsed) or not tool.is_concurrency_safe(parsed):
            raise ValueError(f"Workflow step is not read-only and concurrency-safe: {step.id}")
        read_cache = getattr(context, "read_file_state", None)
        if read_cache is not None:
            read_cache.invalidate(expand_path(parsed.file_path))
        result = await execute_tool(
            {
                "id": f"workflow-{uuid.uuid4().hex}",
                "name": step.tool,
                "input": arguments,
            },
            context,
            can_use_tool=can_use_tool,
        )
        content = result.result_message.content
        if len(content) > MAX_WORKFLOW_RESULT_CHARS:
            content = content[:MAX_WORKFLOW_RESULT_CHARS] + "\n[workflow step output truncated]"
        step_outputs[step.id] = content
        results.append({
            "step_id": step.id,
            "tool": step.tool,
            "status": "failed" if result.is_error else "succeeded",
            "result": content,
        })
        if result.is_error:
            succeeded = False
            break

    output = {"steps": step_outputs}
    if succeeded and definition.output_schema is not None:
        succeeded = Draft202012Validator(definition.output_schema).is_valid(output)
        if not succeeded:
            results.append({
                "step_id": "<workflow-output>",
                "status": "failed",
                "result": "Workflow output failed its declared JSON Schema.",
            })
    return {"workflow_id": definition.id, "version": definition.version, "results": results, "output": output}, succeeded


async def _propose(
    input: WorkflowProposeInput,
    context: ToolUseContext,
    _can_use_tool: Any = None,
    _parent_message: Any = None,
    _on_progress: Any = None,
) -> ToolResult:
    store, session_id, project_root = _session_context(context)
    definition = input.definition.model_dump(mode="json")
    store.save_workflow_definition(project_root, session_id, definition)
    return ToolResult(data={
        "workflow_id": input.definition.id,
        "version": input.definition.version,
        "status": "quarantined",
        "step_count": len(input.definition.steps),
    })


async def _run(
    input: WorkflowRunInput,
    context: ToolUseContext,
    can_use_tool: Any = None,
    _parent_message: Any = None,
    _on_progress: Any = None,
) -> ToolResult:
    store, session_id, project_root = _session_context(context)
    raw_definition = store.get_workflow_definition(
        project_root, input.workflow_id, input.version, session_id,
    )
    if raw_definition is None:
        raise ValueError("Workflow is not available in this session or promoted project registry")
    definition = WorkflowDefinition.model_validate(raw_definition)
    result, succeeded = await _execute_definition(
        definition, input.inputs, context, can_use_tool,
    )
    store.record_workflow_run(
        project_root,
        definition.id,
        definition.version,
        session_id,
        _query_id(context),
        succeeded,
    )
    result["status"] = "succeeded" if succeeded else "failed"
    return ToolResult(data=result)


async def _promote(
    input: WorkflowPromoteInput,
    context: ToolUseContext,
    can_use_tool: Any = None,
    _parent_message: Any = None,
    _on_progress: Any = None,
) -> ToolResult:
    store, session_id, project_root = _session_context(context)
    raw_definition = store.get_workflow_definition(
        project_root, input.workflow_id, input.version, session_id,
    )
    if raw_definition is None:
        raise ValueError("Workflow is not available in this session")
    definition = WorkflowDefinition.model_validate(raw_definition)

    for test_case in definition.tests:
        result, succeeded = await _execute_definition(
            definition,
            test_case.inputs,
            context,
            can_use_tool,
        )
        if not succeeded:
            raise ValueError(
                f"Workflow test failed for {definition.id} v{definition.version}"
            )
        observed = {item["step_id"]: item.get("result", "") for item in result["results"]}
        if any(
            expected not in observed.get(step_id, "")
            for step_id, expected in test_case.expected_contains.items()
        ):
            raise ValueError(
                f"Workflow test output mismatch for {definition.id} v{definition.version}"
            )

    usage = store.promote_workflow_definition(
        project_root, definition.id, definition.version, session_id,
    )
    return ToolResult(data={
        "workflow_id": definition.id,
        "version": definition.version,
        "status": "promoted",
        "usage": usage,
    })


async def _list(
    _input: WorkflowListInput,
    context: ToolUseContext,
    _can_use_tool: Any = None,
    _parent_message: Any = None,
    _on_progress: Any = None,
) -> ToolResult:
    store, session_id, project_root = _session_context(context)
    definitions = store.list_workflow_definitions(project_root, session_id)
    return ToolResult(data={
        "workflows": [
            {
                "id": item["id"],
                "version": item["version"],
                "status": item["status"],
                "description": item["definition"].get("description", ""),
            }
            for item in definitions
        ],
    })


def _map_result(data: dict[str, Any], tool_use_id: str) -> dict[str, Any]:
    return {
        "type": "tool_result",
        "tool_use_id": tool_use_id,
        "content": json.dumps(data, ensure_ascii=True)[:MAX_WORKFLOW_RESULT_CHARS],
    }


def _permission(prompt: str):
    async def check(_input, _context):
        return PermissionAskDecision(message=prompt)
    return check


def _read_only(_input: Any) -> bool:
    return True


def _safe(_input: Any) -> bool:
    return True


def _build_tool(name: str, description: str, input_schema: type[BaseModel], call, permission=None):
    return build_tool(
        name=name,
        description=description,
        prompt=description,
        input_schema=input_schema,
        call=call,
        map_tool_result_to_tool_result_block_param=_map_result,
        render_tool_use_message=lambda _input, _options: name,
        max_result_size_chars=MAX_WORKFLOW_RESULT_CHARS,
        is_read_only=_read_only,
        is_concurrency_safe=_safe,
        check_permissions=permission,
        validate_input=lambda _input, _context: ValidationResult(result=True),
    )


WorkflowProposeTool = _build_tool(
    "WorkflowPropose",
    "Create a versioned read-only workflow in this session's quarantine.",
    WorkflowProposeInput,
    _propose,
    _permission("Store this workflow candidate in the current session quarantine?"),
)
WorkflowRunTool = _build_tool(
    "WorkflowRun",
    "Run an available read-only workflow and record its result.",
    WorkflowRunInput,
    _run,
    _permission("Run this read-only workflow?"),
)
WorkflowPromoteTool = _build_tool(
    "WorkflowPromote",
    "Test and promote a quarantined workflow after repeated successful use.",
    WorkflowPromoteInput,
    _promote,
    _permission("Run this workflow's declared tests and promote it project-wide if they pass?"),
)
WorkflowListTool = _build_tool(
    "WorkflowList",
    "List session-quarantined and project-promoted workflows.",
    WorkflowListInput,
    _list,
    lambda _input, _context: PermissionAllowDecision(),
)


__all__ = [
    "WorkflowDefinition", "WorkflowListTool", "WorkflowPromoteTool",
    "WorkflowProposeTool", "WorkflowRunTool", "WorkflowStep", "WorkflowTestCase",
]