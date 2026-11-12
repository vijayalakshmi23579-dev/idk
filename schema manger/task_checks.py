"""Deterministic, non-executable acceptance checks for Task workers."""
from __future__ import annotations

import json
import os
from typing import Any

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError

from agent_contracts import AcceptanceCheck, JoinCheck


def _payload(check: AcceptanceCheck) -> dict[str, Any] | None:
    if check.kind == "exists":
        return None
    try:
        payload = json.loads(check.spec)
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid {check.kind} acceptance spec") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{check.kind} acceptance spec must be a JSON object")
    return payload


def acceptance_path(check: AcceptanceCheck) -> str | None:
    """Return the confined file target, or None for a structured output check."""
    payload = _payload(check)
    if payload is None:
        path = check.spec
    elif payload.get("target") == "output":
        if check.kind == "exists":
            raise ValueError("An exists check cannot target structured output")
        return None
    else:
        if payload.get("target", "file") != "file":
            raise ValueError("Acceptance target must be 'file' or 'output'")
        path = payload.get("path")
        if not isinstance(path, str) or not path.strip():
            raise ValueError(f"{check.kind} acceptance spec requires a relative path")

    return path


def validate_acceptance_check(check: AcceptanceCheck) -> str | None:
    path = acceptance_path(check)
    payload = _payload(check)
    if check.kind == "exists":
        return path

    assert payload is not None
    if check.kind == "schema":
        schema = payload.get("schema")
        if not isinstance(schema, dict):
            raise ValueError("Schema acceptance spec requires a JSON Schema object")
        try:
            Draft202012Validator.check_schema(schema)
        except SchemaError as error:
            raise ValueError("Acceptance check contains an invalid JSON Schema") from error
        return path

    if not any(key in payload for key in ("contains", "equals_text", "equals")):
        raise ValueError("Assert acceptance spec requires contains, equals_text, or equals")
    return path


def _resolve_file(root: str, relative_path: str) -> str:
    canonical_root = os.path.realpath(os.path.abspath(root))
    candidate = os.path.realpath(os.path.join(canonical_root, relative_path))
    try:
        if os.path.commonpath((canonical_root, candidate)) != canonical_root:
            raise ValueError("Acceptance path is outside the project")
    except ValueError as error:
        if str(error) == "Acceptance path is outside the project":
            raise
        raise ValueError("Acceptance path is outside the project") from error
    return candidate


def _json_path_value(value: Any, path: str) -> tuple[bool, Any]:
    current = value
    for part in path.split("."):
        if not part:
            continue
        if isinstance(current, dict) and part in current:
            current = current[part]
        elif isinstance(current, list) and part.isdigit() and int(part) < len(current):
            current = current[int(part)]
        else:
            return False, None
    return True, current


def evaluate_acceptance(
    check: AcceptanceCheck,
    project_root: str,
    output: dict[str, Any],
) -> bool:
    """Evaluate one already-validated check without executing model-authored code."""
    try:
        path = validate_acceptance_check(check)
        payload = _payload(check)
        if check.kind == "exists":
            return os.path.isfile(_resolve_file(project_root, path or ""))

        assert payload is not None
        is_output = payload.get("target") == "output"
        if is_output:
            value: Any = output
            content = output if isinstance(output, str) else json.dumps(output, ensure_ascii=True)
        else:
            file_path = _resolve_file(project_root, path or "")
            with open(file_path, encoding="utf-8") as source:
                content = source.read()
            value = None

        if check.kind == "schema":
            instance = value if is_output else json.loads(content)
            return Draft202012Validator(payload["schema"]).is_valid(instance)
        if "contains" in payload:
            return str(payload["contains"]) in content
        if "equals_text" in payload:
            return content == str(payload["equals_text"])

        if not is_output:
            value = json.loads(content)
        found, actual = _json_path_value(value, str(payload.get("json_path", "")))
        return found and actual == payload["equals"]
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return False


def validate_join_check(check: JoinCheck) -> None:
    if check.kind != "schema":
        return
    try:
        Draft202012Validator.check_schema(check.schema_definition or {})
    except SchemaError as error:
        raise ValueError("Join check contains an invalid JSON Schema") from error


def evaluate_join_check(
    check: JoinCheck,
    dependencies: dict[str, Any],
) -> bool:
    if check.kind == "schema":
        return Draft202012Validator(check.schema_definition or {}).is_valid(dependencies)
    assert check.left is not None and check.right is not None
    left = dependencies.get(check.left.dependency_id)
    right = dependencies.get(check.right.dependency_id)
    left_found, left_value = _json_path_value(left, check.left.json_path)
    right_found, right_value = _json_path_value(right, check.right.json_path)
    return left_found and right_found and left_value == right_value


__all__ = [
    "acceptance_path", "evaluate_acceptance", "evaluate_join_check",
    "validate_acceptance_check", "validate_join_check",
]