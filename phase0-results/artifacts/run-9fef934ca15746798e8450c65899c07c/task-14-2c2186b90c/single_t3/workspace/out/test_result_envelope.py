"""Tests for the result envelope builder."""

import pytest

from result_envelope import build_result, SCHEMA_VERSION


def test_result_envelope_has_schema_version_and_node_id():
    """build_result must return a dict with correct schema_version and node_id."""
    node_id = "node-42"
    summary = "Task completed successfully"
    artifacts = {"output.txt": "hello world"}

    result = build_result(node_id, summary, artifacts)

    assert isinstance(result, dict)
    assert result["schema_version"] == SCHEMA_VERSION
    assert result["node_id"] == node_id
    assert result["summary"] == summary
    assert result["artifacts"] == artifacts
