"""Tests for the result envelope schema fields."""

from result_envelope import build_result, SCHEMA_VERSION


def test_build_result_has_schema_version_and_node_id():
    envelope = build_result("node-42", "processed successfully", {"count": 3})

    assert envelope["schema_version"] == SCHEMA_VERSION
    assert envelope["node_id"] == "node-42"
