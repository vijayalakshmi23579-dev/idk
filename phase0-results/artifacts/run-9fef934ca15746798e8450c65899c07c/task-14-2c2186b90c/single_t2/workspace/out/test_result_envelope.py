"""Test suite for the result envelope module."""

import pytest

from result_envelope import build_result, SCHEMA_VERSION


def test_schema_version():
    """The built result must contain the expected schema_version."""
    result = build_result(node_id="node-1", summary="test", artifacts={})
    assert result["schema_version"] == SCHEMA_VERSION


def test_node_id():
    """The built result must contain the provided node_id."""
    result = build_result(node_id="my-node", summary="test", artifacts={})
    assert result["node_id"] == "my-node"


def test_summary_and_artifacts():
    """Summary and artifacts are stored correctly."""
    result = build_result(
        node_id="n1",
        summary="all good",
        artifacts={"log": "abc.txt"},
    )
    assert result["summary"] == "all good"
    assert result["artifacts"] == {"log": "abc.txt"}