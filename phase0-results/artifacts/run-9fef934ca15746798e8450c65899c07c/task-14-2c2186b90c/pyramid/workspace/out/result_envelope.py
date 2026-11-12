"""Result envelope builder producing versioned result dictionaries."""

SCHEMA_VERSION = 1


def build_result(node_id, summary, artifacts):
    """Return a versioned result dictionary.

    Args:
        node_id: Identifier of the node producing the result.
        summary: Human-readable summary of the result.
        artifacts: Artifact payload associated with the result.

    Returns:
        A dictionary containing schema_version, node_id, summary, and artifacts.
    """
    return {
        "schema_version": SCHEMA_VERSION,
        "node_id": node_id,
        "summary": summary,
        "artifacts": artifacts,
    }
