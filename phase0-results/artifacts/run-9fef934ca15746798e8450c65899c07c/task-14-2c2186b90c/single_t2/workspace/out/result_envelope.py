"""Result envelope module providing a versioned build_result function."""

SCHEMA_VERSION = "1.0.0"


def build_result(node_id: str, summary: str, artifacts: dict) -> dict:
    """Build a versioned result envelope dictionary.

    Args:
        node_id: Identifier for the node producing the result.
        summary: A short human‑readable summary of the result.
        artifacts: Dict of arbitrary artifact data associated with the result.

    Returns:
        A dictionary with keys: schema_version, node_id, summary, artifacts.
    """
    return {
        "schema_version": SCHEMA_VERSION,
        "node_id": node_id,
        "summary": summary,
        "artifacts": artifacts,
    }