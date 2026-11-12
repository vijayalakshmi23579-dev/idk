"""Result envelope builder.

Provides :func:`build_result` which constructs a versioned result dictionary
for task workers in the schema manager benchmark.
"""

SCHEMA_VERSION = "1.0.0"


def build_result(node_id, summary, artifacts):
    """Build a versioned result dictionary.

    Parameters
    ----------
    node_id : str
        Identifier of the node producing this result.
    summary : str
        Human-readable summary of the node's output.
    artifacts : dict
        Arbitrary artifacts produced by the node.

    Returns
    -------
    dict
        A versioned result dictionary containing ``schema_version``, ``node_id``,
        ``summary``, and ``artifacts`` fields.
    """
    return {
        "schema_version": SCHEMA_VERSION,
        "node_id": node_id,
        "summary": summary,
        "artifacts": artifacts,
    }
