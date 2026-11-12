"""Offline tests for configured worker model tiers."""
import os
import tempfile
from pathlib import Path
from unittest.mock import patch

from model_router import (
    load_tier_configuration,
    resolve_next_task_model,
    resolve_task_model,
    resolve_task_tier,
)


def main():
    assert [resolve_task_tier(value) for value in range(1, 6)] == [
        "T1", "T1", "T2", "T2", "T3",
    ]
    assert resolve_task_tier(5, "T1") == "T1"

    with tempfile.TemporaryDirectory() as temporary:
        config_path = Path(temporary) / "tiers.yaml"
        config_path.write_text(
            "version: 1\n"
            "tiers:\n"
            "  T1:\n    model: test/cheap\n"
            "  T2:\n    model: test/mid\n"
            "  T3:\n    model: ACTIVE_MODEL\n",
            encoding="utf-8",
        )
        config = load_tier_configuration(config_path)
        assert config.tiers["T1"].model == "test/cheap"

        with patch.dict(os.environ, {"AGENT_WORKER_TIERS_CONFIG": str(config_path)}):
            assert resolve_task_model(1) == "test/cheap"
            assert resolve_task_model(4) == "test/mid"
            assert resolve_task_model(5, active_model="test/parent") == "test/parent"
            assert resolve_next_task_model(
                "T1", "test/cheap", active_model="test/parent",
            ) == ("T2", "test/mid")
            assert resolve_next_task_model(
                "T2", "test/mid", active_model="test/parent",
            ) == ("T3", "test/parent")
            assert resolve_next_task_model(
                "T3", "test/parent", active_model="test/parent",
            ) is None
            with patch("api.get_active_model", return_value="test/strong"):
                assert resolve_task_model(5) == "test/strong"

    print("Worker tier routing tests passed")


if __name__ == "__main__":
    main()