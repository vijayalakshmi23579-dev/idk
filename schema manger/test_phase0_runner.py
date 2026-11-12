"""Offline tests for serial Phase 0 execution and its safety boundaries."""
import asyncio
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import api
from phase0_eval import EvaluationCheck, EvaluationDataset, summarize_results
from phase0_gate import Phase0Report
from phase0_runner import (
    _check_passes,
    build_review_template,
    finalize_report,
    run_dataset,
)
from phase0_eval import validate_report_matches_dataset


async def main() -> None:
    shapes = [
        "breadth_research", "multi_file_code",
        "data_pipeline", "sequential_reasoning",
    ]
    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        fixture = root / "fixtures"
        fixture.mkdir()
        (fixture / "source.txt").write_text("pilot fixture evidence", encoding="utf-8")
        dataset_path = root / "dataset.json"
        dataset_path.write_text(json.dumps({
            "schema_version": "1.0",
            "tasks": [
                {
                    "task_id": f"case-{index}",
                    "shape": shapes[index % len(shapes)],
                    "prompt": f"Analyze fixture case {index}",
                    "task_eligible_for_wide": index == 0,
                    "fixture": "fixtures",
                    "acceptance": [{
                        "kind": "assert",
                        "spec": json.dumps({
                            "path": "source.txt",
                            "contains": "pilot fixture",
                        }),
                    }],
                }
                for index in range(8)
            ],
        }), encoding="utf-8")
        dataset = EvaluationDataset.model_validate_json(
            dataset_path.read_text(encoding="utf-8")
        )
        requested_models: list[str] = []

        async def fake_stream(**kwargs):
            requested_models.append(kwargs["model"])
            yield {
                "type": "runtime_activity",
                "area": "model",
                "status": "started",
                "model": kwargs["model"],
            }
            if kwargs.get("query_source") == "subagent_task":
                yield {"type": "text_delta", "text": "Worker completed"}
                yield {
                    "type": "message_stop",
                    "stop_reason": "end_turn",
                    "usage": {"input_tokens": 11, "output_tokens": 5},
                }
                return
            available_tools = {tool.name for tool in kwargs["tools"]}
            has_task_result = any(
                getattr(message, "type", None) == "tool_result"
                and getattr(message, "tool_use_id", None) == "phase0-task-call"
                for message in kwargs["messages"]
            )
            if "Task" in available_tools and not has_task_result:
                yield {
                    "type": "tool_use",
                    "id": "phase0-task-call",
                    "name": "Task",
                    "input": {
                        "tasks": [
                            {
                                "id": f"branch-{index}",
                                "instruction": "Summarize the available fixture.",
                                "complexity": 1,
                            }
                            for index in range(3)
                        ],
                        "estimate": {
                            "estimated_steps": 12,
                            "sequential_dependency": False,
                            "independent_branches": 3,
                            "read_heavy": True,
                            "disjoint_writes": False,
                            "uncertain_steps": False,
                        },
                    },
                }
                yield {
                    "type": "message_stop",
                    "stop_reason": "tool_use",
                    "usage": {"input_tokens": 11, "output_tokens": 5},
                }
                return
            yield {"type": "text_delta", "text": "Completed from the fixture"}
            yield {
                "type": "message_stop",
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 11, "output_tokens": 5},
            }

        tier_config = root / "tiers.json"
        tier_config.write_text(json.dumps({
            "version": 1,
            "tiers": {
                "T1": {"model": "adapter-test/worker-small"},
                "T2": {"model": "adapter-test/model-mid"},
                "T3": {"model": "ACTIVE_MODEL"},
            },
        }), encoding="utf-8")
        with patch.dict("os.environ", {
            "AGENT_WORKER_TIERS_CONFIG": str(tier_config),
        }), patch.object(api, "stream", fake_stream):
            results = await run_dataset(
                dataset,
                dataset_path,
                t2_model="adapter-test/model-mid",
                t3_model="custom-endpoint/model-strong",
                artifacts_dir=root / "artifacts",
                timeout_seconds=5,
            )

        assert len(results) == 24
        assert requested_models.count("adapter-test/model-mid") == 8
        assert requested_models.count("custom-endpoint/model-strong") == 24
        assert requested_models.count("adapter-test/worker-small") == 24
        assert all(result.checks_passed for result in results)
        assert all(
            result.tokens == 16 for result in results
            if result.baseline != "pyramid"
        )
        assert all(
            result.tokens == 80 for result in results
            if result.baseline == "pyramid"
        )
        assert all(
            result.api_calls == 1 for result in results
            if result.baseline != "pyramid"
        )
        assert all(
            result.api_calls == 5 for result in results
            if result.baseline == "pyramid"
        )
        assert all(result.task_invocations == 0 for result in results if result.baseline != "pyramid")
        pyramid_results = [result for result in results if result.baseline == "pyramid"]
        assert all(result.task_invocations == 1 for result in pyramid_results)
        assert all(result.child_api_calls == 3 for result in pyramid_results)
        assert all(result.child_model_ids == ["adapter-test/worker-small"] for result in pyramid_results)
        assert all(result.task_worker_count == 3 for result in pyramid_results)
        assert all(result.task_worker_successes == 3 for result in pyramid_results)
        pyramid_usage = summarize_results(results)["pyramid:breadth_research"]
        assert pyramid_usage["task_invocations"] == 2
        assert pyramid_usage["child_api_calls"] == 6
        assert pyramid_usage["task_worker_successes"] == 6
        assert pyramid_usage["child_model_ids"] == ["adapter-test/worker-small"]
        assert all(result.cost_usd is None for result in results)
        assert all(not result.human_reviewed for result in results)
        assert all(Path(result.response_path).is_file() for result in results)
        assert all(Path(result.artifact_dir).is_dir() for result in results)
        assert all(
            (Path(result.artifact_dir) / "workspace").is_dir()
            for result in results
        )

        review_doc = build_review_template(
            results, {"case-0", "case-1", "case-2"},
        )
        for review in review_doc["reviews"]:
            review["human_quality_score"] = 0.9
            review["human_review_notes"] = "Reviewed against the case criteria."
        report_path = root / "report.json"
        review_path = root / "report.review.json"
        report_path.write_text(json.dumps({
            "schema_version": "1.0",
            "results": [result.model_dump() for result in results],
        }), encoding="utf-8")
        review_path.write_text(json.dumps(review_doc), encoding="utf-8")
        reviewed_path = finalize_report(report_path, review_path)
        reviewed_report = Phase0Report.model_validate_json(
            reviewed_path.read_text(encoding="utf-8")
        )
        validate_report_matches_dataset(dataset, reviewed_report)
        reviewed = [result for result in reviewed_report.results if result.human_reviewed]
        assert len(reviewed) == 9
        assert all(result.human_quality_score == 0.9 for result in reviewed)
        reviewed_metrics = summarize_results(reviewed_report.results)
        assert all(
            summary["mean_human_quality"] == 0.9
            for summary in reviewed_metrics.values()
            if summary["human_reviewed_runs"]
        )

        outside = root.parent / "outside-phase0.txt"
        outside.write_text("not in the workspace", encoding="utf-8")
        try:
            assert not _check_passes(
                EvaluationCheck(kind="exists", spec="../outside-phase0.txt"),
                fixture,
            )
            assert not _check_passes(
                EvaluationCheck(kind="command", spec="touch /tmp/phase0-unsafe"),
                fixture,
            )
            assert _check_passes(
                EvaluationCheck(
                    kind="assert",
                    spec=json.dumps({
                        "target": "response",
                        "contains": "specific evidence",
                    }),
                ),
                fixture,
                "Answer with specific evidence.",
            )
        finally:
            outside.unlink(missing_ok=True)

    print("Phase 0 runner, configurable model IDs, and safe checks passed")


if __name__ == "__main__":
    asyncio.run(main())