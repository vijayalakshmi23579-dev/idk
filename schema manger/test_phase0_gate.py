"""Offline tests for the Phase 0 gate on Layer-2 Task workers."""
import asyncio
import json
import tempfile
from pathlib import Path

from agent_tasks import AgentTasksTool
from execute import execute_tool
from main import build_full_system_prompt
from query import _tools_for_context
from registry import register
from phase0_gate import load_phase0_report
from tool import ToolUseContext


SHAPES = [
    "breadth_research",
    "multi_file_code",
    "data_pipeline",
    "sequential_reasoning",
]


def build_complete_report() -> dict:
    results = []
    for index in range(30):
        for baseline in ("single_t2", "single_t3", "pyramid"):
            results.append({
                "task_id": f"case-{index:02d}",
                "shape": SHAPES[index % len(SHAPES)],
                "baseline": baseline,
                "task_eligible_for_wide": index == 0,
                "checks_passed": True,
                "quality_score": 0.8,
                "tokens": 1000 + index,
                "api_calls": 3,
                "duration_seconds": 10.0,
                "cost_usd": 0.01,
                "task_invocations": 1 if baseline == "pyramid" and index == 0 else 0,
                "child_api_calls": 3 if baseline == "pyramid" and index == 0 else 0,
                "child_provider_attempts": 3 if baseline == "pyramid" and index == 0 else 0,
                "child_model_ids": (
                    ["test/worker"] if baseline == "pyramid" and index == 0 else []
                ),
                "task_worker_count": 3 if baseline == "pyramid" and index == 0 else 0,
                "task_worker_successes": 3 if baseline == "pyramid" and index == 0 else 0,
                "task_worker_failures": 0,
                "human_reviewed": index < 3,
            })
    return {"schema_version": "1.0", "results": results}


async def main():
    register(AgentTasksTool)
    with tempfile.TemporaryDirectory() as temporary:
        report_path = Path(temporary) / "phase0.json"
        report_path.write_text(json.dumps(build_complete_report()), encoding="utf-8")
        report = load_phase0_report(report_path)
        assert len(report.results) == 90

        system_prompt = build_full_system_prompt()
        assert "Task(tasks)" in system_prompt
        assert "join_checks" in system_prompt
        assert "Unresolved escalation or contract-change" in system_prompt
        assert "one stronger-tier" in system_prompt
        assert "separate Task calls are separate runs" in system_prompt
        assert '"estimated_steps":12' in system_prompt
        assert "<=8-step estimate intentionally returns `mode=single`" in system_prompt
        assert any(tool.name == "Task" for tool in _tools_for_context(ToolUseContext()))
        invalid_call = await execute_tool(
            {"id": "task-schema-check", "name": "Task", "input": {}},
            ToolUseContext(),
        )
        assert invalid_call.is_error
        assert "disabled" not in invalid_call.result_message.content

        incomplete_path = Path(temporary) / "incomplete.json"
        incomplete = build_complete_report()
        incomplete["results"].pop()
        incomplete_path.write_text(json.dumps(incomplete), encoding="utf-8")
        try:
            load_phase0_report(incomplete_path)
            raise AssertionError("incomplete baseline report should fail")
        except ValueError as error:
            assert "incomplete or invalid" in str(error)

    print("Phase 0 report validation and Task availability tests passed")


if __name__ == "__main__":
    asyncio.run(main())