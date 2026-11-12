"""Phase 0 dataset validation, result metrics, and Pyramid kill criteria."""
from __future__ import annotations

import json
import math
import argparse
from collections import defaultdict
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from phase0_gate import (
    BaselineName,
    BaselineResult,
    MIN_PILOT_CASE_COUNT,
    Phase0Report,
    REQUIRED_CASE_COUNT,
    TaskShape,
)


CheckKind = Literal["command", "assert", "exists", "schema"]
PARALLEL_SHAPES = frozenset({
    "breadth_research", "multi_file_code", "data_pipeline",
})


class EvaluationCheck(BaseModel):
    kind: CheckKind
    spec: str = Field(min_length=1)


class EvaluationTask(BaseModel):
    task_id: str = Field(min_length=1)
    shape: TaskShape
    prompt: str = Field(min_length=1)
    acceptance: list[EvaluationCheck] = Field(min_length=1)
    fixture: str | None = None
    task_eligible_for_wide: bool = False


class EvaluationDataset(BaseModel):
    schema_version: Literal["1.0"]
    tasks: list[EvaluationTask] = Field(
        min_length=MIN_PILOT_CASE_COUNT,
        max_length=REQUIRED_CASE_COUNT,
    )

    @classmethod
    def model_validate_json(cls, json_data: str | bytes, **kwargs):
        dataset = super().model_validate_json(json_data, **kwargs)
        ids = [task.task_id for task in dataset.tasks]
        if len(set(ids)) != len(ids):
            raise ValueError("Dataset task IDs must be unique")
        shapes = {task.shape for task in dataset.tasks}
        required = {
            "breadth_research", "multi_file_code",
            "data_pipeline", "sequential_reasoning",
        }
        if shapes != required:
            raise ValueError("Dataset must cover all four required task shapes")
        return dataset


def load_dataset(path: str) -> EvaluationDataset:
    with open(path, encoding="utf-8") as source:
        return EvaluationDataset.model_validate_json(source.read())


def load_report(path: str) -> Phase0Report:
    with open(path, encoding="utf-8") as source:
        return Phase0Report.model_validate_json(source.read())


def validate_report_matches_dataset(
    dataset: EvaluationDataset,
    report: Phase0Report,
) -> None:
    expected = {
        task.task_id: (task.shape, task.task_eligible_for_wide)
        for task in dataset.tasks
    }
    report_ids = {result.task_id for result in report.results}
    if report_ids != set(expected):
        raise ValueError("Report task IDs do not match the evaluation dataset")
    if any(
        expected[result.task_id] != (
            result.shape, result.task_eligible_for_wide,
        )
        for result in report.results
    ):
        raise ValueError("Report task metadata does not match the evaluation dataset")


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, int((len(ordered) - 1) * percentile + 0.5)))
    return ordered[index]


def _cost_per_success(results: list[BaselineResult]) -> float | None:
    successful = [result for result in results if result.checks_passed]
    if not successful or any(result.cost_usd is None for result in results):
        return None
    return sum(result.cost_usd for result in results) / len(successful)


def summarize_results(results: list[BaselineResult]) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[BaselineResult]] = defaultdict(list)
    for result in results:
        groups[f"{result.baseline}:{result.shape}"].append(result)

    summaries: dict[str, dict[str, Any]] = {}
    for key, group in sorted(groups.items()):
        human_scores = [
            result.human_quality_score for result in group
            if result.human_quality_score is not None
        ]
        instrumented = [
            result for result in group if result.child_api_calls is not None
        ]
        summaries[key] = {
            "cases": len(group),
            "success_rate": sum(result.checks_passed for result in group) / len(group),
            "mean_quality": sum(result.quality_score for result in group) / len(group),
            "human_reviewed_runs": len(human_scores),
            "mean_human_quality": (
                sum(human_scores) / len(human_scores) if human_scores else None
            ),
            "total_tokens": sum(result.tokens for result in group),
            "total_api_calls": sum(result.api_calls for result in group),
            "task_telemetry_runs": len(instrumented),
            "task_invocations": (
                sum(result.task_invocations or 0 for result in instrumented)
                if instrumented else None
            ),
            "child_api_calls": (
                sum(result.child_api_calls or 0 for result in instrumented)
                if instrumented else None
            ),
            "child_provider_attempts": (
                sum(result.child_provider_attempts or 0 for result in instrumented)
                if instrumented else None
            ),
            "task_worker_count": (
                sum(result.task_worker_count or 0 for result in instrumented)
                if instrumented else None
            ),
            "task_worker_successes": (
                sum(result.task_worker_successes or 0 for result in instrumented)
                if instrumented else None
            ),
            "task_worker_failures": (
                sum(result.task_worker_failures or 0 for result in instrumented)
                if instrumented else None
            ),
            "child_model_ids": (
                sorted({
                    model_id for result in instrumented
                    for model_id in (result.child_model_ids or [])
                })
                if instrumented else None
            ),
            "p50_seconds": _percentile([result.duration_seconds for result in group], 0.50),
            "p95_seconds": _percentile([result.duration_seconds for result in group], 0.95),
            "cost_per_success_usd": _json_number(_cost_per_success(group)),
        }
    return summaries


def assess_pyramid(report: Phase0Report) -> dict[str, Any]:
    """Apply the stated 30% cost-per-success kill criterion to parallel shapes."""
    by_task: dict[str, dict[str, BaselineResult]] = defaultdict(dict)
    for result in report.results:
        by_task[result.task_id][result.baseline] = result

    pyramid_results = [
        task_results["pyramid"]
        for task_results in by_task.values()
        if "pyramid" in task_results
        and task_results["pyramid"].shape in PARALLEL_SHAPES
    ]
    expected = sum(
        1 for task_results in by_task.values()
        if task_results["single_t2"].shape in PARALLEL_SHAPES
    )
    if len(pyramid_results) != expected or expected == 0:
        return {
            "eligible": False,
            "reason": "Pyramid results are missing for parallel-shaped evaluation tasks.",
        }

    baseline_candidates = [
        [task_results[baseline] for task_results in by_task.values()
         if task_results[baseline].shape in PARALLEL_SHAPES]
        for baseline in ("single_t2", "single_t3")
    ]
    baseline_candidates = [candidate for candidate in baseline_candidates if candidate]
    best_baseline = max(
        baseline_candidates,
        key=lambda group: (
            sum(result.quality_score for result in group) / len(group),
            -(_cost_per_success(group) or 0.0),
        ),
    )

    baseline_success = sum(result.checks_passed for result in best_baseline) / len(best_baseline)
    pyramid_success = sum(result.checks_passed for result in pyramid_results) / len(pyramid_results)
    baseline_quality = sum(result.quality_score for result in best_baseline) / len(best_baseline)
    pyramid_quality = sum(result.quality_score for result in pyramid_results) / len(pyramid_results)
    baseline_cost = _cost_per_success(best_baseline)
    pyramid_cost = _cost_per_success(pyramid_results)
    cost_reduction = (
        1.0 - pyramid_cost / baseline_cost
        if baseline_cost is not None and pyramid_cost is not None and baseline_cost > 0
        else None
    )
    quality_preserved = (
        pyramid_success >= baseline_success
        and pyramid_quality >= baseline_quality
    )
    passes = quality_preserved and cost_reduction is not None and cost_reduction >= 0.30
    return {
        "eligible": True,
        "passes_kill_criteria": passes,
        "best_single_baseline": best_baseline[0].baseline,
        "baseline_success_rate": baseline_success,
        "pyramid_success_rate": pyramid_success,
        "baseline_mean_quality": baseline_quality,
        "pyramid_mean_quality": pyramid_quality,
        "baseline_cost_per_success_usd": _json_number(baseline_cost),
        "pyramid_cost_per_success_usd": _json_number(pyramid_cost),
        "cost_reduction": cost_reduction,
        "reason": (
            "Pyramid meets quality and 30% cost-per-success criteria."
            if passes else
            "Keep Pyramid opt-in: cost data is incomplete or quality/cost criteria were not met."
        ),
    }


def assess_efficiency(report: Phase0Report) -> dict[str, Any]:
    """Compare latency and token use without requiring provider price data."""
    by_task: dict[str, dict[str, BaselineResult]] = defaultdict(dict)
    for result in report.results:
        by_task[result.task_id][result.baseline] = result

    parallel_tasks = {
        task_id: results for task_id, results in by_task.items()
        if results["single_t2"].shape in PARALLEL_SHAPES
    }
    if not parallel_tasks or any("pyramid" not in results for results in parallel_tasks.values()):
        return {
            "eligible": False,
            "reason": "A complete Task-DAG result is required for every parallel-shaped task.",
        }

    candidates = [
        [results[baseline] for results in parallel_tasks.values()]
        for baseline in ("single_t2", "single_t3")
    ]
    baseline = max(
        candidates,
        key=lambda group: (
            sum(result.checks_passed for result in group) / len(group),
            sum(result.quality_score for result in group) / len(group),
            -sum(result.duration_seconds for result in group),
            -sum(result.tokens for result in group),
        ),
    )
    pyramid = [results["pyramid"] for results in parallel_tasks.values()]
    baseline_seconds = sum(result.duration_seconds for result in baseline)
    pyramid_seconds = sum(result.duration_seconds for result in pyramid)
    baseline_tokens = sum(result.tokens for result in baseline)
    pyramid_tokens = sum(result.tokens for result in pyramid)
    baseline_success = sum(result.checks_passed for result in baseline) / len(baseline)
    pyramid_success = sum(result.checks_passed for result in pyramid) / len(pyramid)
    baseline_quality = sum(result.quality_score for result in baseline) / len(baseline)
    pyramid_quality = sum(result.quality_score for result in pyramid) / len(pyramid)
    quality_preserved = (
        pyramid_success >= baseline_success
        and pyramid_quality >= baseline_quality
    )
    latency_reduction = (
        1.0 - pyramid_seconds / baseline_seconds if baseline_seconds > 0 else None
    )
    token_reduction = (
        1.0 - pyramid_tokens / baseline_tokens if baseline_tokens > 0 else None
    )
    return {
        "eligible": True,
        "best_single_baseline": baseline[0].baseline,
        "parallel_tasks": len(parallel_tasks),
        "baseline_success_rate": baseline_success,
        "pyramid_success_rate": pyramid_success,
        "baseline_mean_quality": baseline_quality,
        "pyramid_mean_quality": pyramid_quality,
        "quality_preserved": quality_preserved,
        "baseline_seconds": baseline_seconds,
        "pyramid_seconds": pyramid_seconds,
        "latency_reduction": latency_reduction,
        "faster_without_quality_loss": (
            quality_preserved and latency_reduction is not None and latency_reduction > 0
        ),
        "baseline_tokens": baseline_tokens,
        "pyramid_tokens": pyramid_tokens,
        "token_reduction": token_reduction,
        "fewer_tokens_without_quality_loss": (
            quality_preserved and token_reduction is not None and token_reduction > 0
        ),
        "reason": (
            "Efficiency is reported as latency and token deltas; cost is not required."
        ),
    }


def _json_number(value: float | None) -> float | None:
    return value if value is not None and math.isfinite(value) else None


def report_json(report: Phase0Report) -> str:
    return json.dumps({
        "schema_version": report.schema_version,
        "metrics": summarize_results(report.results),
        "pyramid_assessment": assess_pyramid(report),
        "pyramid_efficiency_assessment": assess_efficiency(report),
    }, indent=2, allow_nan=False)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", required=True,
        help=f"Versioned JSON dataset with {MIN_PILOT_CASE_COUNT}-{REQUIRED_CASE_COUNT} tasks",
    )
    parser.add_argument("--report", required=True, help="Collected baseline/Pyramid results JSON")
    parser.add_argument("--output", help="Write summary JSON to this path instead of stdout")
    args = parser.parse_args(argv)
    try:
        dataset = load_dataset(args.dataset)
        report = load_report(args.report)
        validate_report_matches_dataset(dataset, report)
        summary = report_json(report)
        if args.output:
            Path(args.output).write_text(summary + "\n", encoding="utf-8")
        else:
            print(summary)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    return 0


__all__ = [
    "EvaluationCheck", "EvaluationDataset", "EvaluationTask",
    "assess_efficiency", "assess_pyramid", "load_dataset", "load_report", "main", "report_json",
    "summarize_results", "validate_report_matches_dataset",
]


if __name__ == "__main__":
    raise SystemExit(main())