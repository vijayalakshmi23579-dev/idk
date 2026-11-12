"""Run serial Phase 0 cases through the configured query/provider adapter."""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import shutil
import tempfile
import time
import uuid
from pathlib import Path
from typing import Literal
from unittest.mock import patch

from jsonschema import Draft202012Validator
from pydantic import BaseModel, Field

import api
from agent_tasks import AgentTasksTool
from messages import create_user_message
from phase0_eval import EvaluationCheck, EvaluationDataset, EvaluationTask
from phase0_gate import BaselineName, BaselineResult, Phase0Report
from query import QueryParams, query
from query_transitions import Terminal
from registry import TOOLS, clear, register
from state import AppState, FileStateCache, ToolPermissionContext
from task_scheduler import FileLeaseRegistry, TaskFileScope
from tool import ToolUseContext
from tools.edit import EditTool
from tools.read import ReadTool
from tools.write import WriteTool


WORKER_TOOLS = frozenset({"Read", "Edit", "Write"})
PYRAMID_TOOLS = WORKER_TOOLS | {"Task"}
SENSITIVE_FIXTURE_NAMES = frozenset({".env", ".env.local", "id_rsa", "id_ed25519"})


class HumanReview(BaseModel):
	task_id: str = Field(min_length=1)
	baseline: BaselineName
	human_quality_score: float = Field(ge=0, le=1)
	human_review_notes: str | None = Field(default=None, max_length=2_000)


class HumanReviewDocument(BaseModel):
	schema_version: Literal["1.0"]
	reviews: list[HumanReview] = Field(min_length=9)


def _inside(root: Path, candidate: Path) -> bool:
	try:
		return os.path.commonpath((str(root), str(candidate))) == str(root)
	except ValueError:
		return False


def _workspace_path(root: Path, relative_path: str) -> Path:
	candidate_path = Path(relative_path)
	if candidate_path.is_absolute():
		raise ValueError("Acceptance paths must be relative to the case workspace")
	resolved = (root / candidate_path).resolve()
	if not _inside(root.resolve(), resolved):
		raise ValueError("Acceptance path is outside the case workspace")
	return resolved


def _check_spec_payload(spec: str) -> dict[str, object]:
	payload = json.loads(spec)
	if not isinstance(payload, dict):
		raise ValueError("Check spec must be a JSON object")
	if payload.get("target") != "response" and not isinstance(payload.get("path"), str):
		raise ValueError("File check spec must include a relative 'path'")
	return payload


def _check_passes(
	check: EvaluationCheck,
	root: Path,
	response: str = "",
) -> bool:
	try:
		if check.kind == "exists":
			return _workspace_path(root, check.spec).is_file()
		if check.kind == "schema":
			payload = _check_spec_payload(check.spec)
			schema = payload.get("schema")
			if not isinstance(schema, dict):
				return False
			Draft202012Validator.check_schema(schema)
			instance = json.loads(
				_workspace_path(root, str(payload["path"])).read_text(encoding="utf-8")
			)
			return Draft202012Validator(schema).is_valid(instance)
		if check.kind == "assert":
			payload = _check_spec_payload(check.spec)
			if payload.get("target") == "response":
				content = response
			else:
				path = _workspace_path(root, str(payload["path"]))
				content = path.read_text(encoding="utf-8")
			if "contains" in payload:
				return str(payload["contains"]) in content
			if "equals_text" in payload:
				return content == str(payload["equals_text"])
			if "equals" in payload:
				value: object = json.loads(content)
				for part in str(payload.get("json_path", "")).split("."):
					if not part:
						continue
					if not isinstance(value, dict) or part not in value:
						return False
					value = value[part]
				return value == payload["equals"]
			return False
		# Arbitrary acceptance commands would execute model-written content on
		# the host. Add command checks only after an OS-level sandbox exists.
		return False
	except (OSError, ValueError, TypeError, json.JSONDecodeError):
		return False


def _declared_write_paths(task: EvaluationTask) -> frozenset[str]:
	paths: set[str] = set()
	for check in task.acceptance:
		if check.kind == "exists":
			paths.add(check.spec)
		elif check.kind in ("schema", "assert"):
			try:
				payload = _check_spec_payload(check.spec)
				if payload.get("target") != "response":
					paths.add(str(payload["path"]))
			except (ValueError, json.JSONDecodeError):
				continue
	return frozenset(paths)


def _copy_fixture(dataset_path: Path, fixture: str | None, destination: Path) -> None:
	if not fixture:
		return
	fixture_path = Path(fixture)
	if fixture_path.is_absolute():
		raise ValueError("Fixture paths must be relative to the dataset file")
	source = (dataset_path.parent / fixture_path).resolve()
	if not _inside(dataset_path.parent.resolve(), source) or not source.is_dir():
		raise ValueError("Fixture must be a directory inside the dataset directory")
	for entry in source.rglob("*"):
		if entry.is_symlink():
			raise ValueError("Fixture directories cannot contain symlinks")
		if entry.name in SENSITIVE_FIXTURE_NAMES or entry.name.startswith(".env."):
			raise ValueError("Fixture contains a credential-like file")
		relative = entry.relative_to(source)
		target = destination / relative
		if entry.is_dir():
			target.mkdir(parents=True, exist_ok=True)
		elif entry.is_file():
			target.parent.mkdir(parents=True, exist_ok=True)
			shutil.copyfile(entry, target)


def _scope_for_case(root: Path, task: EvaluationTask) -> tuple[TaskFileScope, FileLeaseRegistry]:
	leases = FileLeaseRegistry(str(root))
	reads = frozenset(
		leases.resolve(str(path.relative_to(root)))
		for path in root.rglob("*") if path.is_file()
	)
	writes = frozenset(leases.resolve(path) for path in _declared_write_paths(task))
	if writes:
		leases.acquire("phase0", writes)
	return TaskFileScope(
		task_id="phase0",
		project_root=str(root),
		read_paths=reads,
		write_paths=writes,
		leases=leases,
	), leases


def _register_case_tools(include_task: bool) -> dict:
	previous = dict(TOOLS)
	clear()
	for tool in (ReadTool, EditTool, WriteTool):
		register(tool)
	if include_task:
		register(AgentTasksTool)
	return previous


def _restore_tools(previous: dict) -> None:
	clear()
	for tool in previous.values():
		register(tool)


def _case_prompt(task: EvaluationTask, root: Path, pyramid: bool) -> str:
	files = sorted(
		str(path.relative_to(root)) for path in root.rglob("*") if path.is_file()
	)
	prefix = (
		"You are coordinating a benchmark task. Estimate the work first. Use Task "
		"only when the deterministic split-gate conditions hold: more than eight "
		"steps, no sequential dependency, and at least three independent branches "
		"with read-heavy or disjoint work. Declare exact file scopes and dependencies. "
		"Otherwise complete the task directly. Do not use shell commands.\n\n"
		if pyramid else
		"Complete this benchmark task directly. Do not delegate or use shell commands.\n\n"
	)
	return (
		prefix
		+ f"Workspace: {root}\nAvailable fixture files: {files}\n\n"
		+ task.prompt
	)


async def _run_trial(
	task: EvaluationTask,
	dataset_path: Path,
	baseline: BaselineName,
	model: str,
	*,
	artifact_workspace: Path,
	timeout_seconds: int,
) -> BaselineResult:
	pyramid = baseline == "pyramid"
	started = time.perf_counter()
	invocations = 0
	provider_attempts = 0
	child_api_calls = 0
	child_provider_attempts = 0
	task_invocations = 0
	task_worker_count = 0
	task_worker_successes = 0
	task_worker_failures = 0
	tokens = 0
	models: set[str] = set()
	child_models: set[str] = set()
	events: list[object] = []

	with tempfile.TemporaryDirectory(prefix="phase0-case-") as temp_dir:
		root = Path(temp_dir).resolve()
		_copy_fixture(dataset_path, task.fixture, root)
		scope, leases = _scope_for_case(root, task)
		previous_tools = _register_case_tools(pyramid)
		try:
			state = AppState(
				messages=[],
				tool_permission_context=ToolPermissionContext(mode="bypassPermissions"),
				read_file_cache=FileStateCache(),
			)
			context = ToolUseContext(
				get_app_state=lambda: state,
				set_app_state=lambda _update: None,
				read_file_state=state.read_file_cache,
			)
			context.allowed_tool_names = PYRAMID_TOOLS if pyramid else WORKER_TOOLS
			context.file_scope = scope
			context.project_root = str(root)
			context.model_override = model
			context.session_id = f"phase0-{task.task_id}-{baseline}"
			context.rendered_system_prompt = "You are a careful, scoped benchmark worker."

			original_stream = api.stream

			async def capture_stream(**kwargs):
				nonlocal invocations, provider_attempts, child_api_calls
				nonlocal child_provider_attempts, tokens
				invocations += 1
				is_child_request = kwargs.get("query_source") == "subagent_task"
				if is_child_request:
					child_api_calls += 1
				requested_model = kwargs.get("model")
				if requested_model:
					models.add(str(requested_model))
					if is_child_request:
						child_models.add(str(requested_model))
				async for event in original_stream(**kwargs):
					actual_model = event.get("model")
					if actual_model:
						models.add(str(actual_model))
						if is_child_request:
							child_models.add(str(actual_model))
					if (
						event.get("type") == "runtime_activity"
						and event.get("area") == "model"
						and event.get("status") == "started"
					):
						provider_attempts += 1
						if is_child_request:
							child_provider_attempts += 1
					if event.get("type") == "message_stop":
						usage = event.get("usage") or {}
						tokens += int(
							usage.get("prompt_tokens", usage.get("input_tokens", 0)) or 0
						)
						tokens += int(
							usage.get("completion_tokens", usage.get("output_tokens", 0)) or 0
						)
					yield event

			async def collect_query() -> None:
				nonlocal task_invocations, task_worker_count
				nonlocal task_worker_successes, task_worker_failures
				async for event in query(QueryParams(
					messages=[create_user_message(_case_prompt(task, root, pyramid))],
					system_prompt=context.rendered_system_prompt,
					tool_use_context=context,
					max_turns=8,
					max_output_tokens_override=8_000,
					query_source="phase0_pilot",
				)):
					events.append(event)
					if getattr(event, "type", None) == "assistant":
						task_invocations += sum(
							getattr(call, "name", None) == "Task"
							for call in getattr(event, "tool_calls", None) or []
						)
					elif getattr(event, "type", None) == "progress":
						data = getattr(event, "data", {})
						if data.get("type") == "task_lifecycle":
							if data.get("status") == "started":
								task_worker_count += 1
							elif data.get("status") == "succeeded":
								task_worker_successes += 1
							elif data.get("status") == "failed":
								task_worker_failures += 1

			with patch.object(api, "stream", capture_stream):
				try:
					await asyncio.wait_for(
						collect_query(), timeout=timeout_seconds,
					)
				except asyncio.TimeoutError:
					events.append(Terminal(reason="phase0_timeout"))
		except Exception as error:
			events.append(Terminal(reason=f"phase0_error:{type(error).__name__}"))
		finally:
			leases.release("phase0")
			_restore_tools(previous_tools)

		terminal = next(
			(event for event in reversed(events) if isinstance(event, Terminal)),
			None,
		)
		response = "\n\n".join(
			str(event.content)
			for event in events
			if getattr(event, "type", None) == "assistant"
			and getattr(event, "content", None)
		)
		checks = [
			_check_passes(check, root, response) for check in task.acceptance
		]
		if terminal is None or terminal.reason != "completed":
			checks = [False] * len(checks)
		elapsed = time.perf_counter() - started
		actual_model = ",".join(sorted(models)) or model
		artifact_workspace.mkdir(parents=True, exist_ok=False)
		shutil.copytree(root, artifact_workspace / "workspace")
		(artifact_workspace / "response.txt").write_text(response, encoding="utf-8")
		(artifact_workspace / "trial.json").write_text(json.dumps({
			"task_id": task.task_id,
			"shape": task.shape,
			"baseline": baseline,
			"model_id": actual_model,
			"checks_passed": all(checks),
			"quality_score": sum(checks) / len(checks),
			"tokens": tokens,
			"api_calls": provider_attempts or invocations,
			"task_invocations": task_invocations,
			"child_api_calls": child_api_calls,
			"child_provider_attempts": child_provider_attempts,
			"child_model_ids": sorted(child_models),
			"task_worker_count": task_worker_count,
			"task_worker_successes": task_worker_successes,
			"task_worker_failures": task_worker_failures,
			"duration_seconds": elapsed,
		}, indent=2) + "\n", encoding="utf-8")
	return BaselineResult(
		task_id=task.task_id,
		shape=task.shape,
		baseline=baseline,
		task_eligible_for_wide=task.task_eligible_for_wide,
		model_id=actual_model,
		checks_passed=all(checks),
		quality_score=sum(checks) / len(checks),
		tokens=tokens,
		api_calls=provider_attempts or invocations,
		task_invocations=task_invocations,
		child_api_calls=child_api_calls,
		child_provider_attempts=child_provider_attempts,
		child_model_ids=sorted(child_models),
		task_worker_count=task_worker_count,
		task_worker_successes=task_worker_successes,
		task_worker_failures=task_worker_failures,
		duration_seconds=elapsed,
		cost_usd=None,
		human_reviewed=False,
		artifact_dir=str(artifact_workspace),
		response_path=str(artifact_workspace / "response.txt"),
	)


async def run_dataset(
	dataset: EvaluationDataset,
	dataset_path: str | os.PathLike[str],
	*,
	t2_model: str,
	t3_model: str,
	artifacts_dir: str | os.PathLike[str] | None = None,
	timeout_seconds: int = 900,
) -> list[BaselineResult]:
	"""Run T2, T3, and current Task-DAG trials serially for each case."""
	if not t2_model.strip() or not t3_model.strip():
		raise ValueError("Both baseline model IDs must be provided")
	source = Path(dataset_path).expanduser().resolve()
	artifact_root = Path(
		artifacts_dir or source.parent / "phase0-artifacts",
	).expanduser().resolve()
	run_root = artifact_root / f"run-{uuid.uuid4().hex}"
	run_root.mkdir(parents=True, exist_ok=False)

	results: list[BaselineResult] = []
	for index, task in enumerate(dataset.tasks, start=1):
		task_key = hashlib.sha256(task.task_id.encode("utf-8")).hexdigest()[:10]
		task_root = run_root / f"task-{index:02d}-{task_key}"
		for baseline, model in (
			("single_t2", t2_model),
			("single_t3", t3_model),
			("pyramid", t3_model),
		):
			results.append(await _run_trial(
				task,
				source,
				baseline,
				model,
				artifact_workspace=task_root / baseline,
				timeout_seconds=timeout_seconds,
			))
	return results


def build_review_template(
	results: list[BaselineResult],
	task_ids: set[str],
) -> dict[str, object]:
	selected = {result.task_id for result in results if result.task_id in task_ids}
	if len(selected) < 3:
		raise ValueError("Human review must cover at least three distinct tasks")
	by_task: dict[str, set[str]] = {}
	reviews = []
	for result in results:
		if result.task_id not in selected:
			continue
		by_task.setdefault(result.task_id, set()).add(result.baseline)
		reviews.append({
			"task_id": result.task_id,
			"baseline": result.baseline,
			"model_id": result.model_id,
			"response_path": result.response_path,
			"artifact_dir": result.artifact_dir,
			"human_quality_score": None,
			"human_review_notes": None,
		})
	if any(set(baselines) != {"single_t2", "single_t3", "pyramid"} for baselines in by_task.values()):
		raise ValueError("Each reviewed task must include all three configurations")
	return {"schema_version": "1.0", "reviews": reviews}


def finalize_report(
	report_path: str | os.PathLike[str],
	reviews_path: str | os.PathLike[str],
	output_path: str | os.PathLike[str] | None = None,
) -> Path:
	source = Path(report_path).expanduser().resolve()
	results_payload = json.loads(source.read_text(encoding="utf-8"))
	results = [
		BaselineResult.model_validate(result)
		for result in results_payload.get("results", [])
	]
	review_document = HumanReviewDocument.model_validate_json(
		Path(reviews_path).expanduser().read_text(encoding="utf-8")
	)
	by_key = {(result.task_id, result.baseline): result for result in results}
	seen: set[tuple[str, str]] = set()
	covered_tasks: set[str] = set()
	for review in review_document.reviews:
		key = (review.task_id, review.baseline)
		if key in seen:
			raise ValueError(f"Duplicate human review: {review.task_id}/{review.baseline}")
		seen.add(key)
		result = by_key.get(key)
		if result is None:
			raise ValueError(f"Review has no matching result: {review.task_id}/{review.baseline}")
		result.human_reviewed = True
		result.human_quality_score = review.human_quality_score
		result.human_review_notes = review.human_review_notes
		covered_tasks.add(review.task_id)
	if len(covered_tasks) < 3:
		raise ValueError("Human review must cover at least three distinct tasks")
	for task_id in covered_tasks:
		if not {"single_t2", "single_t3", "pyramid"}.issubset(
			baseline for reviewed_task, baseline in seen if reviewed_task == task_id
		):
			raise ValueError(f"Task {task_id} must be reviewed in all three configurations")

	report = Phase0Report(schema_version="1.0", results=results)
	destination = Path(output_path).expanduser().resolve() if output_path else source.with_name(
		source.stem + "-reviewed" + source.suffix,
	)
	destination.parent.mkdir(parents=True, exist_ok=True)
	destination.write_text(json.dumps({
		"schema_version": report.schema_version,
		"results": [result.model_dump() for result in report.results],
	}, indent=2, allow_nan=False) + "\n", encoding="utf-8")
	return destination


def main(argv: list[str] | None = None) -> int:
	parser = argparse.ArgumentParser(description=__doc__)
	commands = parser.add_subparsers(dest="command", required=True)
	run_parser = commands.add_parser("run", help="Run benchmark trials")
	run_parser.add_argument("--dataset", required=True, help="Phase 0 JSON dataset")
	run_parser.add_argument("--report", required=True, help="Write the result JSON here")
	run_parser.add_argument("--t2-model", required=True, help="Model ID for the single-T2 baseline")
	run_parser.add_argument("--t3-model", required=True, help="Model ID for the single-T3 and Pyramid parent")
	run_parser.add_argument("--artifacts-dir", help="Retain case workspaces under this directory")
	run_parser.add_argument(
		"--review-task", action="append", default=[],
		help="Task ID to include in the review sheet; defaults to the first three",
	)
	run_parser.add_argument("--timeout-seconds", type=int, default=900)
	review_parser = commands.add_parser("finalize", help="Apply completed human scores and validate the report")
	review_parser.add_argument("--report", required=True, help="Initial runner report JSON")
	review_parser.add_argument("--reviews", required=True, help="Completed review JSON")
	review_parser.add_argument("--output", help="Final report path; defaults to <report>-reviewed.json")
	args = parser.parse_args(argv)
	try:
		if args.command == "finalize":
			destination = finalize_report(args.report, args.reviews, args.output)
			print(f"Validated reviewed Phase 0 report: {destination}")
			return 0
		if args.timeout_seconds < 1:
			parser.error("--timeout-seconds must be positive")
		dataset_path = Path(args.dataset).expanduser().resolve()
		dataset = EvaluationDataset.model_validate_json(
			dataset_path.read_text(encoding="utf-8")
		)
		reviewed_task_ids = set(args.review_task) or {
			task.task_id for task in dataset.tasks[:3]
		}
		if len(reviewed_task_ids) < 3:
			raise ValueError("Select at least three tasks for human review")
		unknown_reviews = reviewed_task_ids - {task.task_id for task in dataset.tasks}
		if unknown_reviews:
			raise ValueError(f"Unknown review tasks: {', '.join(sorted(unknown_reviews))}")
		results = asyncio.run(run_dataset(
			dataset,
			dataset_path,
			t2_model=args.t2_model,
			t3_model=args.t3_model,
			artifacts_dir=args.artifacts_dir,
			timeout_seconds=args.timeout_seconds,
		))
		report_path = Path(args.report).expanduser().resolve()
		report_path.parent.mkdir(parents=True, exist_ok=True)
		report_path.write_text(json.dumps({
			"schema_version": "1.0",
			"results": [result.model_dump() for result in results],
		}, indent=2, allow_nan=False) + "\n", encoding="utf-8")
		review_path = report_path.with_name(report_path.stem + ".review.json")
		review_path.write_text(json.dumps(
			build_review_template(results, reviewed_task_ids),
			indent=2,
		) + "\n", encoding="utf-8")
	except (OSError, ValueError) as error:
		parser.error(str(error))
	print(f"Wrote {len(results)} trial results to {report_path}")
	print(f"Retained trial workspaces and review sheet: {review_path}")
	print("Costs remain unknown. Fill in the review scores, then run the finalize command.")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())