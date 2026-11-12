"""Browse retained Phase 0 trials and enter human review scores in a TUI."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from textual import on
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.widgets import Button, Footer, Header, Input, Static

from phase0_eval import EvaluationDataset


BASELINE_ORDER = {"single_t2": 0, "single_t3": 1, "pyramid": 2}


def load_review_data(
	dataset_path: str | Path,
	report_path: str | Path,
	reviews_path: str | Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
	dataset = EvaluationDataset.model_validate_json(
		Path(dataset_path).expanduser().read_text(encoding="utf-8")
	)
	report = json.loads(Path(report_path).expanduser().read_text(encoding="utf-8"))
	reviews_doc = json.loads(Path(reviews_path).expanduser().read_text(encoding="utf-8"))
	tasks = {task.task_id: task for task in dataset.tasks}
	results = {
		(result["task_id"], result["baseline"]): result
		for result in report.get("results", [])
	}
	rows = []
	for review in reviews_doc.get("reviews", []):
		key = (review.get("task_id"), review.get("baseline"))
		if key[0] not in tasks:
			raise ValueError(f"Review references unknown task: {key[0]}")
		if key not in results:
			raise ValueError(f"Review has no matching result: {key[0]}/{key[1]}")
		rows.append({
			**review,
			"prompt": tasks[key[0]].prompt,
			"acceptance": [check.model_dump() for check in tasks[key[0]].acceptance],
			"result": results[key],
		})
	rows.sort(key=lambda row: (
		next(i for i, task in enumerate(dataset.tasks) if task.task_id == row["task_id"]),
		BASELINE_ORDER.get(row["baseline"], 99),
	))
	if not rows:
		raise ValueError("Review sheet contains no trial entries")
	return reviews_doc, rows


def write_review_data(path: str | Path, document: dict[str, Any]) -> None:
	destination = Path(path).expanduser().resolve()
	destination.parent.mkdir(parents=True, exist_ok=True)
	temporary = destination.with_suffix(destination.suffix + ".tmp")
	temporary.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
	temporary.replace(destination)


class Phase0ReviewApp(App):
	TITLE = "PHASE 0 REVIEW"
	CSS = """
	Screen { layout: vertical; }
	#detail { height: 1fr; padding: 0 1; }
	#title { height: 2; text-style: bold; color: $accent; }
	#views { height: 3; }
	#views Button { margin-right: 1; }
	#content-scroll { height: 1fr; border: round $primary; padding: 0 1; }
	#content { width: 1fr; }
	#rubric { height: 1; color: $text-muted; }
	#score-row { height: 3; }
	#score { width: 16; }
	#notes { width: 1fr; }
	#actions { height: 3; align-horizontal: right; }
	#actions Button { margin-left: 1; }
	#status { height: 1; padding: 0 1; background: $panel; }
	"""
	BINDINGS = [
		Binding("ctrl+s", "save", "Save score"),
		Binding("q", "quit", "Quit"),
	]

	def __init__(self, review_path: str | Path, document: dict[str, Any], rows: list[dict[str, Any]]):
		super().__init__()
		self.review_path = Path(review_path).expanduser().resolve()
		self.document = document
		self.rows = rows
		self.index = 0

	def compose(self) -> ComposeResult:
		yield Header()
		with Vertical(id="detail"):
			yield Static(id="title")
			with Horizontal(id="views"):
				yield Button("Response", id="response", variant="primary")
				yield Button("Prompt", id="prompt")
				yield Button("Checks", id="checks")
				yield Button("Files", id="files")
			with VerticalScroll(id="content-scroll"):
				yield Static(id="content", markup=False)
			yield Static(
				"Score: 1 correct | .75 minor issue | .5 partial | .25 mostly wrong | 0 unusable",
				id="rubric",
			)
			with Horizontal(id="score-row"):
				yield Input(placeholder="Score 0.00-1.00", id="score")
				yield Input(placeholder="Review notes (optional)", id="notes")
			with Horizontal(id="actions"):
				yield Button("Previous", id="previous")
				yield Button("Save score", id="save", variant="primary")
				yield Button("Next", id="next")
			yield Static(id="status")
		yield Footer()

	def on_mount(self) -> None:
		for row in self.rows:
			result = row["result"]
			row["automatic_status"] = "PASS" if result["checks_passed"] else "FAIL"
		self._show_row(0)

	@staticmethod
	def _score_label(value: Any) -> str:
		return "-" if value is None else f"{float(value):.2f}"

	def _show_row(self, index: int) -> None:
		self.index = max(0, min(index, len(self.rows) - 1))
		row = self.rows[self.index]
		self.query_one("#title", Static).update(
			f"{self.index + 1}/{len(self.rows)}  {row['task_id']}  |  "
			f"{row['baseline']}  |  {row.get('model_id') or 'model unknown'}"
		)
		self.view = "response"
		self._show_view("response")
		self.query_one("#score", Input).value = (
			"" if row.get("human_quality_score") is None
			else f"{row['human_quality_score']:.2f}"
		)
		self.query_one("#notes", Input).value = row.get("human_review_notes") or ""
		self._update_status()

	def _show_view(self, view: str) -> None:
		row = self.rows[self.index]
		if view == "prompt":
			content = "TASK PROMPT\n\n" + row["prompt"]
		elif view == "checks":
			result = row["result"]
			checks = "\n".join(
				f"{check['kind']}: {check['spec']}"
				for check in row["acceptance"]
			)
			content = (
				f"AUTOMATED: {row['automatic_status']}  "
				f"score={result['quality_score']:.2f}  "
				f"time={result['duration_seconds']:.1f}s  "
				f"tokens={result['tokens']}  calls={result['api_calls']}\n\n"
				f"ACCEPTANCE CHECKS\n{checks}"
			)
		elif view == "files":
			workspace = Path(row.get("artifact_dir") or "") / "workspace"
			try:
				files = sorted(
					str(path.relative_to(workspace))
					for path in workspace.rglob("*") if path.is_file()
				)
			except OSError:
				files = []
			content = f"WORKSPACE\n{workspace}\n\nFILES\n" + ("\n".join(files) or "(none)")
		else:
			response_path = Path(row.get("response_path") or "")
			try:
				content = response_path.read_text(encoding="utf-8")
			except OSError:
				content = "[Response file is missing or unreadable.]"
			if not content:
				content = "[Empty response]"
		self.view = view
		self.query_one("#content", Static).update(content)

	def _update_status(self, message: str | None = None) -> None:
		completed = sum(row.get("human_quality_score") is not None for row in self.rows)
		text = f"Reviewed {completed}/{len(self.rows)} | Saved to {self.review_path}"
		if message:
			text += f" | {message}"
		self.query_one("#status", Static).update(text)

	def _save_current(self) -> bool:
		row = self.rows[self.index]
		score_text = self.query_one("#score", Input).value.strip()
		notes = self.query_one("#notes", Input).value.strip() or None
		if not score_text:
			return True
		try:
			score = float(score_text)
		except ValueError:
			self._update_status("Score must be a number from 0 to 1")
			return False
		if not 0 <= score <= 1:
			self._update_status("Score must be between 0 and 1")
			return False
		row["human_quality_score"] = score
		row["human_review_notes"] = notes
		for saved in self.document["reviews"]:
			if saved["task_id"] == row["task_id"] and saved["baseline"] == row["baseline"]:
				saved["human_quality_score"] = score
				saved["human_review_notes"] = notes
				break
		write_review_data(self.review_path, self.document)
		self._update_status("Saved")
		return True

	def action_save(self) -> None:
		self._save_current()

	def action_previous(self) -> None:
		if self._save_current():
			self._show_row((self.index - 1) % len(self.rows))

	def action_next(self) -> None:
		if self._save_current():
			self._show_row((self.index + 1) % len(self.rows))

	@on(Button.Pressed)
	def on_button_pressed(self, event: Button.Pressed) -> None:
		if event.button.id in {"response", "prompt", "checks", "files"}:
			self._show_view(event.button.id)
		elif event.button.id == "save":
			self.action_save()
		elif event.button.id == "previous":
			self.action_previous()
		elif event.button.id == "next":
			self.action_next()


def main(argv: list[str] | None = None) -> int:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--dataset", required=True)
	parser.add_argument("--report", required=True)
	parser.add_argument("--reviews", required=True)
	args = parser.parse_args(argv)
	try:
		document, rows = load_review_data(args.dataset, args.report, args.reviews)
	except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
		parser.error(str(error))
	Phase0ReviewApp(args.reviews, document, rows).run()
	return 0


if __name__ == "__main__":
	raise SystemExit(main())