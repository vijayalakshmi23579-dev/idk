"""Offline tests for the Phase 0 terminal review portal."""
import asyncio
import json
import tempfile
from pathlib import Path

from textual.widgets import Input, Static

from phase0_review import Phase0ReviewApp, load_review_data


async def main() -> None:
	with tempfile.TemporaryDirectory() as temp_dir:
		root = Path(temp_dir)
		dataset_path = root / "dataset.json"
		report_path = root / "report.json"
		reviews_path = root / "report.review.json"
		shapes = [
			"breadth_research", "multi_file_code",
			"data_pipeline", "sequential_reasoning",
		]
		tasks = [{
			"task_id": f"case-{index}",
			"shape": shapes[index % len(shapes)],
			"prompt": f"Prompt {index}",
			"acceptance": [{"kind": "assert", "spec": "{\"target\":\"response\",\"contains\":\"ok\"}"}],
		} for index in range(8)]
		dataset_path.write_text(json.dumps({
			"schema_version": "1.0", "tasks": tasks,
		}), encoding="utf-8")
		results = []
		reviews = []
		for task in tasks:
			for baseline in ("single_t2", "single_t3", "pyramid"):
				case_dir = root / "artifacts" / task["task_id"] / baseline
				workspace = case_dir / "workspace"
				workspace.mkdir(parents=True)
				response = case_dir / "response.txt"
				response.write_text("ok", encoding="utf-8")
				results.append({
					"task_id": task["task_id"],
					"shape": task["shape"],
					"baseline": baseline,
					"model_id": "test/model",
					"checks_passed": True,
					"quality_score": 1.0,
					"tokens": 10,
					"api_calls": 1,
					"duration_seconds": 1.0,
					"cost_usd": None,
					"human_reviewed": False,
					"artifact_dir": str(case_dir),
					"response_path": str(response),
				})
				if index := int(task["task_id"].split("-")[1]) < 3:
					reviews.append({
						"task_id": task["task_id"],
						"baseline": baseline,
						"model_id": "test/model",
						"response_path": str(response),
						"artifact_dir": str(case_dir),
						"human_quality_score": None,
						"human_review_notes": None,
					})
		report_path.write_text(json.dumps({
			"schema_version": "1.0", "results": results,
		}), encoding="utf-8")
		reviews_path.write_text(json.dumps({
			"schema_version": "1.0", "reviews": reviews,
		}), encoding="utf-8")
		document, rows = load_review_data(dataset_path, report_path, reviews_path)
		assert len(rows) == 9
		app = Phase0ReviewApp(reviews_path, document, rows)
		async with app.run_test() as pilot:
			assert "case-0" in str(app.query_one("#title", Static).render())
			assert app.query_one("#detail").size.width >= 70
			assert app.query_one("#content-scroll").size.width >= 60
			assert "ok" in str(app.query_one("#content", Static).render())
			app._show_view("prompt")
			assert "Prompt 0" in str(app.query_one("#content", Static).render())
			app.query_one("#score", Input).value = "0.75"
			app.query_one("#notes", Input).value = "Correct and complete"
			await pilot.press("ctrl+s")
			saved = json.loads(reviews_path.read_text(encoding="utf-8"))
			assert saved["reviews"][0]["human_quality_score"] == 0.75
			assert saved["reviews"][0]["human_review_notes"] == "Correct and complete"
			app.action_next()
			assert "single_t3" in str(app.query_one("#title", Static).render())
	print("Phase 0 terminal review portal tests passed")


if __name__ == "__main__":
	asyncio.run(main())