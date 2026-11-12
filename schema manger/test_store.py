"""test_store.py — no network. Verifies append-only log + DAG + plan."""
import os
import stat
import tempfile
from pathlib import Path

from store import Store


def main():
    tmp = tempfile.mkdtemp()
    db = os.path.join(tmp, "test.db")
    store = Store(db)
    sid = "test-session"
    assert stat.S_IMODE(os.stat(db).st_mode) & 0o077 == 0
    print("✅ Store database is owner-only OK")

    # ── 1. Append messages ──
    m1 = store.append_message(sid, "user", "hello world")
    m2 = store.append_message(sid, "assistant", "hi there")
    m3 = store.append_message(sid, "user", "how are you?")
    assert store.count_messages(sid) == 3
    print("✅ Append messages OK")

    # ── 2. get_message ──
    row = store.get_message(m2)
    assert row["role"] == "assistant"
    assert row["content"] == "hi there"
    print("✅ get_message OK")

    # ── 3. Summary with source pointers ──
    s1 = store.append_summary(
        sid, 0, "User greeted and asked about assistant",
        source_msg_ids=[m1, m2, m3],
    )
    root = store.get_root_summary(sid)
    assert root["id"] == s1
    assert root["depth"] == 0
    print("✅ Append summary OK")

    # ── 4. Expand recovers originals byte-for-byte ──
    originals = store.expand(s1)
    assert len(originals) == 3
    assert originals[0]["content"] == "hello world"
    assert originals[1]["content"] == "hi there"
    assert originals[2]["content"] == "how are you?"
    print("✅ Expand summary → originals OK")

    # ── 5. Two-level DAG (summary of summaries) ──
    s2 = store.append_summary(
        sid, 1, "Meta-summary covering the greeting exchange",
        source_summary_ids=[s1],
    )
    root2 = store.get_root_summary(sid)
    assert root2["id"] == s2
    assert root2["depth"] == 1
    # Expanding the root reaches all original messages
    deep = store.expand(s2)
    assert len(deep) == 3
    assert deep[0]["content"] == "hello world"
    print("✅ Two-level DAG expand OK")

    # ── 6. Plan save + load ──
    store.save_plan(
        sid,
        objective="Test the store",
        items=[
            {"content": "read code", "status": "completed"},
            {"content": "write test", "status": "in_progress"},
            {"content": "run test", "status": "pending"},
        ],
        focus="write test",
    )
    plan = store.get_plan(sid)
    assert plan["objective"] == "Test the store"
    assert len(plan["items"]) == 3
    assert plan["items"][1]["status"] == "in_progress"
    assert plan["focus"] == "write test"
    print("✅ Plan save/get OK")

    # ── 7. Plan persists across reopen ──
    store.close()
    store = Store(db)
    plan = store.get_plan(sid)
    assert plan is not None
    assert plan["objective"] == "Test the store"
    print("✅ Plan persists across reopen OK")

    # ── 8. Messages also persist ──
    assert store.count_messages(sid) == 3
    print("✅ Messages persist across reopen OK")

    # ── 8a. Task lifecycle events persist separately from conversation text ──
    task_event = {
        "schema_version": "1.0",
        "type": "task_lifecycle",
        "event_id": "event-1",
        "run_id": "run-1",
        "attempt_id": "attempt-1",
        "task_id": "worker-a",
        "status": "succeeded",
        "needs": [],
        "occurred_at": 123.0,
        "model_id": None,
        "duration_ms": 12,
    }
    store.append_task_event(sid, "run-1", task_event)
    store.save_task_run(sid, "run-1", tmp, {
        "schema_version": "1.0",
        "run_id": "run-1",
        "status": "completed",
    })
    store.close()
    store = Store(db)
    assert store.get_task_events(sid, "run-1") == [task_event]
    assert store.get_task_run("run-1", tmp) == {
        "schema_version": "1.0",
        "run_id": "run-1",
        "status": "completed",
    }
    assert store.get_task_run("run-1", os.path.join(tmp, "other-project")) is None
    assert store.get_task_run("run-1", tmp, "other-session") is None
    assert store.get_task_run("unknown-run", tmp) is None
    store.save_task_run(sid, "run-progress", tmp, {
        "schema_version": "1.0",
        "run_id": "run-progress",
        "run_status": "running",
        "task_specs": {"worker-b": {"schema_version": "1.0"}},
        "progress": {},
    })
    for event_id, status in (("event-start", "started"), ("event-finish", "succeeded")):
        store.append_task_event(sid, "run-progress", {
            "schema_version": "1.0",
            "type": "task_lifecycle",
            "event_id": event_id,
            "run_id": "run-progress",
            "attempt_id": "attempt-b",
            "task_id": "worker-b",
            "status": status,
            "needs": [],
            "occurred_at": 124.0,
        })
    progress_run = store.get_task_run("run-progress", tmp, sid)
    assert progress_run["run_status"] == "finalizing"
    assert progress_run["progress"]["worker-b"]["status"] == "succeeded"
    store.save_task_run(sid, "run-interrupted", tmp, {
        "schema_version": "1.0",
        "run_id": "run-interrupted",
        "run_status": "running",
        "task_specs": {"worker-c": {"schema_version": "1.0"}},
        "progress": {},
    })
    store.append_task_event(sid, "run-interrupted", {
        "schema_version": "1.0",
        "type": "task_lifecycle",
        "event_id": "event-interrupted-start",
        "run_id": "run-interrupted",
        "attempt_id": "attempt-c",
        "task_id": "worker-c",
        "status": "started",
        "needs": [],
        "occurred_at": 125.0,
    })
    store.close()
    store = Store(db)
    interrupted_run = store.get_task_run("run-interrupted", tmp, sid)
    assert interrupted_run["run_status"] == "running"
    assert interrupted_run["progress"]["worker-c"]["status"] == "started"
    assert store.get_task_events(sid, "unknown-run") == []
    try:
        store.append_task_event(sid, "run-1", {
            "type": "task_lifecycle", "status": "started",
        })
        raise AssertionError("incomplete Task events should be rejected")
    except ValueError:
        pass
    print("✅ Task lifecycle events persist by run without transcript content")

    # ── 8b. Content-addressed artifacts persist and deduplicate ──
    source_path = os.path.join(tmp, "artifact.bin")
    Path(source_path).write_bytes(b"artifact payload")
    artifact_one = store.put_artifact(source_path)
    artifact_two = store.put_artifact(source_path)
    assert artifact_one == artifact_two
    stored_path = store.get_artifact_path(artifact_one["storage_key"])
    assert stored_path is not None
    assert Path(stored_path).read_bytes() == b"artifact payload"
    store.close()
    store = Store(db)
    assert store.get_artifact_path(artifact_one["storage_key"]) == stored_path
    assert store.get_artifact_path("../outside") is None
    print("✅ Content-addressed artifacts persist and deduplicate")

    # ── 9. get_view assembles plan + summary + tail ──
    view = store.get_view(sid, fresh_tail=10)
    assert view[0]["type"] == "system_plan"
    assert view[1]["type"] == "system_summary"
    assert view[1]["summary_id"] == s2  # highest-depth root
    assert view[2]["type"] == "message"
    assert len(view) == 2 + 3  # plan + summary + 3 messages
    assert "Current Plan" in view[0]["content"]
    assert "Test the store" in view[0]["content"]
    print("✅ get_view assembly OK")

    # ── 10. Fresh tail cap ──
    view_small = store.get_view(sid, fresh_tail=2)
    msgs = [v for v in view_small if v["type"] == "message"]
    assert len(msgs) == 2
    print("✅ Fresh tail cap OK")

    # ── 11. FTS5 search ──
    hits = store.search("hello", sid)
    assert len(hits) == 1
    assert hits[0]["content"] == "hello world"
    # multi-term
    hits2 = store.search("how", sid)
    assert len(hits2) == 1
    print("✅ FTS5 search OK")

    # ── 12. Active view keeps complete recent rounds within budget ──
    store.append_message(sid, "user", "first followup")
    store.append_message(
        sid, "assistant", "", tool_calls=[
            {"id": "read-1", "name": "Read", "input": {"file_path": "a.py"}},
        ],
    )
    store.append_message(sid, "tool", "older tool output " * 100, tool_use_id="read-1")
    store.append_message(sid, "user", "latest question")
    store.append_message(sid, "assistant", "latest answer")

    active = store.get_active_view_messages(sid, budget_tokens=1_000)
    assert active[0].subtype == "compact_boundary"
    assert active[1].is_compact_summary is True
    assert any(
        getattr(message, "tool_use_id", None) == "read-1" for message in active
    )
    assert any(
        getattr(message, "id", None) == "read-1"
        for item in active if getattr(item, "type", None) == "assistant"
        for message in item.tool_calls
    )

    bounded = store.get_active_view_messages(sid, budget_tokens=60)
    assert any(getattr(message, "content", None) == "latest question" for message in bounded)
    assert not any(
        "older tool output" in getattr(message, "content", "")
        for message in bounded
    )
    print("✅ Budgeted active view preserves recent tool rounds OK")

    # ── 13. Append-only invariant: no destructive methods exist ──
    assert not hasattr(store, "update_message")
    assert not hasattr(store, "delete_message")
    assert not hasattr(store, "clear_messages")
    assert not hasattr(store, "truncate")
    print("✅ Append-only invariant OK")

    # ── 14. Summary must have at least one source ──
    try:
        store.append_summary(sid, 0, "orphan", source_msg_ids=None,
                              source_summary_ids=None)
        raise AssertionError("should have raised")
    except ValueError:
        pass
    print("✅ Orphan summary rejected OK")

    store.close()
    print("\n🎉 All store.py tests passed")


main()