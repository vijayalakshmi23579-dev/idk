"""test_compact.py — contract-based tests. No network."""
import asyncio
import os

import compact
from compact import (
    MAX_OUTPUT_TOKENS_FOR_SUMMARY, AUTOCOMPACT_BUFFER_TOKENS,
    WARNING_THRESHOLD_BUFFER_TOKENS, ERROR_THRESHOLD_BUFFER_TOKENS,
    MANUAL_COMPACT_BUFFER_TOKENS, MAX_CONSECUTIVE_AUTOCOMPACT_FAILURES,
    POST_COMPACT_MAX_FILES_TO_RESTORE, POST_COMPACT_TOKEN_BUDGET,
    POST_COMPACT_MAX_TOKENS_PER_FILE, TIME_BASED_MC_CLEARED_MESSAGE,
    IMAGE_MAX_TOKEN_SIZE, COMPACTABLE_TOOLS, PTL_RETRY_MARKER,
    rough_token_count, calculate_tool_result_tokens, estimate_message_tokens,
    token_count_with_estimation, group_messages_by_api_round,
    microcompact, collect_compactable_tool_ids,
    get_effective_context_window_size, get_auto_compact_threshold,
    calculate_token_warning_state, is_auto_compact_enabled, should_auto_compact,
    get_compact_prompt, format_compact_summary, get_compact_user_summary_message,
    CompactionResult, build_post_compact_messages,
    strip_images_from_messages, truncate_head_for_ptl_retry,
    compact_conversation, autocompact,
)
from messages import (
    ToolUseBlock,
    create_assistant_message, create_tool_result, create_user_message,
)
from tool import ToolUseContext


def make_ctx(read_files=None):
    ctx = ToolUseContext()
    from state import FileStateCache
    cache = FileStateCache()
    for path, (content, mtime) in (read_files or {}).items():
        cache.set(path, content, mtime)
    ctx.read_file_state = cache
    return ctx


def _build_pairs(n):
    """Realistic conversation: n [user, assistant] pairs of ~2000 chars each."""
    out = []
    for i in range(n):
        out.append(create_user_message(f"u{i} " + "x" * 2000))
        out.append(create_assistant_message(content=f"a{i} " + "y" * 2000))
    return out


async def main():
    # ── 1. Constants exact ──
    assert MAX_OUTPUT_TOKENS_FOR_SUMMARY == 20_000
    assert AUTOCOMPACT_BUFFER_TOKENS == 13_000
    assert WARNING_THRESHOLD_BUFFER_TOKENS == 20_000
    assert ERROR_THRESHOLD_BUFFER_TOKENS == 20_000
    assert MANUAL_COMPACT_BUFFER_TOKENS == 3_000
    assert MAX_CONSECUTIVE_AUTOCOMPACT_FAILURES == 3
    assert POST_COMPACT_MAX_FILES_TO_RESTORE == 5
    assert POST_COMPACT_TOKEN_BUDGET == 50_000
    assert POST_COMPACT_MAX_TOKENS_PER_FILE == 5_000
    assert TIME_BASED_MC_CLEARED_MESSAGE == "[Old tool result content cleared]"
    assert IMAGE_MAX_TOKEN_SIZE == 2000
    print("✅ Constants OK")

    # ── 2. COMPACTABLE_TOOLS whitelist exact ──
    expected = {"Read", "Bash", "PowerShell", "Grep", "Glob",
                "WebSearch", "WebFetch", "Edit", "Write"}
    assert set(COMPACTABLE_TOOLS) == expected
    assert "TodoWrite" not in COMPACTABLE_TOOLS
    assert "Agent" not in COMPACTABLE_TOOLS
    print("✅ COMPACTABLE_TOOLS whitelist OK")

    # ── 3. Token estimation ──
    assert rough_token_count("") == 0
    assert rough_token_count("abcd") == 1
    assert rough_token_count("a" * 40) == 10
    assert calculate_tool_result_tokens("a" * 40) == 10
    assert calculate_tool_result_tokens([{"type": "image", "data": "x" * 100}]) == 2000
    assert calculate_tool_result_tokens([
        {"type": "text", "text": "a" * 40},
        {"type": "image"},
    ]) == 10 + 2000
    assert estimate_message_tokens([create_user_message("a" * 120)]) == 40
    print("✅ Token estimation (4/3 pad) OK")

    # ── 4. token_count_with_estimation prefers usage ──
    am = create_assistant_message(content="hi")
    am.usage = {"input_tokens": 5000, "cache_read_input_tokens": 300,
                "cache_creation_input_tokens": 200}
    assert token_count_with_estimation([am]) == 5500
    assert token_count_with_estimation([create_user_message("a" * 400)]) == 134
    print("✅ token_count_with_estimation OK")

    # ── 5. Effective window + threshold + env overrides ──
    for var in ("CLAUDE_CODE_AUTO_COMPACT_WINDOW", "CLAUDE_AUTOCOMPACT_PCT_OVERRIDE",
                "CLAUDE_CODE_BLOCKING_LIMIT_OVERRIDE", "DISABLE_COMPACT",
                "DISABLE_AUTO_COMPACT"):
        os.environ.pop(var, None)
    assert get_effective_context_window_size("claude-sonnet-4-5") == 180_000
    assert get_auto_compact_threshold("claude-sonnet-4-5") == 167_000
    os.environ["CLAUDE_CODE_AUTO_COMPACT_WINDOW"] = "100000"
    assert get_effective_context_window_size("claude-sonnet-4-5") == 80_000
    assert get_auto_compact_threshold("claude-sonnet-4-5") == 67_000
    os.environ.pop("CLAUDE_CODE_AUTO_COMPACT_WINDOW")
    os.environ["CLAUDE_AUTOCOMPACT_PCT_OVERRIDE"] = "50"
    assert get_auto_compact_threshold("claude-sonnet-4-5") == 90_000
    os.environ.pop("CLAUDE_AUTOCOMPACT_PCT_OVERRIDE")
    print("✅ Thresholds + env overrides OK")

    # ── 6. Warning state + blocking limit ──
    s = calculate_token_warning_state(170_000, "claude-sonnet-4-5")
    assert s["is_above_auto_compact_threshold"] is True
    assert s["percent_left"] >= 0
    s = calculate_token_warning_state(50_000, "claude-sonnet-4-5")
    assert s["is_above_auto_compact_threshold"] is False
    assert s["is_at_blocking_limit"] is False
    assert calculate_token_warning_state(177_000, "claude-sonnet-4-5")["is_at_blocking_limit"] is True
    os.environ["CLAUDE_CODE_BLOCKING_LIMIT_OVERRIDE"] = "60000"
    assert calculate_token_warning_state(60_000, "claude-sonnet-4-5")["is_at_blocking_limit"] is True
    os.environ.pop("CLAUDE_CODE_BLOCKING_LIMIT_OVERRIDE")
    print("✅ Warning state + blocking limit OK")

    # ── 7. is_auto_compact_enabled env switches ──
    assert is_auto_compact_enabled() is True
    os.environ["DISABLE_AUTO_COMPACT"] = "1"
    assert is_auto_compact_enabled() is False
    os.environ.pop("DISABLE_AUTO_COMPACT")
    os.environ["DISABLE_COMPACT"] = "1"
    assert is_auto_compact_enabled() is False
    os.environ.pop("DISABLE_COMPACT")
    print("✅ is_auto_compact_enabled OK")

    # ── 8. should_auto_compact recursion guards ──
    big = [create_user_message("a" * 700_000)]
    assert should_auto_compact(big, "claude-sonnet-4-5") is True
    assert should_auto_compact(big, "claude-sonnet-4-5", "compact") is False
    assert should_auto_compact(big, "claude-sonnet-4-5", "session_memory") is False
    assert should_auto_compact([create_user_message("hi")], "claude-sonnet-4-5") is False
    print("✅ should_auto_compact + recursion guards OK")

    # ── 9. Prompts ──
    p = get_compact_prompt()
    assert p.startswith("CRITICAL: Respond with TEXT ONLY")
    assert p.endswith("you will fail the task.")
    for header in ("1. Primary Request and Intent", "2. Key Technical Concepts",
                   "3. Files and Code Sections", "4. Errors and fixes",
                   "5. Problem Solving", "6. All user messages",
                   "7. Pending Tasks", "8. Current Work", "9. Optional Next Step"):
        assert header in p, f"missing section {header}"
    assert "<analysis>" in p and "<summary>" in p
    assert "Additional Instructions:\nfocus on tests" in get_compact_prompt("focus on tests")
    print("✅ Compact prompt OK")

    # ── 10. format_compact_summary ──
    raw = "<analysis>scratchpad\nmore</analysis>\n<summary>1. A\n2. B</summary>\n\n\nend"
    f = format_compact_summary(raw)
    assert "scratchpad" not in f
    assert f.startswith("Summary:")
    assert "1. A" in f and "\n\n\n" not in f
    assert format_compact_summary("plain text").strip() == "plain text"
    print("✅ format_compact_summary OK")

    # ── 11. Summary user message ──
    m = get_compact_user_summary_message("x", True, "/tmp/t.jsonl", True)
    assert "being continued from a previous conversation" in m
    assert "/tmp/t.jsonl" in m
    assert "Recent messages are preserved verbatim." in m
    assert "do not acknowledge the summary" in m
    print("✅ get_compact_user_summary_message OK")

    # ── 12. build_post_compact_messages order ──
    b = create_user_message("boundary")
    s1, s2 = create_user_message("s1"), create_user_message("s2")
    k1 = create_user_message("k1")
    a1 = create_user_message("a1")
    h1 = create_user_message("h1")
    r = CompactionResult(boundary_marker=b, summary_messages=[s1, s2],
                         attachments=[a1], hook_results=[h1],
                         messages_to_keep=[k1])
    assert build_post_compact_messages(r) == [b, s1, s2, k1, a1, h1]
    assert build_post_compact_messages(
        CompactionResult(boundary_marker=b, summary_messages=[s1],
                         attachments=[], hook_results=[])
    ) == [b, s1]
    print("✅ build_post_compact_messages order OK")

    # ── 13. strip_images_from_messages ──
    um = create_user_message([
        {"type": "text", "text": "look"},
        {"type": "image", "source": {"data": "xx"}},
        {"type": "document", "source": {"data": "yy"}},
        {"type": "tool_result", "tool_use_id": "t1", "content": [
            {"type": "text", "text": "ok"},
            {"type": "image"},
        ]},
    ])
    out = strip_images_from_messages([um])
    blocks = out[0].content
    assert blocks[1] == {"type": "text", "text": "[image]"}
    assert blocks[2] == {"type": "text", "text": "[document]"}
    assert blocks[3]["content"][1] == {"type": "text", "text": "[image]"}
    plain = create_user_message("hello")
    assert strip_images_from_messages([plain])[0] is plain
    print("✅ strip_images_from_messages OK")

    # ── 14. group_messages_by_api_round ──
    u = create_user_message("u")
    a1 = create_assistant_message(content="a1")
    a2 = create_assistant_message(content="a2")
    tr = create_tool_result("t1", "res")
    a3 = create_assistant_message(content="a3")
    assert [len(g) for g in group_messages_by_api_round([u, a1, a2, tr, a3])] == [1, 3, 1]
    # round_id path
    u2 = create_user_message("u")
    a1b = create_assistant_message(content="a1")
    a2b = create_assistant_message(content="a2")
    a3b = create_assistant_message(content="a3")
    for msg, rid in ((u2, "r1"), (a1b, "r1"), (a2b, "r1"), (a3b, "r2")):
        msg.round_id = rid
    assert [len(g) for g in group_messages_by_api_round([u2, a1b, a2b, a3b])] == [1, 2, 1]
    print("✅ group_messages_by_api_round OK")

    # ── 15. collect_compactable_tool_ids + microcompact ──
    a_with_tools = create_assistant_message(tool_calls=[
        ToolUseBlock(id="tu_read", name="Read", input={}),
        ToolUseBlock(id="tu_todo", name="TodoWrite", input={}),
        ToolUseBlock(id="tu_bash", name="Bash", input={}),
    ])
    assert collect_compactable_tool_ids([a_with_tools]) == ["tu_read", "tu_bash"]

    old_a = create_assistant_message(tool_calls=[
        ToolUseBlock(id=f"old_{i}", name="Read", input={}) for i in range(4)
    ])
    seq = [old_a]
    for i in range(4):
        seq.append(create_tool_result(f"old_{i}", "X" * 4000))
        seq.append(create_assistant_message(tool_calls=[
            ToolUseBlock(id=f"mid_{i}", name="Read", input={})]))
        seq.append(create_tool_result(f"mid_{i}", "Y" * 4000))
    out = await microcompact(seq, None, "repl_main_thread")
    cleared = [m for m in out["messages"]
               if getattr(m, "type", None) == "tool_result"
               and m.content == TIME_BASED_MC_CLEARED_MESSAGE]
    assert len(cleared) >= 1
    last_tr = [m for m in out["messages"]
               if getattr(m, "type", None) == "tool_result"][-1]
    assert last_tr.content != TIME_BASED_MC_CLEARED_MESSAGE
    print("✅ microcompact OK")

    # ── 16. truncate_head_for_ptl_retry — CONTRACT + edge cases ──
    # Contract:
    #   • Returns None when < 2 groups (can't preserve user-first invariant)
    #   • Otherwise: new list, first element is user-role,
    #     at least one original message dropped, deterministic
    #   • Passing a marker-prepended list strips the marker first (idempotent)

    # 16a. Empty → None
    assert truncate_head_for_ptl_retry([], "x") is None

    # 16b. One group (adjacent assistants) → None
    one_group = [create_assistant_message(content="a"),
                 create_assistant_message(content="b")]
    assert truncate_head_for_ptl_retry(one_group, "x") is None

    # 16c. Two groups → trims to one + prepends user-role marker
    two_groups = [create_user_message("u"),
                  create_assistant_message(content="a1"),
                  create_assistant_message(content="a2")]
    tiny = truncate_head_for_ptl_retry(two_groups, "garbage")
    assert tiny is not None
    assert tiny is not two_groups
    assert getattr(tiny[0], "type", None) == "user"

    # 16d. Realistic large conversation (5 pairs, ~500 tokens per message)
    groups_in = _build_pairs(5)
    first_uuid = groups_in[0].uuid
    original_tokens = estimate_message_tokens(groups_in)
    original_uuids = {getattr(m, "uuid", None) for m in groups_in if getattr(m, "uuid", None)}

    # Well-formed PTL: gap drives drop count
    trunc_gap = truncate_head_for_ptl_retry(
        groups_in, "Prompt is too long: 5000 tokens > 100 maximum"
    )
    assert trunc_gap is not None
    assert trunc_gap is not groups_in
    assert getattr(trunc_gap[0], "type", None) == "user"
    # Contract: at least one ORIGINAL message dropped
    surviving_uuids = {getattr(m, "uuid", None) for m in trunc_gap}
    assert first_uuid not in surviving_uuids
    assert len(surviving_uuids & original_uuids) < len(original_uuids)
    # Contract: tokens decreased (with large messages)
    assert estimate_message_tokens(trunc_gap) < original_tokens

    # Unparseable PTL: 20% fallback
    trunc_20 = truncate_head_for_ptl_retry(groups_in, "garbage")
    assert trunc_20 is not None
    assert trunc_20 is not groups_in
    assert getattr(trunc_20[0], "type", None) == "user"
    surviving_20 = {getattr(m, "uuid", None) for m in trunc_20}
    assert first_uuid not in surviving_20
    assert estimate_message_tokens(trunc_20) < original_tokens

        # Determinism: same input → same observable output (content + type).
    # NOT UUIDs — the marker created on each call is a new object with
    # a fresh uuid; that is expected and correct. What must be stable is
    # the sequence the model will see.
    trunc_20b = truncate_head_for_ptl_retry(groups_in, "garbage")
    sig = lambda msgs: [(getattr(m, "type", None), getattr(m, "content", None))
                        for m in msgs]
    assert sig(trunc_20) == sig(trunc_20b), (
        f"non-deterministic output: {sig(trunc_20)} != {sig(trunc_20b)}"
    )

    # Idempotent marker strip: a marker-prepended input re-truncates cleanly
    already_marked = [create_user_message(PTL_RETRY_MARKER, is_meta=True)] + list(groups_in[2:])
    trunc_again = truncate_head_for_ptl_retry(already_marked, "garbage")
    if trunc_again is not None:
        markers = [m for m in trunc_again
                   if getattr(m, "type", None) == "user"
                   and getattr(m, "content", "") == PTL_RETRY_MARKER]
        assert len(markers) <= 1
    print("✅ truncate_head_for_ptl_retry (contract + 7 edge cases) OK")

    # ── 17. compact_conversation end-to-end (summary mocked) ──
    original = compact._stream_compact_summary

    async def fake_summary(_msgs, _req, _pre, _model):
        return ("<analysis>thinking...</analysis>\n<summary>"
                "1. Primary Request: build it\n"
                "8. Current Work: porting compact.py\n"
                "</summary>")

    compact._stream_compact_summary = fake_summary
    try:
        ctx = make_ctx(read_files={
            "/a.py": ("A" * 100, 300),
            "/b.py": ("B" * 100, 200),
            "/c.py": ("C" * 100, 100),
        })
        msgs = [
            create_user_message("write code"),
            create_assistant_message(tool_calls=[
                ToolUseBlock(id="t1", name="Read", input={})]),
            create_tool_result("t1", "file contents"),
            create_assistant_message(content="done"),
        ]
        result = await compact_conversation(
            msgs, ctx, None, suppress_follow_up_questions=True,
            is_auto_compact=True,
        )
        built = build_post_compact_messages(result)
        assert built[0] is result.boundary_marker
        assert "being continued from a previous conversation" in built[1].content
        assert "build it" in built[1].content
        assert "thinking..." not in built[1].content
        assert len(result.attachments) == 3
        assert result.attachments[0].attachment["filename"] == "/a.py"
        assert ctx.read_file_state.files == {}
        print("✅ compact_conversation end-to-end OK")
    finally:
        compact._stream_compact_summary = original

    # ── 18. autocompact circuit breaker ──
    out2 = await autocompact(
        [create_user_message("a" * 700_000)], make_ctx(), None,
        "repl_main_thread", {"consecutiveFailures": 3}, 0,
    )
    assert out2 == {"compaction_result": None, "consecutive_failures": None}
    out3 = await autocompact(
        [create_user_message("small")], make_ctx(), None,
        "repl_main_thread", None, 0,
    )
    assert out3["compaction_result"] is None
    print("✅ autocompact circuit breaker OK")

    print("\n🎉 All compact.py tests passed")


asyncio.run(main())