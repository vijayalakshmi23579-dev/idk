"""test_smoke.py — pre-flight validation. No API calls. No network.

Imports every module, registers tools, converts schemas, builds contexts.
If any wire is unconnected, this fails fast with a clear line.
"""
import asyncio
import os
import sys


def check(name, fn):
    try:
        fn()
        print(f"✅ {name}")
    except Exception as e:
        print(f"❌ {name}: {type(e).__name__}: {e}")
        raise


async def async_check(name, coro_fn):
    try:
        await coro_fn()
        print(f"✅ {name}")
    except Exception as e:
        print(f"❌ {name}: {type(e).__name__}: {e}")
        raise


def main():
    # ── 1. Every module imports ──
    def imports():
        import messages, tool, state, system_context, registry
        import permission_rules, denial_tracking, path_safety, permission_types
        import permissions, classifier, classifier_decision
        import api
        import execute, streaming_executor, tool_orchestration
        import query_config, query_transitions, query_deps
        import query_token_budget, query_stop_hooks
        import query, compact
        import memory_injector, memory_tools, main, todo
        import tools.read, tools.edit, tools.write, tools.bash
    check("All modules import", imports)

    def cli_tools_registered():
        from registry import find
        expected = (
            "Read", "Edit", "Write", "Bash", "TodoWrite", "Task",
            "MemorySearch", "MemoryExpand", "WorkflowPropose", "WorkflowRun",
            "WorkflowList", "WorkflowPromote",
        )
        missing = [name for name in expected if find(name) is None]
        assert not missing, f"CLI tools not registered: {missing}"
    check("CLI tool registration", cli_tools_registered)

    # ── 2. Read registers and is findable ──
    def read_tool():
        from tools.read import ReadTool, ReadInput
        from registry import register, find, all_tools
        register(ReadTool)
        found = find("Read")
        assert found is not None, "Read not in registry"
        assert found.name == "Read"
        assert any(t.name == "Read" for t in all_tools())
        assert found.is_read_only(ReadInput(file_path="/tmp/x")) is True
        assert found.is_concurrency_safe(ReadInput(file_path="/tmp/x")) is True
    check("Read registers + is findable", read_tool)

    # ── 3. Tool schema converts to OpenAI format ──
    def schema():
        from tool import to_openai_schema
        from registry import find
        tool = find("Read")
        schema = to_openai_schema(tool)
        assert schema["type"] == "function"
        assert schema["function"]["name"] == "Read"
        assert "properties" in schema["function"]["parameters"]
        assert "file_path" in schema["function"]["parameters"]["properties"]
    check("Tool schema → OpenAI format", schema)

    # ── 4. Message factory + to_openai ──
    def messages():
        from messages import (
            create_user_message, create_assistant_message, create_tool_result,
            to_openai, ToolUseBlock,
        )
        u = create_user_message("hello")
        a = create_assistant_message(
            tool_calls=[ToolUseBlock(id="t1", name="Read", input={"file_path": "/x"})],
            stop_reason="tool_use",
        )
        tr = create_tool_result("t1", "file contents")
        conv = to_openai([u, a, tr])
        assert conv[0] == {"role": "user", "content": "hello"}
        assert conv[1]["role"] == "assistant"
        assert conv[1]["tool_calls"][0]["function"]["name"] == "Read"
        assert conv[2]["role"] == "tool"
        assert conv[2]["tool_call_id"] == "t1"
        # compact-summary factory path
        cs = create_user_message("summary", is_compact_summary=True)
        assert cs.is_compact_summary is True
    check("Message factories + to_openai", messages)

    # ── 5. Model info fallback chain ──
    def model_info():
        from query import get_context_window_for_model, get_max_output_tokens_for_model
        assert get_context_window_for_model("claude-sonnet-4-5") == 200_000
        assert get_context_window_for_model("gemini-2.5-flash") == 1_048_576
        assert get_context_window_for_model("groq/llama-3.3-70b-versatile") == 131_072
        assert get_context_window_for_model("unknown") == 200_000
        assert get_max_output_tokens_for_model("claude-sonnet-4-5") > 0
    check("Model info (multi-tier)", model_info)

    # ── 6. Compact thresholds compute ──
    def compact():
        from compact import (
            get_auto_compact_threshold, calculate_token_warning_state,
            should_auto_compact, is_auto_compact_enabled,
        )
        for var in ("CLAUDE_CODE_AUTO_COMPACT_WINDOW", "CLAUDE_AUTOCOMPACT_PCT_OVERRIDE"):
            os.environ.pop(var, None)
        assert get_auto_compact_threshold("claude-sonnet-4-5") == 167_000
        assert is_auto_compact_enabled() is True
        s = calculate_token_warning_state(50_000, "claude-sonnet-4-5")
        assert s["is_above_auto_compact_threshold"] is False
        from messages import create_user_message
        assert should_auto_compact(
            [create_user_message("hi")], "claude-sonnet-4-5"
        ) is False
    check("Compact thresholds", compact)

    # ── 7. Permission context + gauntlet constructs ──
    async def perms():
        from permissions import has_permissions_to_use_tool
        from registry import find
        from state import AppState, FileStateCache, ToolPermissionContext
        from tool import ToolUseContext
        state = AppState(
            messages=[],
            tool_permission_context=ToolPermissionContext(mode="bypassPermissions"),
            read_file_cache=FileStateCache(),
        )
        ctx = ToolUseContext(
            get_app_state=lambda: state,
            set_app_state=lambda f: None,
            read_file_state=state.read_file_cache,
        )
        tool = find("Read")
        decision = await has_permissions_to_use_tool(
            tool, {"file_path": "/tmp/x"}, ctx, tool_use_id="t1",
        )
        assert decision.behavior == "allow"
    asyncio.run(async_check("Permission gauntlet constructs", perms))

    # ── 8. Execute pipeline (Read, real file) ──
    async def execute_read():
        import tempfile
        from execute import execute_tool
        from registry import find
        from state import AppState, FileStateCache, ToolPermissionContext
        from tool import ToolUseContext
        # Create a temp file
        fd, path = tempfile.mkstemp(suffix=".txt")
        with os.fdopen(fd, "w") as f:
            f.write("line one\nline two\nline three\n")
        state = AppState(
            messages=[],
            tool_permission_context=ToolPermissionContext(mode="bypassPermissions"),
            read_file_cache=FileStateCache(),
        )
        ctx = ToolUseContext(
            get_app_state=lambda: state,
            set_app_state=lambda f: None,
            read_file_state=state.read_file_cache,
        )
        result = await execute_tool(
            {"id": "t1", "name": "Read", "input": {"file_path": path}},
            ctx,
        )
        assert result.is_error is False, f"Read failed: {result.result_message.content[:100]}"
        assert "line two" in str(result.result_block["content"])
        os.unlink(path)
    asyncio.run(async_check("Execute pipeline (Read, real file)", execute_read))

    # ── 9. Query loop constructs (NoopDeps, no API) ──
    async def query_construct():
        from messages import create_assistant_message, create_user_message
        from query import QueryParams, query
        from query_deps import NoopDeps
        from query_transitions import Terminal
        from state import AppState, FileStateCache, ToolPermissionContext
        from tool import ToolUseContext
        deps = NoopDeps(responses=[
            create_assistant_message(content="hello back", stop_reason="end_turn"),
        ])
        state = AppState(
            messages=[],
            tool_permission_context=ToolPermissionContext(mode="bypassPermissions"),
            read_file_cache=FileStateCache(),
        )
        ctx = ToolUseContext(
            get_app_state=lambda: state,
            set_app_state=lambda f: None,
            read_file_state=state.read_file_cache,
        )
        params = QueryParams(
            messages=[create_user_message("hi")],
            system_prompt="You are helpful",
            tool_use_context=ctx,
            deps=deps.as_deps(),
        )
        events = []
        async for e in query(params):
            events.append(e)
        term = next((e for e in reversed(events) if isinstance(e, Terminal)), None)
        assert term is not None, "no Terminal emitted"
        assert term.reason == "completed"
        assert any(
            hasattr(e, "type") and getattr(e, "type") == "assistant"
            for e in events
        )
    asyncio.run(async_check("Query loop constructs (NoopDeps)", query_construct))

    # ── 10. Environment check ──
    def env_check():
        found_key = any(
            os.environ.get(k) for k in
            ("GROQ_API_KEY", "GEMINI_API_KEY", "NVIDIA_API_KEY", "OPENROUTER_API_KEY")
        )
        if found_key:
            print("   (at least one provider key found in env — main.py will work)")
        else:
            print("   ⚠️  no provider API key found in env")
            print("   Set one of: GROQ_API_KEY, GEMINI_API_KEY,")
            print("                NVIDIA_API_KEY, OPENROUTER_API_KEY")
            print("   (in .env or export ...)")
    check("Environment", env_check)

    print("\n🎉 All smoke tests passed. The framework is wired correctly.")
    print("   Next: python main.py \"read the file <path>\"")


main()