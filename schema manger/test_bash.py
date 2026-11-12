"""test_bash.py — no network. Real commands in subprocesses."""
import asyncio
import os
import tempfile

from registry import register, clear
from execute import execute_tool
from tools.bash import (
    BashTool, BashInput, is_read_only_command, is_destructive_command,
    split_compound_command, interpret_exit_code,
    DEFAULT_TIMEOUT_MS, MAX_TIMEOUT_MS,
)
from state import AppState, FileStateCache, ToolPermissionContext
from tool import ToolUseContext


def make_ctx(mode="bypassPermissions"):
    state = AppState(
        messages=[],
        tool_permission_context=ToolPermissionContext(mode=mode),
        read_file_cache=FileStateCache(),
    )
    return ToolUseContext(
        get_app_state=lambda: state, set_app_state=lambda f: None,
        read_file_state=state.read_file_cache,
    )


async def _bash(ctx, command, on_progress=None, **kwargs):
    return await execute_tool(
        {"id": "tu", "name": "Bash",
         "input": {"command": command, **kwargs}},
        ctx, on_progress=on_progress,
    )


async def main():
    clear()
    register(BashTool)

    # ═══ 1. Compound split ═══
    assert split_compound_command("ls && pwd") == ["ls", "pwd"]
    assert split_compound_command("echo 'a && b'") == ["echo 'a && b'"]
    assert split_compound_command("ls; pwd | wc -l") == ["ls", "pwd", "wc -l"]
    print("✅ Compound split OK")

    # ═══ 2. is_read_only classifier ═══
    assert is_read_only_command("ls") is True
    assert is_read_only_command("ls -la /tmp") is True
    assert is_read_only_command("cat foo.py") is True
    assert is_read_only_command("grep -r pattern .") is True
    assert is_read_only_command("git status") is True
    assert is_read_only_command("git log --oneline") is True
    # Compound: ALL parts must be read-only
    assert is_read_only_command("ls && pwd") is True
    assert is_read_only_command("ls && rm foo") is False
    # git subcommand check
    assert is_read_only_command("git push origin main") is False
    assert is_read_only_command("git reset --hard") is False
    # Redirects imply writing
    assert is_read_only_command("ls > /tmp/out") is False
    print("✅ is_read_only classifier OK")

    # ═══ 2b. Heredoc content is not classified as a filesystem path ═══
    heredoc_input = BashInput(command=(
        "cat > api.md <<'EOF'\n"
        "Example contains ... in prose\n"
        "EOF"
    ))
    heredoc_decision = await BashTool.check_permissions(heredoc_input, make_ctx())
    assert heredoc_decision.behavior == "allow", heredoc_decision.message
    suspicious_path_input = BashInput(command="cat '.../api.md'")
    path_decision = await BashTool.check_permissions(suspicious_path_input, make_ctx())
    assert path_decision.behavior == "deny"
    print("✅ Heredoc path-safety classification OK")

    # ═══ 3. is_destructive ═══
    assert is_destructive_command("rm -rf build") is True
    assert is_destructive_command("ls") is False
    assert is_destructive_command("mv a b") is True
    print("✅ is_destructive OK")

    # ═══ 4. Exit code interpretation ═══
    assert interpret_exit_code("grep foo file", 1) == "exit 1: no matches found (not an error)"
    assert interpret_exit_code("grep foo file", 0) is None
    assert interpret_exit_code("ls", 2) == "exit 2"
    assert "differences present" in (interpret_exit_code("git diff", 1) or "")
    print("✅ interpret_exit_code OK")

    # ═══ 5. Simple command ═══
    ctx = make_ctx()
    res = await _bash(ctx, "echo hello")
    assert not res.is_error, res.result_message.content
    assert "hello" in res.result_message.content
    print("✅ Simple echo OK")

    # ═══ 6. Nonzero exit that's NOT an error (grep no matches) ═══
    tmp = tempfile.mkdtemp()
    empty = os.path.join(tmp, "empty.txt")
    open(empty, "w").close()
    ctx = make_ctx()
    res = await _bash(ctx, f"grep zzz {empty}")
    # grep 1 = no matches → is_error=False per semantics
    assert not res.is_error, res.result_message.content
    assert "not an error" in res.result_message.content
    print("✅ grep no-matches treated as non-error OK")

    ctx = make_ctx()
    res = await _bash(ctx, f'grep "not found" "{empty}"')
    assert not res.is_error, res.result_message.content
    print("✅ Path safety ignores command whitespace OK")

    # ═══ 7. Nonzero exit that IS an error ═══
    ctx = make_ctx()
    res = await _bash(ctx, "false")
    assert res.is_error
    assert "exit 1" in res.result_message.content.lower() or "Exit code" in res.result_message.content
    print("✅ genuine failure marked as error OK")

    # ═══ 8. stderr captured ═══
    ctx = make_ctx()
    res = await _bash(ctx, "echo out; echo err >&2; exit 1")
    assert "err" in res.result_message.content
    assert res.is_error
    print("✅ stderr captured OK")

    # ═══ 9. Timeout interrupts ═══
    ctx = make_ctx()
    res = await _bash(ctx, "sleep 5", timeout=200)
    assert res.is_error
    assert "interrupted" in res.result_message.content.lower()
    print("✅ Timeout interrupts OK")

    # ═══ 10. Abort signal interrupts a running process ═══
    class AbortController:
        def __init__(self):
            self.aborted = False
            self.event = asyncio.Event()

        async def wait(self):
            await self.event.wait()

        def abort(self):
            self.aborted = True
            self.event.set()

    ctx = make_ctx()
    abort_controller = AbortController()
    ctx.abort_controller = abort_controller
    task = asyncio.create_task(_bash(ctx, "sleep 5"))
    await asyncio.sleep(0.1)
    abort_controller.abort()
    res = await asyncio.wait_for(task, timeout=2)
    assert res.is_error
    assert "interrupted" in res.result_message.content.lower()
    print("✅ Abort signal interrupts running process OK")

    # ═══ 11. Output chunks reach the progress callback ═══
    progress = []
    ctx = make_ctx()
    res = await _bash(
        ctx, "printf 'first\\n'; sleep 0.1; printf 'second\\n'",
        on_progress=progress.append,
    )
    assert not res.is_error
    assert any("first" in item["output"] for item in progress)
    assert any("second" in item["output"] for item in progress)
    print("✅ Incremental output progress OK")

    # ═══ 12. Destructive path refused ═══
    ctx = make_ctx()
    res = await _bash(ctx, "rm -rf /")
    assert res.is_error
    assert "dangerous removal path" in res.result_message.content
    print("✅ Dangerous removal refused OK")

    # ═══ 13. Empty command rejected ═══
    ctx = make_ctx()
    res = await _bash(ctx, "")
    assert res.is_error
    assert "Empty command" in res.result_message.content
    print("✅ Empty command rejected OK")

    # ═══ 14. Timeout > max rejected ═══
    ctx = make_ctx()
    res = await _bash(ctx, "ls", timeout=MAX_TIMEOUT_MS + 1)
    assert res.is_error
    assert "exceeds the maximum" in res.result_message.content
    print("✅ Timeout > max rejected OK")

    # ═══ 15. Output truncation ═══
    ctx = make_ctx()
    # 40K chars of output > 30K limit
    res = await _bash(ctx, "python3 -c \"print('x' * 40000)\"")
    assert not res.is_error
    assert "output truncated" in res.result_message.content.lower()
    assert "Read to see more" in res.result_message.content
    print("✅ Output truncation OK")

    # ═══ 16. is_silent_command ═══
    from tools.bash import is_silent_command
    assert is_silent_command("mv a b") is True
    assert is_silent_command("echo hi") is False
    print("✅ is_silent_command OK")

    # ═══ 17. Tool metadata ═══
    assert BashTool.name == "Bash"
    assert BashTool.max_result_size_chars == 30_000
    assert BashTool.is_read_only(BashInput(command="ls")) is True
    assert BashTool.is_read_only(BashInput(command="rm foo")) is False
    assert BashTool.is_concurrency_safe(BashInput(command="ls")) is True
    assert BashTool.is_concurrency_safe(BashInput(command="rm foo")) is False
    assert BashTool.is_destructive(BashInput(command="rm foo")) is True
    assert BashTool.to_auto_classifier_input(BashInput(command="ls")) == "ls"
    print("✅ Tool metadata OK")

    print("\n🎉 All bash.py tests passed")


asyncio.run(main())