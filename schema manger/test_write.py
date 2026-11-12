"""test_write.py — real files in a temp dir, no network."""
import asyncio
import os
import tempfile

from registry import register, clear as clear_registry
from execute import execute_tool
from tools.read import ReadTool, ReadInput
from tools.write import WriteTool, FileWriteInput, FILE_WRITE_TOOL_NAME
from state import AppState, FileStateCache, ToolPermissionContext
from tool import ToolUseContext


def make_ctx():
    state = AppState(
        messages=[],
        tool_permission_context=ToolPermissionContext(mode="bypassPermissions"),
        read_file_cache=FileStateCache(),
    )
    return ToolUseContext(
        get_app_state=lambda: state, set_app_state=lambda f: None,
        read_file_state=state.read_file_cache,
    )


async def _write(ctx, path, content):
    return await execute_tool(
        {"id": "tu", "name": FILE_WRITE_TOOL_NAME,
         "input": {"file_path": path, "content": content}},
        ctx,
    )


async def main():
    clear_registry()
    register(ReadTool)
    register(WriteTool)
    tmp = tempfile.mkdtemp()

    # ═══ 1. Create a new file (no read-gate for new files) ═══
    ctx = make_ctx()
    p = os.path.join(tmp, "new.md")
    res = await _write(ctx, p, "# Hello\n\nbody\n")
    assert not res.is_error, res.result_message.content
    assert res.result_message.content == f"File created successfully at: {p}"
    with open(p) as f:
        assert f.read() == "# Hello\n\nbody\n"
    print("✅ Create new file OK")

    # ═══ 2. Read-cache updated after create ═══
    cached = ctx.read_file_state.get(os.path.abspath(p))
    assert cached is not None
    assert cached["content"] == "# Hello\n\nbody\n"
    assert cached["offset"] is None and cached["limit"] is None
    print("✅ Read cache updated OK")

    # ═══ 3. Overwrite without read → errorCode 2 ═══
    ctx2 = make_ctx()
    res = await _write(ctx2, p, "overwrite\n")
    assert res.is_error
    assert "has not been read yet" in res.result_message.content
    print("✅ Overwrite read-gate OK")

    # ═══ 4. Read then overwrite → update ═══
    ctx3 = make_ctx()
    await ReadTool.call(ReadInput(file_path=p), ctx3)
    res = await _write(ctx3, p, "# Hello\n\nNEW body\n")
    assert not res.is_error, res.result_message.content
    assert res.result_message.content == f"The file {p} has been updated successfully."
    # The output includes a structured diff on update
    assert res.result_block["content"].startswith("The file")
    with open(p) as f:
        assert f.read() == "# Hello\n\nNEW body\n"
    print("✅ Overwrite after read OK")

    # ═══ 5. Always writes LF (source L133-138) ═══
    ctx4 = make_ctx()
    crlf = os.path.join(tmp, "crlf.txt")
    with open(crlf, "wb") as f:
        f.write(b"old\r\ncontent\r\n")
    await ReadTool.call(ReadInput(file_path=crlf), ctx4)
    res = await _write(ctx4, crlf, "new\ncontent\n")
    assert not res.is_error
    with open(crlf, "rb") as f:
        raw = f.read()
    assert b"\r\n" not in raw, "Write must not preserve CRLF"
    assert raw == b"new\ncontent\n"
    print("✅ Write always uses LF OK")

    # ═══ 6. TOCTOU: file modified after read → errorCode 3 ═══
    ctx5 = make_ctx()
    toctou = os.path.join(tmp, "toctou.txt")
    with open(toctou, "w") as f:
        f.write("original\n")
    await ReadTool.call(ReadInput(file_path=toctou), ctx5)
    import time
    time.sleep(0.01)
    with open(toctou, "a") as f:
        f.write("external\n")
    os.utime(toctou, (time.time() + 10, time.time() + 10))
    res = await _write(ctx5, toctou, "changed\n")
    assert res.is_error
    assert "modified since read" in res.result_message.content
    print("✅ TOCTOU detected OK")

    # ═══ 7. mkdir auto-creates parent directories ═══
    ctx6 = make_ctx()
    nested = os.path.join(tmp, "a", "b", "c", "deep.md")
    res = await _write(ctx6, nested, "deep\n")
    assert not res.is_error, res.result_message.content
    assert os.path.exists(nested)
    print("✅ mkdir parent creation OK")

    # ═══ 8. Tool metadata matches source ═══
    assert WriteTool.name == "Write"
    assert WriteTool.max_result_size_chars == 100_000
    assert WriteTool.strict is True
    inp = FileWriteInput(file_path="/x", content="hi")
    assert WriteTool.is_read_only(inp) is False
    assert WriteTool.is_concurrency_safe(inp) is False
    assert WriteTool.get_path(inp) == "/x"
    assert WriteTool.to_auto_classifier_input(inp) == "/x: hi"
    assert WriteTool.extract_search_text({"type": "create"}) == ""
    print("✅ Tool metadata OK")

    # ═══ 9. backfillObservableInput expands ~ and relative ═══
    d = {"file_path": "~/foo.md", "content": "x"}
    WriteTool.backfill_observable_input(d)
    assert d["file_path"].startswith(os.path.expanduser("~"))
    d2 = {"file_path": "rel.md", "content": "x"}
    WriteTool.backfill_observable_input(d2)
    assert os.path.isabs(d2["file_path"])
    print("✅ backfillObservableInput OK")

    # ═══ 10. Heredoc content with "..." does NOT trip a permission error ═══
    # This is the regression test for the false positive you hit.
    ctx7 = make_ctx()
    md = os.path.join(tmp, "doc.md")
    content = (
        "# Doc\n\n"
        "- **`stream(messages, tools, model, ...)`:** the primary interface.\n"
        "Done.\n"
    )
    res = await _write(ctx7, md, content)
    assert not res.is_error, res.result_message.content
    with open(md) as f:
        assert "..." in f.read()
    print("✅ Content containing '...' writes without false positive OK")

    print("\n🎉 All write.py tests passed")


asyncio.run(main())