"""test_edit.py — real files in a temp dir, no network.

Every assertion is an invariant, not an internal implementation detail.
"""
import asyncio
import os
import tempfile

from messages import ToolUseBlock  # noqa: F401 (import-path regression guard)
from state import AppState, FileStateCache, ToolPermissionContext
from tool import ToolUseContext
from tools.edit import (
    EditTool, FileEditInput, FILE_UNEXPECTEDLY_MODIFIED_ERROR,
    LEFT_DOUBLE_CURLY_QUOTE, RIGHT_DOUBLE_CURLY_QUOTE,
    normalize_quotes, find_actual_string, preserve_quote_style,
    apply_edit_to_file, get_patch_for_edit, read_file_for_edit,
    write_text_content,
)

from registry import register, clear
from execute import execute_tool
from tools.read import ReadTool, ReadInput

def setup():
    clear()
    register(ReadTool)
    register(EditTool)




def make_ctx():
    state = AppState(
        messages=[],
        tool_permission_context=ToolPermissionContext(mode="bypassPermissions"),
        read_file_cache=FileStateCache(),
    )
    return ToolUseContext(
        get_app_state=lambda: state,
        set_app_state=lambda f: None,
        read_file_state=state.read_file_cache,
    )



async def _read(ctx, path, **kwargs):
    return await ReadTool.call(ReadInput(file_path=path, **kwargs), ctx)

async def _edit(ctx, path, old, new, replace_all=False):
    return await execute_tool(
        {"id": "tu", "name": "Edit",
         "input": {"file_path": path, "old_string": old,
                   "new_string": new, "replace_all": replace_all}},
        ctx,
    )


async def main():
    setup()
    tmp = tempfile.mkdtemp()

    # ═══ 1. Quote normalization (pure functions) ═══
    assert normalize_quotes(f"don{LEFT_DOUBLE_CURLY_QUOTE}t") == 'don"t'
    assert find_actual_string('say "hi"', 'say "hi"') == 'say "hi"'
    # curly in file, straight from model → returns curly substring
    curly_file = f'say {LEFT_DOUBLE_CURLY_QUOTE}hi{RIGHT_DOUBLE_CURLY_QUOTE}'
    assert find_actual_string(curly_file, 'say "hi"') == curly_file
    assert find_actual_string("nothing", "absent") is None
    print("✅ Quote normalization OK")

    # ═══ 2. preserveQuoteStyle applies curly style to new_string ═══
    preserved = preserve_quote_style(
        'say "hi"', curly_file, 'say "bye"',
    )
    assert LEFT_DOUBLE_CURLY_QUOTE in preserved
    assert RIGHT_DOUBLE_CURLY_QUOTE in preserved
    # No normalization happened → unchanged
    assert preserve_quote_style("a", "a", "b") == "b"
    print("✅ preserveQuoteStyle OK")

    # ═══ 3. applyEditToFile: replace + replace_all ═══
    assert apply_edit_to_file("foo foo", "foo", "bar") == "bar foo"
    assert apply_edit_to_file("foo foo", "foo", "bar", replace_all=True) == "bar bar"
    # Deletion strips orphaned trailing newline
    assert apply_edit_to_file("a\nb\nc\n", "b", "") == "a\nc\n"
    # Deletion without trailing newline: plain replace
    assert apply_edit_to_file("a\nb", "b", "") == "a\n"
    print("✅ applyEditToFile OK")

    # ═══ 4. get_patch_for_edit: hunks + updated ═══
    patch, updated = get_patch_for_edit(
        "/x", "line1\nline2\nline3\n", "line2", "LINE2",
    )
    assert updated == "line1\nLINE2\nline3\n"
    assert len(patch) == 1
    h = patch[0]
    assert all(k in h for k in ("oldStart", "oldLines", "newStart", "newLines", "lines"))
    assert any(l.startswith("-line2") for l in h["lines"])
    assert any(l.startswith("+LINE2") for l in h["lines"])
    # Identical content → error
    try:
        get_patch_for_edit("/x", "same", "same", "same")
        raise AssertionError("should raise on no-op")
    except RuntimeError:
        pass
    # old_string is substring of previous new_string → error (multi-edit path)
    from tools.edit import get_patch_for_edits
    try:
        get_patch_for_edits("/x", "abc", [
            {"old_string": "a", "new_string": "xyz", "replace_all": False},
            {"old_string": "xy", "new_string": "q", "replace_all": False},
        ])
        raise AssertionError("should raise on substring collision")
    except RuntimeError as e:
        assert "substring" in str(e)
    print("✅ getPatchForEdit(s) OK")

    # ═══ 5. Edit without read → read-gate errorCode 6 ═══
    p = os.path.join(tmp, "unread.txt")
    with open(p, "w") as f:
        f.write("hello world\n")
    ctx = make_ctx()
    res = await _edit(ctx, p, "hello", "goodbye")
    assert res.is_error
    assert "File has not been read yet" in res.result_message.content
    print("✅ Read-gate (not read) OK")

    # ═══ 6. Partial read does NOT satisfy the read-gate ═══
    ctx = make_ctx()
    big = os.path.join(tmp, "big.txt")
    with open(big, "w") as f:
        f.write("\n".join(f"line{i}" for i in range(100)) + "\n")
    await _read(ctx, big, offset=1, limit=10)  # partial
    res = await _edit(ctx, big, "line5", "LINE5")
    assert res.is_error
    assert "File has not been read yet" in res.result_message.content
    print("✅ Read-gate (partial read rejected) OK")

    # ═══ 7. Full read → edit succeeds ═══
    ctx = make_ctx()
    p2 = os.path.join(tmp, "greet.txt")
    with open(p2, "w") as f:
        f.write("hello world\nsecond line\n")
    await _read(ctx, p2)
    res = await _edit(ctx, p2, "hello", "goodbye")
    assert not res.is_error, res.result_message.content
    with open(p2) as f:
        assert f.read() == "goodbye world\nsecond line\n"
    # Read cache updated with post-edit content
    cached = ctx.read_file_state.get(os.path.abspath(p2))
    assert cached is not None
    assert "goodbye world" in cached["content"]
    assert cached["offset"] is None and cached["limit"] is None
    print("✅ Full read → edit → cache update OK")

    # ═══ 8. TOCTOU: file modified after read → errorCode 7 ═══
    ctx = make_ctx()
    p3 = os.path.join(tmp, "toctou.txt")
    with open(p3, "w") as f:
        f.write("original\n")
    await _read(ctx, p3)
    import time
    time.sleep(0.01)
    with open(p3, "a") as f:
        f.write("external change\n")
    os.utime(p3, (time.time() + 10, time.time() + 10))
    res = await _edit(ctx, p3, "original", "changed")
    assert res.is_error
    assert "modified since read" in res.result_message.content
    print("✅ TOCTOU detected OK")

    # ═══ 9. old_string == new_string → errorCode 1 ═══
    ctx = make_ctx()
    p4 = os.path.join(tmp, "noop.txt")
    with open(p4, "w") as f:
        f.write("same\n")
    await _read(ctx, p4)
    res = await _edit(ctx, p4, "same", "same")
    assert res.is_error
    assert "exactly the same" in res.result_message.content
    print("✅ old_string == new_string rejected OK")

    # ═══ 10. String not found → errorCode 8 ═══
    ctx = make_ctx()
    p5 = os.path.join(tmp, "missing.txt")
    with open(p5, "w") as f:
        f.write("only this\n")
    await _read(ctx, p5)
    res = await _edit(ctx, p5, "does not exist", "replacement")
    assert res.is_error
    assert "String to replace not found" in res.result_message.content
    print("✅ String not found rejected OK")

    # ═══ 11. Uniqueness: 2 matches + replace_all=False → errorCode 9 ═══
    ctx = make_ctx()
    p6 = os.path.join(tmp, "dup.txt")
    with open(p6, "w") as f:
        f.write("foo\nfoo\n")
    await _read(ctx, p6)
    res = await _edit(ctx, p6, "foo", "bar")
    assert res.is_error
    assert "Found 2 matches" in res.result_message.content
    print("✅ Uniqueness rejected OK")

    # ═══ 12. replace_all=True replaces all ═══
    ctx = make_ctx()
    p7 = os.path.join(tmp, "all.txt")
    with open(p7, "w") as f:
        f.write("foo\nfoo\nfoo\n")
    await _read(ctx, p7)
    res = await _edit(ctx, p7, "foo", "bar", replace_all=True)
    assert not res.is_error
    with open(p7) as f:
        assert f.read() == "bar\nbar\nbar\n"
    print("✅ replace_all OK")

    # ═══ 13. File creation (old_string='', file doesn't exist) ═══
    ctx = make_ctx()
    newp = os.path.join(tmp, "created.txt")
    res = await _edit(ctx, newp, "", "brand new content\n")
    assert not res.is_error, res.result_message.content
    assert os.path.exists(newp)
    with open(newp) as f:
        assert f.read() == "brand new content\n"
    print("✅ File creation OK")

    # ═══ 14. File creation refused when file exists and non-empty ═══
    ctx = make_ctx()
    existing = os.path.join(tmp, "exists.txt")
    with open(existing, "w") as f:
        f.write("already here\n")
    res = await _edit(ctx, existing, "", "overwrite\n")
    assert res.is_error
    assert "already exists" in res.result_message.content
    print("✅ File creation refused over existing OK")

    # ═══ 15. Curly quotes in file, straight in old_string → preserved style ═══
    ctx = make_ctx()
    p8 = os.path.join(tmp, "curly.txt")
    with open(p8, "w", encoding="utf-8") as f:
        f.write(f'msg = {LEFT_DOUBLE_CURLY_QUOTE}hello{RIGHT_DOUBLE_CURLY_QUOTE}\n')
    await _read(ctx, p8)
    res = await _edit(ctx, p8, 'msg = "hello"', 'msg = "goodbye"')
    assert not res.is_error, res.result_message.content
    with open(p8, encoding="utf-8") as f:
        out = f.read()
    assert LEFT_DOUBLE_CURLY_QUOTE in out and "goodbye" in out
    assert '"' not in out  # straight quotes were converted to curly
    print("✅ Curly quote preservation OK")

    # ═══ 16. .ipynb → errorCode 5 ═══
    # ═══ 16. .ipynb → errorCode 5 ═══
    ctx = make_ctx()
    nb = os.path.join(tmp, "notebook.ipynb")
    with open(nb, "w") as f:
        f.write("{}\n")
    # REMOVED: await _read(ctx, nb)
    # ReadTool refuses .ipynb, and the .ipynb check in validate_input
    # comes BEFORE the read-gate, so no read is needed to reach it.
    res = await _edit(ctx, nb, "{}", '{"cells":[]}')
    assert res.is_error
    assert "Jupyter Notebook" in res.result_message.content
    print("✅ .ipynb rejected OK")

    # ═══ 17. Line-ending preservation: CRLF in → CRLF out ═══
    ctx = make_ctx()
    crlf = os.path.join(tmp, "crlf.txt")
    with open(crlf, "wb") as f:
        f.write(b"first\r\nsecond\r\n")
    await _read(ctx, crlf)
    res = await _edit(ctx, crlf, "second", "SECOND")
    assert not res.is_error, res.result_message.content
    with open(crlf, "rb") as f:
        raw = f.read()
    assert b"\r\n" in raw and b"SECOND" in raw
    print("✅ CRLF preservation OK")

    # ═══ 18. Tool metadata matches source ═══
    assert EditTool.name == "Edit"
    assert EditTool.max_result_size_chars == 100_000
    assert EditTool.strict is True
    assert EditTool.is_read_only(FileEditInput(
        file_path="/x", old_string="a", new_string="b")) is False
    assert EditTool.is_concurrency_safe(FileEditInput(
        file_path="/x", old_string="a", new_string="b")) is False
    inp = FileEditInput(file_path="/x", old_string="a", new_string="b")
    assert EditTool.get_path(inp) == "/x"
    assert EditTool.to_auto_classifier_input(inp) == "/x: b"
    print("✅ Tool metadata OK")

    # ═══ 19. backfillObservableInput expands ~ and relative ═══
    inp_dict = {"file_path": "~/foo.txt", "old_string": "a", "new_string": "b"}
    EditTool.backfill_observable_input(inp_dict)
    assert inp_dict["file_path"].startswith(os.path.expanduser("~"))
    rel = {"file_path": "rel.txt", "old_string": "a", "new_string": "b"}
    EditTool.backfill_observable_input(rel)
    assert os.path.isabs(rel["file_path"])
    print("✅ backfillObservableInput OK")

        # ═══ 20. map_tool_result_to_tool_result_block_param ═══
    ctx = make_ctx()
    p9 = os.path.join(tmp, "map.txt")
    with open(p9, "w") as f:
        f.write("before\n")
    await _read(ctx, p9)
    res = await _edit(ctx, p9, "before", "after")
    # execute_tool already called the mapper; res.result_block IS the
    # wire-format tool_result. Don't re-map.
    block = res.result_block
    assert block["type"] == "tool_result"
    assert block["tool_use_id"] == "tu"     # helper uses id="tu"
    assert "has been updated" in block["content"]
    print("✅ Result mapping OK")


asyncio.run(main())