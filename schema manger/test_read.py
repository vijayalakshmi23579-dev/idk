"""test_read.py — real files in a temp dir, no network."""
import asyncio
import os
import tempfile

from messages import ToolUseBlock  # noqa: F401  (import-path regression guard)
from permission_types import PermissionAllowDecision
from state import AppState, FileStateCache, ToolPermissionContext
from tool import ToolUseContext
from tools.read import (
    ReadTool, ReadInput, FILE_UNCHANGED_STUB, MAX_LINES_TO_READ,
    BLOCKED_DEVICE_PATHS, add_line_numbers, expand_path,
    is_blocked_device_path, find_similar_file, _validate_input,
)


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


async def main():
    tmp = tempfile.mkdtemp()
    p = os.path.join(tmp, "sample.py")
    with open(p, "w") as f:
        f.write("line one\nline two\nline three\n")

    ctx = make_ctx()

    # ── 1. Line numbers: compact format is the 3P default ──
    assert add_line_numbers("a\nb", 1) == "1\ta\n2\tb"
    assert add_line_numbers("a\nb", 10) == "10\ta\n11\tb"
    os.environ["CLAUDE_CODE_COMPACT_LINE_PREFIX"] = "0"
    padded = add_line_numbers("a", 7)
    assert padded == "     7→a", padded
    os.environ.pop("CLAUDE_CODE_COMPACT_LINE_PREFIX")
    assert add_line_numbers("", 1) == ""
    print("✅ add_line_numbers (compact + padded) OK")

    # ── 2. Full text read ──
    r = await ReadTool.call(ReadInput(file_path=p), ctx)
    assert r.data["type"] == "text"
    assert r.data["file"]["numLines"] == 3
    assert r.data["file"]["totalLines"] == 3
    assert r.data["file"]["startLine"] == 1
    assert r.data["file"]["content"] == "1\tline one\n2\tline two\n3\tline three"
    block = ReadTool.map_tool_result_to_tool_result_block_param(r.data, "t1")
    assert block["type"] == "tool_result" and block["tool_use_id"] == "t1"
    assert "line two" in block["content"]
    print("✅ Full text read OK")

    # ── 3. offset + limit ──
    r = await ReadTool.call(ReadInput(file_path=p, offset=2, limit=1), ctx)
    assert r.data["file"]["content"] == "2\tline two"
    assert r.data["file"]["numLines"] == 1
    assert r.data["file"]["totalLines"] == 3
    print("✅ offset/limit OK")

    # ── 4. Dedup: same range + unchanged mtime → file_unchanged ──
    ctx2 = make_ctx()
    first = await ReadTool.call(ReadInput(file_path=p), ctx2)
    assert first.data["type"] == "text"
    second = await ReadTool.call(ReadInput(file_path=p), ctx2)
    assert second.data["type"] == "file_unchanged"
    block = ReadTool.map_tool_result_to_tool_result_block_param(second.data, "t2")
    assert block["content"] == FILE_UNCHANGED_STUB
    # a DIFFERENT range is not deduped
    third = await ReadTool.call(ReadInput(file_path=p, offset=2, limit=1), ctx2)
    assert third.data["type"] == "text"
    # touching the file invalidates the dedup
    ctx3 = make_ctx()
    await ReadTool.call(ReadInput(file_path=p), ctx3)
    with open(p, "a") as f:
        f.write("line four\n")
    os.utime(p, (1, 1))
    after = await ReadTool.call(ReadInput(file_path=p), ctx3)
    assert after.data["type"] == "text"
    assert "line four" in after.data["file"]["content"]
    print("✅ Dedup (range + mtime) OK")

    # ── 5. Empty file → warning in place of contents ──
    e = os.path.join(tmp, "empty.txt")
    open(e, "w").close()
    r = await ReadTool.call(ReadInput(file_path=e), make_ctx())
    assert r.data["file"]["totalLines"] == 0
    assert "contents are empty" in r.data["file"]["content"]
    print("✅ Empty file warning OK")

    # ── 6. MAX_LINES_TO_READ default cap ──
    big = os.path.join(tmp, "big.txt")
    with open(big, "w") as f:
        f.write("\n".join(f"l{i}" for i in range(1, MAX_LINES_TO_READ + 501)))
    r = await ReadTool.call(ReadInput(file_path=big), make_ctx())
    assert r.data["file"]["numLines"] == MAX_LINES_TO_READ
    assert r.data["file"]["totalLines"] == MAX_LINES_TO_READ + 500
    print("✅ MAX_LINES_TO_READ cap OK")

    # ── 7. Directory → clear error ──
    try:
        await ReadTool.call(ReadInput(file_path=tmp), make_ctx())
        raise AssertionError("should have raised")
    except ValueError as ex:
        assert "is a directory" in str(ex)
        assert "Bash" in str(ex)
    print("✅ Directory error OK")

    # ── 8. ENOENT → similar-file suggestion ──
    try:
        await ReadTool.call(
            ReadInput(file_path=os.path.join(tmp, "sample.ts")), make_ctx(),
        )
        raise AssertionError("should have raised")
    except FileNotFoundError as ex:
        msg = str(ex)
        assert "File does not exist." in msg
        assert "Note: your current working directory is" in msg
        assert "Did you mean sample.py?" in msg
    print("✅ File-not-found + similar-file suggestion OK")

    # ── 9. validateInput: binary ext, device file, pages ──
    v = _validate_input(ReadInput(file_path=os.path.join(tmp, "a.exe")), None)
    assert v.result is False and v.error_code == 4
    v = _validate_input(ReadInput(file_path="/dev/zero"), None)
    assert v.result is False and v.error_code == 9
    assert is_blocked_device_path("/dev/stdin") and is_blocked_device_path("/proc/1/fd/1")
    assert not is_blocked_device_path("/dev/null")
    v = _validate_input(ReadInput(file_path=p, pages="1-25"), None)
    assert v.result is False and v.error_code == 7
    v = _validate_input(ReadInput(file_path=p, pages="bad"), None)
    assert v.result is False and v.error_code == 7
    v = _validate_input(ReadInput(file_path=p, pages="1-5"), None)
    assert v.result is True
    # images and PDFs are exempt from the binary-ext rejection
    assert _validate_input(ReadInput(file_path=os.path.join(tmp, "x.png")), None).result is True
    assert _validate_input(ReadInput(file_path=os.path.join(tmp, "x.pdf")), None).result is True
    print("✅ validateInput (binary/device/pages) OK")

    # ── 10. Image → base64 image_url block ──
    png = os.path.join(tmp, "tiny.png")
    with open(png, "wb") as f:
        f.write(b"\x89PNG\r\n\x1a\n" + b"\x00" * 32)
    r = await ReadTool.call(ReadInput(file_path=png), make_ctx())
    assert r.data["type"] == "image"
    assert r.data["file"]["mediaType"] == "image/png"
    blk = ReadTool.map_tool_result_to_tool_result_block_param(r.data, "t3")
    assert blk["content"][0]["type"] == "image_url"
    assert blk["content"][0]["image_url"]["url"].startswith("data:image/png;base64,")
    print("✅ Image read + image_url block OK")

    # ── 11. Tool metadata matches source ──
    assert ReadTool.name == "Read"
    assert ReadTool.max_result_size_chars == float("inf")
    assert ReadTool.strict is True
    assert ReadTool.is_read_only(ReadInput(file_path=p)) is True
    assert ReadTool.is_concurrency_safe(ReadInput(file_path=p)) is True
    assert ReadTool.is_search_or_read_command(ReadInput(file_path=p))["isRead"] is True
    assert ReadTool.to_auto_classifier_input(ReadInput(file_path=p)) == p
    assert ReadTool.get_path(ReadInput(file_path=p)) == p
    assert ReadTool.extract_search_text({"type": "text"}) == ""
    print("✅ Tool metadata OK")

    # ── 12. backfill expands ~ and relative paths ──
    inp = {"file_path": "~/x.txt"}
    ReadTool.backfill_observable_input(inp)
    assert inp["file_path"].startswith(os.path.expanduser("~"))
    inp2 = {"file_path": "rel.txt"}
    ReadTool.backfill_observable_input(inp2)
    assert os.path.isabs(inp2["file_path"])
    print("✅ backfillObservableInput OK")

    # ── 13. check_permissions allows reads (deny rules still win at 1a) ──
    res = await ReadTool.check_permissions(ReadInput(file_path=p), make_ctx())
    assert isinstance(res, PermissionAllowDecision)
    print("✅ check_permissions allow OK")

    print("\n🎉 All read.py tests passed")


asyncio.run(main())