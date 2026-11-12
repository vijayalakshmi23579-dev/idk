"""test_classifier.py — mock litellm so we don't hit the network."""
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

# Mock before importing classifier
import sys
mock_litellm = MagicMock()
mock_litellm.acompletion = AsyncMock()
sys.modules["litellm"] = mock_litellm

from classifier import (
    classify_yolo_action, format_action_for_classifier,
    _parse_decision, _format_transcript,
    CLASSIFIER_MODEL, STAGE_1_MAX_TOKENS,
)
from messages import create_user_message, create_assistant_message, create_tool_result
from tool import Tool


class _FakeTool:
    def __init__(self, name): self.name = name


def mock_response(text: str, in_tok=100, out_tok=10):
    """Build a fake LiteLLM response."""
    r = MagicMock()
    r.choices = [MagicMock()]
    r.choices[0].message.content = text
    r.usage = MagicMock()
    r.usage.prompt_tokens = in_tok
    r.usage.completion_tokens = out_tok
    r.usage.cache_read_input_tokens = 50
    r.usage.cache_creation_input_tokens = 0
    return r


async def main():
    # ── Action formatting ──
    assert "Bash" in format_action_for_classifier("Bash", {"command": "ls"})
    assert "ls" in format_action_for_classifier("Bash", {"command": "ls"})
    # Truncation
    big = {"data": "x" * 5000}
    result = format_action_for_classifier("Big", big)
    assert len(result) < 1100
    assert result.endswith("...")
    print("✅ Action formatting OK")

    # ── Transcript formatting ──
    msgs = [
        create_user_message("read the file"),
        create_assistant_message("ok"),
        create_tool_result("t1", "file contents"),
    ]
    text, too_long = _format_transcript(msgs)
    assert "USER" in text
    assert "ASSISTANT" in text
    assert "RESULT" in text
    assert too_long is False
    print("✅ Transcript formatting OK")

    # ── Decision parsing ──
    should_block, reason = _parse_decision("<decision>allow</decision>")
    assert should_block is False
    should_block, reason = _parse_decision("<decision>deny</decision>")
    assert should_block is True
    # Fail-closed on malformed
    should_block, reason = _parse_decision("garbage output")
    assert should_block is True
    assert "missing" in reason.lower()
    # With reasoning
    should_block, reason = _parse_decision(
        "<reasoning>This rm -rf / is dangerous</reasoning>\n<decision>deny</decision>"
    )
    assert should_block is True
    assert "dangerous" in reason.lower()
    print("✅ Decision parsing OK")

    # ── Stage 1 allows → immediate return ──
    mock_litellm.acompletion.reset_mock()
    mock_litellm.acompletion.return_value = mock_response("<decision>allow</decision>")
    result = await classify_yolo_action(
        msgs, "Bash: ls -la", [_FakeTool("Bash")], None,
    )
    assert result["should_block"] is False
    assert result["stage"] == "fast"
    assert result["unavailable"] is False
    assert mock_litellm.acompletion.call_count == 1  # only Stage 1
    print("✅ Stage 1 allow OK")

    # ── Stage 1 denies → Stage 2 runs ──
    mock_litellm.acompletion.reset_mock()
    mock_litellm.acompletion.side_effect = [
        mock_response("<decision>deny</decision>"),                     # Stage 1
        mock_response("<reasoning>dangerous</reasoning><decision>deny</decision>"),  # Stage 2
    ]
    result = await classify_yolo_action(
        msgs, "Bash: rm -rf /", [_FakeTool("Bash")], None,
    )
    assert result["should_block"] is True
    assert result["stage"] == "thinking"
    assert "dangerous" in result["reason"].lower()
    assert mock_litellm.acompletion.call_count == 2  # both stages
    print("✅ Stage 2 deny OK")

    # ── Stage 1 denies but Stage 2 allows ──
    mock_litellm.acompletion.reset_mock()
    mock_litellm.acompletion.side_effect = [
        mock_response("<decision>deny</decision>"),
        mock_response("<reasoning>safe on reflection</reasoning><decision>allow</decision>"),
    ]
    result = await classify_yolo_action(
        msgs, "Bash: cat /tmp/foo", [_FakeTool("Bash")], None,
    )
    assert result["should_block"] is False
    assert result["stage"] == "thinking"
    print("✅ Stage 2 override OK")

    # ── API failure on Stage 1 → unavailable ──
    mock_litellm.acompletion.reset_mock()
    mock_litellm.acompletion.side_effect = Exception("rate limited")
    result = await classify_yolo_action(
        msgs, "Bash: ls", [_FakeTool("Bash")], None,
    )
    assert result["unavailable"] is True
    assert result["should_block"] is False  # fail-open at classifier; caller decides
    assert result["stage"] == "unavailable"
    print("✅ API failure → unavailable OK")

    # ── API failure on Stage 2 → trust Stage 1 (deny) ──
    mock_litellm.acompletion.reset_mock()
    mock_litellm.acompletion.side_effect = [
        mock_response("<decision>deny</decision>"),
        Exception("stage 2 timed out"),
    ]
    result = await classify_yolo_action(
        msgs, "Bash: rm -rf /", [_FakeTool("Bash")], None,
    )
    assert result["should_block"] is True  # trust stage 1
    assert result["stage"] == "fast"
    print("✅ Stage 2 failure → trust Stage 1 OK")

    # ── Transcript too long ──
    huge_msgs = [create_user_message("x" * 10000) for _ in range(20)]
    result = await classify_yolo_action(
        huge_msgs, "Bash: ls", [_FakeTool("Bash")], None,
    )
    assert result["transcript_too_long"] is True
    assert result["unavailable"] is False
    print("✅ Transcript too long OK")

    # ── Usage accumulation ──
    mock_litellm.acompletion.reset_mock()
    mock_litellm.acompletion.side_effect = [
        mock_response("<decision>deny</decision>", in_tok=100, out_tok=10),
        mock_response("<decision>deny</decision>", in_tok=200, out_tok=20),
    ]
    result = await classify_yolo_action(
        msgs, "Bash: rm -rf /", [_FakeTool("Bash")], None,
    )
    assert result["usage"]["input_tokens"] == 300
    assert result["usage"]["output_tokens"] == 30
    assert result["stage1_usage"]["input_tokens"] == 100
    assert result["stage2_usage"]["input_tokens"] == 200
    print("✅ Usage accumulation OK")

    print(f"\n🎉 All classifier.py tests passed (model={CLASSIFIER_MODEL}, "
          f"stage1_max_tokens={STAGE_1_MAX_TOKENS})")


asyncio.run(main())