"""test_permissions.py"""
import asyncio
from pydantic import BaseModel
from permission_types import (
    PermissionAllowDecision, PermissionAskDecision, PermissionDenyDecision,
    PermissionPassthroughDecision, RuleReason, ModeReason, SafetyCheckReason,
    ClassifierReason, PermissionResult as make_result,
)
from permission_rules import PermissionRule, PermissionRuleValue
from denial_tracking import create_denial_tracking_state, record_denial
from state import AppState, ToolPermissionContext, FileStateCache
from tool import Tool, ToolUseContext, ToolResult, build_tool
from permissions import (
    has_permissions_to_use_tool, check_rule_based_permissions,
    get_allow_rules, get_deny_rules, get_ask_rules,
    get_deny_rule_for_tool, get_ask_rule_for_tool, tool_always_allowed_rule,
    create_permission_request_message,
)


class In(BaseModel):
    command: str = ""


def make_ctx(mode="default", allow=None, deny=None, ask=None):
    state = AppState(
        messages=[],
        tool_permission_context=ToolPermissionContext(
            mode=mode,
            always_allow_rules=allow or {},
            always_deny_rules=deny or {},
            always_ask_rules=ask or {},
        ),
        read_file_cache=FileStateCache(),
    )
    return ToolUseContext(
        get_app_state=lambda: state,
        set_app_state=lambda f: None,
    )


def make_tool(name="Bash", check_result=None):
    async def call(args, ctx, can=None, parent=None, on_progress=None):
        return ToolResult(data="ok")

    async def check_perms(args, ctx):
        if check_result is not None:
            return check_result
        return PermissionPassthroughDecision(message="check")

    return build_tool(
        name=name, description="x", prompt="x", input_schema=In,
        call=call, check_permissions=check_perms,
        max_result_size_chars=1000,
        map_tool_result_to_tool_result_block_param=lambda o, i: {},
        render_tool_use_message=lambda i, o: "",
    )


# append to test_permissions.py, in main()
async def _test_hard_deny():
    from permissions import has_permissions_to_use_tool
    from registry import find, register
    from state import AppState, FileStateCache, ToolPermissionContext
    from tool import ToolUseContext
    from pydantic import BaseModel
    from tool import build_tool, ToolResult
    from permission_types import PermissionPassthroughDecision

    class BashIn(BaseModel):
        command: str = ""

    async def call(args, ctx, can=None, parent=None, on_progress=None):
        return ToolResult(data={"stdout": "", "stderr": "", "code": 0})

    register(build_tool(
        name="Bash", description="x", prompt="x", input_schema=BashIn,
        call=call, max_result_size_chars=30_000,
        map_tool_result_to_tool_result_block_param=lambda o, i: {"content": ""},
        render_tool_use_message=lambda i, o: "",
        check_permissions=lambda a, c: _passthrough_decision(),
    ))

    state = AppState(
        messages=[],
        tool_permission_context=ToolPermissionContext(mode="bypassPermissions"),
        read_file_cache=FileStateCache(),
    )
    ctx = ToolUseContext(
        get_app_state=lambda: state, set_app_state=lambda f: None,
        read_file_state=state.read_file_cache,
    )

    # Hard-denied even in bypass mode
    tool = find("Bash")
    for cmd in ("rm -rf /", "rm -rf ~", "mkfs.ext4 /dev/sda", "git push --force origin main"):
        d = await has_permissions_to_use_tool(tool, {"command": cmd}, ctx)
        assert d.behavior == "deny", f"{cmd} should be hard-denied"
        assert "hard-coded safety rule" in d.message
    # Normal command passes through
    d = await has_permissions_to_use_tool(tool, {"command": "ls -la"}, ctx)
    assert d.behavior == "allow"
    print("✅ Hard deny patterns OK")

async def _passthrough_decision():
    from permission_types import PermissionPassthroughDecision
    return PermissionPassthroughDecision(message="ok")


async def main():
    await _test_hard_deny()
    
    # ── 1a. deny rule ──
    tool = make_tool()
    ctx = make_ctx(deny={"userSettings": ["Bash"]})
    result = await has_permissions_to_use_tool(tool, {"command": "ls"}, ctx)
    assert result.behavior == "deny"
    assert isinstance(result.decision_reason, RuleReason)
    print("✅ 1a. deny rule OK")

    # ── 1b. ask rule ──
    ctx = make_ctx(ask={"userSettings": ["Bash"]})
    result = await has_permissions_to_use_tool(tool, {"command": "ls"}, ctx)
    assert result.behavior == "ask"
    assert isinstance(result.decision_reason, RuleReason)
    print("✅ 1b. ask rule OK")

    # ── 1d. tool deny ──
    deny_result = PermissionDenyDecision(
        message="blocked", decision_reason=RuleReason(),
    )
    tool = make_tool(check_result=deny_result)
    ctx = make_ctx()
    result = await has_permissions_to_use_tool(tool, {"command": "ls"}, ctx)
    assert result.behavior == "deny"
    print("✅ 1d. tool deny OK")

    # ── 1g. safetyCheck ──
    safety = PermissionAskDecision(
        message="sensitive path",
        decision_reason=SafetyCheckReason(reason=".git/", classifier_approvable=True),
    )
    tool = make_tool(check_result=safety)
    ctx = make_ctx()
    result = await has_permissions_to_use_tool(tool, {"command": "ls"}, ctx)
    assert result.behavior == "ask"
    assert result.decision_reason.type == "safetyCheck"
    print("✅ 1g. safetyCheck OK")

    # ── 2a. bypass mode ──
    tool = make_tool(check_result=PermissionAskDecision(message="ask"))
    ctx = make_ctx(mode="bypassPermissions")
    result = await has_permissions_to_use_tool(tool, {"command": "ls"}, ctx)
    assert result.behavior == "allow"
    assert isinstance(result.decision_reason, ModeReason)
    print("✅ 2a. bypassPermissions OK")

    # ── 2b. always-allow rule ──
    ctx = make_ctx(allow={"userSettings": ["Bash"]})
    result = await has_permissions_to_use_tool(tool, {"command": "ls"}, ctx)
    assert result.behavior == "allow"
    assert isinstance(result.decision_reason, RuleReason)
    print("✅ 2b. always-allow OK")

    # ── 3. passthrough → ask ──
    tool = make_tool(check_result=PermissionPassthroughDecision(message="p"))
    ctx = make_ctx()
    result = await has_permissions_to_use_tool(tool, {"command": "ls"}, ctx)
    assert result.behavior == "ask"
    print("✅ 3. passthrough → ask OK")

    # ── dontAsk mode ──
    tool = make_tool(check_result=PermissionAskDecision(message="ask"))
    ctx = make_ctx(mode="dontAsk")
    result = await has_permissions_to_use_tool(tool, {"command": "ls"}, ctx)
    assert result.behavior == "deny"
    print("✅ dontAsk OK")

    # ── headless auto-deny ──
    state = AppState(
        messages=[],
        tool_permission_context=ToolPermissionContext(
            mode="default", should_avoid_permission_prompts=True,
        ),
    )
    ctx = ToolUseContext(
        get_app_state=lambda: state,
        set_app_state=lambda f: None,
    )
    tool = make_tool(check_result=PermissionAskDecision(message="ask"))
    result = await has_permissions_to_use_tool(tool, {"command": "ls"}, ctx)
    assert result.behavior == "deny"
    print("✅ headless auto-deny OK")

    # ── checkRuleBasedPermissions (BashTool pre-check) ──
    ctx = make_ctx(deny={"userSettings": ["Bash"]})
    r = await check_rule_based_permissions(tool, {"command": "ls"}, ctx)
    assert r is not None and r.behavior == "deny"
    ctx = make_ctx()  # no rules
    tool2 = make_tool(check_result=PermissionPassthroughDecision(message="p"))
    r = await check_rule_based_permissions(tool2, {"command": "ls"}, ctx)
    assert r is None
    print("✅ check_rule_based_permissions OK")

    # ── createPermissionRequestMessage for each reason ──
    assert "requires approval" in create_permission_request_message(
    "Bash", RuleReason(rule=PermissionRule("userSettings", "deny", PermissionRuleValue("Bash")))
).lower()

    assert "mode" in create_permission_request_message(
        "Bash", ModeReason(mode="plan")
    ).lower()
    assert "sensitive" in create_permission_request_message(
        "Bash", SafetyCheckReason(reason="sensitive file")
    ).lower()
    print("✅ createPermissionRequestMessage OK")

    print("\n🎉 All permissions.py tests passed")


asyncio.run(main())