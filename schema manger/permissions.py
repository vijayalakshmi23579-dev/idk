"""
permissions.py — the 7-stage permission gauntlet.

Extracted from src/utils/permissions/permissions.ts (385 lines of source,
~470 lines in Python with full type handling and no analytics).

Two entry points:
  - has_permissions_to_use_tool()  — the full gauntlet + mode transformations
  - check_rule_based_permissions() — the rule-only subset (used by BashTool's
                                      own checkPermissions to pre-check rules)

The 7 stages (inner), each bypass-immune unless marked:
  1a. deny rule             [BYPASS-IMMUNE]
  1b. ask rule              (sandbox auto-allow can skip)
  1c. tool.check_permissions
  1d. tool deny             [BYPASS-IMMUNE]
  1e. requiresUserInteraction [BYPASS-IMMUNE]
  1f. content ask rule      [BYPASS-IMMUNE]
  1g. safetyCheck           [BYPASS-IMMUNE]
  2a. bypassPermissions mode
  2b. always-allow rule
  3.  passthrough → ask

Then (outer):
  - dontAsk mode       : ask → deny
  - auto mode          : ask → classifier (acceptEdits fast-path, allowlist,
                         then two-stage LLM)
  - headless prompts   : ask → PermissionRequest hooks, then auto-deny
"""
from __future__ import annotations

import logging
from typing import Any, Awaitable, Callable

from permission_types import (
    PermissionAllowDecision, PermissionAskDecision, PermissionDecision,
    PermissionDecisionReason, PermissionDenyDecision, PermissionMode,
    PermissionPassthroughDecision, PermissionResult, PermissionUpdate,
    RuleReason, ModeReason, SubcommandResultsReason, PermissionPromptToolReason,
    HookReason, AsyncAgentReason, SandboxOverrideReason, ClassifierReason,
    WorkingDirReason, SafetyCheckReason, OtherReason,
    permission_mode_title, PermissionResult as make_result,
)
from permission_rules import (
    PermissionRule, RULE_SOURCES,
    parse_rule, rule_value_to_string,
    tool_matches_rule_value, filter_denied_agent_types,
)
from denial_tracking import (
    DenialTrackingState, DENIAL_LIMITS,
    create_denial_tracking_state, record_denial, record_success,
    should_fallback_to_prompting, hit_total_limit, reset_denials,
    apply_in_place,
)
from path_safety import check_path_safety_for_auto_edit
from state import AppState, ToolPermissionContext
from tool import Tool, ToolUseContext


logger = logging.getLogger(__name__)

# permissions.py — add near the top, check before ALL stages
HARD_DENY_PATTERNS: tuple[tuple[str, str], ...] = (
    # (tool_name, pattern) — pattern is glob or "exact:<string>" or "prefix:<str>"
    ("Bash", "exact:rm -rf /"),
    ("Bash", "exact:rm -rf /*"),
    ("Bash", "exact:rm -rf ~"),
    ("Bash", "prefix:mkfs"),
    ("Bash", "prefix:dd if="),
    ("Bash", "exact:chmod -R 777 /"),
    ("Bash", "prefix:git push --force origin main"),
    ("Bash", "exact::(){ :|:& };:"),
    ("Edit", "glob:~/.ssh/**"),
    ("Edit", "glob:~/.aws/**"),
    ("Edit", "glob:~/.bashrc"),
    ("Edit", "glob:~/.zshrc"),
    ("Edit", "glob:~/.profile"),
    ("Write", "glob:/etc/**"),
    ("Write", "glob:~/.ssh/**"),
)

import fnmatch, os

def _expand(p: str) -> str:
    return os.path.expanduser(p) if p.startswith("~") else p

def _matches_hard_deny(tool_name: str, input: dict) -> str | None:
    """Return the matched pattern label, or None."""
    if tool_name == "Bash":
        cmd = (input.get("command") or "").strip()
        for tn, pat in HARD_DENY_PATTERNS:
            if tn != "Bash":
                continue
            if pat.startswith("exact:"):
                if cmd == pat[6:]:
                    return pat
            elif pat.startswith("prefix:"):
                if cmd.startswith(pat[7:]):
                    return pat
            elif pat.startswith("glob:"):
                if fnmatch.fnmatchcase(cmd, pat[5:]):
                    return pat
    elif tool_name in ("Edit", "Write"):
        path = _expand(input.get("file_path") or "")
        for tn, pat in HARD_DENY_PATTERNS:
            if tn != tool_name:
                continue
            if pat.startswith("glob:"):
                if fnmatch.fnmatchcase(path, _expand(pat[5:])):
                    return pat
    return None

# ── Classifier injection (built in classifier.py, file #5) ───────────────
# Falls back to "unavailable" if classifier.py doesn't exist yet. This
# lets permissions.py be tested before classifier.py is built.
try:
    from classifier import classify_yolo_action, format_action_for_classifier
except ImportError:
    async def classify_yolo_action(*args, **kwargs):
        return {
            "should_block": False,
            "reason": "classifier not available",
            "unavailable": True,
            "model": "none",
        }

    def format_action_for_classifier(tool_name: str, input: dict) -> str:
        return f"{tool_name}({input})"


# ── Rule accessors ────────────────────────────────────────────────────────

def _rules_by_behavior(context: ToolPermissionContext, behavior: str) -> list[PermissionRule]:
    """Flatten rules from all sources into a list of PermissionRule objects."""
    source_map = {
        "allow": context.always_allow_rules,
        "deny": context.always_deny_rules,
        "ask": context.always_ask_rules,
    }.get(behavior, {})

    result: list[PermissionRule] = []
    for source in RULE_SOURCES:
        for rule_string in source_map.get(source, []):
            try:
                value = parse_rule(rule_string)
            except ValueError:
                logger.warning("Ignoring invalid %s rule: %r", behavior, rule_string)
                continue
            result.append(PermissionRule(
                source=source,
                rule_behavior=behavior,
                rule_value=value,
            ))
    return result


def get_allow_rules(context: ToolPermissionContext) -> list[PermissionRule]:
    return _rules_by_behavior(context, "allow")


def get_deny_rules(context: ToolPermissionContext) -> list[PermissionRule]:
    return _rules_by_behavior(context, "deny")


def get_ask_rules(context: ToolPermissionContext) -> list[PermissionRule]:
    return _rules_by_behavior(context, "ask")


def _find_tool_level_rule(
    rules: list[PermissionRule], tool_name: str
) -> PermissionRule | None:
    """First rule whose rule_value matches the tool at the tool level."""
    for rule in rules:
        if tool_matches_rule_value(tool_name, rule.rule_value):
            return rule
    return None


def get_deny_rule_for_tool(context: ToolPermissionContext, tool_name: str) -> PermissionRule | None:
    return _find_tool_level_rule(get_deny_rules(context), tool_name)


def get_ask_rule_for_tool(context: ToolPermissionContext, tool_name: str) -> PermissionRule | None:
    return _find_tool_level_rule(get_ask_rules(context), tool_name)


def tool_always_allowed_rule(context: ToolPermissionContext, tool_name: str) -> PermissionRule | None:
    return _find_tool_level_rule(get_allow_rules(context), tool_name)


def get_deny_rule_for_agent(
    context: ToolPermissionContext, agent_tool_name: str, agent_type: str
) -> PermissionRule | None:
    """Specific agent denied via Agent(agentType) syntax."""
    for rule in get_deny_rules(context):
        if (
            rule.rule_value.tool_name == agent_tool_name
            and rule.rule_value.rule_content == agent_type
        ):
            return rule
    return None


def filter_denied_agents(
    agents: list[dict], context: ToolPermissionContext, agent_tool_name: str
) -> list[dict]:
    denied = filter_denied_agent_types(get_deny_rules(context), agent_tool_name)
    return [a for a in agents if a.get("agent_type") not in denied]


def get_rule_by_contents_for_tool(
    context: ToolPermissionContext, tool_name: str, behavior: str
) -> dict[str, PermissionRule]:
    """Map rule_content → PermissionRule for a specific tool + behavior."""
    source_rules = {
        "allow": get_allow_rules,
        "deny": get_deny_rules,
        "ask": get_ask_rules,
    }.get(behavior)
    if source_rules is None:
        return {}
    result: dict[str, PermissionRule] = {}
    for rule in source_rules(context):
        if (
            rule.rule_value.tool_name == tool_name
            and rule.rule_value.rule_content is not None
            and rule.rule_behavior == behavior
        ):
            result[rule.rule_value.rule_content] = rule
    return result


# ── Permission request message ────────────────────────────────────────────

def create_permission_request_message(
    tool_name: str,
    decision_reason: PermissionDecisionReason | None = None,
) -> str:
    """Human-readable explanation of why we're asking.

    Exact port of createPermissionRequestMessage. Handles every
    PermissionDecisionReason variant.
    """
    if decision_reason is not None:
        rtype = decision_reason.type

        if rtype == "classifier":
            return (
                f"Classifier '{decision_reason.classifier}' requires approval "
                f"for this {tool_name} command: {decision_reason.reason}"
            )

        if rtype == "hook":
            if decision_reason.reason:
                return (
                    f"Hook '{decision_reason.hook_name}' blocked this action: "
                    f"{decision_reason.reason}"
                )
            return (
                f"Hook '{decision_reason.hook_name}' requires approval for "
                f"this {tool_name} command"
            )

        if rtype == "rule":
            rule_string = rule_value_to_string(decision_reason.rule.rule_value)
            return (
                f"Permission rule '{rule_string}' from "
                f"{decision_reason.rule.source} requires approval for this "
                f"{tool_name} command"
            )

        if rtype == "subcommandResults":
            needs: list[str] = []
            for cmd, result in decision_reason.reasons.items():
                behavior = getattr(result, "behavior", None)
                if behavior in ("ask", "passthrough"):
                    needs.append(cmd)
            if needs:
                n = len(needs)
                word = "part" if n == 1 else "parts"
                verb = "requires" if n == 1 else "require"
                return (
                    f"This {tool_name} command contains multiple operations. "
                    f"The following {n} {word} {verb} approval: "
                    f"{', '.join(needs)}"
                )
            return (
                f"This {tool_name} command contains multiple operations that "
                f"require approval"
            )

        if rtype == "permissionPromptTool":
            return (
                f"Tool '{decision_reason.permission_prompt_tool_name}' requires "
                f"approval for this {tool_name} command"
            )

        if rtype == "sandboxOverride":
            return "Run outside of the sandbox"

        if rtype == "workingDir":
            return decision_reason.reason

        if rtype == "safetyCheck":
            return decision_reason.reason

        if rtype == "other":
            return decision_reason.reason

        if rtype == "mode":
            title = permission_mode_title(decision_reason.mode)
            return (
                f"Current permission mode ({title}) requires approval for "
                f"this {tool_name} command"
            )

        if rtype == "asyncAgent":
            return decision_reason.reason

    return (
        f"Claude requested permissions to use {tool_name}, but you haven't "
        f"granted it yet."
    )


# ── Denial persistence (local vs appState) ───────────────────────────────

def _persist_denial_state(
    context: ToolUseContext, new_state: DenialTrackingState
) -> None:
    """Subagents use localDenialTracking (set_app_state is no-op).
    Main thread writes to appState.denial_tracking.
    """
    if context.local_denial_tracking is not None:
        apply_in_place(context.local_denial_tracking, new_state)
    else:
        def updater(prev: AppState) -> AppState:
            if getattr(prev, "denial_tracking", None) is new_state:
                return prev
            new_app = AppState(
                messages=prev.messages,
                tool_permission_context=prev.tool_permission_context,
                read_file_cache=prev.read_file_cache,
                tasks=prev.tasks,
                agent_name_registry=prev.agent_name_registry,
                total_cost_usd=prev.total_cost_usd,
                session_start_time=prev.session_start_time,
                turn_count=prev.turn_count,
                mode=prev.mode,
            )
            new_app.denial_tracking = new_state  # type: ignore[attr-defined]
            return new_app
        context.set_app_state(updater)


# ── Main gauntlet ─────────────────────────────────────────────────────────

async def _has_permissions_to_use_tool_inner(
    tool: Tool, input: dict, context: ToolUseContext
) -> PermissionDecision:
    """Stages 1a → 3. No mode transformation. The caller wraps this."""
    if getattr(context.abort_controller, "aborted", False):
        raise RuntimeError("Aborted")

    app_state = context.get_app_state()
    ctx = app_state.tool_permission_context

    # ── 1a. Tool-level deny rule (BYPASS-IMMUNE) ──
    deny_rule = get_deny_rule_for_tool(ctx, tool.name)
    if deny_rule is not None:
        return PermissionDenyDecision(
            message=f"Permission to use {tool.name} has been denied.",
            decision_reason=RuleReason(rule=deny_rule),
        )

    # ── 1b. Tool-level ask rule ──
    ask_rule = get_ask_rule_for_tool(ctx, tool.name)
    if ask_rule is not None:
        # Sandbox auto-allow exception: if Bash + sandbox + auto-allow +
        # should sandbox, skip the ask rule and let the tool decide.
        can_sandbox_auto_allow = _can_sandbox_auto_allow(tool, input)
        if not can_sandbox_auto_allow:
            return PermissionAskDecision(
                message=create_permission_request_message(
                    tool.name, RuleReason(rule=ask_rule)
                ),
                decision_reason=RuleReason(rule=ask_rule),
            )

    # ── 1c. Tool-specific permission check ──
    tool_result: PermissionResult = PermissionPassthroughDecision(
        message=create_permission_request_message(tool.name)
    )
    try:
        parsed = tool.input_schema(**input) if input else tool.input_schema()
        tool_result = await tool.check_permissions(parsed, context)
    except Exception as e:
        logger.error("Tool %s check_permissions raised: %s", tool.name, e)

    # ── 1d. Tool deny (BYPASS-IMMUNE) ──
    if tool_result.behavior == "deny":
        return tool_result

    # ── 1e. requiresUserInteraction (BYPASS-IMMUNE) ──
    if (
        getattr(tool, "requires_user_interaction", None)
        and tool.requires_user_interaction()
        and tool_result.behavior == "ask"
    ):
        return tool_result

    # ── 1f. Content-specific ask rule (BYPASS-IMMUNE) ──
    if tool_result.behavior == "ask":
        dr = getattr(tool_result, "decision_reason", None)
        if (
            dr is not None
            and dr.type == "rule"
            and getattr(dr.rule, "rule_behavior", None) == "ask"
        ):
            return tool_result

    # ── 1g. Safety check (BYPASS-IMMUNE) ──
    if tool_result.behavior == "ask":
        dr = getattr(tool_result, "decision_reason", None)
        if dr is not None and dr.type == "safetyCheck":
            return tool_result

    # ── 2a. Bypass permissions mode ──
    # Re-read app_state — mode may have changed during tool check.
    app_state = context.get_app_state()
    ctx = app_state.tool_permission_context
    should_bypass = (
        ctx.mode == "bypassPermissions"
        or (ctx.mode == "plan" and ctx.is_bypass_permissions_mode_available)
    )
    if should_bypass:
        return PermissionAllowDecision(
            updated_input=_updated_input(tool_result, input),
            decision_reason=ModeReason(mode=ctx.mode),
        )

    # ── 2b. Always-allow rule ──
    allow_rule = tool_always_allowed_rule(ctx, tool.name)
    if allow_rule is not None:
        return PermissionAllowDecision(
            updated_input=_updated_input(tool_result, input),
            decision_reason=RuleReason(rule=allow_rule),
        )

    # ── 3. Passthrough → ask ──
    if tool_result.behavior == "passthrough":
        return PermissionAskDecision(
            message=create_permission_request_message(
                tool.name, getattr(tool_result, "decision_reason", None)
            ),
            decision_reason=getattr(tool_result, "decision_reason", None),
            suggestions=getattr(tool_result, "suggestions", None),
            blocked_path=getattr(tool_result, "blocked_path", None),
            pending_classifier_check=getattr(tool_result, "pending_classifier_check", None),
        )

    return tool_result


def _updated_input(permission_result: PermissionResult, fallback: dict) -> dict:
    """Extract updatedInput, or fall back to the original input."""
    return getattr(permission_result, "updated_input", None) or fallback


def _can_sandbox_auto_allow(tool: Tool, input: dict) -> bool:
    """Sandbox auto-allow exception: only Bash + sandbox enabled + auto-allow."""
    if tool.name != "Bash":
        return False
    try:
        from path_safety import expand_tilde
        # Sandbox module is optional — no-op if not present
        from sandbox import SandboxManager  # type: ignore
        if not SandboxManager.is_sandboxing_enabled():
            return False
        if not SandboxManager.is_auto_allow_bash_if_sandboxed_enabled():
            return False
        # shouldUseSandbox is a Bash-specific helper — placeholder
        return True
    except ImportError:
        return False


# ── Outer wrapper: mode transformation + classifier ───────────────────────

async def has_permissions_to_use_tool(
    tool: Tool,
    input: dict,
    context: ToolUseContext,
    assistant_message: Any = None,
    tool_use_id: str = "",
) -> PermissionDecision:

    # ── HARD DENY — checked before every stage, bypass-immune ──
    matched = _matches_hard_deny(tool.name, input)
    if matched is not None:
        return PermissionDenyDecision(
            message=(
                f"This command matches a hard-coded safety rule ({matched}). "
                f"Even bypassPermissions mode cannot override this. If you "
                f"genuinely need to do this, run it manually outside the agent."
            ),
            decision_reason=OtherReason(reason=f"hard_deny:{matched}"),
        )

    """The full gauntlet. Wraps _inner with mode transformation."""
    result = await _has_permissions_to_use_tool_inner(tool, input, context)

    # Reset consecutive denials on any allowed tool use in auto mode.
    if result.behavior == "allow":
        app_state = context.get_app_state()
        if app_state.tool_permission_context.mode == "auto":
            current = context.local_denial_tracking or getattr(app_state, "denial_tracking", None)
            if current is not None and current.consecutive_denials > 0:
                _persist_denial_state(context, record_success(current))
        return result

    if result.behavior != "ask":
        return result

    # ── dontAsk: ask → deny ──
    app_state = context.get_app_state()
    if app_state.tool_permission_context.mode == "dontAsk":
        return PermissionDenyDecision(
            message=(
                f"Permission to use {tool.name} has been denied because "
                f"Claude Code is running in don't ask mode."
            ),
            decision_reason=ModeReason(mode="dontAsk"),
        )

    # ── auto mode: classifier path ──
    if app_state.tool_permission_context.mode == "auto":
        return await _apply_auto_mode_classifier(
            tool, input, context, assistant_message, tool_use_id, result
        )

    # ── headless: PermissionRequest hooks, then auto-deny ──
    if app_state.tool_permission_context.should_avoid_permission_prompts:
        hook_decision = await _run_headless_permission_hooks(
            tool, input, tool_use_id, context, result
        )
        if hook_decision is not None:
            return hook_decision
        return PermissionDenyDecision(
            message=(
                f"Permission to use {tool.name} has been denied. "
                f"IMPORTANT: You may attempt to accomplish this action using "
                f"other tools that might naturally be used. But you should not "
                f"attempt to work around this denial in malicious ways."
            ),
            decision_reason=AsyncAgentReason(
                reason="Permission prompts are not available in this context"
            ),
        )

    return result


async def _apply_auto_mode_classifier(
    tool: Tool,
    input: dict,
    context: ToolUseContext,
    assistant_message: Any,
    tool_use_id: str,
    ask_result: PermissionAskDecision,
) -> PermissionDecision:
    """The auto-mode classifier path. Extracted for readability."""
    app_state = context.get_app_state()
    ctx = app_state.tool_permission_context

    # Non-classifier-approvable safetyCheck → hard deny (never auto-approved).
    dr = ask_result.decision_reason
    if dr is not None and dr.type == "safetyCheck" and not dr.classifier_approvable:
        if ctx.should_avoid_permission_prompts:
            return PermissionDenyDecision(
                message=ask_result.message,
                decision_reason=AsyncAgentReason(
                    reason="Safety check requires interactive approval and "
                           "permission prompts are not available in this context"
                ),
            )
        return ask_result

    # requiresUserInteraction → return ask (never auto-approved).
    if getattr(tool, "requires_user_interaction", None) and tool.requires_user_interaction():
        return ask_result

    denial_state = (
        context.local_denial_tracking
        or getattr(app_state, "denial_tracking", None)
        or create_denial_tracking_state()
    )

    # ── acceptEdits fast-path ──
    if tool.name not in ("Agent", "REPL"):
        try:
            parsed = tool.input_schema(**input) if input else tool.input_schema()
            # Override mode to acceptEdits for the re-check
            from types import SimpleNamespace
            accept_ctx = SimpleNamespace(**{
                k: getattr(context, k)
                for k in dir(context)
                if not k.startswith("_") and k != "get_app_state"
            })
            accept_state = SimpleNamespace(
                messages=app_state.messages,
                tool_permission_context=SimpleNamespace(
                    mode="acceptEdits",
                    additional_working_directories=ctx.additional_working_directories,
                    always_allow_rules=ctx.always_allow_rules,
                    always_deny_rules=ctx.always_deny_rules,
                    always_ask_rules=ctx.always_ask_rules,
                    is_bypass_permissions_mode_available=ctx.is_bypass_permissions_mode_available,
                    should_avoid_permission_prompts=ctx.should_avoid_permission_prompts,
                ),
                read_file_cache=app_state.read_file_cache,
                tasks=app_state.tasks,
                agent_name_registry=app_state.agent_name_registry,
                total_cost_usd=app_state.total_cost_usd,
                session_start_time=app_state.session_start_time,
                turn_count=app_state.turn_count,
                mode="acceptEdits",
            )
            accept_ctx.get_app_state = lambda: accept_state
            accept_result = await tool.check_permissions(parsed, accept_ctx)
            if accept_result.behavior == "allow":
                _persist_denial_state(context, record_success(denial_state))
                return PermissionAllowDecision(
                    updated_input=getattr(accept_result, "updated_input", None) or input,
                    decision_reason=ModeReason(mode="auto"),
                )
        except Exception:
            pass  # Fall through to classifier

    # ── Safe-tool allowlist ──
    try:
        from classifier_decision import is_auto_mode_allowlisted_tool
        if is_auto_mode_allowlisted_tool(tool.name):
            _persist_denial_state(context, record_success(denial_state))
            return PermissionAllowDecision(
                updated_input=input,
                decision_reason=ModeReason(mode="auto"),
            )
    except ImportError:
        pass  # No allowlist module yet

    # ── Run the classifier ──
    action = format_action_for_classifier(tool.name, input)
    try:
        classifier_result = await classify_yolo_action(
            context.messages, action, context.options.tools, ctx,
            getattr(context.abort_controller, "signal", None),
        )
    except Exception as e:
        logger.error("Classifier raised: %s", e)
        classifier_result = {
            "should_block": False, "reason": str(e),
            "unavailable": True, "model": "none",
        }

    # transcriptTooLong → fall back to manual (or abort if headless)
    if classifier_result.get("transcript_too_long"):
        if ctx.should_avoid_permission_prompts:
            raise RuntimeError(
                "Agent aborted: auto mode classifier transcript exceeded "
                "context window in headless mode"
            )
        return PermissionAskDecision(
            message=ask_result.message,
            decision_reason=OtherReason(
                reason="Auto mode classifier transcript exceeded context "
                       "window — falling back to manual approval"
            ),
        )

    # unavailable → fail closed (deny) or fail open (ask)
    if classifier_result.get("unavailable"):
        # fail closed by default
        return PermissionDenyDecision(
            message=(
                f"Auto mode classifier unavailable. Wait briefly and try "
                f"again. Read-only operations still work."
            ),
            decision_reason=ClassifierReason(
                classifier="auto-mode", reason="Classifier unavailable"
            ),
        )

    # blocked → record denial, check limits, deny
    if classifier_result.get("should_block"):
        new_state = record_denial(denial_state)
        _persist_denial_state(context, new_state)
        if should_fallback_to_prompting(new_state):
            if ctx.should_avoid_permission_prompts:
                raise RuntimeError(
                    "Agent aborted: too many classifier denials in headless mode"
                )
            warning = (
                f"{new_state.total_denials} actions were blocked this session. "
                f"Please review the transcript before continuing."
                if hit_total_limit(new_state)
                else f"{new_state.consecutive_denials} consecutive actions "
                     f"were blocked. Please review the transcript before continuing."
            )
            if hit_total_limit(new_state):
                _persist_denial_state(context, reset_denials(new_state))
            return PermissionAskDecision(
                message=warning + f"\n\nLatest blocked action: {classifier_result.get('reason', '')}",
                decision_reason=ClassifierReason(
                    classifier="auto-mode",
                    reason=warning,
                ),
            )
        return PermissionDenyDecision(
            message=(
                f"Permission for this action has been denied. Reason: "
                f"{classifier_result.get('reason', '')}. If you have other "
                f"tasks that don't depend on this action, continue working "
                f"on those."
            ),
            decision_reason=ClassifierReason(
                classifier="auto-mode",
                reason=classifier_result.get("reason", ""),
            ),
        )

    # Allowed by classifier
    _persist_denial_state(context, record_success(denial_state))
    return PermissionAllowDecision(
        updated_input=input,
        decision_reason=ClassifierReason(
            classifier="auto-mode",
            reason=classifier_result.get("reason", ""),
        ),
    )


async def _run_headless_permission_hooks(
    tool: Tool, input: dict, tool_use_id: str,
    context: ToolUseContext, ask_result: PermissionAskDecision,
) -> PermissionDecision | None:
    """Run PermissionRequest hooks for headless agents. Stub — full impl later."""
    # TODO: implement execute_permission_request_hooks when hooks module exists
    return None


# ── Rule-only subset (for BashTool's own pre-check) ──────────────────────

async def check_rule_based_permissions(
    tool: Tool, input: dict, context: ToolUseContext
) -> PermissionAskDecision | PermissionDenyDecision | None:
    """The rule-based steps only. Used by BashTool.checkPermissions to
    pre-check before delegating to the gauntlet. Returns None if no rule
    objects.

    Exact port of checkRuleBasedPermissions. Does NOT run the classifier,
    dontAsk/auto transformations, headless hooks, or bypass/allow rules.
    """
    app_state = context.get_app_state()
    ctx = app_state.tool_permission_context

    # 1a. deny rule
    deny_rule = get_deny_rule_for_tool(ctx, tool.name)
    if deny_rule is not None:
        return PermissionDenyDecision(
            message=f"Permission to use {tool.name} has been denied.",
            decision_reason=RuleReason(rule=deny_rule),
        )

    # 1b. ask rule
    ask_rule = get_ask_rule_for_tool(ctx, tool.name)
    if ask_rule is not None:
        if not _can_sandbox_auto_allow(tool, input):
            return PermissionAskDecision(
                message=create_permission_request_message(
                    tool.name, RuleReason(rule=ask_rule)
                ),
                decision_reason=RuleReason(rule=ask_rule),
            )

    # 1c. tool check
    tool_result: PermissionResult = PermissionPassthroughDecision(
        message=create_permission_request_message(tool.name)
    )
    try:
        parsed = tool.input_schema(**input) if input else tool.input_schema()
        tool_result = await tool.check_permissions(parsed, context)
    except Exception as e:
        logger.error("Tool %s check_permissions raised: %s", tool.name, e)

    # 1d. tool deny
    if tool_result.behavior == "deny":
        return tool_result

    # 1f. content ask rule
    if tool_result.behavior == "ask":
        dr = getattr(tool_result, "decision_reason", None)
        if (
            dr is not None
            and dr.type == "rule"
            and getattr(dr.rule, "rule_behavior", None) == "ask"
        ):
            return tool_result

    # 1g. safetyCheck
    if tool_result.behavior == "ask":
        dr = getattr(tool_result, "decision_reason", None)
        if dr is not None and dr.type == "safetyCheck":
            return tool_result

    return None