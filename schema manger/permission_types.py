"""
permission_types.py — the full permission type hierarchy.

Extracted from src/types/permissions.ts. Every decision variant, every
decision reason variant, every mode — exact port, no approximations.

A PermissionResult is one of:
  - PermissionAllowDecision   {behavior: 'allow', ...}
  - PermissionAskDecision     {behavior: 'ask', message, ...}
  - PermissionDenyDecision    {behavior: 'deny', message, decisionReason}
  - PermissionPassthroughDecision  {behavior: 'passthrough', ...}

Tools return PermissionResult from check_permissions(). The gauntlet
in permissions.py consumes them and produces the final PermissionDecision.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Union


# ── Modes (exact from types/permissions.ts) ──────────────────────────────

ExternalPermissionMode = Literal[
    "acceptEdits",
    "bypassPermissions",
    "default",
    "dontAsk",
    "plan",
]

InternalPermissionMode = Literal[
    "acceptEdits",
    "bypassPermissions",
    "default",
    "dontAsk",
    "plan",
    "auto",    # ant-only, feature-gated behind TRANSCRIPT_CLASSIFIER
    "bubble",  # internal
]

PermissionMode = InternalPermissionMode

EXTERNAL_PERMISSION_MODES: tuple[ExternalPermissionMode, ...] = (
    "acceptEdits", "bypassPermissions", "default", "dontAsk", "plan",
)

INTERNAL_PERMISSION_MODES: tuple[PermissionMode, ...] = (
    *EXTERNAL_PERMISSION_MODES, "auto",
)

PERMISSION_MODE_TITLES: dict[PermissionMode, str] = {
    "default": "Default",
    "plan": "Plan Mode",
    "acceptEdits": "Accept edits",
    "bypassPermissions": "Bypass Permissions",
    "dontAsk": "Don't Ask",
    "auto": "Auto mode",
    "bubble": "Bubble",
}


def permission_mode_title(mode: PermissionMode) -> str:
    return PERMISSION_MODE_TITLES.get(mode, "Default")


# ── Behavior ──────────────────────────────────────────────────────────────

PermissionBehavior = Literal["allow", "deny", "ask"]


# ── Rule source (exact ordering from types/permissions.ts) ───────────────

PermissionRuleSource = Literal[
    "userSettings",
    "projectSettings",
    "localSettings",
    "flagSettings",
    "policySettings",
    "cliArg",
    "command",
    "session",
]


# ── Permission update (for suggestions) ───────────────────────────────────

PermissionUpdateDestination = Literal[
    "userSettings", "projectSettings", "localSettings", "session", "cliArg",
]


@dataclass
class PermissionUpdate:
    """One update operation. `type` determines which fields are used."""
    type: Literal[
        "addRules", "replaceRules", "removeRules",
        "setMode", "addDirectories", "removeDirectories",
    ]
    destination: PermissionUpdateDestination
    rules: list[Any] | None = None        # PermissionRuleValue list
    behavior: PermissionBehavior | None = None
    mode: ExternalPermissionMode | None = None
    directories: list[str] | None = None


# ── Decision reasons (11 variants, exact from source) ────────────────────

@dataclass
class RuleReason:
    type: Literal["rule"] = "rule"
    rule: Any = None   # PermissionRule


@dataclass
class ModeReason:
    type: Literal["mode"] = "mode"
    mode: PermissionMode = "default"


@dataclass
class SubcommandResultsReason:
    type: Literal["subcommandResults"] = "subcommandResults"
    reasons: dict[str, Any] = field(default_factory=dict)   # cmd → PermissionResult


@dataclass
class PermissionPromptToolReason:
    type: Literal["permissionPromptTool"] = "permissionPromptTool"
    permission_prompt_tool_name: str = ""
    tool_result: Any = None


@dataclass
class HookReason:
    type: Literal["hook"] = "hook"
    hook_name: str = ""
    hook_source: str | None = None
    reason: str | None = None


@dataclass
class AsyncAgentReason:
    type: Literal["asyncAgent"] = "asyncAgent"
    reason: str = ""


@dataclass
class SandboxOverrideReason:
    type: Literal["sandboxOverride"] = "sandboxOverride"
    reason: Literal["excludedCommand", "dangerouslyDisableSandbox"] = "excludedCommand"


@dataclass
class ClassifierReason:
    type: Literal["classifier"] = "classifier"
    classifier: str = ""
    reason: str = ""


@dataclass
class WorkingDirReason:
    type: Literal["workingDir"] = "workingDir"
    reason: str = ""


@dataclass
class SafetyCheckReason:
    type: Literal["safetyCheck"] = "safetyCheck"
    reason: str = ""
    classifier_approvable: bool = True


@dataclass
class OtherReason:
    type: Literal["other"] = "other"
    reason: str = ""


PermissionDecisionReason = Union[
    RuleReason, ModeReason, SubcommandResultsReason, PermissionPromptToolReason,
    HookReason, AsyncAgentReason, SandboxOverrideReason, ClassifierReason,
    WorkingDirReason, SafetyCheckReason, OtherReason,
]


# ── Decisions ─────────────────────────────────────────────────────────────

@dataclass
class PermissionAllowDecision:
    behavior: Literal["allow"] = "allow"
    updated_input: dict | None = None
    user_modified: bool | None = None
    decision_reason: PermissionDecisionReason | None = None
    tool_use_id: str | None = None
    accept_feedback: str | None = None
    content_blocks: list | None = None


@dataclass
class PermissionAskDecision:
    behavior: Literal["ask"] = "ask"
    message: str = ""
    updated_input: dict | None = None
    decision_reason: PermissionDecisionReason | None = None
    suggestions: list[PermissionUpdate] | None = None
    blocked_path: str | None = None
    metadata: dict | None = None
    is_bash_security_check_for_misparsing: bool | None = None
    pending_classifier_check: dict | None = None
    content_blocks: list | None = None


@dataclass
class PermissionDenyDecision:
    behavior: Literal["deny"] = "deny"
    message: str = ""
    decision_reason: PermissionDecisionReason = field(default_factory=OtherReason)
    tool_use_id: str | None = None


@dataclass
class PermissionPassthroughDecision:
    behavior: Literal["passthrough"] = "passthrough"
    message: str = ""
    decision_reason: PermissionDecisionReason | None = None
    suggestions: list[PermissionUpdate] | None = None
    blocked_path: str | None = None
    pending_classifier_check: dict | None = None


PermissionDecision = Union[
    PermissionAllowDecision, PermissionAskDecision, PermissionDenyDecision,
]

PermissionResult = Union[PermissionDecision, PermissionPassthroughDecision]


# ── Factory (backward-compat with old PermissionResult(behavior=...) style) ─

def PermissionResult(behavior: str, **kwargs) -> PermissionResult:  # noqa: N802
    """Factory: PermissionResult('allow', updated_input={...}) → AllowDecision.

    Keeps the old constructor style working while exposing the full type
    hierarchy. Tools that want fine control can construct the decision
    dataclasses directly.
    """
    if behavior == "allow":
        return PermissionAllowDecision(**kwargs)
    if behavior == "ask":
        return PermissionAskDecision(**kwargs)
    if behavior == "deny":
        return PermissionDenyDecision(**kwargs)
    if behavior == "passthrough":
        return PermissionPassthroughDecision(**kwargs)
    raise ValueError(f"Unknown permission behavior: {behavior!r}")