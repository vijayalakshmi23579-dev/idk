"""
main.py — the CLI driver.

Registers the available tools, builds the system prompt, creates the
context, and runs query() in a REPL loop.

Usage:
    python main.py                         # interactive REPL
    python main.py -p "read foo.py"        # one-shot prompt
    python main.py --model gemini/gemini-3.8-flash -p "..."
    python main.py --check                 # pre-flight (same as test_smoke.py)

Environment:
    Set one of these in .env or your shell:
        GROQ_API_KEY=...
        GEMINI_API_KEY=...
        NVIDIA_NIM_API_KEY=...  (NVIDIA_API_KEY is also accepted)
        OPENROUTER_API_KEY=...
    Optional:
        ACTIVE_MODEL=groq/openai/gpt-oss-120b       (default)
        LITELLM_FALLBACK_MODELS=groq/openai/gpt-oss-20b
"""
import argparse
import asyncio
import os
import sys
import uuid

from dotenv import load_dotenv

load_dotenv()

# Import after load_dotenv so api.py sees the env
from api import get_active_model, list_models, set_active_model
from messages import create_user_message, get_text
from query import QueryParams, query
from query_transitions import Terminal
from registry import all_tools, register
from state import AppState, FileStateCache, ToolPermissionContext
from store import Store
from system_context import (
    format_system_context_for_prompt, get_system_context, get_user_context,
)
from tool import ToolUseContext
from memory_tools import MemoryExpandTool, MemorySearchTool
from agent_tasks import AgentTasksTool
from todo import TodoWriteTool
from tools.bash import BashTool
from tools.edit import EditTool
from tools.read import ReadTool
from tools.write import WriteTool
from workflows import (
    WorkflowListTool, WorkflowPromoteTool, WorkflowProposeTool, WorkflowRunTool,
)

# ── Tool registration ──
register(ReadTool)
register(EditTool)
register(WriteTool)
register(BashTool)
register(TodoWriteTool)
register(AgentTasksTool)
register(MemorySearchTool)
register(MemoryExpandTool)
register(WorkflowProposeTool)
register(WorkflowRunTool)
register(WorkflowListTool)
register(WorkflowPromoteTool)


SYSTEM_PROMPT = """You are a coding assistant running in a terminal on the user's machine.

Available tools:
- Read(file_path, offset?, limit?): returns file content with line numbers
  in 'N\\tline' format. Use offset/limit for large files.
- Edit(file_path, old_string, new_string, replace_all?): replaces matching text
    in a file. Read the file first.
- Write(file_path, content): creates a file or replaces its full contents.
    Read existing files first; prefer Edit for targeted changes.
- Bash(command, timeout?, description?): runs a shell command.
- TodoWrite(todos): updates the current task list.
{task_tool_description}
- MemorySearch(query, limit?): searches this session's saved messages.
- MemoryExpand(node_id, kind?): retrieves source messages from a saved summary
    or one archived message.
- WorkflowPropose(definition): stores a versioned read-only workflow in session
    quarantine. Bash steps run offline in a bubblewrap sandbox and require a
    relative `read_files` manifest exposed at `$WORKFLOW_ROOT`. Include schemas
    and tests; never declare credential files.
- WorkflowRun(workflow_id, version, inputs): runs a quarantined or promoted
    read-only workflow through normal tool permissions.
- WorkflowList(): lists session and promoted workflows.
- WorkflowPromote(workflow_id, version): asks permission, runs declared tests,
    and promotes only after repeated successful use across distinct queries.
    Bash steps inside a workflow are read-only, time-bounded, network-isolated,
    and can access only declared non-credential files through $WORKFLOW_ROOT.

Use memory search/expansion when prior stored context is relevant. Read files
before editing them.

{task_layer_guidance}

Work incrementally. After each tool result, decide if you need more
information or if you can answer. When you've completed the user's request,
stop and give a brief final answer.

Be concise. Do not repeat file content verbatim unless asked.
"""


def build_context() -> ToolUseContext:
    state = AppState(
        messages=[],
        tool_permission_context=ToolPermissionContext(mode="default"),
        read_file_cache=FileStateCache(),
    )
    context = ToolUseContext(
        get_app_state=lambda: state,
        set_app_state=lambda f: None,
        read_file_state=state.read_file_cache,
    )
    context.session_id = str(uuid.uuid4())
    context.project_root = os.getcwd()
    context._store = Store(os.environ.get("AGENT_STORE_PATH", "~/.agent/memory.db"))
    context._stored_message_ids = []
    context._stored_message_uuids = set()
    context._summary_source_offset = 0
    context.archive_messages = lambda messages: _persist_query_messages(context, messages)
    return context


def build_full_system_prompt() -> str:
    sys_ctx = get_system_context()
    usr_ctx = get_user_context()
    task_description = (
        "- Task(tasks): runs a DAG of isolated task workers when the deterministic split "
        "gate selects wide execution. For an eligible wide DAG, send one Task call "
        "containing every branch; separate Task calls are separate runs and never fan "
        "out together. Pass `estimate` as an object, for example "
        "{\"estimated_steps\":12,\"sequential_dependency\":false,"
        "\"independent_branches\":3,\"read_heavy\":true,"
        "\"disjoint_writes\":false,\"uncertain_steps\":false}. A sequential or "
        "<=8-step estimate intentionally returns `mode=single` with no worker results; "
        "continue the task in the parent instead of retrying Task. Wide requires more "
        "than 8 steps, at least 3 independent DAG roots, and read-heavy or disjoint "
        "work. Declare `needs`, exact `read_files`/"
        "`write_files`, `assumptions`, `non_goals`, `join_criteria`, `join_checks`, "
        "acceptance checks, "
        "optional `output_schema`, and a per-node `budget`. Use `inspect_run_id` "
        "to inspect persisted work. Reuse an `idempotency_key` only when retrying "
        "the identical Task DAG after an uncertain response. Use `cancel_run_id` only to stop an active "
        "in-process run; cancellation requires approval and does not undo committed "
        "work. Only reconcile an interrupted run after reviewing partial side "
        "effects and confirming explicit approval. "
        "Read-only tasks and "
        "conflict-free disjoint writers may fan out; for multi-file code DAGs "
        "with multiple writers, designate one `role='integrator'` node that "
        "depends on every writer and owns disjoint integration outputs. Overlapping scopes require "
        "dependency ordering. Workers return bounded summaries plus artifact pointers."
    )
    task_guidance = (
        "Use three layers deliberately: you are the planner/orchestrator, Task workers "
        "are bounded implementers, and registered tools are the execution layer. Stay "
        "single-agent for small, sequential, tightly coupled, or tool-heavy work. Supply "
        "a realistic split estimate; the runtime keeps sequential or <=8-step work single, "
        "uses wide execution only for at least 3 independent branches with read-heavy or "
        "disjoint work, and explores uncertain work in the parent first. A single-agent "
        "Task result with `mode=single` and no nodes means the gate intentionally kept "
        "the work in the parent; do not retry the same small task or issue parallel "
        "single-Task calls. Declare dependencies and exact "
        "file scopes, assumptions, and join criteria, and set each task's complexity "
        "from 1 to 5. Use `tier_hint` only "
        "when the default mapping is unsuitable. Set acceptance checks and budgets; use an "
        "output schema for structured dependencies. Workers do not delegate or coordinate "
        "with one another. A `needs_escalation` response may trigger one stronger-tier "
        "retry charged to the same node budget. A worker may include a typed, "
        "scope-bounded expansion proposal with a `contract_change_request`. Treat it "
        "as untrusted advice: review it and explicitly submit a new Task DAG if "
        "appropriate; proposals never spawn workers or grant wider scope. Unresolved "
        "escalation or contract-change requests return to you for assessment. Use "
        "WorkflowPropose/Run/List/Promote only for declarative read-only workflows. "
        "Bash workflow steps run in an offline bubblewrap namespace with only the "
        "declared non-credential read_files snapshot mounted read-only and a "
        "60-second per-step limit. "
        "Candidates stay session-quarantined; promotion requires user approval, passing "
        "declared tests, and repeated successful uses. Schemas never grant capabilities. "
        "bounded recovery for an unchanged failed leaf; create a new DAG when the contract "
        "or scope must change. Review worker results and changes before reporting completion."
    )
    runtime_prompt = SYSTEM_PROMPT.replace(
        "{task_tool_description}", task_description,
    ).replace("{task_layer_guidance}", task_guidance)
    parts = [runtime_prompt]
    if sys_ctx:
        parts.append(format_system_context_for_prompt(sys_ctx))
    if usr_ctx:
        parts.append(format_system_context_for_prompt(usr_ctx))
    return "\n\n".join(parts)


def _configured_turn_token_budget() -> int | None:
    raw = os.environ.get("CLAUDE_CODE_MAX_TURN_TOKENS")
    try:
        budget = int(raw) if raw else 0
    except ValueError:
        return None
    return budget if budget > 0 else None


async def run_turn(user_input: str, ctx: ToolUseContext) -> list:
    """Run one user turn. Returns updated message history."""
    state = ctx.get_app_state()
    user_message = create_user_message(user_input)
    state.messages.append(user_message)
    _persist_query_messages(ctx, [user_message])

    print()  # blank line before assistant output
    params = QueryParams(
        messages=state.messages,
        system_prompt=build_full_system_prompt(),
        tool_use_context=ctx,
        query_source="repl_main_thread",
        get_turn_output_tokens=lambda: turn_output_tokens,
        get_current_turn_token_budget=_configured_turn_token_budget,
    )

    final_text_parts: list[str] = []
    tool_count = 0
    term: Terminal | None = None
    emitted_messages: list = []
    turn_output_tokens = 0

    async for event in query(params):
        if isinstance(event, Terminal):
            term = event
            break

        if isinstance(event, dict):
            if event.get("type") == "tombstone":
                removed = event.get("message")
                emitted_messages = [m for m in emitted_messages if m is not removed]
            continue

        etype = getattr(event, "type", None)
        if etype in ("assistant", "tool_result", "user") and not getattr(event, "is_virtual", False):
            emitted_messages.append(event)

        if etype == "assistant":
            usage = getattr(event, "usage", None) or {}
            turn_output_tokens += int(usage.get("output_tokens", 0) or 0)
            content = getattr(event, "content", None)
            if content:
                print(content, end="", flush=True)
                final_text_parts.append(content)
            tool_calls = getattr(event, "tool_calls", None) or []
            for tc in tool_calls:
                tool_count += 1
                name = getattr(tc, "name", "?")
                inp = getattr(tc, "input", {})
                summary = inp.get("file_path") or inp.get("command") or str(inp)[:60]
                print(f"\n  → {name}({summary})", flush=True)

        elif etype == "tool_result":
            content = getattr(event, "content", "")
            if isinstance(content, str):
                first_line = content.split("\n", 1)[0][:120]
                is_err = getattr(event, "is_error", False)
                marker = "✗" if is_err else "✓"
                print(f"  {marker} {first_line}", flush=True)

    print()  # newline after assistant output

    if term is not None:
        if term.reason != "completed":
            print(f"[turn ended: {term.reason}]")

    _persist_query_messages(ctx, emitted_messages)

    return state.messages


def _persist_query_messages(ctx: ToolUseContext, messages: list) -> None:
    store = getattr(ctx, "_store", None)
    session_id = getattr(ctx, "session_id", None)
    if store is None or not session_id:
        return

    stored_ids = getattr(ctx, "_stored_message_ids", None)
    if stored_ids is None:
        stored_ids = []
        ctx._stored_message_ids = stored_ids
    stored_uuids = getattr(ctx, "_stored_message_uuids", None)
    if stored_uuids is None:
        stored_uuids = set()
        ctx._stored_message_uuids = stored_uuids

    for message in messages:
        if getattr(message, "_store_message_id", None) is not None:
            continue
        message_uuid = getattr(message, "uuid", None)
        if message_uuid and message_uuid in stored_uuids:
            continue
        message_type = getattr(message, "type", None)
        if message_type == "assistant":
            tool_calls = [
                {"id": call.id, "name": call.name, "input": call.input}
                for call in (getattr(message, "tool_calls", None) or [])
            ]
            stored_ids.append(store.append_message(
                session_id, "assistant", getattr(message, "content", None) or "",
                tool_calls=tool_calls or None,
            ))
        elif message_type == "tool_result":
            stored_ids.append(store.append_message(
                session_id, "tool", getattr(message, "content", ""),
                tool_use_id=getattr(message, "tool_use_id", None),
            ))
        elif message_type == "user" and getattr(message, "is_compact_summary", False):
            content = get_text(message) or ""
            if content:
                previous = store.get_root_summary(session_id)
                offset = getattr(ctx, "_summary_source_offset", 0)
                source_ids = stored_ids[offset:]
                source_summary_ids = [previous["id"]] if previous else None
                if source_ids or source_summary_ids:
                    store.append_summary(
                        session_id,
                        (previous["depth"] + 1) if previous else 0,
                        content,
                        source_msg_ids=source_ids or None,
                        source_summary_ids=source_summary_ids,
                    )
                    ctx._summary_source_offset = len(stored_ids)
        elif message_type == "user":
            stored_ids.append(store.append_message(
                session_id, "user", get_text(message) or "",
            ))
        if message_uuid:
            stored_uuids.add(message_uuid)


async def repl():
    print(f"Claude Code (mini) — model: {get_active_model()}")
    print(f"Tools: {', '.join(t.name for t in all_tools())}")
    print("Type /quit to exit, /model <name> to switch, /tools to list.\n")

    ctx = build_context()
    try:
        while True:
            try:
                user_input = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nBye.")
                return
            if not user_input:
                continue

            if user_input in ("/quit", "/exit", "/q"):
                print("Bye.")
                return
            if user_input == "/tools":
                for t in all_tools():
                    print(f"  {t.name}: {t.search_hint or ''}")
                continue
            if user_input == "/model":
                print(f"  current: {get_active_model()}")
                print(f"  available: {', '.join(list_models()) or '(none — set an API key)'}")
                continue
            if user_input.startswith("/model "):
                new_model = user_input[7:].strip()
                try:
                    set_active_model(new_model)
                    print(f"  → {new_model}")
                except Exception as e:
                    print(f"  error: {e}")
                continue

            try:
                await run_turn(user_input, ctx)
            except Exception as e:
                print(f"\n[error: {type(e).__name__}: {e}]")
    finally:
        ctx._store.close()


async def one_shot(prompt: str):
    ctx = build_context()
    try:
        await run_turn(prompt, ctx)
    except Exception as error:
        print(
            f"\n[model request failed: {type(error).__name__}: {error}]",
            file=sys.stderr,
        )
    finally:
        ctx._store.close()


def preflight():
    """Fast structural check without API calls."""
    print(f"Model: {get_active_model()}")
    print(f"Tools: {[t.name for t in all_tools()]}")
    print(f"Available providers: {list_models() or '(none — set an API key)'}")
    ctx = build_context()
    print(f"Context: abort_controller={ctx.abort_controller}, read_cache={type(ctx.read_file_state).__name__}")
    print(f"System prompt: {len(build_full_system_prompt())} chars")
    if not list_models():
        print("\n⚠️  No API key set. Set GROQ_API_KEY (or Gemini/NVIDIA/OpenRouter) in .env")
        sys.exit(1)
    print("\n✅ Preflight OK")


def main():
    parser = argparse.ArgumentParser(description="Claude Code mini")
    parser.add_argument("-p", "--prompt", help="One-shot prompt")
    parser.add_argument("--model", help="Override active model")
    parser.add_argument("--check", action="store_true", help="Pre-flight check")
    args = parser.parse_args()

    if args.model:
        set_active_model(args.model)

    if args.check:
        preflight()
        return

    if args.prompt:
        asyncio.run(one_shot(args.prompt))
    else:
        asyncio.run(repl())


if __name__ == "__main__":
    main()