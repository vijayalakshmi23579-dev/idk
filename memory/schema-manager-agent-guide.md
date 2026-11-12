# Schema Manager Agent Guide

Last updated: 2026-11-12. This guide describes the checked-in Python runtime and the current evidence state. For the active roadmap, see `memory/plan_mode.md`; for a compact status snapshot, see `memory/schema-manager-overview.md`.

## 1. Purpose and design

Schema Manager is a provider-agnostic coding-agent runtime for one local user. It is not a Claude Code source replica. Its architecture has three practical layers:

- **Planner/orchestrator:** the parent model receives the user request, chooses single-agent or Task-DAG work, declares task contracts and dependencies, integrates results, and reports completion.
- **Isolated workers:** Task nodes receive fresh query/message state, configured model tiers, bounded budgets, exact file scopes, and dependency outputs. They do not chat with one another or edit the parent's plan.
- **Deterministic execution:** registered tools, schemas, permissions, path scopes, scheduling, budgets, acceptance checks, persistence, and workflow execution are enforced by Python code.

Keep single-agent execution as the fallback. Parallel execution is an eligibility and capability choice, not a goal by itself.

## 2. Request flow

1. `main.py` or `ui.py` builds a `ToolUseContext`, registers tools, and constructs the system prompt.
2. `query.py` maintains the conversation, injects active plan/summary/project memory, applies context shaping, and calls the model through `query_deps.py`.
3. `api.py` converts messages and tool schemas to the LiteLLM interface, normalizes streaming events, applies provider concurrency and optional RPM/TPM limits, and performs bounded retries/fallbacks.
4. Model tool calls enter `StreamingToolExecutor` or the serial fallback in `tool_orchestration.py`.
5. `execute.py` validates tool availability and input, checks task file scope, runs the permission gauntlet, executes the tool, and maps its result back to the model.
6. `query.py` appends results, continues the loop, or terminates with a reason.

Provider IDs are configuration values. `worker_tiers.yaml` maps T1/T2/T3 to arbitrary LiteLLM model IDs; T3 may use `ACTIVE_MODEL`. `model_router.py` is static complexity-to-tier routing plus a single stronger-tier retry for a typed `needs_escalation`. It is not learned routing.

## 3. Task DAG execution

The model-facing `Task` tool accepts up to eight nodes per call. The deterministic split gate keeps work in the parent when there is no estimate, work is sequential, or estimated size is at most eight steps. Wide mode requires more than eight steps, at least three independent branches, and read-heavy work or disjoint writes. Uncertain work is explored by the parent first.

A Task DAG declares node IDs, `needs`, exact `read_files`/`write_files`, assumptions, non-goals, join criteria/checks, output schemas, acceptance checks, complexity/tier hints, and budgets. The scheduler:

- Rejects unknown dependencies, cycles, paths outside the project, unordered overlapping write/write or read/write scopes, and invalid integrator declarations.
- Runs ready nodes concurrently, up to four at a time. A dependent starts only after all declared dependencies succeed and receives their outputs. Failed prerequisites block dependents; independent branches can continue.
- Uses process-local write leases and stages declared writes. Git-root projects use detached worktrees. Successful outputs are committed only after optimistic workspace hash checks.
- Persists a per-node file-commit intent and before/after hashes. Reviewed reconciliation can finalize a node only when every declared output matches; partial or conflicting output stays failed. It never automatically reruns an interrupted worker.

Workers with no `write_files` receive `Read` only. Workers with write scopes receive `Read`, `Edit`, and `Write`, restricted to declared paths. Task workers do not receive Bash. Every separate Task invocation requires its own permission decision; approval is not sticky across batches.

The special `role="integrator"` is for file-writing integration: it must depend on every writer and own a disjoint write scope. For a read-only synthesis node, use an ordinary worker with `needs` on its inputs instead of declaring the integrator role.

Task cancellation applies to active in-process runs, stops pending work, and discards uncommitted staging; completed commits remain. Interrupted runs require inspection and explicit side-effect review before reconciliation. Request idempotency keys deduplicate identical whole-DAG submissions; they are not a general external-side-effect ledger.

## 4. Tools, permissions, and safety

`tool.py` defines the common tool contract; `registry.py` holds explicitly registered tools. Main filesystem tools are `Read`, `Edit`, and `Write`; `Bash` is available to the parent application. `execute.py` checks tool enablement, schema, validation, file scope, permissions, abort state, and loop guards before execution.

Permission modes, explicit rules, tool-specific checks, path safety, and hard-deny patterns are applied in `permissions.py`. Ordinary file edits use read-before-write gates and stale-file checks. These are application safeguards, not OS isolation.

General user Bash and Task workers are not OS-sandboxed. Do not grant Bash or generated executable tools to Task workers without a separately verified OS sandbox and secret isolation.

## 5. Reusable workflows

`workflows.py` stores versioned workflow definitions in session quarantine. Definitions have input/output schemas, tests, provenance, and explicit promotion. Promotion requires successful uses across distinct queries, passing declared tests, and user approval; the workflow list reports usage counts and a promotion-readiness suggestion.

Workflow steps currently support:

- `Read`, with the existing Read tool's behavior. If a `read_files` manifest is supplied, Read paths must be within it. Legacy Read-only workflows can omit this manifest, so they are not necessarily confined to a file subset.
- `Bash`, only with an explicit relative `read_files` manifest. The runtime rejects credential-like paths, copies declared regular files into a temporary snapshot, and runs an allowlisted read-only command through bubblewrap and `prlimit`. The namespace has no network and exposes only the copied files plus required read-only system binaries. CPU, process count, open files, temporary storage, per-file size, and output are bounded. Missing isolation utilities fail closed. Writes, nested workflows, and generated executable sequences are not supported.

This workflow sandbox does not change ordinary Bash behavior or grant Bash to Task workers.

## 6. Provider, state, and memory

`api.py` uses LiteLLM across configured providers. It classifies provider errors, honors bounded retry delays, fails fast on hard quota failures, uses explicitly configured alternate models for eligible errors, and records normalized model/provider activity. RPM/TPM limits are optional; TPM admission is approximate and should be calibrated with observed usage.

`store.py` keeps an append-only SQLite conversation log, FTS5 search, reversible summary DAG, plans, Task runs/events, artifacts, workflow definitions, and workflow usage. Project artifacts use content hashes. `memory_injector.py` adds bounded plans, summaries, and selected project-memory files to queries; `MemorySearch` and `MemoryExpand` are session-scoped tools. `TodoWrite` state is also mirrored to the store.

## 7. Text UI and permission prompts

`ui.py` provides the Textual chat, tool cards, plan panel, activity notices, and Allow/Deny permission screen. Permission requests are serialized so concurrent workers get a visible decision one at a time. The activity status returns from “Waiting for permission” after the last decision. A new Task call still requires a new approval.

The focused `test_ui.py` passed after this change. The full offline suite passed before the final UI-only update; rerun it before treating the whole current working tree as fully validated.

## 8. Current live evidence

There are three different observations; do not combine them into a single success claim:

- **Region-totals single case:** all acceptance checks passed on the active T3 model in 15.9 seconds, 20,508 tokens, and four API calls. The model estimated six steps and made zero Task calls. This proves single-agent correctness on that fixture, not fan-out.
- **User-observed UI Task run:** screenshots show an eight-node wide batch with six successes, two approximately 60-second timeouts (`main.py` and `task_scheduler.py`), and a second batch starting. No final integrator result is shown. This demonstrates that fan-out started, but not that the whole audit completed. The “Waiting for permission” label may be stale if workers have already started; each separate Task call still needs approval.
- **Scripted audit attempts:** the first used incorrectly prefixed file paths; the corrected attempt did not successfully execute workers because the parent changed the read-only integrator contract and reached the turn cap. These attempts are not successful Task-DAG evidence.

The historical 30-task/90-trial report predates child telemetry and cannot establish worker utilization. The one-case run and the screenshots are not matched baselines. No quality, latency, or efficiency conclusion about Pyramid should be made from them.

## 9. Remaining roadmap and next checks

- Rerun the full offline script suite after the latest UI permission change.
- Review Task budgets and grouping for large-file audits; the screenshot's two failures timed out at about 60 seconds. Do not raise limits blindly; compare per-node complexity, file scope, tool errors, and observed duration.
- Capture completion of a read-only DAG, including child model IDs, all node outcomes, dependency results, and final synthesis. An integrator that only writes is not appropriate for a read-only audit; use a normal dependent worker.
- Calibrate the explicit file-writing integrator workflow and approximate token/provider-attempt budgets before broadening code-writing DAGs.
- Keep workflows read-only. General Bash/Task OS isolation, write-capable workflows, learned routing, and speculation remain gated; do not claim them complete.
- Run the full telemetry-valid matched benchmark and ablations only at the roadmap's final evaluation gate. One live case is not enough to promote a mode by task class.

## 10. Useful local checks

From `schema manger/`, the offline script suite is:

```bash
for test in test_*.py; do PYTHONDONTWRITEBYTECODE=1 python "$test" || exit $?; done
```

A single Task integration test is `PYTHONDONTWRITEBYTECODE=1 python test_task_query_integration.py`. Never print or copy provider credentials from `.env`.
