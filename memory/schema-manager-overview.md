# Schema Manager Current State

Last updated: 2026-11-12. This snapshot is the source of truth for the current working tree; older roadmap notes below are historical and may be superseded.

## Current Session Handoff (2026-11-12)

The current objective is a provider-agnostic Pyramid for one local coding agent. Keep single-agent execution as a first-class path. Do not run another live model/benchmark call unless the user explicitly requests it. Learned routing, speculation, and benchmark-based promotion remain evidence-gated; a single live case cannot establish comparative quality or efficiency.

- **Current uncommitted implementation:** writable Task nodes persist prepared/committed file-output manifests with before/after SHA-256 hashes. Reviewed reconciliation finalizes a node only if every declared output matches, without rerunning its worker; partial or conflicting outputs remain failed for review. Offline tests cover full and partial commits, corrupt manifests, and recovery-depth exhaustion.
- **Workflow capability:** reusable workflows support `Read` and read-only `Bash` in a bubblewrap namespace. Bash receives only a copied manifest of relative, non-credential project files at `$WORKFLOW_ROOT`, has no network, is bounded by CPU/process/open-file/tmp/file-size/output limits, and fails closed if `bwrap` or `prlimit` is missing. Writes, nested workflows, ordinary user Bash, and Task-worker Bash remain outside this sandbox. Workflow lists expose successful-run counts and a promotion-readiness suggestion; promotion still requires declared tests and user approval.
- **Permission UI fix:** `ui.py` serializes concurrent permission requests through one modal queue and restores the status bar after the last decision. `test_ui.py` passed after this change, including two queued prompts, Allow/Deny controls, and status restoration. The entire offline script suite passed before this UI-only change; it has not yet been rerun afterward. Static diagnostics and `git diff --check` passed after the UI change.
- **User-observed live Task run:** screenshots showed an eight-node wide Task batch with six successes and two worker timeouts at about 60 seconds (`main.py` and `task_scheduler.py`), followed by a second batch starting. This demonstrates that the UI run reached Task fan-out, but the screenshots do not show completion of the second batch or final synthesis. The bottom-bar text `Waiting for permission` could be stale; each separate Task call independently asks approval. Check for an actual modal and lifecycle completion, not the status label alone. `Task b2.permissions` is a worker ID, not proof by itself of a permission prompt.
- **One live Phase 0 case:** `pipeline-01-region-totals` passed all acceptance checks using `nvidia_nim/nvidia/nemotron-3.5-super-vl-120b-a12b`: 4 API calls, 20,508 tokens, 15.94 seconds, zero Task calls/child requests. The model estimated six steps, below the `>8` wide threshold. This is a single-agent correctness result, not a Pyramid fan-out result. Artifacts: `phase0-results/one-case-live/pipeline-01-region-totals/pyramid/`.
- **Our own scripted Task probes:** the first failed to read paths because it prefixed them with `schema manger/` while `project_root` already pointed there. The corrected probe still did not run workers: the model repeatedly added a write scope for its proposed integrator, conflicting with the read-only guard, then hit the two-turn cap. Do not count those probes as successful Task execution.
- **Worktree:** changes are uncommitted. Modified source/tests include `agent_tasks.py`, `task_scheduler.py`, `ui.py`, `workflows.py`, `tools/bash.py`, `tool.py`, `store.py`, and `main.py`, with matching offline tests and roadmap notes. The new `os_sandbox.py` and `phase0-results/one-case-live/` are untracked. Preserve these changes; do not reset or discard them.

**Next checks:** rerun `PYTHONDONTWRITEBYTECODE=1 python test_ui.py`, then the full `for test in test_*.py; do PYTHONDONTWRITEBYTECODE=1 python "$test" || exit $?; done` suite from `schema manger/`. For a live fan-out claim, inspect the user’s final batch and integrator results; do not launch another live call without explicit authorization.

## What it is
A Python coding-agent runtime under `schema manger/`. The query loop is provider-agnostic through LiteLLM; internal messages and registered tools are converted at the API boundary. Do not treat it as a Claude Code source replica: several behaviors are local implementations or explicit deviations.

## Request flow
1. `query.py` maintains conversation state, shapes context, injects plan/summary/relevant file memories, calls the model, and continues or terminates after tools.
2. `query_deps.py` forwards the query's selected model, explicit fallback, source, and token limit to `api.stream()`.
3. `api.py` normalizes provider events through LiteLLM and applies bounded retry/fallback policy. Registered tool schemas are sent in the request; `tool.py` emits standard OpenAI function schemas.
4. Model tool calls go through `StreamingToolExecutor` or `tool_orchestration.py`. The executor applies concurrency-safety policy; tool-batch concurrency is separate from the Task DAG scheduler.
5. `execute.py` validates calls, checks permissions, prompts when needed, applies safety gates, executes the tool, and returns a model-facing result.
6. `permissions.py` applies hard denies, configured rules, tool-specific checks, permission modes, and auto-mode classification.

## Main building blocks
- `tool.py`, `registry.py`, `tools/`: shared tool contract, explicit registry, and filesystem/Bash tools.
- `todo.py`, `agent_tasks.py`, `task_scheduler.py`: model-invoked plan updates, isolated Task child query loops, validated DAG scheduling, dependency payloads, and file leases.
- `messages.py`, `api.py`, `query_deps.py`: message types, LiteLLM streaming/error policy, and model adapter.
- `state.py`: per-session in-memory state; `store.py`: append-only SQLite history, FTS5 search, summary DAG, and persisted plans.
- `compact.py`, `query_token_budget.py`: token estimates, compaction, and continuation controls.
- `system_context.py`, `memdir.py`, `memory_injector.py`, `memory_tools.py`: system context, selectable file memories, per-turn injection, MemorySearch, and MemoryExpand.
- `ui.py`: Textual chat UI, plan panel, tool cards, permission modal, and runtime activity notices.
- `path_safety.py`, `permission_rules.py`, `denial_tracking.py`, `loop_guard.py`: safety and repeated-action controls.

## Maturity notes
The core query, API, tool execution, permissions, filesystem tools, persistence, Task DAG, memory injection, and Textual UI are implemented and covered by offline script tests. Task workers receive fresh contexts and scoped Read/Edit/Write tools; Bash is unavailable to workers. The runtime validates worker outputs, optionally verifies them in a read-only context, and returns bounded summaries/artifact references. Offline tests prove scripted contracts and routing, not that a live model will choose tools correctly or that the verifier improves quality.

General user Bash and Task workers are not OS-sandboxed; `path_safety.py` and permission gates remain application safeguards, not isolation. Workflow Bash is a separate bubblewrap/prlimit path: it has no network, receives a copied read-only snapshot of declared non-credential files, enforces CPU/process/file-descriptor/storage/file-size bounds, and fails closed if either isolation utility is unavailable. Some hooks/context-recovery areas in `compact.py` remain partial. Provider concurrency and optional RPM/TPM budgets are enforced in the API adapter. RPM/TPM default to unlimited unless configured; TPM admission uses an approximate prompt-token estimate reconciled against returned usage. There is no ToolSearch/deferred-schema expansion or FastMCP client integration. Writable Task nodes persist per-node file-commit intent and declared-path before/after hashes; reviewed restart reconciliation finalizes fully matching outputs without rerunning the worker. Partial or conflicting outputs remain failed for inspection and bounded recovery. This does not provide transactional idempotency for arbitrary external effects.

`Tool.prompt` is currently not included by `to_openai_schema()`; the provider receives tool description and parameters. `should_defer` is metadata only. Full schemas for the tools available in the current query context are sent each request, so do not claim Anthropic `defer_loading` or `tool_reference` behavior here.

Runtime activity notices now show model request/retry/fallback/failure status; Task child lifecycle and dependencies; TodoWrite count updates; automatic memory source/count; and explicit memory operation counts/source IDs. They intentionally omit prompt text, API keys, memory body, and raw provider error payloads. The PlanPanel is separate from this activity stream.

The current editor buffer `schema manger/experiments/stream_writer.py` is not present on disk: the `experiments/` directory is absent from the checkout. It is an unverified scratch loop that sleeps synchronously and appends `output_log.txt`; do not treat it as production or as implemented async streaming.

## Current provider behavior
- `requirements.txt` currently pins `litellm==1.105.0`; provider credentials are read from environment/`.env` and must never be copied into this note.
- `api.py` retries transient 429/5xx/network failures a bounded number of times, honors `Retry-After` up to its configured wait budget, treats hard-quota 429 as fail-fast, and does not retry a 404 on the same model.
- Cross-model failover uses explicitly configured alternates (`fallback_model` or `LITELLM_FALLBACK_MODELS`) for 404, overload, and connection failures. Rate limits do not trigger cross-model failover.
- Provider errors are normalized to safe query-facing messages. Context overflow maps to the existing compaction recovery signal.
- `AGENT_PROVIDER_MAX_RPM` and `AGENT_PROVIDER_MAX_TPM` configure global rate limits; provider-specific `AGENT_PROVIDER_MAX_RPM_<PROVIDER>` and `AGENT_PROVIDER_MAX_TPM_<PROVIDER>` override them. Unset or zero means unlimited. TPM admission reserves an approximate prompt estimate plus the requested output cap, then reconciles actual usage; configure limits with headroom because token estimates vary by tokenizer/provider. The originally reported NVIDIA 404 still needs a redacted provider response/request ID to identify its cause. On 2026-11-10, minimal live smoke calls succeeded for Bolt E4B and Super VL 120B. Lightning 30B succeeded with a 256-token output cap; smaller caps ended during reasoning without visible answer text. These were connectivity checks, not quality or latency benchmarks.
- A prior MCP compatibility report involved LiteLLM's optional experimental MCP modules, not this folder's regular `litellm.acompletion` path. This project has no MCP client use; do not upgrade/pin MCP solely based on that optional import without validating package constraints.

## Current orchestration and verification
- The Task node has opt-in `verification_mode="independent"`. After worker schema/acceptance checks, it runs a fresh-context verifier with a distinct configured model, Read-only access, bounded calls/tokens/time, and a typed verdict; rejection fails the node and blocks dependents. This is offline-tested but not live quality-calibrated. One evidence-triggered stronger-tier retry, bounded per-leaf recovery, typed scope-bounded expansion proposals, and deterministic dependency/schema/path-equality join checks are also implemented. Expansion proposals return to L1 and are never auto-executed.
- TodoWrite persists the plan to AppState and SQLite; the UI PlanPanel reads the stored plan. Completed plans are cleared. Strict caps are opt-in.
- Task accepts up to eight DAG nodes; the scheduler allows up to four concurrent ready workers, validates dependencies and file scopes, and serializes conflicting writers. Writable nodes in Git-root projects use detached Git worktrees; successful staged files are committed through optimistic hash checks and failed/cancelled staging is discarded. Per-node commit intent and output hashes are persisted before/after workspace replacement; reviewed reconciliation finalizes a fully matching commit without replay, while partial or conflicting commits stay failed. An optional `integrator` role is validated: it must depend on every writer and have a disjoint output scope. Task workers still lack Bash. Multi-file code DAGs should remain opt-in pending partial-write recovery validation and end-to-end integrator calibration.
- Task nodes accept complexity (1-5), an optional T1/T2/T3 hint, non-goals, output JSON Schema, deterministic file/output acceptance checks, per-node and aggregate budgets, verification mode, and escalation limits. `worker_tiers.yaml` maps T1/T2 to configured models and T3 to `ACTIVE_MODEL`. Workers return validated summaries, structured outputs, usage, verification results, and artifact references; dependents receive validated dependency outputs rather than raw worker transcripts.
- Task run controls include inspection, approval-gated cancellation of an active in-process run, and reviewed reconciliation after interruption. Cancellation stops pending nodes, cancels active workers, discards uncommitted staging, and preserves completed writes. Restarted runs are not automatically resumed or replayed; fully hash-matched file commits can be finalized from their persisted result without rerunning, while partial writes still require user review. Optional request-level idempotency keys deduplicate identical DAG submissions and reject changed payloads. Per-node journal guarantees cover declared file outputs only, not arbitrary external side effects.
- Task remains available. Its deterministic split gate defaults to single-agent when no estimate is supplied, the work is sequential, or estimated steps are at most eight. Wide execution requires at least three independent branches and a read-heavy or disjoint-write estimate; DAG root counts are checked against the estimate. Uncertain work returns to the parent for exploration. `phase0_gate.py` and `phase0_eval.py` validate and score supplied data; `phase0_runner.py` runs serial trials for single-T2, single-T3, and the Task-enabled T3 configuration. Each trial runs in a fresh temporary workspace, then retains a review copy and final response. The runner exposes scoped Read/Edit/Write tools, no Bash, and static `exists`/`assert`/`schema` checks; response assertions support read-only research/reasoning cases, while command checks fail closed until an OS-level sandbox exists. It accepts model IDs at runtime, records parent/child model IDs, Task invocation counts, worker lifecycle counts, token/call/latency metrics, and leaves cost unknown. A review sheet contains all three configurations for selected tasks; scores and notes are applied by the `finalize` command, which validates the report. The evaluator accepts 8-30 tasks covering all four shapes and requires paired single-T2/single-T3 results plus at least three human-reviewed tasks. Its efficiency assessment compares matched quality, latency, and token counts without requiring prices; the older 30% cost kill criterion remains available but is not the active objective. The provisional 30-task repo-derived dataset completed one live 90-trial run and was human-reviewed across 12 entries. That historical report predates Task/child telemetry, which is now represented as null rather than zero, so it cannot verify worker utilization or subagent efficiency.
- Provider requests share per-provider concurrency semaphores and optional RPM/TPM buckets across model, compaction, and classifier calls. `AGENT_PROVIDER_MAX_CONCURRENCY` defaults to four; `AGENT_PROVIDER_MAX_CONCURRENCY_<PROVIDER>` overrides an individual provider. `AGENT_PROVIDER_MAX_RPM` and `AGENT_PROVIDER_MAX_TPM` configure global budgets; provider-specific `AGENT_PROVIDER_MAX_RPM_<PROVIDER>` and `AGENT_PROVIDER_MAX_TPM_<PROVIDER>` override them. Unset or zero means unlimited. TPM admission reserves an approximate prompt estimate plus the requested output cap, then reconciles actual usage; configure with headroom because token estimates vary by tokenizer/provider.
- Task lifecycle events append to SQLite `task_events` when a session Store is configured. Events are scoped by session/run and store task ID, status, dependencies, duration, and safe error type, not prompts, outputs, or raw provider payloads. This is durable logging, not crash-resume or transactional coupling to scheduler state.
- Memory injection runs per query from the active plan, root summary, and selected project-memory files. MemorySearch/MemoryExpand are model-callable and session-scoped. The automatic injection and explicit tool calls now produce safe UI activity notices.
- On 2026-11-11, the scheduler lifecycle test was corrected to assert that independent siblings both complete before their dependent starts, without requiring a deterministic sibling completion order. `test_task_query_integration.py` verifies parent tool selection, provider-adapter model routing, child progress visibility, dependencies, and worker tool filtering with a scripted provider. `test_phase0_runner.py` and `test_phase0_review.py` cover the runner, report finalization, and Textual score-entry workflow offline. The full offline script suite passes after adding child telemetry and legacy-report handling. A live 90-trial run completed and was human-reviewed across 12 entries (one task per shape, all three configurations). Automated acceptance passed for 15/30 single-T2, 20/30 single-T3, and 19/30 Task-enabled T3 runs. On the 23 parallel-shaped tasks, Task-enabled T3 success was 12/23 versus 13/23 for single-T3; mean automated quality was 0.681 versus 0.725. Task-enabled T3 took 382s and 244,373 tokens in aggregate, versus 267s and 177,450 tokens for single-T3: no quality-preserving latency or token win on this dataset. Prices remain null. That historical run lacks Task/child-role telemetry; the report now represents those metrics as null, not zero. The runner records Task invocations, child requests/model IDs, and worker lifecycle counts for future runs. Report, review, summary, and artifacts are under `/workspaces/idk/phase0-results`.
- On 2026-11-11, the scheduler lifecycle test was corrected to assert that independent siblings both complete before their dependent starts, without requiring a deterministic sibling completion order. `test_task_query_integration.py` verifies parent tool selection, provider-adapter model routing, child progress visibility, dependencies, and worker tool filtering with a scripted provider. `test_phase0_runner.py` and `test_phase0_review.py` cover the runner, report finalization, and Textual score-entry workflow offline. The full offline script suite passes after adding child telemetry, legacy-report handling, and provider rate-limit tests. A live 90-trial run completed and was human-reviewed across 12 entries (one task per shape, all three configurations). Automated acceptance passed for 15/30 single-T2, 20/30 single-T3, and 19/30 Task-enabled T3 runs. On the 23 parallel-shaped tasks, Task-enabled T3 success was 12/23 versus 13/23 for single-T3; mean automated quality was 0.681 versus 0.725. Task-enabled T3 took 382s and 244,373 tokens in aggregate, versus 267s and 177,450 tokens for single-T3: no quality-preserving latency or token win on this dataset. Prices remain null. That historical run lacks Task/child-role telemetry; the report now represents those metrics as null, not zero. Report, review, summary, and artifacts are under `/workspaces/idk/phase0-results`.


```bash
python phase0_review.py --dataset path/to/dataset.json --report path/to/report.json --reviews path/to/report.review.json
```

The Textual review portal browses each selected task/configuration, shows prompt, automated checks, response, retained workspace files, model ID, token count, latency, and score input. **Previous**/**Next** move between entries; `Ctrl+S` or **Save score** persists the current score/notes immediately; `q` exits. It performs no provider calls.
## Phase 0 Pilot Workflow
Run trials with model IDs supplied at runtime; worker tiers continue to come from `worker_tiers.yaml` or `AGENT_WORKER_TIERS_CONFIG`.

```bash
cd 'schema manger'
python phase0_runner.py run --dataset path/to/dataset.json --report path/to/report.json --t2-model provider/model-id --t3-model provider/model-id
```

This writes `report.review.json` and retains trial workspaces under `phase0-artifacts/` beside the dataset by default. An answer assertion can use `{"target":"response","contains":"expected evidence"}`; file checks use a relative `path`. Open the review portal:

```bash
python phase0_review.py --dataset path/to/dataset.json --report path/to/report.json --reviews path/to/report.review.json
```

Use the view buttons to inspect the response, prompt, checks, and workspace files. Score each selected configuration from 0 to 1 using the on-screen rubric; scores and notes save immediately. Then finalize without rerunning models:

```bash
python phase0_runner.py finalize --report path/to/report.json --reviews path/to/report.review.json
```

The finalized report is written as `report-reviewed.json`. Run `python phase0_eval.py --dataset path/to/dataset.json --report path/to/report-reviewed.json` to calculate quality, latency, token-efficiency, and legacy cost metrics. Costs remain null unless actual prices are supplied; command acceptance checks are intentionally unsupported without an OS-level sandbox.

## Next Work
1. Calibrate the explicit integrator workflow and, if the scheduler commit protocol changes, add a process-kill test for interruption during a multi-file replacement. Current offline coverage checks complete hash-matched recovery, partial-write refusal, corrupt manifests, and recovery-depth limits; recovery remains review-gated and never replays a worker automatically.
2. Calibrate aggregate token/provider-attempt estimates and live verifier quality. Keep verifier opt-in until its false-accept/reject behavior is measured.
3. Keep multi-file code DAGs opt-in while validating the explicit integrator flow on representative integration cases; leases remain process-local even though writable Git tasks use detached worktrees.
4. Keep workflow Bash limited to offline, read-only bubblewrap snapshots of declared non-credential files. Do not add workflow writes, nested workflows, or Bash access for Task workers without stronger isolation.
5. After implementation phases are complete, run the full telemetry-valid matched evaluation, including designated wide-eligible work, held-out cases, human review, and feature ablations. The earlier live comparison predates child telemetry and cannot establish worker efficiency.
6. Only after the implementation/evaluation gates, consider learned routing and read-only speculation. Do not start the live campaign early; the active roadmap places it last.

## Session Handoff (2026-11-10)
- NVIDIA tier config was updated and resolved successfully: T1 `nvidia_nim/nvidia/nemotron-3.5-bolt-e4b`, T2 `nvidia_nim/nvidia/nemotron-3.5-lightning-30b-a3b`, T3 `nvidia_nim/nvidia/nemotron-3.5-super-vl-120b-a12b` via `ACTIVE_MODEL`. The model router is configuration-driven; do not hard-code these model IDs into runtime logic or tests.
- Live smoke results: Bolt 30 input/2 output tokens; Lightning 34/136 at a 256-token cap (earlier 8- and 64-token calls ended before visible text); Super VL 32/33. These prove endpoint connectivity only. No live benchmark or Task worker evaluation was run.
- Phase 0 cost-per-success now counts spend from failed attempts and divides by successful runs. `BaselineResult` has optional `model_id` and optional `cost_usd`; unknown cost remains JSON null and cannot satisfy the 30% cost-reduction gate.
- Pilot dataset/report minimum is eight tasks, with all four task shapes, paired single-T2/single-T3 results, and human review of at least three tasks. Full target remains 30. Tests for Phase 0 and Task tier routing pass.
- No real experiment task corpus or comparative benchmark results are present in the current checkout. A serial pilot runner now exists in `phase0_runner.py`; its offline test uses synthetic fixture cases. Do not expose Bash to benchmark agents until a real sandbox exists.
- A provider key was included in a chat attachment. The user plans to rotate keys later. Never store, repeat, or print credential values; avoid further live calls unless explicitly requested.
- The scheduler event-order assertion was corrected on 2026-11-11, and the complete offline script loop passed afterward. The scripted query and Phase 0 runner tests use fake provider responses; they are architecture/contract tests, not live-model evaluations.

## Historical Notes
The sections in the prior version of this document below described an earlier memory/context implementation phase. Preserve those details as history, but use the current-state sections above for present behavior.

The initial architecture review excluded `main.py`; it was inspected and updated during the implementation phase below.

## Historical Implementation Roadmap (superseded by current state above)

The persistence checks passed before implementation: `test_todo.py` and `test_store.py`.

### Completed in this phase
- Hardened `memdir.py`: absolute/local path validation, project-setting override rejection, symlink-escape protection, project-root keyed caching, remote memory directory support, one-day freshness warnings, and SHA-256 no-op writes.
- Added per-turn context assembly for the persisted plan, summary, and up to five relevant file memories. Memory selection is deduplicated across a session and routed through the query's existing single system-reminder injection.
- Added read-only `MemorySearch` and `MemoryExpand` tools for FTS5 search and exact DAG expansion.
- Connected CLI registration and SQLite persistence. Query messages are archived before context shaping; compaction summaries become summary-DAG nodes, and expansion retains chronological order.
- Made microcompacted tool results point to their archived message IDs, so `MemoryExpand` can recover them. Preserved loop-guard state through streaming tool contexts and registered memory reads as safe/idempotent.
- Synchronized the compacted active transcript back to the CLI's message history and connected `CLAUDE_CODE_MAX_TURN_TOKENS` to the existing diminishing-returns guard.
- Fixed two verified execution defects encountered in regression tests: Bash path safety was evaluating whole command strings as paths, and `execute.py` discarded the tool mapper's `is_error` result.

### Remaining before calling the context-window phase complete
1. `Store.get_active_view_messages()` now supplies the query after a summary exists, using a 40K default budget and keeping recent user/tool groups intact. Its estimate is character-based, and one indivisible newest group can exceed the budget; validate with the provider tokenizer before claiming the ~29K target.
2. Before the first persisted summary exists, the full transcript still relies on the existing auto-compaction trigger. Reconcile the cited 87% figure with the repository's current 167K threshold before changing it.
3. Load user-level memory settings in CLI setup. Project-level memory-path settings must remain rejected; the resolver supports an explicit user setting but the CLI does not load one yet.

Team sync, AST indexing, sandboxing, HRR, and automatic memory extraction remain separate deferred projects. The complete offline regression suite passed after implementation; persistence, query compaction, memory injection, memory tools, active views, and recovery pointers also have focused tests.

## Historical Next Session Notes (superseded where noted above)
- `plans.py` is now correctly named but remains unused: no imports or runtime call sites. Keep it deferred until a real Markdown Plan Mode lifecycle (enter/exit/resume/display) is implemented; do not merge it into the active SQLite `TodoWrite` plan path. Remove it only if Plan Mode is abandoned.
- Provider config in `api.py`: short `nvidia/...` names normalize to `nvidia_nim/nvidia/...`; legacy `NVIDIA_API_KEY` maps to LiteLLM's `NVIDIA_NIM_API_KEY`; Gemini catalog uses `gemini-3.8-flash`. The old statement that cross-provider fallbacks only happen after rate limits is superseded; use the current provider behavior section above.
- API tests and smoke tests passed after those provider changes. The most recent `python main.py -p "hi"` terminal status was exit code 0; inspect the actual response if live provider behavior needs confirmation.
- Provider credentials were exposed in a chat attachment. Rotate those keys; never copy their values into this note or another chat.
- The full offline suite passed after the context adapter and provider changes. No commit was made; this workspace note is intended to be included if the user commits the work.
