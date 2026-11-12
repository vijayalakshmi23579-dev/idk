# Pyramid Implementation Plan v3

Updated: 2026-11-12. This is the active roadmap for a highly capable personal coding agent. Research claims inform experiments; they are not workload-specific guarantees or fixed acceptance thresholds.

## Goal

Build the strongest useful provider-agnostic pyramid for one user's local agent: an intelligent L1 planner, isolated and adaptively routed L2 workers, and deterministic L3 tools and reusable workflows. Use available models and compute where they improve capability, but require every added layer to earn its latency, token, and complexity cost. Continue improving the single-agent path in parallel; a negative fan-out result must not cancel useful verification, tools, or context improvements.

## Architecture Decisions

- **L1 owns the global plan.** With the full task context, it emits a DAG of contracts, dependencies, ownership boundaries, assumptions, and join criteria. Do not add a separate divider model. L1 may inspect results and revise the plan.
- **L2 workers are isolated executors.** Each gets a fresh context, a configured model, an explicit task contract, bounded budgets, and only its granted tools. It may make local sequential steps inside that contract, but cannot widen its scope, edit the parent plan, or spawn nested Task workers by default. A newly discovered independent branch becomes a structured escalation to L1.
- **L3 is deterministic.** Permissions, scheduling, leases, budgets, schemas, verification checks, persistence, and workflow execution are code, not shared model memory.
- **Use all three execution modes deliberately.** Single-agent is the baseline for sequential, tightly coupled, small, or tool-heavy work. Wide execution runs independent ready DAG nodes concurrently. Recursive recovery re-decomposes only a failed or newly understood leaf. These modes can compose; recursion is not a rival architecture to wide DAG execution.
- **Split on expected net value, not agent count.** The current gate (sequential or <=8 estimated steps stays single; wide needs at least 3 branches plus read-heavy work or disjoint writes) is a conservative eligibility filter, not proof that fan-out is worthwhile. The real decision must account for expected success, parallel wall time, planning/join/verifier overhead, provider queueing and limits, and extra calls/tokens. Do not hard-code a 15-minute cutoff or paper-derived fan-out as universal truth; calibrate against this agent's workload.
- **Bound concurrency and ownership.** Begin with the existing maximum of 8 DAG nodes and 4 concurrent tasks. Declare exact file scopes. The scheduler may overlap disjoint writers only after rejecting unordered write/write and read/write overlaps; dependencies serialize declared read-after-write work. Keep multi-file code DAGs opt-in and add staging/worktree integration before expanding them.
- **Route in distinct stages.** Keep complexity-to-tier mapping configurable (already present). Add deterministic escalation only for typed evidence such as invalid output, failed acceptance, verifier rejection, loop detection, or exhausted tool-error budget. Consider learned per-step routing only after telemetry is sufficient to train and compare it against fixed-tier baselines. Keep model IDs in configuration and log actual IDs per run.
- **Use a local blackboard, not agent chat.** SQLite plus an in-process queue is the right coordination mechanism for this single-process agent. Persist versioned task events, contracts, summaries, and artifact references. Redis Streams is a possible future transport if workers become multi-process; Redis Pub/Sub is not a durable control log.
- **Keep workflows distinct from schemas.** JSON Schema/Pydantic describes inputs and outputs; it does not turn a tool-call sequence into a safe executable workflow. A reusable workflow needs a versioned implementation, capability/permission manifest, input/output schemas, tests, provenance, and explicit promotion. New workflows remain quarantined and user-approved.
- **Delay speculation.** If evaluated later, a cheap proposer may start only read-only, cancellable work. Its result is provisional and cannot commit or influence writes until checked by the authoritative worker/planner.

The exact benefit of multi-agent systems, model routing, verifier depth, and spawn thresholds is task- and model-dependent. Treat figures from papers and vendor evaluations as hypotheses. Prefer matched local measurements and preserve useful single-agent improvements even if wide execution is not worthwhile for a task class.

## Current State

- **Phase 0 evidence:** A 30-task repo-derived set ran in three configurations (90 live trials), with 12 outputs across four task shapes human-scored. Automated acceptance was single-T2 15/30, single-T3 20/30, Task-enabled T3 19/30. On 23 parallel-shaped cases, Task-enabled T3 was 12/23 versus 13/23 for single-T3; mean automated quality was 0.681 vs 0.725, latency 382s vs 267s, and tokens 244,373 vs 177,450. Cost is unknown and is not the current objective. This is a negative signal for that run, not yet a definitive verdict on every task class.
- **Phase 0 limitation:** The report predates child telemetry and cannot establish which Task calls ran or which models children used. The runner now records Task invocations, child requests/model IDs, and lifecycle counts. A telemetry-enabled matched run is required before interpreting Pyramid efficiency.
- **Recent pilot artifacts:** The inspected run contains the first eight research tasks; the inspected Pyramid trial recorded zero Task launches. Those tasks are not marked wide-eligible, so zero is expected and this pilot does not measure worker execution. The designated region-totals case still needs to be included in a matched run.
- **Orchestration foundations present:** deterministic split gate; validated Task DAG; configurable complexity-to-tier routing; per-node budgets; output-schema validation; file scopes and process-local leases; provider concurrency and configurable RPM/TPM controls; Phase 0 runner/reviewer; child telemetry; SQLite Task lifecycle events.
- **Completed 2026-11-12:** Task specs, results, artifacts, and lifecycle events have versioned schemas; structured outputs survive DAG dependencies and reach L1; completed run specs, outcomes, recovery links, and aggregate usage persist in project-scoped SQLite and restore by `run_id`; recovery is depth- and scope-bounded and cannot raise the original aggregate token/call/time ceilings. Worker lifecycle events include event/run/attempt IDs, timestamps, and model IDs. Declared dependency schemas and deterministic file/output acceptance are enforced. Worker artifacts are retained in a local content-addressed store and linked by SHA-256 while workspace copies remain available. The Phase 0 gate requires all three configurations, telemetry, reviewed matrices, and actual child launches for wide-eligible cases. `pipeline-01-region-totals` is the first designated three-branch candidate.
- **Current implementation gaps:** interrupted runs persist `running`/`finalizing` status and per-node progress atomically with each lifecycle event, and L1 can inspect that state after Store reopen; they are intentionally not auto-replayed because a writer may have partially changed files. Manual reconciliation/cancellation and a side-effect idempotency ledger remain open. Failed nodes are conservatively charged their full declared limits. Conflict-checked disjoint file writers can overlap, but leases remain process-local and multi-file code DAGs still need staging/worktree integration. Arbitrary command checks remain disabled without an OS sandbox.
- **Verifier and recovery status:** the opt-in independent fresh-context verifier has Read-only access, bounded budgets, and a typed verdict, but has not been live quality-calibrated. Offline tests cover valid, rejected, malformed, unavailable, and timed-out verdicts. Failed-leaf re-decomposition can load completed runs after Store reopen, preserves aggregate budgets, and cannot widen file scopes. Candidate-content prompt-injection/cancellation coverage, automatic tier escalation/advisor, semantic join verification beyond declared schemas, in-flight crash recovery, and worktree/staging integration remain gaps.
- **Not implemented:** reusable workflow quarantine/promotion, OS sandboxing, speculation, and learned routing. Existing model selection is configurable static complexity routing, not learned routing.
- **Security boundary:** path and permission checks are application safeguards, not OS isolation. Task workers currently receive Read, Edit, and Write, not Bash. Do not grant workers Bash or execute generated tools without a real OS-level sandbox and secret isolation.

## Implementation Sequence

Complete and validate implementation phases first, using offline tests and mocked providers continuously. Defer live benchmark spending until the final calibration gate below. Keep wide execution opt-in; do not make it the default globally. If the final benchmark shows no repeatable benefit for a task class, retain useful contracts, routing, verification, and tools while keeping fan-out experimental.

### Phase 1: Pyramid Contracts And Evidence Path

1. **Make the contracts executable end to end.** Task, lifecycle event, result, and artifact envelopes are versioned and validated at their boundaries. Continue to version verifier verdicts and represent success, failure, escalation, and contract-change requests distinctly.
2. **Make the evaluation gate trustworthy.** Require the intended baseline matrix and telemetry in the validator. Add explicit checks that a gate-eligible case actually launched workers. Keep “not measured” distinct from zero.
3. **Broaden deterministic acceptance safely.** Confined file assertions, file JSON Schema checks, structured-output assertions, and structured-output schema checks are implemented. Add any further non-arbitrary checks needed by real tasks, and improve coding cases so file presence alone cannot masquerade as behavioral correctness. Keep arbitrary commands disabled until an OS sandbox exists.
4. **Complete local run records.** Completed Task run specs, outcomes, recovery links, and budget usage persist in SQLite and restore by `run_id`; lifecycle events carry schema, event/attempt/run IDs, timestamps, and model IDs, and progress snapshots update atomically with each event. A read-only Task inspection returns interrupted progress and scopes; automatic replay is rejected. Next add an explicit user-approved reconcile/cancel action and document how partially written files are handled. Keep crash recovery useful for this personal single-process runtime; do not build multi-user event infrastructure.
5. **Persist artifacts by content identity.** Worker artifacts are copied into a private SHA-256-addressed local store with size and workspace-path references. Add integrity verification on reads and garbage collection for unreferenced blobs; keep full bytes out of L1 context.
6. **Measure budgets and routing.** Retain configurable complexity tiers; record estimated and actual tokens/calls/time. Exercise RPM/TPM admission against observed usage and keep conservative headroom because token estimates are approximate.

**Exit:** malformed contracts and failed deterministic checks cannot reach dependents; report validation rejects missing matrix/telemetry; structured outputs survive parent and dependency boundaries; local events and artifact references survive restart; budget accounting is observable.

### Phase 2: Verification, Adaptive Routing, And Recovery

1. **Harden the existing opt-in verifier.** Offline tests now cover malformed verdicts, rejection, timeout, and provider unavailability. Add candidate-content prompt-injection and cancellation cases, then measure false accepts/rejects on representative tasks before enabling it broadly.
2. **Verify joins.** Declared dependency output schemas are now checked before a dependent worker starts, and mismatches fail the node. Add frozen-interface/semantic join checks where real multi-artifact tasks demonstrate the need.
3. **Add bottom-up escalation.** For typed failures only, retry within budget, route the failed leaf to a stronger configured tier or verifier, or return an escalation envelope to L1. Preserve attempt IDs, outputs, usage, and diagnostics.
4. **Extend bounded failed-leaf re-decomposition.** L1 can now reload completed runs and submit a recovery DAG within the original aggregate token/call/time ceilings; a read-only inspect operation exposes interrupted progress. Add an explicit reconciliation path and tests for depth exhaustion, cancellation, corrupt records, and partially applied file writes. Keep depth capped at 2; children cannot spawn children by default, and successful branches remain intact.
5. **Harden code-writing DAGs.** The scheduler now permits conflict-free disjoint file writers and rejects unordered read/write or write/write overlaps. Add staging/worktree isolation and one integrator as the only writer to the shared branch before broadening multi-file code DAGs.

**Exit:** invalid, rejected, or inconsistent results are blocked; escalation is bounded and observable; cancellation leaves no untracked writes; verifier quality is measured; sequential controls remain single-agent.

### Phase 3: Reusable Workflows And Tools

Mine the event history for repeated successful sequences as suggestions. Define a versioned workflow manifest containing implementation identity, input/output schemas, allowed tools/capabilities, tests, provenance, and status. New workflows remain session-scoped and quarantined; test without secrets and with an OS sandbox; require explicit user promotion and pinned versions. Never allow self-promotion or infer executable semantics from JSON Schema alone.

**Exit:** a candidate passes isolated tests, is reused successfully across distinct tasks, stays within declared capabilities, and is explicitly approved. Otherwise keep it session-scoped or discard it.

### Phase 4: Learned Routing And Speculation

Use stable, labeled telemetry to compare a learned per-step router with fixed-tier baselines. Add a cheap proposer only if it improves held-out tasks. Limit speculation to read-only, cancellable work; do not let provisional results commit writes or become authoritative before verification. Run ablations for routing, fan-out, verifier, and recovery separately so their contribution is identifiable.

**Exit:** every added mechanism shows repeatable quality-preserving improvement on held-out personal workloads. Retain advanced capabilities that improve capability or reliability even if they do not reduce tokens; remove or disable mechanisms whose cost is not justified by measured value.

### Final Phase: Live Calibration And Promotion

1. Run the matched single-T2, single-T3, and Task-enabled T3 comparison with fixed model IDs, the designated region aggregation case, and representative personal coding tasks. Keep sequential, tightly coupled, and tool-heavy controls.
2. Verify actual child calls, worker model IDs, attempts, lifecycle events, tokens, latency, and human review across all four task shapes. The earlier summary without child telemetry is not sufficient evidence.
3. Report paired per-task and per-shape outcomes. If a candidate was marked wide-eligible but did not launch workers, fix the harness/prompt and rerun that case; do not call it a negative parallel result.
4. Enable wide execution only for task classes with repeatable quality-preserving benefit. Otherwise leave it opt-in and keep improving the single-agent path.

**Exit:** a complete, telemetry-valid matched report with review coverage and clear per-class decisions. Price is optional; unknown cost stays null. No live benchmark is a prerequisite for the implementation phases above.

## Personal-Agent Safety Boundary

This project does not need multi-user tenancy, Redis, cross-process locks, compliance audit infrastructure, or production SLOs. Keep those out of scope unless deployment changes. Do retain strict worker tool scopes, explicit user approval for promotion, cancellation and budget limits, and the distinction between application checks and OS isolation. Do not expose Bash or generated executable tools to workers until OS-level sandboxing and secret isolation exist.

## Immediate Next Work

1. Run the telemetry-enabled matched benchmark with the designated region aggregation case and representative personal coding tasks. Keep fan-out opt-in and report per-shape outcomes.
2. Add verifier prompt-injection/cancellation cases and tests for recovery depth, corrupt records, and crash behavior; live-calibrate the verifier before enabling it broadly.
3. Add artifact integrity verification/garbage collection and an explicit user-approved reconcile/cancel flow for interrupted runs; do not auto-replay file-writing work.
4. Add more naturally wide-eligible cases and reassess scheduler support for disjoint writers before claiming write-parallel speedup.