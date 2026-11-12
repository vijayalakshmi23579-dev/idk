# Schema Manager Benchmark Notes

This document is a fixed evidence pack for the Phase 0 research cases.

- The runtime query loop owns conversation state, context shaping, compaction, and tool continuation.
- The API adapter converts internal messages and tools for LiteLLM and normalizes provider streaming events.
- Tool execution validates inputs, checks permissions, runs safety checks, and maps results back to the model.
- Task workers have isolated query state. A Task DAG can have at most eight nodes; ready read-only workers run concurrently, and writers are serialized.
- Task contracts can declare file scopes, dependencies, non-goals, JSON output schemas, and per-node budgets.
- The deterministic split gate defaults to one agent when no estimate is supplied, work is sequential, or estimated steps are at most eight. Wide execution requires at least three independent branches and read-heavy work or disjoint writes.
- Provider calls have bounded concurrency and retry/fallback behavior. RPM and TPM quotas are not enforced.
- Filesystem path checks and permission gates are application safeguards. The runtime has no OS-level shell sandbox.
- Session messages are append-only in SQLite, with FTS5 search, summary DAGs, and recoverable source messages.
- Phase 0 reports include success checks, quality, tokens, calls, latency, model IDs, optional cost, and human review. Unknown prices remain null.
- Tool schemas currently include descriptions and parameters. Tool prompt guidance and deferred-schema loading are not provider-visible behaviors.
- The benchmark runner exposes scoped Read/Edit/Write tools, never Bash, and does not execute dataset-supplied commands.