# Runtime Alpha v0.6

## Current Version

`Runtime Alpha v0.6`

## Stage Position

The project is no longer a simple LangGraph demo. The current stage is:

- Agent Runtime Engine
- Task DAG Runtime
- Workflow Orchestrator Alpha

## Completed Modules

### P0 Parser Repair Chain

- Invalid JSON repair
- Schema repair
- Missing field repair
- Dependency repair
- `parser_repair_history`

### P1 Executor Stability Layer

- Retry policy
- Timeout policy
- State guard
- Executor crash isolation
- `task_failures` metadata

### P2 Router Permission and Capability Layer

- `ToolCapability`
- `PermissionContext`
- `routing_history`
- Concurrency gate
- Explicit capability registration

### P3 LLM Layer

- `app/llm/`
- Retry
- Fallback
- Circuit breaker
- Structured output
- Function calling
- `llm_calls`

### P4 Observability Hardening

- `RuntimeTraceSnapshot`
- `ReplaySnapshot`
- `DebugSnapshot`
- `MetricsSnapshot`
- `safe_observe`
- Recorder isolation

## Runtime Capabilities

The current Runtime already includes:

- Task DAG Runtime
- Parser Repair
- Queue Scheduling
- Router Policy
- Executor Stability
- LLM Retry / Fallback / Circuit Breaker
- Runtime Trace
- Replay Snapshot
- Metrics Snapshot
- Debug Snapshot
- Core Regression Test Suite

## Verified Results

- `rag310` environment is healthy
- Environment self-check passed
  - `python -c "import asyncio, random, secrets, ssl; print('ok')"`
- Smoke test passed
  - `python -m pytest tests/test_simple_flow.py -q`
  - Result: `1 passed`
- Core regression passed
  - Result: `112 passed`
- `Runtime Alpha v0.6` is ready to freeze

## Not Recommended Right Now

The following are not recommended as immediate next work:

- RAG
- Memory
- Multi-Agent
- Streaming
- Tool Marketplace
- Dashboard
- Distributed Runtime
- PostgreSQL / Redis / Milvus integration

Reason:

The current priority is to freeze Alpha, stabilize the test baseline, and record known debt before expanding scope.

## Next Stage Suggestion

The next stage can move toward `P5 Persistence Foundation`.

The first step of P5 should not be writing database integration directly. It should start with:

- Persistence data model
- `request` table
- `task` table
- `task_event` table
- `trace_snapshot` table
- `replay_snapshot` table
- Projection / DTO layer
