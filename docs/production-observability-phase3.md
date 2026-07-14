# Phase 3 - Production Observability & Runtime Reliability

## Scope

Phase 3 keeps the existing LangGraph runtime flow unchanged:

- no DAG redesign
- no Queue redesign
- no Executor rewrite
- no Supervisor / Planner / Executor flow rewrite

This phase only adds production observability and runtime reliability around the current execution path.

## Delivered Capabilities

### 1. Replay System

- Added local execution trace persistence under `outputs/runtime_traces/`
- Each `request_id` stores an immutable event log
- Stored trace content includes:
  - node execution order
  - task state transitions
  - tool calls
  - context snapshots
- Added replay engine:
  - `replay(request_id)` through `RuntimeReplayEngine`
  - full replay
  - step-by-step replay
  - HTTP replay through `POST /run?replay=true`

Replay is deterministic reconstruction only. It does not re-execute tools or LLM calls.

### 2. Metrics Persistence Layer

- Added in-memory metrics collector
- Per-request metrics now persist beyond log output
- Collected metrics include:
  - `task_success_rate`
  - `dag_correctness_rate`
  - `retry_rate`
  - `retry_count`
  - `context_consistency_rate`
  - latency breakdown:
    - `supervisor`
    - `planner`
    - `queue`
    - `executor`
    - `total`
- Export entrypoint:
  - `app.api.service.export_metrics()`

### 3. LLM Reliability Layer

- Centralized retry behavior in `LLMClient.generate()`
- Added exponential backoff retry policy
- Retained per-provider timeout handling through the shared abstraction
- Added transparent failover client:
  - primary provider
  - secondary provider
  - fallback path through the same `LLMClient` interface
- Circuit breaker state remains enforced per provider client

## Runtime Integration Points

### Trace Persistence

Trace persistence is attached to the existing checkpoint save path:

- graph node checkpoint save
- queue checkpoint save
- executor checkpoint save

This preserves current runtime architecture while capturing replayable artifacts.

### Metrics Recording

Metrics are recorded only after a real runtime execution completes.

Replay mode does not record new execution metrics.

### HTTP Gateway

`POST /run` now supports:

- `debug=true`
- `replay=true`

Replay requests can provide:

- `request_id`
- or `session_id` to locate the latest persisted trace

## Added Tests

- `tests/test_replay_system.py`
- `tests/test_metrics_persistence.py`
- `tests/test_failover_llm_client.py`

These tests extend the existing suite and are designed to remain backward compatible with earlier runtime validations.
