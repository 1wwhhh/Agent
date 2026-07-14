# Runtime Changes

## Runtime Task Type Ownership Refactor

### Background

Planner output occasionally used a tool name as `task_type`. The concrete failure that motivated this change was:

- `tool = rag_batch_summarize_tool`
- `task_type = rag_batch_summarize_tool`

The valid Runtime task type is `rag_batch_summary`. Trusting Planner output allowed invalid tasks to reach Router and Executor.

### Previous Compatibility Layer

A temporary alias existed:

```python
TASK_TYPE_ALIASES = {
    "rag_batch_summarize_tool": "rag_batch_summary",
}
```

This alias remains only as a compatibility shim for old checkpoints and old Planner artifacts. New tools must declare task type ownership through `ToolCapability.supported_task_types` and `ToolCapability.default_task_type`. Do not add new aliases.

### New Ownership Model

- Planner decides the tool.
- Runtime decides the final `task_type` from the selected tool's `ToolCapability`.
- Router validates the resolved task strictly.

Runtime no longer treats Planner as the trusted source for final `task_type`.

### Capability Contract

`ToolCapability` now includes:

- `supported_task_types: list[str]`
- `default_task_type: str | None`

Rules:

- Empty `supported_task_types` remains valid for historical Router configuration semantics.
- Runtime Resolver rejects a planned task that uses a tool with empty `supported_task_types`.
- A single supported task type automatically becomes `default_task_type` when no default is provided.
- `default_task_type` must be inside `supported_task_types`.
- A single-type tool cannot declare a different default.

### Runtime Resolver Behavior

For single-task-type tools, Runtime overwrites Planner output with the only legal task type. This covers wrong, missing, or tool-name-shaped Planner output.

For multi-task-type tools, Runtime does not guess. Planner must provide an explicit task type that is in `supported_task_types`.

Resolver emits DEBUG trace logs with:

- `task_id`
- `tool`
- `planner_task_type`
- `resolved_task_type`

### Coverage

Runtime task type resolution is applied in these entry points:

- Parser normal parse path before `state.set_planned_tasks(...)`
- Parser checkpoint/resume path before the early return
- Simple task construction before planned tasks enter Queue
- Queue bootstrap before `TaskQueue.initialize(...)`
- Checkpoint resume hydrate path before `TaskQueue.hydrate(...)`
- Executor bootstrap when it has to recreate Queue

### RAG Chain Result

The RAG path now resolves:

```text
rag_search_tool -> rag_batch_summarize_tool -> text_generate_tool
```

When Planner emits:

```text
tool = rag_batch_summarize_tool
task_type = rag_batch_summarize_tool
```

Runtime resolves it before Queue/Router/Executor to:

```text
tool = rag_batch_summarize_tool
task_type = rag_batch_summary
```

### Tests

Targeted validation command:

```bash
conda run -n Agent python -m pytest -q tests/test_runtime_graph_rag_flow.py tests/test_rag_planning_decision.py tests/test_router_permissions.py
```

Result before this close-out pass:

```text
48 passed
```

The close-out pass adds a full Queue/Executor RAG DAG test and keeps the same ownership rules.

Close-out validation result:

```text
49 passed
```

Full test collection currently fails on unrelated historical debt documented in `docs/known_debt.md`:

- `app.tools.ppt_render` missing
- `app.ppt` missing
- `app.rag.query_parser.build_search_payload` missing/export drift
