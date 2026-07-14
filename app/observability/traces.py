from __future__ import annotations

import asyncio
import json
from pathlib import Path
from threading import Lock
from typing import Any

from app.schemas.observability import PersistedExecutionTrace, PersistedTraceEvent, RequestMetricsSnapshot
from app.state import LangGraphState
from app.observability.compact import compact_observability_payload


class LocalExecutionTraceStore:
    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory)
        self._lock = Lock()
        self._sequence_by_request: dict[str, int] = {}

    async def record_request_event(
        self,
        *,
        request_id: str,
        session_id: str,
        layer: str,
        event: str,
        metadata: dict[str, Any] | None = None,
    ) -> PersistedTraceEvent:
        return await asyncio.to_thread(
            self._record_request_event_sync,
            request_id,
            session_id,
            layer,
            event,
            metadata or {},
        )

    async def record_state_event(
        self,
        *,
        state: LangGraphState,
        source_layer: str,
        event: str,
        metadata: dict[str, Any] | None = None,
    ) -> PersistedTraceEvent:
        payload = {
            "request_id": state.context.runtime.request_id,
            "session_id": state.context.runtime.session_id,
            "layer": source_layer,
            "event": event,
            "phase": state.phase.value,
            "current_node": state.current_node,
            "last_completed_node": state.last_completed_node,
            "supervisor_route": state.supervisor_route,
            "node_execution_order": list(state.metadata.get("completed_nodes", [])),
            "task_states": _build_task_states(state),
            "task_graph": _build_task_graph(state),
            "execution_history": compact_observability_payload(
                [item.model_dump(mode="json") for item in state.context.execution_history]
            ),
            "tool_calls": compact_observability_payload(_build_tool_calls(state)),
            "context_snapshot": compact_observability_payload(state.context.model_dump(mode="json")),
            "metadata": metadata or {},
        }
        return await asyncio.to_thread(self._record_state_event_sync, payload)

    async def attach_metrics(self, request_id: str, metrics: RequestMetricsSnapshot) -> None:
        await asyncio.to_thread(self._attach_metrics_sync, request_id, metrics)

    async def load_trace(self, request_id: str) -> PersistedExecutionTrace:
        return await asyncio.to_thread(self._load_trace_sync, request_id)

    async def load_latest_trace(
        self,
        *,
        request_id: str | None = None,
        session_id: str | None = None,
    ) -> PersistedExecutionTrace | None:
        return await asyncio.to_thread(self._load_latest_trace_sync, request_id, session_id)

    async def reset(self) -> None:
        await asyncio.to_thread(self._reset_sync)

    def _record_request_event_sync(
        self,
        request_id: str,
        session_id: str,
        layer: str,
        event: str,
        metadata: dict[str, Any],
    ) -> PersistedTraceEvent:
        with self._lock:
            self.directory.mkdir(parents=True, exist_ok=True)
            trace = self._load_trace_locked(request_id=request_id, session_id=session_id)
            sequence = self._next_sequence_locked(request_id)
            entry = PersistedTraceEvent(
                request_id=request_id,
                session_id=session_id,
                sequence=sequence,
                layer=layer,
                event=event,
                metadata=metadata,
            )
            trace.events.append(entry)
            trace.updated_at = entry.timestamp
            self._save_trace_locked(trace)
            return entry

    def _record_state_event_sync(self, payload: dict[str, Any]) -> PersistedTraceEvent:
        request_id = str(payload["request_id"])
        session_id = str(payload["session_id"])
        with self._lock:
            self.directory.mkdir(parents=True, exist_ok=True)
            trace = self._load_trace_locked(request_id=request_id, session_id=session_id)
            sequence = self._next_sequence_locked(request_id)
            entry = PersistedTraceEvent(
                request_id=request_id,
                session_id=session_id,
                sequence=sequence,
                layer=str(payload["layer"]),
                event=str(payload["event"]),
                phase=payload.get("phase"),
                current_node=payload.get("current_node"),
                last_completed_node=payload.get("last_completed_node"),
                supervisor_route=payload.get("supervisor_route"),
                node_execution_order=list(payload.get("node_execution_order", [])),
                task_states=dict(payload.get("task_states", {})),
                task_graph=dict(payload.get("task_graph", {})),
                execution_history=list(payload.get("execution_history", [])),
                tool_calls=list(payload.get("tool_calls", [])),
                context_snapshot=dict(payload.get("context_snapshot", {})),
                metadata=dict(payload.get("metadata", {})),
            )
            trace.events.append(entry)
            trace.updated_at = entry.timestamp
            self._save_trace_locked(trace)
            return entry

    def _attach_metrics_sync(self, request_id: str, metrics: RequestMetricsSnapshot) -> None:
        with self._lock:
            self.directory.mkdir(parents=True, exist_ok=True)
            trace = self._load_trace_locked(request_id=request_id, session_id=metrics.session_id)
            trace.metrics = metrics
            trace.updated_at = metrics.recorded_at
            self._save_trace_locked(trace)

    def _load_trace_sync(self, request_id: str) -> PersistedExecutionTrace:
        with self._lock:
            path = self._trace_path(request_id)
            if not path.exists():
                raise FileNotFoundError(f"trace not found for request_id={request_id}")
            trace = PersistedExecutionTrace.model_validate(json.loads(path.read_text("utf-8")))
            self._sequence_by_request[request_id] = max((event.sequence for event in trace.events), default=0)
            return trace

    def _load_latest_trace_sync(
        self,
        request_id: str | None,
        session_id: str | None,
    ) -> PersistedExecutionTrace | None:
        with self._lock:
            if request_id:
                path = self._trace_path(request_id)
                if not path.exists():
                    return None
                return PersistedExecutionTrace.model_validate(json.loads(path.read_text("utf-8")))
            if session_id is None or not self.directory.exists():
                return None

            latest: PersistedExecutionTrace | None = None
            for path in self.directory.glob("*.json"):
                try:
                    trace = PersistedExecutionTrace.model_validate(json.loads(path.read_text("utf-8")))
                except Exception:
                    continue
                if trace.session_id != session_id:
                    continue
                if latest is None or trace.updated_at > latest.updated_at:
                    latest = trace
            return latest

    def _reset_sync(self) -> None:
        with self._lock:
            self._sequence_by_request.clear()
            if self.directory.exists():
                for path in self.directory.glob("*.json"):
                    path.unlink()

    def _load_trace_locked(self, *, request_id: str, session_id: str) -> PersistedExecutionTrace:
        path = self._trace_path(request_id)
        if not path.exists():
            return PersistedExecutionTrace(request_id=request_id, session_id=session_id)
        raw = path.read_text("utf-8")
        if not raw.strip():
            return PersistedExecutionTrace(request_id=request_id, session_id=session_id)
        trace = PersistedExecutionTrace.model_validate(json.loads(raw))
        self._sequence_by_request[request_id] = max((event.sequence for event in trace.events), default=0)
        return trace

    def _save_trace_locked(self, trace: PersistedExecutionTrace) -> None:
        path = self._trace_path(trace.request_id)
        payload = json.dumps(trace.model_dump(mode="json"), ensure_ascii=False, indent=2)
        path.write_text(payload, "utf-8")

    def _next_sequence_locked(self, request_id: str) -> int:
        sequence = self._sequence_by_request.get(request_id, 0) + 1
        self._sequence_by_request[request_id] = sequence
        return sequence

    def _trace_path(self, request_id: str) -> Path:
        return self.directory / f"{request_id}.json"


def _build_task_states(state: LangGraphState) -> dict[str, dict[str, Any]]:
    return {
        task_id: {
            "status": str(task.status.value if hasattr(task.status, "value") else task.status),
            "retry_count": task.retry_count,
            "max_retry": task.max_retry,
            "depends_on": list(task.depends_on),
            "output_key": task.output_key,
            "tool": task.tool,
            "idempotency_key": task.idempotency_key,
        }
        for task_id, task in state.context.tasks.items()
    }


def _build_task_graph(state: LangGraphState) -> dict[str, Any]:
    tasks = list(state.context.tasks.values())
    return {
        "nodes": [task.model_dump(mode="json") for task in tasks],
        "edges": [
            {"from": dependency, "to": task.task_id}
            for task in tasks
            for dependency in task.depends_on
        ],
    }


def _build_tool_calls(state: LangGraphState) -> list[dict[str, Any]]:
    tasks_by_id = state.context.tasks
    task_results = state.context.task_results
    tool_calls: list[dict[str, Any]] = []
    for record in state.context.tool_call_chain:
        task = tasks_by_id.get(record.task_id) if record.task_id else None
        output = None
        if task is not None and task.output_key in task_results:
            output = task_results.get(task.output_key)
        elif isinstance(record.metadata, dict):
            output = record.metadata.get("output")
        tool_calls.append(
            {
                "tool_name": record.tool_name,
                "task_id": record.task_id,
                "input": record.metadata.get("input") if isinstance(record.metadata, dict) else None,
                "output": output,
                "status": record.status,
                "timestamp": record.started_at.isoformat(),
                "latency_ms": record.latency_ms,
            }
        )
    return tool_calls
