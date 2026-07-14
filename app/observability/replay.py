from __future__ import annotations

from app.observability.traces import LocalExecutionTraceStore
from app.schemas.observability import ReplayMode, ReplayResult, ReplayStep


class RuntimeReplayEngine:
    def __init__(self, *, trace_store: LocalExecutionTraceStore) -> None:
        self.trace_store = trace_store

    async def replay(
        self,
        request_id: str,
        *,
        mode: ReplayMode = ReplayMode.FULL,
    ) -> ReplayResult:
        trace = await self.trace_store.load_trace(request_id)
        steps = [
            ReplayStep(
                sequence=event.sequence,
                layer=event.layer,
                event=event.event,
                phase=event.phase,
                task_states=event.task_states,
                task_graph=event.task_graph,
                tool_calls=event.tool_calls,
                context_snapshot=event.context_snapshot,
                timestamp=event.timestamp,
            )
            for event in trace.events
        ]
        stateful_steps = [step for step in steps if step.context_snapshot or step.task_states or step.task_graph]
        last_step = stateful_steps[-1] if stateful_steps else (steps[-1] if steps else None)
        final_output = {}
        task_states = {}
        trace_summary = {
            "event_count": len(steps),
            "node_execution_order": trace.events[-1].node_execution_order if trace.events else [],
            "supervisor_route": trace.events[-1].supervisor_route if trace.events else None,
            "metrics": trace.metrics.model_dump(mode="json") if trace.metrics is not None else None,
        }
        if last_step is not None:
            final_output = last_step.context_snapshot.get("final_output") or {}
            task_states = last_step.task_states

        if mode == ReplayMode.STEP_BY_STEP:
            return ReplayResult(
                request_id=trace.request_id,
                session_id=trace.session_id,
                mode=mode,
                steps=steps,
                final_output=final_output,
                task_states=task_states,
                trace_summary=trace_summary,
            )

        return ReplayResult(
            request_id=trace.request_id,
            session_id=trace.session_id,
            mode=mode,
            steps=steps,
            final_output=final_output,
            task_states=task_states,
            trace_summary=trace_summary,
        )
