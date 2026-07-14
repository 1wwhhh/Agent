from __future__ import annotations

from threading import RLock
from typing import Any

from app.schemas.observability import RequestMetricsSnapshot


class RuntimeMetricsCollector:
    def __init__(self) -> None:
        self._lock = RLock()
        self._metrics_by_request: dict[str, RequestMetricsSnapshot] = {}

    def record(self, snapshot: RequestMetricsSnapshot) -> None:
        with self._lock:
            self._metrics_by_request[snapshot.request_id] = snapshot

    def export_metrics(self) -> dict[str, Any]:
        with self._lock:
            snapshots = list(self._metrics_by_request.values())

        total_requests = len(snapshots)
        if total_requests == 0:
            return {
                "total_requests": 0,
                "averages": {},
                "requests": {},
            }

        latency_keys = sorted({key for item in snapshots for key in item.latency.keys()})
        averages = {
            "task_success_rate": sum(item.task_success_rate for item in snapshots) / total_requests,
            "dag_correctness_rate": sum(item.dag_correctness_rate for item in snapshots) / total_requests,
            "retry_rate": sum(item.retry_rate for item in snapshots) / total_requests,
            "retry_count": sum(item.retry_count for item in snapshots) / total_requests,
            "context_consistency_rate": sum(item.context_consistency_rate for item in snapshots) / total_requests,
            "latency": {
                key: sum(item.latency.get(key, 0.0) for item in snapshots) / total_requests for key in latency_keys
            },
        }
        return {
            "total_requests": total_requests,
            "averages": averages,
            "requests": {
                item.request_id: item.model_dump(mode="json")
                for item in snapshots
            },
        }

    def reset(self) -> None:
        with self._lock:
            self._metrics_by_request.clear()
