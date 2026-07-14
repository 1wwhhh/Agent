from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any
from uuid import uuid4

from app.observability import LocalExecutionTraceStore
from app.schemas.checkpoint import RuntimeCheckpoint
from app.utils import configure_runtime_logger, runtime_log

# 本地储存
class LocalCheckpointStore:
    """基于本地文件系统的检查点存储。"""

    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory)

    async def save(self, checkpoint: RuntimeCheckpoint) -> Path:
        """把检查点保存到本地 JSON 文件。"""
        await asyncio.to_thread(self.directory.mkdir, parents=True, exist_ok=True)
        path = self.directory / f"{checkpoint.request_id}__{checkpoint.checkpoint_id}.json"
        payload = checkpoint.model_dump(mode="json")
        await asyncio.to_thread(path.write_text, json.dumps(payload, ensure_ascii=False, indent=2), "utf-8")
        return path

    async def load(self, checkpoint_id: str) -> RuntimeCheckpoint:
        """按检查点 ID 读取快照。"""
        pattern = f"*__{checkpoint_id}.json"
        matches = list(self.directory.glob(pattern))
        if not matches:
            raise FileNotFoundError(f"未找到检查点：{checkpoint_id}")
        raw = await asyncio.to_thread(matches[0].read_text, "utf-8")
        return RuntimeCheckpoint.model_validate(json.loads(raw))

    async def load_latest(
        self,
        *,
        request_id: str | None = None,
        session_id: str | None = None,
    ) -> RuntimeCheckpoint | None:
        """读取指定请求或会话下最新的一条检查点。"""
        if not self.directory.exists():
            return None

        def _load_latest_sync() -> RuntimeCheckpoint | None:
            candidates: list[RuntimeCheckpoint] = []
            for path in self.directory.glob("*.json"):
                try:
                    checkpoint = RuntimeCheckpoint.model_validate(json.loads(path.read_text("utf-8")))
                except Exception:
                    continue
                if request_id is not None and checkpoint.request_id != request_id:
                    continue
                if session_id is not None and checkpoint.session_id != session_id:
                    continue
                candidates.append(checkpoint)
            if not candidates:
                return None
            candidates.sort(key=lambda item: item.created_at)
            return candidates[-1]

        return await asyncio.to_thread(_load_latest_sync)


class RuntimeCheckpointManager:
    """统一管理运行时检查点的保存、加载与恢复。"""

    def __init__(
        self,
        *,
        store: LocalCheckpointStore,
        enabled: bool = True,
        trace_store: LocalExecutionTraceStore | None = None,
    ) -> None:
        self.store = store
        self.enabled = enabled
        self.trace_store = trace_store
        self.logger = configure_runtime_logger()

    async def save_state(
        self,
        state,
        *,
        source_layer: str,
        event: str,
        metadata: dict[str, Any] | None = None,
    ) -> RuntimeCheckpoint | None:
        """保存当前 LangGraphState 为检查点。"""
        if not self.enabled:
            return None

        checkpoint = RuntimeCheckpoint(
            checkpoint_id=uuid4().hex,
            request_id=state.context.runtime.request_id,
            session_id=state.context.runtime.session_id,
            source_layer=source_layer,
            event=event,
            snapshot_payload=state.serialize_for_checkpoint(),
            metadata=metadata or {},
        )
        path = await self.store.save(checkpoint)
        state.metadata["last_checkpoint_id"] = checkpoint.checkpoint_id
        state.metadata["last_checkpoint_path"] = str(path)
        state.context.set_shared_value(
            "last_checkpoint",
            {
                "checkpoint_id": checkpoint.checkpoint_id,
                "path": str(path),
                "source_layer": source_layer,
                "event": event,
            },
        )
        runtime_log(
            layer="checkpoint",
            event="success",
            data={
                "checkpoint_id": checkpoint.checkpoint_id,
                "path": str(path),
                "source_layer": source_layer,
                "event": event,
            },
            logger=self.logger,
        )
        if self.trace_store is not None:
            await self.trace_store.record_state_event(
                state=state,
                source_layer=source_layer,
                event=event,
                metadata={
                    "checkpoint_id": checkpoint.checkpoint_id,
                    "checkpoint_path": str(path),
                    **(metadata or {}),
                },
            )
        return checkpoint

    async def load_checkpoint(self, checkpoint_id: str) -> RuntimeCheckpoint:
        """读取指定检查点。"""
        checkpoint = await self.store.load(checkpoint_id)
        runtime_log(
            layer="checkpoint",
            event="execute",
            data={"checkpoint_id": checkpoint_id, "mode": "load"},
            logger=self.logger,
        )
        return checkpoint

    async def load_latest_checkpoint(
        self,
        *,
        request_id: str | None = None,
        session_id: str | None = None,
    ) -> RuntimeCheckpoint | None:
        """读取最近一条检查点。"""
        checkpoint = await self.store.load_latest(request_id=request_id, session_id=session_id)
        if checkpoint is not None:
            runtime_log(
                layer="checkpoint",
                event="execute",
                data={
                    "checkpoint_id": checkpoint.checkpoint_id,
                    "mode": "load_latest",
                    "request_id": checkpoint.request_id,
                    "session_id": checkpoint.session_id,
                },
                logger=self.logger,
            )
        return checkpoint

    async def restore_state(self, checkpoint: RuntimeCheckpoint):
        """从检查点恢复 LangGraphState。"""
        from app.state import LangGraphState
        from app.schemas.graph import GraphPhase

        state = LangGraphState.from_checkpoint_payload(checkpoint.snapshot_payload)
        if state.phase == GraphPhase.FAILED:
            state.phase = GraphPhase.PARSED if state.planned_tasks else GraphPhase.INITIALIZED
            state.final_response = None
            state.context.final_output = None
            state.agent_state.final_output_ready = False
        state.metadata["resume_from_checkpoint"] = True
        state.metadata.pop("interrupt_after_task_count", None)
        state.metadata["restored_checkpoint"] = {
            "checkpoint_id": checkpoint.checkpoint_id,
            "source_layer": checkpoint.source_layer,
            "event": checkpoint.event,
            "created_at": checkpoint.created_at.isoformat(),
        }
        runtime_log(
            layer="checkpoint",
            event="success",
            data={
                "checkpoint_id": checkpoint.checkpoint_id,
                "mode": "restore",
                "phase": state.phase.value,
            },
            logger=self.logger,
        )
        return state
