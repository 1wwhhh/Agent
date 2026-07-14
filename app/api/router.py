"""FastAPI 路由层，只负责接收请求并调用 Runtime。"""

from __future__ import annotations

from fastapi import APIRouter, Query

from app.api.schemas import AgentRequest, AgentResponse
from app.api.service import run_runtime

router = APIRouter(tags=["runtime"])


@router.post("/run", response_model=AgentResponse)
async def run_agent(
    request: AgentRequest,
    debug: bool = Query(default=False),
    replay: bool = Query(default=False),
) -> AgentResponse:
    """统一运行入口，HTTP 层不参与任务规划、调度或执行。"""
    return await run_runtime(request, debug=debug, replay=replay)
