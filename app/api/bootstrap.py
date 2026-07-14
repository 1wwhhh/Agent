from __future__ import annotations

from app.adapters import ModelRouter
from app.api.runtime import build_runtime_components
from app.api.schemas import AgentRequest
from app.schemas.model import RuntimeLLMConfig
from app.utils import load_project_env

# 读模型配置
async def build_env_runtime_components(_: AgentRequest):
    load_project_env()
    config = RuntimeLLMConfig.from_env()
    client = ModelRouter().build_client(config)
    return await build_runtime_components(client=client)
