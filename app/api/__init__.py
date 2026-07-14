from app.api.schemas import AgentRequest, AgentResponse, AgentTaskState
from app.api.runtime import RuntimeComponents, build_graph_runtime, build_runtime_components
from app.api.service import export_metrics, run_runtime

__all__ = [
    "AgentRequest",
    "AgentResponse",
    "AgentTaskState",
    "RuntimeComponents",
    "app",
    "build_graph_runtime",
    "build_runtime_components",
    "export_metrics",
    "router",
    "run_runtime",
]


def __getattr__(name: str):
    if name == "app":
        from app.api.server import app

        return app
    if name == "router":
        from app.api.router import router

        return router
    raise AttributeError(f"module 'app.api' has no attribute {name!r}")
