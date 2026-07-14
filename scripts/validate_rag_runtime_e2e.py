from __future__ import annotations

import asyncio
import importlib.util
import json
import os
import sys
import types
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import Field

ROOT = Path(__file__).resolve().parents[1]
APP_DIR = ROOT / "app"
OUTPUT_PATH = ROOT / "outputs" / "rag_runtime_e2e_result.json"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _ensure_package(name: str, path: Path) -> types.ModuleType:
    module = sys.modules.get(name)
    if module is None:
        module = types.ModuleType(name)
        module.__path__ = [str(path)]
        sys.modules[name] = module
    return module


def _load_module(name: str, path: Path) -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"failed to load module spec for {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _bootstrap_runtime_imports() -> dict[str, types.ModuleType]:
    executor_pkg = _ensure_package("app.executor", APP_DIR / "executor")
    router_pkg = _ensure_package("app.router", APP_DIR / "router")
    queue_pkg = _ensure_package("app.queue", APP_DIR / "queue")

    exceptions_mod = _load_module("app.executor.exceptions", APP_DIR / "executor" / "exceptions.py")
    _load_module("app.executor.retry_policy", APP_DIR / "executor" / "retry_policy.py")
    permissions_mod = _load_module("app.router.permissions", APP_DIR / "router" / "permissions.py")
    capability_mod = _load_module("app.router.capability", APP_DIR / "router" / "capability.py")
    task_router_mod = _load_module("app.router.task_router", APP_DIR / "router" / "task_router.py")
    queue_mod = _load_module("app.queue.task_queue", APP_DIR / "queue" / "task_queue.py")
    task_executor_mod = _load_module("app.executor.task_executor", APP_DIR / "executor" / "task_executor.py")

    setattr(router_pkg, "PermissionContext", permissions_mod.PermissionContext)
    setattr(router_pkg, "TaskRouter", task_router_mod.TaskRouter)
    setattr(router_pkg, "TaskRouterError", task_router_mod.TaskRouterError)
    setattr(router_pkg, "RouterConfigurationError", task_router_mod.RouterConfigurationError)

    setattr(queue_pkg, "TaskQueue", queue_mod.TaskQueue)
    setattr(queue_pkg, "TaskQueueError", queue_mod.TaskQueueError)

    setattr(executor_pkg, "TaskExecutor", task_executor_mod.TaskExecutor)
    setattr(executor_pkg, "TaskExecutorError", task_executor_mod.TaskExecutorError)
    setattr(executor_pkg, "RetryableToolError", exceptions_mod.RetryableToolError)
    setattr(executor_pkg, "NonRetryableToolError", exceptions_mod.NonRetryableToolError)

    return {
        "capability": capability_mod,
        "task_router": task_router_mod,
        "task_queue": queue_mod,
        "task_executor": task_executor_mod,
    }


def _check_runtime_registration() -> dict[str, bool]:
    runtime_source = (APP_DIR / "api" / "runtime.py").read_text(encoding="utf-8")
    return {
        "rag_tool_import_present": "from app.tools.rag_search import RAGSearchTool" in runtime_source,
        "rag_tool_instantiation_present": "rag_tool = RAGSearchTool(client=client)" in runtime_source,
        "rag_tool_registration_present": "(rag_tool, capability_from_tool(rag_tool))" in runtime_source,
        "rag_batch_tool_import_present": "from app.tools.rag_batch_summarize import RAGBatchSummarizeTool" in runtime_source,
        "rag_batch_tool_registration_present": "(rag_batch_summarize_tool, capability_from_tool(rag_batch_summarize_tool))"
        in runtime_source,
    }


def _build_plan(query: str, top_k: int) -> list[dict[str, Any]]:
    return [
        {
            "task_id": "task_1",
            "task_name": "search_knowledge_base",
            "description": "Search the company knowledge base for relevant material.",
            "task_type": "rag_search",
            "tool": "rag_search_tool",
            "input": {
                "query": query,
                "top_k": top_k,
            },
            "output_key": "rag_context",
            "depends_on": [],
            "priority": 1,
            "tags": ["rag", "search"],
            "status": "PENDING",
            "retry_count": 0,
            "max_retry": 1,
            "timeout": 60,
        },
        {
            "task_id": "task_2",
            "task_name": "summarize_rag_batches",
            "description": "Summarize the retrieved RAG batches.",
            "task_type": "rag_batch_summary",
            "tool": "rag_batch_summarize_tool",
            "input": {
                "query": query,
                "rag_output_key": "rag_context",
            },
            "output_key": "rag_summary",
            "depends_on": ["task_1"],
            "priority": 2,
            "tags": ["rag", "summary"],
            "status": "PENDING",
            "retry_count": 0,
            "max_retry": 1,
            "timeout": 1800,
        },
        {
            "task_id": "task_3",
            "task_name": "generate_answer_from_rag_summary",
            "description": "Generate the final answer from summarized RAG evidence.",
            "task_type": "text_generation",
            "tool": "text_generate_tool",
            "input": {
                "prompt": "Please answer from the summarized RAG evidence. If the evidence is insufficient, say so clearly.",
                "context": "{{rag_summary.text}}",
                "style": "clear",
                "audience": "business_user",
            },
            "output_key": "final_result",
            "depends_on": ["task_2"],
            "priority": 3,
            "tags": ["llm", "generation"],
            "status": "PENDING",
            "retry_count": 0,
            "max_retry": 1,
            "timeout": 75,
        },
    ]


def _parse_iso_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    normalized = value.replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(normalized)
    except ValueError:
        return None


async def _run_validation(*, query: str, top_k: int) -> dict[str, Any]:
    modules = _bootstrap_runtime_imports()

    from app.adapters import ModelRouter
    from app.schemas.context import AgentState, ContextStore, RuntimeContext
    from app.schemas.model import RuntimeLLMConfig
    from app.schemas.task import TaskModel
    from app.tools.llm_reason import LLMReasonTool
    from app.tools.rag_batch_summarize import RAGBatchSummarizeTool
    from app.tools.rag_search import RAGSearchTool
    from app.tools.text_generate import TextGenerateTool
    from app.utils import load_project_env

    class InspectingTextGenerateTool(TextGenerateTool):
        debug_capture: dict[str, Any] = Field(default_factory=dict, exclude=True)

        def resolve_payload_templates(
            self,
            *,
            payload: dict[str, Any],
            context: ContextStore | None = None,
        ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
            resolved_payload, injection_results = super().resolve_payload_templates(payload=payload, context=context)
            self.debug_capture["resolved_payload"] = resolved_payload
            self.debug_capture["template_injections"] = injection_results
            return resolved_payload, injection_results

        def log_prompt_render(
            self,
            *,
            payload: dict[str, Any],
            rendered_prompt,
            context: ContextStore | None,
            injection_results: list[dict[str, Any]],
        ) -> None:
            self.debug_capture["prompt_payload"] = payload
            self.debug_capture["rendered_prompt"] = {
                "name": rendered_prompt.name,
                "version": rendered_prompt.version,
                "system_prompt": rendered_prompt.system_prompt,
                "user_prompt": rendered_prompt.user_prompt,
            }
            self.debug_capture["output_key_injections"] = injection_results
            super().log_prompt_render(
                payload=payload,
                rendered_prompt=rendered_prompt,
                context=context,
                injection_results=injection_results,
            )

    load_project_env()

    rag_base_url = os.getenv("RAG_BASE_URL", "").strip()
    runtime_checks = _check_runtime_registration()
    client = ModelRouter().build_client(RuntimeLLMConfig.from_env())

    try:
        router = modules["task_router"].TaskRouter()
        capability_from_tool = modules["capability"].capability_from_tool
        rag_tool = RAGSearchTool(client=client)
        rag_batch_tool = RAGBatchSummarizeTool(client=client)
        reason_tool = LLMReasonTool(client=client)
        text_tool = InspectingTextGenerateTool(client=client)
        await router.register_tools(
            [
                (rag_tool, capability_from_tool(rag_tool)),
                (rag_batch_tool, capability_from_tool(rag_batch_tool)),
                (reason_tool, capability_from_tool(reason_tool)),
                (text_tool, capability_from_tool(text_tool)),
            ]
        )

        runtime = RuntimeContext(
            request_id="req_rag_runtime_e2e",
            session_id="sess_rag_runtime_e2e",
            user_input=f"请回答：{query}",
        )
        context = ContextStore(runtime=runtime)
        state = AgentState(context=context)
        tasks = [TaskModel.model_validate(task) for task in _build_plan(query=query, top_k=top_k)]

        queue = modules["task_queue"].TaskQueue(context=context, state=state, max_concurrency=2)
        await queue.initialize(tasks)
        executor = modules["task_executor"].TaskExecutor(context=context, queue=queue, router=router)
        execution_results = await executor.execute_until_complete()

        rendered_context, replacements = context.render_template_string("{{rag_summary.text}}")
        rag_context = context.task_results.get("rag_context")
        rag_summary = context.task_results.get("rag_summary")
        final_result = context.task_results.get("final_result")
        task_1 = context.tasks.get("task_1")
        task_2 = context.tasks.get("task_2")
        task_3 = context.tasks.get("task_3")
        result_by_task = {result.task_id: result for result in execution_results}
        task_1_result = result_by_task.get("task_1")
        task_2_result = result_by_task.get("task_2")
        task_3_result = result_by_task.get("task_3")

        task_1_finished_at = _parse_iso_datetime(task_1_result.finished_at.isoformat()) if task_1_result is not None else None
        task_2_finished_at = _parse_iso_datetime(task_2_result.finished_at.isoformat()) if task_2_result is not None else None
        task_3_started_at = _parse_iso_datetime(task_3_result.started_at.isoformat()) if task_3_result is not None else None

        joined_context = rag_context.get("joined_context", "") if isinstance(rag_context, dict) else ""
        rag_summary_text = rag_summary.get("text", "") if isinstance(rag_summary, dict) else ""
        resolved_payload = text_tool.debug_capture.get("resolved_payload", {})
        rendered_prompt = text_tool.debug_capture.get("rendered_prompt", {})
        rendered_user_prompt = str(rendered_prompt.get("user_prompt", ""))

        llm_request = (
            task_3_result.tool_result.metadata.get("request", {})
            if task_3_result is not None and task_3_result.tool_result is not None
            else {}
        )
        llm_raw_response = (
            task_3_result.tool_result.metadata.get("raw_response", {})
            if task_3_result is not None and task_3_result.tool_result is not None
            else {}
        )
        llm_messages = llm_request.get("messages", []) if isinstance(llm_request, dict) else []
        llm_user_prompt = ""
        if isinstance(llm_messages, list) and len(llm_messages) > 1 and isinstance(llm_messages[1], dict):
            llm_user_prompt = str(llm_messages[1].get("content", ""))

        final_result_matches_tool_output = (
            task_3_result is not None
            and task_3_result.tool_result is not None
            and final_result == task_3_result.tool_result.output
        )

        return {
            "runtime_registration_checks": runtime_checks,
            "rag_base_url": rag_base_url,
            "registered_tools": await router.list_tools(enabled_only=True),
            "manual_plan": _build_plan(query=query, top_k=top_k),
            "task_statuses": {task_id: str(task.status) for task_id, task in context.tasks.items()},
            "task_execution_checks": {
                "task_1_output_key": task_1.output_key if task_1 is not None else None,
                "task_2_depends_on": list(task_2.depends_on) if task_2 is not None else [],
                "task_2_output_key": task_2.output_key if task_2 is not None else None,
                "task_3_depends_on": list(task_3.depends_on) if task_3 is not None else [],
                "task_3_executed": task_3_result is not None,
                "task_3_success": bool(task_3_result.success) if task_3_result is not None else False,
                "task_3_failed": bool(task_3_result is not None and not task_3_result.success),
                "task_3_tool_result_present": task_3_result is not None and task_3_result.tool_result is not None,
                "task_3_started_after_task_2_finished": (
                    task_2_finished_at is not None and task_3_started_at is not None and task_3_started_at >= task_2_finished_at
                ),
                "task_2_started_after_task_1_finished": (
                    task_1_finished_at is not None
                    and task_2_result is not None
                    and _parse_iso_datetime(task_2_result.started_at.isoformat()) is not None
                    and _parse_iso_datetime(task_2_result.started_at.isoformat()) >= task_1_finished_at
                ),
            },
            "execution_result_count": len(execution_results),
            "execution_results": [result.model_dump(mode="json") for result in execution_results],
            "rag_context_present": isinstance(rag_context, dict),
            "rag_context_chunk_count": len(rag_context.get("chunks", [])) if isinstance(rag_context, dict) else 0,
            "rag_summary_present": isinstance(rag_summary, dict),
            "rag_joined_context_preview": str(joined_context)[:1200],
            "template_resolution": {
                "resolved": "{{rag_summary.text}}" not in rendered_context,
                "preview": rendered_context[:1200],
                "replacements": replacements,
            },
            "prompt_injection_checks": {
                "resolved_payload": resolved_payload,
                "resolved_payload_context_preview": str(resolved_payload.get("context", ""))[:1200],
                "resolved_payload_contains_placeholder": "{{rag_summary.text}}" in str(resolved_payload.get("context", "")),
                "rendered_prompt": rendered_prompt,
                "rendered_user_prompt_contains_placeholder": "{{rag_summary.text}}" in rendered_user_prompt,
                "rendered_user_prompt_contains_summary_text": bool(rag_summary_text)
                and rag_summary_text[:200] in rendered_user_prompt,
                "joined_context_preview": str(joined_context)[:1200],
            },
            "llm_generation_checks": {
                "llm_called": bool(llm_request),
                "llm_request": llm_request,
                "llm_user_prompt_preview": llm_user_prompt[:1200],
                "llm_response_text_preview": str(
                    (((llm_raw_response.get("choices") or [{}])[0].get("message") or {}).get("content", ""))
                )[:1200],
                "final_result_matches_tool_output": final_result_matches_tool_output,
                "final_result_text_preview": str(final_result.get("text", ""))[:1200]
                if isinstance(final_result, dict)
                else "",
            },
            "final_result_present": isinstance(final_result, dict),
            "final_result": final_result,
            "success": isinstance(final_result, dict) and bool(final_result.get("text")),
        }
    finally:
        await client.aclose()


async def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    query = os.getenv("RAG_E2E_QUERY", "今年公司都做了哪些申报")
    top_k_raw = os.getenv("RAG_E2E_TOP_K", "3").strip()
    try:
        top_k = int(top_k_raw)
    except ValueError:
        top_k = 3

    result = await _run_validation(query=query, top_k=max(1, top_k))
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result.get("success") else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
