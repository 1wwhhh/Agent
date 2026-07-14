from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from app.planner import ParserRepairExhaustedError, RepairPipeline, TaskParser
from app.state import LangGraphState
from tests.support.models import RuntimeTestInput, TaskSpec
from tests.support.runtime_runner import run_runtime
from tests.support.tools import ParserRepairTestLLMClient, build_planner_json


def _base_task_specs() -> list[TaskSpec]:
    return [
        TaskSpec(
            task_id="task_1",
            task_name="repair_target",
            description="A task repaired by the parser repair chain.",
            tool="text_generate_tool",
            prompt="Repair me",
            output_key="repair_output",
        )
    ]


def _build_runtime_input(
    name: str,
    *,
    planner_raw_output: str,
    parser_repair_outputs: list[str] | None = None,
) -> RuntimeTestInput:
    runtime_input = RuntimeTestInput(
        name=name,
        user_input="Repair the planner output.",
        force_route="COMPLEX_TASK",
        planner_raw_output=planner_raw_output,
        task_specs=_base_task_specs(),
    )
    runtime_input.parser_repair_outputs = parser_repair_outputs or []
    return runtime_input


def _valid_plan_json(name: str, *, task_specs: list[TaskSpec] | None = None) -> str:
    runtime_input = RuntimeTestInput(
        name=f"{name}_valid_plan",
        user_input="Repair the planner output.",
        force_route="COMPLEX_TASK",
        task_specs=task_specs or _base_task_specs(),
    )
    return build_planner_json(runtime_input)


def _missing_field_raw_output() -> str:
    payload = json.loads(_valid_plan_json("missing_field"))
    del payload["tasks"][0]["task_id"]
    return json.dumps(payload)


def _invalid_schema_raw_output() -> str:
    payload = json.loads(_valid_plan_json("invalid_schema"))
    payload["tasks"][0]["input"] = "not-an-object"
    return json.dumps(payload)


def _invalid_dependency_raw_output() -> str:
    payload = json.loads(_valid_plan_json("invalid_dependency"))
    payload["tasks"][0]["depends_on"] = ["ghost_task"]
    return json.dumps(payload)


def _cyclic_dependency_raw_output() -> str:
    task_specs = [
        TaskSpec(
            task_id="task_1",
            task_name="first_task",
            description="First task in cycle.",
            tool="text_generate_tool",
            prompt="First",
            output_key="output_1",
        ),
        TaskSpec(
            task_id="task_2",
            task_name="second_task",
            description="Second task in cycle.",
            tool="text_generate_tool",
            prompt="Second",
            output_key="output_2",
        ),
    ]
    payload = json.loads(_valid_plan_json("cyclic_dependency", task_specs=task_specs))
    payload["tasks"][0]["depends_on"] = ["task_2"]
    payload["tasks"][1]["depends_on"] = ["task_1"]
    return json.dumps(payload)


def _invalid_json_raw_output() -> str:
    return '{"goal": "broken", "tasks": ['


def _markdown_wrapped_valid_plan(name: str) -> str:
    return f"```json\n{_valid_plan_json(name)}\n```"


def _assert_history_contract(history_entry: dict[str, object], *, violated: bool) -> None:
    assert history_entry["output_contract_violation"] is violated
    if violated:
        assert history_entry["violation_reason"] == "repair_output_not_strict_json"
    else:
        assert history_entry["violation_reason"] is None


@pytest.mark.asyncio
async def test_invalid_json_repair_succeeds_on_raw_planner_output_path():
    runtime_input = _build_runtime_input(
        "parser_repair_invalid_json",
        planner_raw_output=_invalid_json_raw_output(),
        parser_repair_outputs=[_valid_plan_json("parser_repair_invalid_json")],
    )
    result = await run_runtime(runtime_input)

    history = result.final_output["metadata"]["parser_repair_history"]
    assert result.final_output["success"] is True
    assert history[0]["repair_type"] == "INVALID_JSON"
    assert history[0]["success"] is True
    _assert_history_contract(history[0], violated=False)
    assert result.execution_trace["repair_llm_call_count"] == 1


@pytest.mark.asyncio
async def test_missing_field_repair_succeeds():
    runtime_input = _build_runtime_input(
        "parser_repair_missing_field",
        planner_raw_output=_missing_field_raw_output(),
        parser_repair_outputs=[_valid_plan_json("parser_repair_missing_field")],
    )
    result = await run_runtime(runtime_input)

    history = result.final_output["metadata"]["parser_repair_history"]
    assert result.final_output["success"] is True
    assert history[0]["repair_type"] == "MISSING_FIELD"
    assert history[0]["success"] is True
    _assert_history_contract(history[0], violated=False)


@pytest.mark.asyncio
async def test_invalid_schema_repair_succeeds():
    runtime_input = _build_runtime_input(
        "parser_repair_invalid_schema",
        planner_raw_output=_invalid_schema_raw_output(),
        parser_repair_outputs=[_valid_plan_json("parser_repair_invalid_schema")],
    )
    result = await run_runtime(runtime_input)

    history = result.final_output["metadata"]["parser_repair_history"]
    assert result.final_output["success"] is True
    assert history[0]["repair_type"] == "INVALID_SCHEMA"
    assert history[0]["success"] is True
    _assert_history_contract(history[0], violated=False)


@pytest.mark.asyncio
async def test_invalid_dependency_repair_succeeds():
    runtime_input = _build_runtime_input(
        "parser_repair_invalid_dependency",
        planner_raw_output=_invalid_dependency_raw_output(),
        parser_repair_outputs=[_valid_plan_json("parser_repair_invalid_dependency")],
    )
    result = await run_runtime(runtime_input)

    history = result.final_output["metadata"]["parser_repair_history"]
    assert result.final_output["success"] is True
    assert history[0]["repair_type"] == "INVALID_DEPENDENCY"
    assert history[0]["success"] is True
    _assert_history_contract(history[0], violated=False)


@pytest.mark.asyncio
async def test_cyclic_dependency_is_classified_as_invalid_dependency():
    runtime_input = _build_runtime_input(
        "parser_repair_cyclic_dependency",
        planner_raw_output=_cyclic_dependency_raw_output(),
        parser_repair_outputs=[_valid_plan_json("parser_repair_cyclic_dependency")],
    )
    result = await run_runtime(runtime_input)

    history = result.final_output["metadata"]["parser_repair_history"]
    assert result.final_output["success"] is True
    assert history[0]["repair_type"] == "INVALID_DEPENDENCY"
    assert history[0]["success"] is True
    _assert_history_contract(history[0], violated=False)


@pytest.mark.asyncio
async def test_retry_exhausted_returns_failed_runtime_and_never_enters_queue():
    runtime_input = _build_runtime_input(
        "parser_repair_retry_exhausted",
        planner_raw_output=_invalid_json_raw_output(),
        parser_repair_outputs=[
            '{"goal": "still-bad", "tasks": [',
            json.dumps({"goal": "still-bad", "tasks": [{"task_name": "missing"}]}),
            _invalid_dependency_raw_output(),
        ],
    )
    result = await run_runtime(runtime_input)

    history = result.final_output["metadata"]["parser_repair_history"]
    failure = result.final_output["metadata"]["failure"]
    completed_nodes = result.final_output["metadata"]["completed_nodes"]
    assert result.final_output["success"] is False
    assert result.final_output["phase"] == "FAILED"
    assert len(history) == 3
    assert "parser repair exhausted" in failure["message"]
    assert "last_repair_type=INVALID_DEPENDENCY" in failure["message"]
    assert "queue" not in completed_nodes
    assert "queue" not in result.execution_trace["node_timings"]


@pytest.mark.asyncio
async def test_parser_repair_exhausted_error_exposes_attributes():
    runtime_input = _build_runtime_input(
        "parser_repair_direct_exhausted_error",
        planner_raw_output=_invalid_json_raw_output(),
        parser_repair_outputs=[
            json.dumps({"goal": "still-bad", "tasks": [{"task_name": "missing"}]}),
        ],
    )
    pipeline = RepairPipeline(
        parser=TaskParser(),
        repair_llm_client=ParserRepairTestLLMClient(test_input=runtime_input),
        max_parse_retry=1,
    )
    state = LangGraphState.create(
        request_id="req_parser_repair_direct_error",
        session_id="sess_parser_repair_direct_error",
        user_input="Repair the planner output.",
    )

    with pytest.raises(ParserRepairExhaustedError) as exc_info:
        await pipeline.run(raw_planner_output=_invalid_json_raw_output(), state=state)

    error = exc_info.value
    assert error.retry_count == 1
    assert error.last_repair_type.value == "MISSING_FIELD"
    assert "TaskPlan schema" in error.last_error_message


@pytest.mark.asyncio
async def test_successful_first_parse_bypasses_repair_llm_and_history():
    valid_output = _valid_plan_json("parser_repair_first_parse_bypass")
    runtime_input = _build_runtime_input(
        "parser_repair_first_parse_bypass",
        planner_raw_output=valid_output,
        parser_repair_outputs=[_invalid_json_raw_output()],
    )
    result = await run_runtime(runtime_input)

    assert result.final_output["success"] is True
    assert "parser_repair_history" not in result.final_output["metadata"]
    assert "parser_repair_history" not in result.context_snapshot["shared_data"]
    assert result.execution_trace["repair_llm_call_count"] == 0


def test_parser_module_does_not_call_llm_directly():
    parser_path = Path("app/planner/parser.py")
    source = parser_path.read_text(encoding="utf-8")
    tree = ast.parse(source)

    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.add(alias.name)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imports.add(node.module)

    assert "app.tools.llm_client" not in imports
    assert "app.schemas.llm" not in imports
    assert "LLMClient" not in source
    assert ".generate(" not in source


@pytest.mark.asyncio
async def test_repair_history_is_visible_in_state_metadata_and_mirrored_to_shared_data():
    runtime_input = _build_runtime_input(
        "parser_repair_history_visibility",
        planner_raw_output=_invalid_json_raw_output(),
        parser_repair_outputs=[_valid_plan_json("parser_repair_history_visibility")],
    )
    result = await run_runtime(runtime_input)

    metadata_history = result.final_output["metadata"]["parser_repair_history"]
    shared_history = result.context_snapshot["shared_data"]["parser_repair_history"]
    assert metadata_history == shared_history
    assert metadata_history[0]["retry_count"] == 1
    assert metadata_history[0]["latency_ms"] >= 0
    _assert_history_contract(metadata_history[0], violated=False)


@pytest.mark.asyncio
async def test_empty_repair_output_retries_until_exhausted():
    runtime_input = _build_runtime_input(
        "parser_repair_empty_output",
        planner_raw_output=_invalid_json_raw_output(),
        parser_repair_outputs=["", "", ""],
    )
    result = await run_runtime(runtime_input)

    history = result.final_output["metadata"]["parser_repair_history"]
    failure = result.final_output["metadata"]["failure"]
    assert result.final_output["success"] is False
    assert len(history) == 3
    assert all("at least 1 character" in str(item["error_message"]) for item in history)
    assert all(item["output_contract_violation"] is True for item in history)
    assert "at least 1 character" in failure["message"]


@pytest.mark.asyncio
async def test_markdown_wrapped_json_is_accepted_and_recorded_as_contract_violation():
    runtime_input = _build_runtime_input(
        "parser_repair_markdown_wrapped",
        planner_raw_output=_invalid_json_raw_output(),
        parser_repair_outputs=[_markdown_wrapped_valid_plan("parser_repair_markdown_wrapped")],
    )
    result = await run_runtime(runtime_input)

    history = result.final_output["metadata"]["parser_repair_history"]
    assert result.final_output["success"] is True
    assert history[0]["success"] is True
    _assert_history_contract(history[0], violated=True)


@pytest.mark.asyncio
async def test_non_task_plan_json_retries_until_exhausted():
    invalid_task_plan_json = json.dumps({"message": "not a task plan"})
    runtime_input = _build_runtime_input(
        "parser_repair_non_task_plan_json",
        planner_raw_output=_invalid_json_raw_output(),
        parser_repair_outputs=[invalid_task_plan_json, invalid_task_plan_json, invalid_task_plan_json],
    )
    result = await run_runtime(runtime_input)

    history = result.final_output["metadata"]["parser_repair_history"]
    failure = result.final_output["metadata"]["failure"]
    assert result.final_output["success"] is False
    assert len(history) == 3
    assert history[0]["repair_type"] == "INVALID_JSON"
    assert history[1]["repair_type"] == "MISSING_FIELD"
    assert history[2]["repair_type"] == "MISSING_FIELD"
    assert all(item["success"] is False for item in history)
    assert all(item["output_contract_violation"] is False for item in history)
    assert "parser repair exhausted" in failure["message"]


@pytest.mark.asyncio
async def test_execution_history_does_not_include_parser_repair_events():
    runtime_input = _build_runtime_input(
        "parser_repair_execution_history",
        planner_raw_output=_invalid_json_raw_output(),
        parser_repair_outputs=[_valid_plan_json("parser_repair_execution_history")],
    )
    result = await run_runtime(runtime_input)

    execution_history = result.execution_trace["execution_history"]
    runtime_task_ids = set(result.task_states)

    assert execution_history
    assert all(record["task_id"] in runtime_task_ids for record in execution_history)
    assert all("repair_type" not in record for record in execution_history)
    assert all("latency_ms" not in record for record in execution_history)
    assert all("timestamp" not in record for record in execution_history)
