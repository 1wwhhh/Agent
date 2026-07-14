from __future__ import annotations

from typing import Iterable

from tests.support.models import PlannerScenario, RuntimeTestInput, TaskBehavior, TaskSpec


class TestCaseGenerator:
    @staticmethod
    def simple_success_case() -> RuntimeTestInput:
        return RuntimeTestInput(
            name="simple_success_case",
            user_input="Write a short greeting.",
            force_route="SIMPLE_TASK",
        )

    @staticmethod
    def complex_flow_case() -> RuntimeTestInput:
        return RuntimeTestInput(
            name="complex_flow_case",
            user_input="Analyze a topic and write a report.",
            force_route="COMPLEX_TASK",
            task_specs=[
                TaskSpec(
                    task_id="task_1",
                    task_name="analyze_topic",
                    description="Analyze the topic.",
                    tool="llm_reason_tool",
                    prompt="Analyze the topic.",
                    output_key="analysis",
                ),
                TaskSpec(
                    task_id="task_2",
                    task_name="write_report",
                    description="Write the report.",
                    tool="text_generate_tool",
                    prompt="Write the report.",
                    output_key="report",
                    depends_on=["task_1"],
                    read_keys=["analysis"],
                    priority=20,
                ),
            ],
        )

    @staticmethod
    def structured_llm_complex_flow_case() -> RuntimeTestInput:
        return RuntimeTestInput(
            name="structured_llm_complex_flow_case",
            user_input="Analyze a topic and write a report.",
            use_supervisor_agent=True,
            use_planner_agent=True,
            task_specs=[
                TaskSpec(
                    task_id="task_1",
                    task_name="analyze_topic",
                    description="Analyze the topic.",
                    tool="llm_reason_tool",
                    prompt="Analyze the topic.",
                    output_key="analysis",
                ),
                TaskSpec(
                    task_id="task_2",
                    task_name="write_report",
                    description="Write the report.",
                    tool="text_generate_tool",
                    prompt="Write the report.",
                    output_key="report",
                    depends_on=["task_1"],
                    read_keys=["analysis"],
                    priority=20,
                ),
            ],
        )

    @staticmethod
    def system_simple_task_cases() -> list[RuntimeTestInput]:
        return [
            RuntimeTestInput(
                name="system_simple_greeting",
                user_input="Write a short greeting.",
                force_route="SIMPLE_TASK",
                use_supervisor_agent=True,
            ),
            RuntimeTestInput(
                name="system_simple_summary",
                user_input="Summarize this request in one sentence.",
                force_route="SIMPLE_TASK",
                use_supervisor_agent=True,
            ),
        ]

    @staticmethod
    def system_complex_dag_cases() -> list[RuntimeTestInput]:
        return [
            RuntimeTestInput(
                name="system_complex_report",
                user_input="Analyze a topic and write a report.",
                force_route="COMPLEX_TASK",
                use_supervisor_agent=True,
                use_planner_agent=True,
                task_specs=[
                    TaskSpec(
                        task_id="task_1",
                        task_name="analyze_topic",
                        description="Analyze the topic.",
                        tool="llm_reason_tool",
                        prompt="Analyze the topic.",
                        output_key="analysis",
                    ),
                    TaskSpec(
                        task_id="task_2",
                        task_name="write_report",
                        description="Write the report.",
                        tool="text_generate_tool",
                        prompt="Write the report.",
                        output_key="report",
                        depends_on=["task_1"],
                        read_keys=["analysis"],
                        priority=20,
                    ),
                ],
            ),
            RuntimeTestInput(
                name="system_complex_parallel_merge",
                user_input="Run parallel analysis branches then merge them.",
                force_route="COMPLEX_TASK",
                use_supervisor_agent=True,
                use_planner_agent=True,
                queue_max_concurrency=3,
                task_specs=[
                    TaskSpec(
                        task_id="task_a",
                        task_name="branch_a",
                        description="Parallel branch A.",
                        tool="llm_reason_tool",
                        prompt="Branch A",
                        output_key="branch_a",
                        delay_seconds=0.02,
                    ),
                    TaskSpec(
                        task_id="task_b",
                        task_name="branch_b",
                        description="Parallel branch B.",
                        tool="text_generate_tool",
                        prompt="Branch B",
                        output_key="branch_b",
                        delay_seconds=0.02,
                    ),
                    TaskSpec(
                        task_id="task_merge",
                        task_name="merge",
                        description="Merge parallel branches.",
                        tool="text_generate_tool",
                        prompt="Merge branches",
                        output_key="merged",
                        depends_on=["task_a", "task_b"],
                        read_keys=["branch_a", "branch_b"],
                        priority=20,
                    ),
                ],
            ),
        ]

    @staticmethod
    def sequential_dag_case(depth: int = 3) -> RuntimeTestInput:
        specs: list[TaskSpec] = []
        for index in range(depth):
            task_id = f"task_{index + 1}"
            depends_on = [f"task_{index}"] if index > 0 else []
            read_keys = [f"output_{index}"] if index > 0 else []
            specs.append(
                TaskSpec(
                    task_id=task_id,
                    task_name=f"sequential_step_{index + 1}",
                    description=f"Sequential step {index + 1}.",
                    tool="text_generate_tool" if index % 2 else "llm_reason_tool",
                    prompt=f"Run step {index + 1}.",
                    output_key=f"output_{index + 1}",
                    depends_on=depends_on,
                    read_keys=read_keys,
                    priority=(index + 1) * 10,
                )
            )
        return RuntimeTestInput(
            name=f"sequential_dag_depth_{depth}",
            user_input="Execute a sequential DAG workflow.",
            force_route="COMPLEX_TASK",
            task_specs=specs,
        )

    @staticmethod
    def parallel_dag_case() -> RuntimeTestInput:
        return RuntimeTestInput(
            name="parallel_dag_case",
            user_input="Run parallel analysis branches then merge them.",
            force_route="COMPLEX_TASK",
            queue_max_concurrency=3,
            task_specs=[
                TaskSpec(
                    task_id="task_a",
                    task_name="branch_a",
                    description="Parallel branch A.",
                    tool="llm_reason_tool",
                    prompt="Branch A",
                    output_key="branch_a",
                    delay_seconds=0.05,
                ),
                TaskSpec(
                    task_id="task_b",
                    task_name="branch_b",
                    description="Parallel branch B.",
                    tool="text_generate_tool",
                    prompt="Branch B",
                    output_key="branch_b",
                    delay_seconds=0.05,
                ),
                TaskSpec(
                    task_id="task_merge",
                    task_name="merge",
                    description="Merge parallel branches.",
                    tool="text_generate_tool",
                    prompt="Merge branches",
                    output_key="merged",
                    depends_on=["task_a", "task_b"],
                    read_keys=["branch_a", "branch_b"],
                    priority=20,
                ),
            ],
        )

    @staticmethod
    def partial_failure_branch_case() -> RuntimeTestInput:
        return RuntimeTestInput(
            name="partial_failure_branch_case",
            user_input="Run two branches where one fails and one succeeds.",
            force_route="COMPLEX_TASK",
            queue_max_concurrency=2,
            task_specs=[
                TaskSpec(
                    task_id="task_fail_root",
                    task_name="failing_root",
                    description="Failing branch root.",
                    tool="llm_reason_tool",
                    prompt="Fail this branch",
                    output_key="failed_branch",
                    behavior=TaskBehavior.TOOL_FAILURE,
                    max_retry=0,
                ),
                TaskSpec(
                    task_id="task_fail_child",
                    task_name="blocked_child",
                    description="Blocked child.",
                    tool="text_generate_tool",
                    prompt="Should be cancelled",
                    output_key="blocked_output",
                    depends_on=["task_fail_root"],
                    max_retry=0,
                ),
                TaskSpec(
                    task_id="task_ok",
                    task_name="healthy_branch",
                    description="Independent successful branch.",
                    tool="text_generate_tool",
                    prompt="Healthy branch",
                    output_key="healthy_output",
                    max_retry=0,
                ),
            ],
        )

    @staticmethod
    def fail_once_case() -> RuntimeTestInput:
        return RuntimeTestInput(
            name="fail_once_case",
            user_input="Analyze with a transient failure.",
            force_route="COMPLEX_TASK",
            task_specs=[
                TaskSpec(
                    task_id="task_retry",
                    task_name="retrying_task",
                    description="Task that fails once then succeeds.",
                    tool="text_generate_tool",
                    prompt="Retry this task",
                    output_key="retry_output",
                    behavior=TaskBehavior.FAIL_ONCE,
                    max_retry=1,
                )
            ],
        )

    @staticmethod
    def timeout_case() -> RuntimeTestInput:
        return RuntimeTestInput(
            name="timeout_case",
            user_input="Run a timeout case.",
            force_route="COMPLEX_TASK",
            task_specs=[
                TaskSpec(
                    task_id="task_timeout",
                    task_name="slow_task",
                    description="Slow task that times out.",
                    tool="text_generate_tool",
                    prompt="This should timeout",
                    output_key="timeout_output",
                    behavior=TaskBehavior.TIMEOUT,
                    timeout=1,
                    max_retry=0,
                    metadata={
                        "timeout_seconds": 1,
                        "tool_timeout_seconds": 2,
                        "executor_timeout_seconds": 3,
                        "timeout_sleep_seconds": 3.5,
                    },
                )
            ],
        )

    @staticmethod
    def tool_exception_case() -> RuntimeTestInput:
        return RuntimeTestInput(
            name="tool_exception_case",
            user_input="Run a tool exception case.",
            force_route="COMPLEX_TASK",
            task_specs=[
                TaskSpec(
                    task_id="task_exception",
                    task_name="exception_task",
                    description="Task that raises a tool exception.",
                    tool="llm_reason_tool",
                    prompt="Raise an exception",
                    output_key="exception_output",
                    behavior=TaskBehavior.TOOL_EXCEPTION,
                    max_retry=0,
                )
            ],
        )

    @staticmethod
    def executor_crash_case() -> RuntimeTestInput:
        return RuntimeTestInput(
            name="executor_crash_case",
            user_input="Run an executor crash case.",
            force_route="COMPLEX_TASK",
            task_specs=[
                TaskSpec(
                    task_id="task_executor_crash",
                    task_name="executor_crash_task",
                    description="Task that triggers executor-side failure after tool success.",
                    tool="text_generate_tool",
                    prompt="Trigger executor crash",
                    output_key="executor_crash_output",
                    behavior=TaskBehavior.EXECUTOR_CRASH,
                    max_retry=0,
                )
            ],
        )

    @staticmethod
    def invalid_json_case() -> RuntimeTestInput:
        return RuntimeTestInput(
            name="invalid_json_case",
            user_input="Generate invalid planner json.",
            force_route="COMPLEX_TASK",
            planner_scenario=PlannerScenario.INVALID_JSON,
            task_specs=[TestCaseGenerator._placeholder_task_spec()],
        )

    @staticmethod
    def invalid_schema_case() -> RuntimeTestInput:
        return RuntimeTestInput(
            name="invalid_schema_case",
            user_input="Generate invalid planner schema.",
            force_route="COMPLEX_TASK",
            planner_scenario=PlannerScenario.INVALID_SCHEMA,
            task_specs=[TestCaseGenerator._placeholder_task_spec()],
        )

    @staticmethod
    def missing_dependency_case() -> RuntimeTestInput:
        return RuntimeTestInput(
            name="missing_dependency_case",
            user_input="Generate a missing dependency plan.",
            force_route="COMPLEX_TASK",
            task_specs=[
                TaskSpec(
                    task_id="task_missing",
                    task_name="missing_dependency_task",
                    description="Task with missing dependency.",
                    tool="llm_reason_tool",
                    prompt="Broken dependency",
                    output_key="broken_output",
                    depends_on=["ghost_task"],
                )
            ],
        )

    @staticmethod
    def context_consistency_case() -> RuntimeTestInput:
        return RuntimeTestInput(
            name="context_consistency_case",
            user_input="Build a result using outputs from previous tasks.",
            force_route="COMPLEX_TASK",
            queue_max_concurrency=2,
            task_specs=[
                TaskSpec(
                    task_id="task_left",
                    task_name="left_input",
                    description="Produce left input.",
                    tool="llm_reason_tool",
                    prompt="Produce left",
                    output_key="left_output",
                ),
                TaskSpec(
                    task_id="task_right",
                    task_name="right_input",
                    description="Produce right input.",
                    tool="text_generate_tool",
                    prompt="Produce right",
                    output_key="right_output",
                ),
                TaskSpec(
                    task_id="task_join",
                    task_name="join_inputs",
                    description="Join left and right inputs.",
                    tool="text_generate_tool",
                    prompt="Join inputs",
                    output_key="joined_output",
                    depends_on=["task_left", "task_right"],
                    read_keys=["left_output", "right_output"],
                    priority=20,
                ),
            ],
        )

    @staticmethod
    def cyclic_dependency_case() -> RuntimeTestInput:
        return RuntimeTestInput(
            name="cyclic_dependency_case",
            user_input="Generate a cyclic task graph.",
            force_route="COMPLEX_TASK",
            task_specs=[
                TaskSpec(
                    task_id="task_cycle_a",
                    task_name="cycle_a",
                    description="Cycle node A.",
                    tool="llm_reason_tool",
                    prompt="Cycle A",
                    output_key="cycle_a_output",
                    depends_on=["task_cycle_b"],
                ),
                TaskSpec(
                    task_id="task_cycle_b",
                    task_name="cycle_b",
                    description="Cycle node B.",
                    tool="text_generate_tool",
                    prompt="Cycle B",
                    output_key="cycle_b_output",
                    depends_on=["task_cycle_a"],
                ),
            ],
        )

    @staticmethod
    def checkpoint_interrupt_case() -> RuntimeTestInput:
        return RuntimeTestInput(
            name="checkpoint_interrupt_case",
            user_input="Run a checkpoint interruption and resume workflow.",
            force_route="COMPLEX_TASK",
            queue_max_concurrency=1,
            task_specs=[
                TaskSpec(
                    task_id="task_stage_1",
                    task_name="stage_1",
                    description="First checkpointed task.",
                    tool="llm_reason_tool",
                    prompt="Stage 1",
                    output_key="stage_1_output",
                    irreversible=True,
                ),
                TaskSpec(
                    task_id="task_stage_2",
                    task_name="stage_2",
                    description="Second checkpointed task.",
                    tool="text_generate_tool",
                    prompt="Stage 2",
                    output_key="stage_2_output",
                    depends_on=["task_stage_1"],
                    read_keys=["stage_1_output"],
                    priority=20,
                ),
            ],
        )

    @staticmethod
    def idempotency_replay_case() -> RuntimeTestInput:
        return RuntimeTestInput(
            name="idempotency_replay_case",
            user_input="Replay an irreversible task through checkpoint restore.",
            force_route="COMPLEX_TASK",
            queue_max_concurrency=1,
            task_specs=[
                TaskSpec(
                    task_id="task_irreversible",
                    task_name="irreversible_task",
                    description="Irreversible task that must not run twice.",
                    tool="text_generate_tool",
                    prompt="Irreversible write",
                    output_key="irreversible_output",
                    irreversible=True,
                )
            ],
        )

    @staticmethod
    def shared_key_conflict_case() -> RuntimeTestInput:
        return RuntimeTestInput(
            name="shared_key_conflict_case",
            user_input="Run two parallel tasks that contend on the same shared key.",
            force_route="COMPLEX_TASK",
            queue_max_concurrency=2,
            task_specs=[
                TaskSpec(
                    task_id="task_conflict_a",
                    task_name="conflict_a",
                    description="Writes shared key from branch A.",
                    tool="llm_reason_tool",
                    prompt="Conflict branch A",
                    output_key="conflict_a_output",
                    behavior=TaskBehavior.SHARED_KEY_WRITE,
                    max_retry=0,
                    metadata={"shared_key": "conflict_key"},
                ),
                TaskSpec(
                    task_id="task_conflict_b",
                    task_name="conflict_b",
                    description="Writes shared key from branch B.",
                    tool="text_generate_tool",
                    prompt="Conflict branch B",
                    output_key="conflict_b_output",
                    behavior=TaskBehavior.SHARED_KEY_WRITE,
                    max_retry=0,
                    metadata={"shared_key": "conflict_key"},
                ),
            ],
        )

    @staticmethod
    def stress_50_tasks_case() -> RuntimeTestInput:
        specs = list(TestCaseGenerator._independent_specs(count=50))
        return RuntimeTestInput(
            name="stress_50_tasks_case",
            user_input="Execute fifty independent tasks.",
            force_route="COMPLEX_TASK",
            queue_max_concurrency=10,
            task_specs=specs,
        )

    @staticmethod
    def deep_dag_case(depth: int = 8) -> RuntimeTestInput:
        return TestCaseGenerator.sequential_dag_case(depth=depth)

    @staticmethod
    def _independent_specs(*, count: int) -> Iterable[TaskSpec]:
        for index in range(count):
            yield TaskSpec(
                task_id=f"task_{index + 1}",
                task_name=f"independent_{index + 1}",
                description=f"Independent task {index + 1}.",
                tool="text_generate_tool" if index % 2 else "llm_reason_tool",
                prompt=f"Independent task {index + 1}",
                output_key=f"output_{index + 1}",
                delay_seconds=0.01 if index % 5 == 0 else 0.0,
                priority=index,
            )

    @staticmethod
    def _placeholder_task_spec() -> TaskSpec:
        return TaskSpec(
            task_id="task_placeholder",
            task_name="placeholder",
            description="Placeholder task for planner failure scenarios.",
            tool="llm_reason_tool",
            prompt="Placeholder",
            output_key="placeholder_output",
        )
