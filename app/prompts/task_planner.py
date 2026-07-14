from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from app.prompts.base import PromptTemplate
from app.prompts.language_policy import chinese_primary_policy_block
from app.prompts.registry import PromptRegistry
from app.schemas.planner import TaskPlan
from app.schemas.task import utc_now

TASK_PLANNER_PROMPT_NAME = "task_planner_prompt"
TASK_PLANNER_PROMPT_VERSION = "v2"


class ToolDefinition(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True, str_strip_whitespace=True)

    name: str = Field(..., min_length=1, description="Stable tool registry name.")
    description: str = Field(..., min_length=1, description="Human-readable tool description.")
    schema_version: str = Field(default="v1", min_length=1, description="Declared tool schema version.")
    input_schema: dict[str, Any] = Field(default_factory=dict, description="JSON schema for task input.")
    output_schema: dict[str, Any] = Field(default_factory=dict, description="JSON schema for task output.")
    supported_task_types: list[str] = Field(default_factory=list, description="Task types accepted by this tool.")
    default_task_type: str | None = Field(default=None, description="Runtime default task type for this tool.")
    supported_tags: list[str] = Field(default_factory=list, description="Routing tags accepted by this tool.")


class PlannerPromptBundle(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    prompt_name: str = Field(..., min_length=1)
    prompt_version: str = Field(..., min_length=1)
    system_prompt: str = Field(..., min_length=1)
    user_prompt: str = Field(..., min_length=1)
    output_schema: dict[str, Any] = Field(..., description="JSON schema enforced by the planner prompt.")
    planning_timestamp: str = Field(..., min_length=1, description="Exact timestamp injected into task creation.")


def get_task_plan_schema() -> dict[str, Any]:
    return TaskPlan.model_json_schema()


def build_task_planner_prompt(
    *,
    user_input: str,
    tools: Sequence[ToolDefinition | dict[str, Any]],
    context_summary: str | None = None,
    planning_timestamp: str | None = None,
) -> PlannerPromptBundle:
    normalized_input = user_input.strip()
    if not normalized_input:
        raise ValueError("user_input must not be empty")

    if not tools:
        raise ValueError("at least one tool definition is required to build the planner prompt")

    resolved_timestamp = planning_timestamp or utc_now().isoformat()
    schema = get_task_plan_schema()
    schema_json = json.dumps(schema, ensure_ascii=True, indent=2)
    tool_catalog = _render_tool_catalog(tools)
    context_block = context_summary.strip() if context_summary else "None"
    registry = build_default_planner_prompt_registry()
    rendered = registry.render(
        TASK_PLANNER_PROMPT_NAME,
        version=TASK_PLANNER_PROMPT_VERSION,
        variables={
            "user_goal": normalized_input,
            "context_summary": context_block,
            "tool_catalog": tool_catalog,
            "planning_timestamp": resolved_timestamp,
            "schema_json": schema_json,
            "ab_case_usage_guidance": _build_ab_case_usage_guidance(),
            "rag_usage_guidance": _build_rag_usage_guidance(),
            "mysql_business_usage_guidance": _build_mysql_business_usage_guidance(),
            "feishu_sync_usage_guidance": _build_feishu_sync_usage_guidance(),
            "feishu_sync_correct_example": _build_feishu_sync_correct_example(resolved_timestamp),
            "feishu_sync_forbidden_example": _build_feishu_sync_forbidden_example(resolved_timestamp),
            "ab_case_dag_example": _build_ab_case_dag_example(resolved_timestamp),
            "rag_dag_example": _build_rag_dag_example(resolved_timestamp),
            "mysql_business_dag_example": _build_mysql_business_dag_example(resolved_timestamp),
        },
    )

    return PlannerPromptBundle(
        prompt_name=rendered.name,
        prompt_version=rendered.version,
        system_prompt=rendered.system_prompt,
        user_prompt=rendered.user_prompt,
        output_schema=schema,
        planning_timestamp=resolved_timestamp,
    )


def _render_tool_catalog(tools: Sequence[ToolDefinition | dict[str, Any]]) -> str:
    rendered_lines: list[str] = []

    for item in tools:
        definition = item if isinstance(item, ToolDefinition) else ToolDefinition.model_validate(item)
        rendered_lines.append(
            "\n".join(
                [
                    f"- {definition.name} (schema_version={definition.schema_version})",
                    f"  description: {definition.description}",
                    f"  input_fields: {', '.join(sorted(definition.input_schema.get('properties', {}).keys())) or 'None'}",
                    f"  output_fields: {', '.join(sorted(definition.output_schema.get('properties', {}).keys())) or 'None'}",
                    f"  supported_task_types: {', '.join(definition.supported_task_types) or 'None'}",
                    f"  default_task_type: {definition.default_task_type or 'None'}",
                    f"  supported_tags: {_format_supported_tags(definition.supported_tags)}",
                ]
            )
        )

    return "\n".join(rendered_lines)


def _format_supported_tags(supported_tags: Sequence[str]) -> str:
    if not supported_tags:
        return "None"
    return json.dumps(list(supported_tags), ensure_ascii=True)


def _build_rag_usage_guidance() -> str:
    return "\n".join(
        [
            "- 运行时已支持 rag_search_tool，可通过现有 RAG /search API 检索公司知识库。",
            "- 当用户询问公司制度、内部知识、内部文档、流程、标准、质量检验、测试验收、来料检验、财务制度、报销流程、HR 制度、业务规则、作业指导书、SOPs、检验规范、contracts、OA、ERP、Wiki，或任何需要参考资料才能可靠回答的问题时，必须使用 rag_search_tool。",
            "- 例外：周报、本周完成、下周计划、上周计划是否完成、部门计划、月度计划、部门自评、完成率、计划追踪、考核统计属于结构化业务分析，优先使用 MySQL 业务工具，不要只用 rag_search_tool。",
            "- rag_search_tool.input.source_type 只能是 pdf、word、ppt、excel 之一；allowed values: pdf, word, ppt, excel。",
            "- 如果用户没有明确要求文件类型，不要设置 source_type。",
            "- Do not use business labels such as company_docs, internal_docs, knowledge_base, docs, document, company_policy, or internal_knowledge as source_type values.",
            "- 内部知识问答必须使用固定链路 rag_search_tool -> rag_batch_summarize_tool -> text_generate_tool。",
            "- rag_search_tool.input.query must be the exact original User Goal text. 不要总结、翻译、改写、缩短、提取关键词，也不要只传文档名或文件名。",
            "- 当使用 rag_search_tool -> rag_batch_summarize_tool -> text_generate_tool 时，the final text_generate_tool input must include rag_grounded: true.",
            "- 对于 tool = rag_batch_summarize_tool，task_type must be rag_batch_summary; never use rag_batch_summarize_tool as task_type.",
            "- 不需要检索且可以直接完成的问题，可以只规划 text_generate_tool 或 llm_reason_tool。",
            "- 公司知识库、流程、标准、质量检验、测试验收、来料检验、SOP 或检验规范类问题，不要只规划 llm_reason_tool。",
            "- Do not invent model_name values. 除非用户明确指定支持的模型，否则从 task.input 中省略 model_name。",
            "- Never output model_name as \"qwen\".",
            "- 占位符 {{rag_summary.text}} 依赖上游 output_key 必须正好是 rag_summary。",
        ]
    )


def _build_ab_case_usage_guidance() -> str:
    return "\n".join(
        [
            "- 运行时已支持 ab_case_search_tool，用于 A/B 案例、AB 案例、A\\B 案例、评分案例、案例打分样例或相似案例检索。",
            "- 业务定义：A案例是好事奖励/正向表扬案例；B案例是坏事惩罚/负向处罚案例。最终回答必须保持这个定义，不能反转。",
            "- 当用户询问某个事件/原因描述是否有相似 A/B 案例、参考什么案例评分、怎么按历史 A/B 案例判断或打分时，必须使用 ab_case_search_tool。",
            "- ab_case_search_tool 只调用 RAG 后端 POST /monthly/cases/search；BGE-M3 向量化、专属 Milvus 检索、example_id 回查 MySQL、similarity 与 ab_case_score_examples 完整字段合并都由 RAG 后端负责。",
            "- ab_case_search_tool.input.query 必须使用原始 User Goal text，不要总结、翻译、缩短或只提关键词。",
            "- 如果用户明确给出事件和原因，可分别填 event_text/reason_text；如果只给完整问题，填 query。只要 event_text/reason_text 非空，RAG 后端会优先使用它们并忽略 query。",
            "- 如果用户明确只问 A案例/奖励案例，设置 case_class=\"A\"；明确只问 B案例/惩罚案例，设置 case_class=\"B\"；不明确则不要设置 case_class。",
            "- A/B 案例问题必须使用固定链路 ab_case_search_tool -> text_generate_tool；最终 text_generate_tool.input.context 必须使用 {{ab_case_results.case_context_text}}。",
            "- 最终 text_generate_tool 应说明检索到的案例偏 A案例（奖励）还是 B案例（惩罚），以及相似点、差异点和可参考的奖惩/评分依据。",
            "- 如果 ab_case_search_tool 返回 low_relevance=true、results 为空或最高相似度不足，最终回答必须明确说明没有找到足够相似的 A/B 案例，不能强行套用案例。",
            "- 不要把 A/B 案例问题交给普通 rag_search_tool，除非用户明确是在问普通文档内容而不是案例库相似案例。",
        ]
    )


def _build_mysql_business_usage_guidance() -> str:
    return "\n".join(
        [
            "- 运行时已支持 7 个 MySQL/业务分析结构化工具：query_weekly_reports、classify_weekly_blockers、compare_weekly_plan_done、judge_weekly_blocker_trace、monthly_department_analysis、compare_dept_plan_completion、query_opl_issues。",
            "- MySQL 工具是业务分析工具，不是知识库搜索；不要让模型写 SQL，不要创建任意 SQL 查询任务。",
            "- 如果用户问题涉及周报、本周完成、下周计划、上周计划是否完成、卡点、风险、求助、部门计划、月度计划、部门自评、完成率、计划追踪、考核统计、OPL、问题清单、问题闭环、解决进展，优先使用 MySQL 工具。",
            "- query_weekly_reports 用于查询个人/部门在日期范围内的 this_week_work/this_week_done、next_week_plan、self_eval、department_plan 明细；record_level=items 查询 weekly_report_items 拆分事项，record_level=reports 查询 weekly_reports 主表一人一周一条并返回员工自填卡点字段 risk_and_help。",
            "- 当用户询问周报卡点、风险、求助、阻塞时，必须先用 query_weekly_reports 查询目标周 weekly_reports 主表：input.item_type=null，input.record_level=\"reports\"，input.include_evidence_text=false，并获取员工自填卡点字段；随后必须调用 classify_weekly_blockers 对该字段做语义分类。不能用字段非空、包含“无卡点”等简单规则直接决定是否为卡点。",
            "- classify_weekly_blockers 会区分 current_blocker、mixed_current_blocker、no_current_blocker、historical_or_resolved、ambiguous、empty；如果“暂无卡点”同时写了具体待协助/阻塞事项，应作为 mixed_current_blocker 并抽取具体卡点。",
            "- 对周报卡点/风险/求助类问题，compare_weekly_plan_done 必须依赖 query_weekly_reports 和 classify_weekly_blockers，并设置 weekly_blocker_classification_output_key 为分类任务 output_key、trace_weeks=2、include_historical_blockers=true；last_week_start/end 仍取目标周上一周，this_week_start/end 取目标周。工具层会只追溯 needs_trace=true 的人员，并按上一周、上上周两个独立窗口追溯。",
            "- compare_weekly_plan_done 之后必须调用 judge_weekly_blocker_trace，输入分类结果和追溯结果，用于判断历史卡点是否在后续周报中解决或仍持续。",
            "- 周报卡点/风险/求助类最终 text_generate_tool.input.context 必须使用 {{weekly_blocker_trace_judgement.weekly_blocker_context_text}}，不要把 {{weekly_reports}}、完整 {{weekly_plan_comparison}} 或完整 {{weekly_blocker_classification}} 全量塞入 context。",
            "- 周报卡点/风险/求助类最终 text_generate_tool.input.prompt 和最终回答必须使用“员工自填卡点”“未确认当前卡点”“根据计划完成证据推断”“历史卡点后续判断”等业务表述，不得出现 risk_and_help、weekly_blocker_context_text、plan_followups 等内部字段名或输出键。",
            "- compare_weekly_plan_done 用于查询上周/某月计划与后续本周完成记录，并在工具层输出 weekly_pairs、plan_followups、pairing_summary、plan_tracking_context_text；计划完成追踪类最终由 text_generate_tool 基于 plan_followups 做 LLM 分批判断与分层汇总，工具层 final_report 只作为失败兜底；必须提供 last_week_start、last_week_end、this_week_start、this_week_end，日期格式 YYYY-MM-DD。",
            "- 当用户问‘上周工作/完成内容有没有按原计划完成’时，目标是判断上周完成是否兑现原计划：last_week_start/end 应取上上周，查询上上周 next_week_plan；this_week_start/end 应取上周，查询上周 this_week_work/this_week_done。不要误用上周计划对比本周完成。",
            "- 对于‘上个月/某月每周计划是否完成、还有哪些未完成、月度每周计划追踪’这类问题，必须只创建 1 个 compare_weekly_plan_done 任务，不能只查月初第一周，也不要拆成多个单周任务；last_week_start/end 必须覆盖该月完整计划日期范围，例如 2026-05-01 到 2026-05-31，this_week_start/end 至少覆盖该月及后续一期周报。",
            "- compare_weekly_plan_done 会按计划 report_date 逐条配对下一期本周完成记录；最终 text_generate_tool.input.context 必须使用 {{weekly_plan_comparison.plan_tracking_context_text}}，不要把完整 {{weekly_plan_comparison}} 全量塞入 context；text_generate_tool 会优先使用 LLM 分批判断与分层汇总，完整明细由后端追加，确定性报告只作兜底。",
            "- monthly_department_analysis 用于部门月度计划、周报、自评和 OPL 问题闭环综合分析；month 格式 YYYY-MM。",
            "- compare_dept_plan_completion 用于核对三七计划书/部门月度计划是否完成；它会查 dept_plan_items，并按负责人、月份和周报日期从 weekly_reports 主表取负责人完整周报原文，同时用 weekly_report_items 仅为非负责人协作周报排序提供辅助线索，按计划负责人姓名直接关联 employee_self_eval_reports/items 中负责人本人月度考核记录，并默认 include_opl=true 纳入 opl_issue_items 的问题闭环证据。普通月度核对 followup_days 用 7 作为月末短宽限；只有用户明确要求‘后续一个月/截至现在/追踪后续完成’时才放宽到 31 或用户指定天数。完成状态必须由 text_generate_tool 后端 LLM 基于 dept_plan_followups 判断。",
            "- 当用户问‘三七计划书中各部门某月计划有没有完成/落地/落实/完成率/哪些没完成’时，必须优先使用 compare_dept_plan_completion，month 用 YYYY-MM，followup_days 用 7，include_opl=true；department 可为空表示各部门；最终 text_generate_tool.input.context 必须使用 {{dept_plan_completion.dept_plan_completion_context_text}}。",
            "- query_opl_issues 用于单独查询 OPL 问题清单、未解决问题、已解决问题、解决措施/最新进展、负责人、优先级和问题闭环统计；如果用户只问 OPL 问题本身，用 query_opl_issues 后接 text_generate_tool。",
            "- OPL 是问题闭环证据：未闭环 OPL 代表风险/卡点，不能单独等同计划未完成；已解决 OPL 只有在解决进展对应计划目标时才可作为闭环辅助证据。",
            "- 如果三七计划/月度部门计划问题同时询问卡点、卡着、卡住、没动、督促、风险或跨部门协作，除 compare_dept_plan_completion 外，还必须增加 query_weekly_reports 查询计划月份 weekly_reports 主表：item_type=null、record_level=\"reports\"、include_evidence_text=false，用有效 risk_and_help 作为员工自填卡点来源；最终生成任务依赖两个 MySQL 任务，并说明没有有效员工自填卡点时只能基于计划完成证据推断需跟进项。",
            "- 用户给出月份如 2026年5月/5月/月度时，若问题是部门综合分析/自评一致性，用 monthly_department_analysis.input.month=YYYY-MM；若问题是计划完成追踪，用 compare_weekly_plan_done 并给出明确 YYYY-MM-DD 日期范围。",
            "- MySQL 工具返回结构化事实和 evidence_text 后，必须接 text_generate_tool 生成面向用户的最终回答；对于计划完成追踪类 compare_weekly_plan_done，使用 LLM 分批判断计划完成状态并分层汇总，工具输出的标准 final_report 只作为 LLM 失败兜底。",
            "- 普通文件内容问答、合同条款、制度内容、PDF/Word/PPT/Excel 原文检索、模糊查文件内容，仍然使用 rag_search_tool / Milvus。",
            "- 不要把周报完成率、计划追踪、部门自评一致性这类结构化统计问题只交给 rag_search_tool。",
        ]
    )


def _build_feishu_sync_usage_guidance() -> str:
    return "\n".join(
        [
            "- FeishuSyncToNasTool 是会向 NAS 写文件的重副作用工具。",
            "- 只有当用户明确要求 sync、download、import、update 或 save 飞书共享文件夹到 NAS 时，才使用 FeishuSyncToNasTool。",
            "- 普通 QA 不要使用 FeishuSyncToNasTool。",
            "- 不要因为用户提到 Feishu 就使用 FeishuSyncToNasTool。",
            "- RAG 问答或摘要请求不要使用 FeishuSyncToNasTool，除非用户明确先要求同步。",
            "- 除非用户在请求中提供真实 folder_url 和真实 nas_dir，否则不要使用 FeishuSyncToNasTool。",
            "- 永远不要把 FEISHU_FOLDER_URL、$FEISHU_FOLDER_URL 或 brace form 写入 task.input。",
            "- 永远不要把 FEISHU_SYNC_NAS_DIR、$FEISHU_SYNC_NAS_DIR 或 brace form 写入 task.input。",
            "- 如果缺少真实 folder_url 或 nas_dir，不要创建 FeishuSyncToNasTool task；创建直接回复任务让用户补充缺失参数。",
            "- 使用 FeishuSyncToNasTool 时，task_type 设为 feishu_sync_to_nas，tags 设为 [\"connector\", \"feishu\", \"nas\", \"heavy\"]，max_retry 设为 0。",
            "- Never output Feishu tags as [\"connector/feishu/nas/heavy\"] or \"connector/feishu/nas/heavy\".",
        ]
    )


def _build_rag_dag_example(planning_timestamp: str) -> str:
    example = {
        "goal": "Answer a user question from the company knowledge base",
        "tasks": [
            {
                "task_id": "task_1",
                "task_name": "search_knowledge_base",
                "description": "Search the company knowledge base for information relevant to the user question.",
                "task_type": "rag_search",
                "tool": "rag_search_tool",
                "input": {
                    "query": "original user question",
                    "top_k": 10,
                },
                "output_key": "rag_context",
                "depends_on": [],
                "priority": 1,
                "tags": ["rag", "search"],
                "status": "PENDING",
                "retry_count": 0,
                "max_retry": 1,
                "timeout": 60,
                "created_at": planning_timestamp,
            },
            {
                "task_id": "task_2",
                "task_name": "summarize_rag_batches",
                "description": "Summarize the retrieved RAG batches.",
                "task_type": "rag_batch_summary",
                "tool": "rag_batch_summarize_tool",
                "input": {
                    "query": "original user question",
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
                "created_at": planning_timestamp,
            },
            {
                "task_id": "task_3",
                "task_name": "generate_answer_from_rag_summary",
                "description": "Generate the final answer from the summarized RAG evidence.",
                "task_type": "text_generation",
                "tool": "text_generate_tool",
                "input": {
                    "prompt": "Please answer from the summarized RAG evidence. If the evidence is insufficient, say so clearly.",
                    "context": "{{rag_summary.text}}",
                    "rag_grounded": True,
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
                "created_at": planning_timestamp,
            },
        ],
    }
    return json.dumps(example, ensure_ascii=True, indent=2)


def _build_ab_case_dag_example(planning_timestamp: str) -> str:
    example = {
        "goal": "查询相似 A/B 奖惩案例并给出参考回答",
        "tasks": [
            {
                "task_id": "task_1",
                "task_name": "search_ab_case_examples",
                "description": "调用 A/B 案例专用检索接口，获取相似奖惩案例、相似度和完整案例字段。",
                "task_type": "ab_case_search",
                "tool": "ab_case_search_tool",
                "input": {
                    "query": "original user question",
                    "top_k": 8,
                },
                "output_key": "ab_case_results",
                "depends_on": [],
                "priority": 1,
                "tags": ["ab_case", "case_search", "rag"],
                "status": "PENDING",
                "retry_count": 0,
                "max_retry": 1,
                "timeout": 60,
                "created_at": planning_timestamp,
            },
            {
                "task_id": "task_2",
                "task_name": "generate_ab_case_answer",
                "description": "根据相似 A/B 奖惩案例生成面向用户的最终回答。",
                "task_type": "text_generation",
                "tool": "text_generate_tool",
                "input": {
                    "prompt": "请基于 A/B 案例检索结果回答用户问题。A案例表示好事奖励，B案例表示坏事惩罚。优先引用相似度高的案例，说明相似点、差异点、偏 A 奖励还是偏 B 惩罚，以及可参考的奖惩/评分依据；如果相关性不足，请明确说明未找到足够相似案例，不要强行套用。",
                    "context": "{{ab_case_results.case_context_text}}",
                    "style": "structured",
                    "audience": "business_user",
                },
                "output_key": "final_result",
                "depends_on": ["task_1"],
                "priority": 2,
                "tags": ["llm", "generation"],
                "status": "PENDING",
                "retry_count": 0,
                "max_retry": 1,
                "timeout": 120,
                "created_at": planning_timestamp,
            },
        ],
    }
    return json.dumps(example, ensure_ascii=False, indent=2)


def _build_mysql_business_dag_example(planning_timestamp: str) -> str:
    example = {
        "goal": "分析用户周计划是否完成",
        "tasks": [
            {
                "task_id": "task_1",
                "task_name": "compare_weekly_plan_done",
                "description": "查询结构化周报记录，获取上一周计划与本周完成内容。",
                "task_type": "compare_weekly_plan_done",
                "tool": "compare_weekly_plan_done",
                "input": {
                    "user_name": "张三",
                    "department": None,
                    "last_week_start": "2026-05-04",
                    "last_week_end": "2026-05-10",
                    "this_week_start": "2026-05-11",
                    "this_week_end": "2026-05-17",
                    "limit": 500,
                },
                "output_key": "weekly_plan_comparison",
                "depends_on": [],
                "priority": 1,
                "tags": ["mysql", "business", "weekly_report"],
                "status": "PENDING",
                "retry_count": 0,
                "max_retry": 1,
                "timeout": 30,
                "created_at": planning_timestamp,
            },
            {
                "task_id": "task_2",
                "task_name": "generate_weekly_plan_answer",
                "description": "根据结构化 MySQL 记录和证据说明计划完成情况。",
                "task_type": "text_generation",
                "tool": "text_generate_tool",
                "input": {
                    "prompt": "请根据 compare_weekly_plan_done 的 plan_tracking_context_text 输出计划完成情况。请基于上游 plan_followups 做语义判断与汇总；不要把完整 weekly_plan_comparison 全量塞入上下文。",
                    "context": "{{weekly_plan_comparison.plan_tracking_context_text}}",
                    "style": "structured",
                    "audience": "business_user",
                },
                "output_key": "final_result",
                "depends_on": ["task_1"],
                "priority": 2,
                "tags": ["llm", "generation"],
                "status": "PENDING",
                "retry_count": 0,
                "max_retry": 1,
                "timeout": 75,
                "created_at": planning_timestamp,
            },
            {
                "task_id": "task_3",
                "task_name": "compare_dept_plan_completion",
                "description": "查询三七计划书计划项，并直连负责人本人月度考核记录和周报证据。",
                "task_type": "compare_dept_plan_completion",
                "tool": "compare_dept_plan_completion",
                "input": {
                    "month": "2026-05",
                    "department": None,
                    "doc_id": None,
                    "include_weekly": True,
                "include_self_eval": True,
                "include_opl": True,
                "followup_days": 7,
                "limit": 1000,
                },
                "output_key": "dept_plan_completion",
                "depends_on": [],
                "priority": 1,
                "tags": ["mysql", "business", "dept_plan"],
                "status": "PENDING",
                "retry_count": 0,
                "max_retry": 1,
                "timeout": 30,
                "created_at": planning_timestamp,
            },
            {
                "task_id": "task_4",
                "task_name": "generate_dept_plan_completion_answer",
                "description": "根据三七计划、周报和负责人本人月度考核记录，由 LLM 判断各部门计划是否完成。",
                "task_type": "text_generation",
                "tool": "text_generate_tool",
                "input": {
                    "prompt": "请根据 compare_dept_plan_completion 的 dept_plan_completion_context_text 输出三七计划完成情况。逐条判断已完成、部分完成、未完成或证据不足，并说明依据；请综合负责人周报、负责人月度考核和 OPL 问题闭环证据；未闭环 OPL 只能作为风险/卡点证据，不要直接等同计划未完成；不要把完整 dept_plan_completion 全量塞入上下文。",
                    "context": "{{dept_plan_completion.dept_plan_completion_context_text}}",
                    "style": "structured",
                    "audience": "business_user",
                },
                "output_key": "final_dept_plan_result",
                "depends_on": ["task_3"],
                "priority": 2,
                "tags": ["llm", "generation"],
                "status": "PENDING",
                "retry_count": 0,
                "max_retry": 1,
                "timeout": 120,
                "created_at": planning_timestamp,
            },
        ],
    }
    return json.dumps(example, ensure_ascii=False, indent=2)


def _build_feishu_sync_correct_example(planning_timestamp: str) -> str:
    example = {
        "goal": "Sync a Feishu shared folder to NAS",
        "tasks": [
            {
                "task_id": "task_1",
                "task_name": "sync_feishu_folder_to_nas",
                "description": "Sync the user-provided Feishu shared folder to the user-provided NAS directory.",
                "task_type": "feishu_sync_to_nas",
                "tool": "FeishuSyncToNasTool",
                "input": {
                    "folder_url": "https://xxx.feishu.cn/drive/folder/abc123",
                    "nas_dir": "/mnt/9goo-dept/9goo-nas/飞书同步",
                    "recursive": True,
                    "overwrite": False,
                },
                "output_key": "sync_result",
                "depends_on": [],
                "priority": 1,
                "tags": ["connector", "feishu", "nas", "heavy"],
                "status": "PENDING",
                "retry_count": 0,
                "max_retry": 0,
                "timeout": 3600,
                "created_at": planning_timestamp,
            }
        ],
    }
    return json.dumps(example, ensure_ascii=False, indent=2)


def _build_feishu_sync_forbidden_example(planning_timestamp: str) -> str:
    example = {
        "goal": "Forbidden Feishu sync planner output",
        "tasks": [
            {
                "task_id": "task_1",
                "task_name": "sync_feishu_folder_to_nas",
                "description": "Do not output this shape.",
                "task_type": "feishu_sync_to_nas",
                "tool": "FeishuSyncToNasTool",
                "input": {
                    "folder_url": "FEISHU_FOLDER_URL",
                    "nas_dir": "FEISHU_SYNC_NAS_DIR",
                },
                "output_key": "sync_result",
                "depends_on": [],
                "priority": 1,
                "tags": ["connector/feishu/nas/heavy"],
                "status": "PENDING",
                "retry_count": 0,
                "max_retry": 0,
                "timeout": 3600,
                "created_at": planning_timestamp,
            }
        ],
    }
    return json.dumps(example, ensure_ascii=False, indent=2)


def get_task_planner_prompt_template() -> PromptTemplate:
    return PromptTemplate(
        name=TASK_PLANNER_PROMPT_NAME,
        version=TASK_PLANNER_PROMPT_VERSION,
        description="Planner prompt that forces executable TaskPlan DAG output.",
        system_template="\n".join(
            [
                chinese_primary_policy_block(role="你是 Agent Runtime System 的 Task Planner。"),
                "你的唯一任务是把用户目标转换成可执行的 Task DAG。",
                "只返回 required function call，不要返回 markdown，不要解释。",
                "function arguments 必须严格满足提供的 JSON schema。",
                "每个 task 都必须 atomic、executable、traceable、retryable，并明确 depends_on 关系。",
                "只能使用 allowed tool catalog 中的工具。",
                "如果用户问题涉及 A/B 案例、AB 案例、A\\B 案例、A案例、B案例、奖励案例、惩罚案例、评分案例、案例打分样例或相似案例检索，必须使用 ab_case_search_tool -> text_generate_tool。A案例表示好事奖励，B案例表示坏事惩罚。",
                "如果用户问题涉及周报、本周完成、下周计划、计划完成率、计划追踪、部门自评、月度考核或部门计划分析，必须优先使用 MySQL 结构化业务工具，再用 text_generate_tool 生成最终回答。",
                "如果问题依赖公司知识、内部文档、公司制度、流程、标准、质量检验、测试验收、来料检验、SOPs 或检验规范，必须使用 RAG 链路 rag_search_tool -> rag_batch_summarize_tool -> text_generate_tool；但周报/计划/自评/完成率等结构化业务分析问题例外，优先 MySQL 工具。",
                "知识库问题必须先检索再生成：create a retrieval-first plan that uses rag_search_tool before final generation.",
                "FeishuSyncToNasTool 是会向 NAS 写文件的重副作用工具。",
                "只有当用户明确要求 sync、download、import、update 或 save 飞书共享文件夹到 NAS 时，才允许使用 FeishuSyncToNasTool。",
                "普通问答、RAG 问答、摘要请求，或只是提到 Feishu 的请求，不要使用 FeishuSyncToNasTool。",
                "除非用户在请求中提供真实 folder_url 和真实 nas_dir，否则不要使用 FeishuSyncToNasTool。",
                "永远不要把 FEISHU_FOLDER_URL、$FEISHU_FOLDER_URL 或 brace form 写入 task.input。",
                "永远不要把 FEISHU_SYNC_NAS_DIR、$FEISHU_SYNC_NAS_DIR 或 brace form 写入 task.input。",
                "如果缺少真实 folder_url 或 nas_dir，不要创建 FeishuSyncToNasTool task；请创建直接回复任务要求用户补充参数。",
                "使用 FeishuSyncToNasTool 时，task_type 必须是 feishu_sync_to_nas，tags 必须是 [\"connector\", \"feishu\", \"nas\", \"heavy\"]，max_retry 必须是 0。",
                "Never output Feishu tags as [\"connector/feishu/nas/heavy\"] or \"connector/feishu/nas/heavy\".",
                "如果不需要检索，可以输出单个直接生成或推理任务；但内部知识库问题不能只用 llm_reason_tool 回答。",
                f"Set every task.status to '{'PENDING'}'.",
                "Set every task.retry_count to 0 unless the runtime explicitly passes a prior retry state.",
                "Set every task.created_at to the exact planning timestamp provided by the runtime.",
                "简单目标可以输出单个 task。",
                "复杂目标必须拆解为带显式 depends_on 的 DAG。",
            ]
        ),
        user_template="\n\n".join(
            [
                "User Goal:",
                "{user_goal}",
                "Runtime Context Summary:",
                "{context_summary}",
                "Allowed Tool Catalog:",
                "{tool_catalog}",
                "A/B Case Search Guidance:",
                "{ab_case_usage_guidance}",
                "RAG Planning Guidance:",
                "{rag_usage_guidance}",
                "MySQL Business Tool Guidance:",
                "{mysql_business_usage_guidance}",
                "Feishu Sync Guidance:",
                "{feishu_sync_usage_guidance}",
                "Planning Rules:",
                "\n".join(
                    [
                        "1. Output must be a single structured function call and nothing else.",
                        "2. The function arguments must satisfy the schema exactly.",
                        "3. Each task must have a unique task_id and output_key.",
                        "4. Each task must use exactly one tool from the allowed catalog.",
                        "5. Task.input must match the selected tool input schema.",
                        "6. Task.output_key and task output contract must match the selected tool output schema.",
                        "7. depends_on must reference only task_ids defined in the same output.",
                        "8. Use an empty array for depends_on when a task has no dependencies.",
                        "9. Use smaller priority numbers for tasks that should be scheduled earlier.",
                        "10. Set task.status to PENDING for all newly planned tasks.",
                        "11. Set task.retry_count to 0 for all newly planned tasks.",
                        "12. Set task.max_retry to a non-negative integer.",
                        "13. Set task.timeout to a positive integer in seconds.",
                        "14. Keep each task single-responsibility and executable by the assigned tool.",
                        "15. Set task.created_at to this exact ISO 8601 UTC timestamp: {planning_timestamp}",
                        "15a. When the user asks about A/B cases, AB cases, A\\B cases, A案例, B案例, reward cases, punishment cases, scoring examples, similar case examples, or asks how to score/judge by historical A/B cases, use ab_case_search_tool first and then text_generate_tool. A案例 means good/reward; B案例 means bad/punishment.",
                        "15b. For ab_case_search_tool, set task_type to ab_case_search, output_key to ab_case_results, tags to supported English tags such as [\"ab_case\", \"case_search\", \"rag\"], and input.query to the exact original User Goal text.",
                        "15c. For A/B case answers, the final text_generate_tool must depend on ab_case_search_tool and use context={{ab_case_results.case_context_text}}.",
                        "15d. If A/B case retrieval returns low_relevance=true, no results, or insufficient similarity, the final answer must say that no sufficiently similar A/B case was found.",
                        "15e. For A/B case final answers, explicitly state whether the retrieved evidence leans toward A案例（奖励） or B案例（惩罚） when the evidence supports it; never reverse A/B meaning.",
                        "16. When the answer should come from company knowledge, internal documents, company policies, company processes, processes, standards, quality inspection, acceptance testing, incoming material inspection, financial policies, reimbursement, HR policies, business rules, work instructions, SOPs, inspection specifications, contracts, OA, ERP, Wiki, procurement, approvals, or other reference material, you must use the fixed RAG chain rag_search_tool -> rag_batch_summarize_tool -> text_generate_tool.",
                        "17. When you use rag_search_tool, set the first task output_key to rag_context.",
                        "18. When you use rag_search_tool, add rag_batch_summarize_tool as the next task and make the final text_generate_tool consume {{rag_summary.text}}.",
                        "18c. When using rag_search_tool, task.input.query must be the exact original User Goal text. Never replace it with keywords, a translated summary, a file name, a document title, or a shortened search phrase.",
                        "18b. Internal knowledge-base questions must not be answered by only llm_reason_tool.",
                        "18a. When using rag_search_tool -> rag_batch_summarize_tool -> text_generate_tool, the final text_generate_tool input must include rag_grounded: true.",
                        "19. rag_search_tool.input.source_type may only be pdf, word, ppt, or excel.",
                        "20. If the user did not explicitly ask for a file type, omit source_type from rag_search_tool.input.",
                        "21. Do not use company_docs, internal_docs, knowledge_base, docs, document, company_policy, or internal_knowledge as source_type values.",
                        "22. 闲聊、通用写作、翻译，或不需要检索即可直接回答的问题，不要强制使用 rag_search_tool。",
                        "22a. 对周报、本周完成、下周计划、卡点、风险、求助、计划完成情况、部门计划、月度计划、自评、完成率、计划追踪、考核统计、OPL、问题清单、问题闭环、解决进展类问题，优先使用 MySQL 业务工具，而不是 rag_search_tool。",
                        "22b. 不要创建任意 SQL 任务。查询 MySQL 业务数据时，只能使用 query_weekly_reports、classify_weekly_blockers、compare_weekly_plan_done、judge_weekly_blocker_trace、monthly_department_analysis、compare_dept_plan_completion 或 query_opl_issues。",
                        "22c. MySQL 业务工具返回记录后，必须使用 text_generate_tool 基于结构化记录和 evidence_text 生成最终面向用户的回答。对于计划完成追踪类 compare_weekly_plan_done，text_generate_tool 应基于 plan_followups 做 LLM 分批判断与分层汇总；工具层 final_report 只作失败兜底。",
                        "22c1. 对周报卡点、风险、求助类问题，必须规划固定链路：query_weekly_reports -> classify_weekly_blockers -> compare_weekly_plan_done -> judge_weekly_blocker_trace -> text_generate_tool。query_weekly_reports 查询目标周 weekly_reports 主表，且 item_type=null、record_level=\"reports\"、include_evidence_text=false。",
                        "22c2. classify_weekly_blockers 必须依赖 query_weekly_reports，并用 weekly_reports_output_key 指向查询任务 output_key。不能用字段非空或包含“无卡点”等简单规则判断；“暂无卡点，但需要软件协助联调”这类混合文本必须交由语义分类。",
                        "22c3. compare_weekly_plan_done 必须依赖 query_weekly_reports 和 classify_weekly_blockers；input.weekly_blocker_classification_output_key 设置为分类任务 output_key，trace_weeks=2，include_historical_blockers=true。last_week_start/end 取目标周上一周，this_week_start/end 取目标周；工具层会按上一周和上上周两个独立窗口追溯 needs_trace=true 的人员。",
                        "22c4. judge_weekly_blocker_trace 必须依赖 classify_weekly_blockers 和 compare_weekly_plan_done，用于判断历史卡点是否已在后续周报中完成/闭环、仍持续、证据不足或不是卡点。",
                        "22c5. 周报卡点/风险/求助类最终生成任务必须使用 input.context={{weekly_blocker_trace_judgement.weekly_blocker_context_text}}；不要把 {{weekly_reports}}、完整 {{weekly_plan_comparison}} 或完整 {{weekly_blocker_classification}} 全量传给 text_generate_tool，避免大上下文导致模型超时。",
                        "22c6. 不要用 item_text/evidence_text 推断出的卡点替换当前有效员工自填卡点；推断只作为未确认当前卡点时的兜底。最终 text_generate_tool.input.prompt 和最终回答必须使用“员工自填卡点”“未确认当前卡点”“根据计划完成证据推断”“历史卡点后续判断”等业务表述，不得出现 risk_and_help、weekly_blocker_context_text、plan_followups 等内部字段名或输出键。",
                        "22d. 对月度部门分析或自评一致性问题，使用 monthly_department_analysis，并传入 YYYY-MM 格式的 month；该工具会返回 OPL 问题闭环汇总。对周度/月度计划完成追踪，必须只创建 1 个 compare_weekly_plan_done 任务，并给出明确 YYYY-MM-DD 日期范围；用户问‘上周工作/完成内容有没有按原计划完成’时，用上上周 next_week_plan 对比上周 this_week_work/this_week_done；月度每周追踪时，last_week_start/end 必须覆盖完整计划月份，不能只覆盖第一周，this_week_start/end 必须覆盖该月以及至少后续一期周报。",
                        "22e. 对 compare_weekly_plan_done 的输出，plan_followups 只是证据。计划完成追踪类最终 text_generate_tool.input.context 必须使用 {{weekly_plan_comparison.plan_tracking_context_text}}，不要把完整 {{weekly_plan_comparison}} 全量传给 text_generate_tool；必须要求 text_generate_tool 逐条判断已完成、部分完成、未完成或证据不足。",
                        "22f. 对三七计划书/部门计划书的月度完成核对，使用 compare_dept_plan_completion，默认 include_opl=true；普通月度核对 followup_days 用 7，只有用户明确要求‘后续一个月/截至现在/追踪后续完成’时才放宽到 31 或用户指定天数。不要用 compare_weekly_plan_done 替代，因为计划来源是 dept_plan_items。最终 text_generate_tool.input.context 必须使用 {{dept_plan_completion.dept_plan_completion_context_text}}，并要求 LLM 基于 dept_plan_followups 中第一层抽取的负责人原文证据、日期、提交人、完成/卡点信息、非负责人协作辅助材料、负责人本人月度考核记录和 OPL 问题闭环证据逐条判断；负责人月度考核记录来自 employee_self_eval_reports/items 按负责人姓名直接查询，不是部门候选搜索。OPL 未闭环问题只能作为风险/卡点证据，不得直接等同计划未完成。",
                        "22f0. 如果用户只问 OPL、问题清单、未解决问题、已解决问题、解决措施、负责人问题统计或优先级问题统计，使用 query_opl_issues 查询结构化问题记录，再用 text_generate_tool 生成最终回答。",
                        "22f1. 如果三七计划/月度部门计划问题同时询问卡点、卡着、卡住、没动、督促、风险或跨部门协作，必须同时规划 query_weekly_reports 查询计划月份 weekly_reports 主表：item_type=null、record_level=\"reports\"、include_evidence_text=false；最终 text_generate_tool 依赖 compare_dept_plan_completion 和 query_weekly_reports，并在 prompt 中要求直接采用有效 risk_and_help，无有效员工自填卡点时再把缺少有效完成证据的计划标为推断需跟进项。",
                        "23. FeishuSyncToNasTool is a heavy side-effect tool that writes files to NAS.",
                        "24. Only use FeishuSyncToNasTool when the user explicitly asks to sync, download, import, update, or save a Feishu shared folder to NAS.",
                        "25. Do not use FeishuSyncToNasTool for normal QA, for RAG question answering, or because the user only mentioned Feishu.",
                        "26. Do not use FeishuSyncToNasTool unless the user provides a real folder_url and a real nas_dir in the request.",
                        "27. When using FeishuSyncToNasTool, set task_type to feishu_sync_to_nas, tags to [\"connector\", \"feishu\", \"nas\", \"heavy\"], and max_retry to 0.",
                        "28. Never output Feishu tags as [\"connector/feishu/nas/heavy\"] or \"connector/feishu/nas/heavy\".",
                        "29. Never put FEISHU_FOLDER_URL, $FEISHU_FOLDER_URL, or the brace form of FEISHU_FOLDER_URL into task.input.",
                        "30. Never put FEISHU_SYNC_NAS_DIR, $FEISHU_SYNC_NAS_DIR, or the brace form of FEISHU_SYNC_NAS_DIR into task.input.",
                        "31. If a real folder_url or nas_dir is missing, do not create a FeishuSyncToNasTool task; create a direct response task asking the user to provide the missing parameters.",
                        "32. task_type must not be a tool name; task_type must be selected from the selected tool's supported_task_types when supported_task_types is declared.",
                        "33. For tool = rag_batch_summarize_tool, task_type must be rag_batch_summary and must never be rag_batch_summarize_tool.",
                        "33a. For tool = rag_batch_summarize_tool, set task.timeout to at least 1800 seconds.",
                        "34. Every task.tags value must come from the selected tool capability supported_tags shown in the tool catalog.",
                        "35. task.tags must use English tags only; never use Chinese tags.",
                        "36. Never use Chinese semantic words as tags, including \"生成\", \"检索\", \"摘要\", or \"推理\".",
                        "37. Do not invent tags. If you are not certain a tag is supported, omit that tag instead of using an unsupported tag.",
                        "38. For text_generate_tool, tags must be [\"llm\", \"generation\"] or [\"llm\", \"generation\", \"text\"].",
                        "39. For text_generate_tool, never output tags as [\"llm\", \"生成\"].",
                        "40. Do not invent model_name values in task.input.",
                        "41. Unless the user explicitly specified a supported model, omit model_name from task.input.",
                        "42. Never output model_name as \"qwen\".",
                    ]
                ),
                "A/B Case DAG Example:",
                "{ab_case_dag_example}",
                "MySQL Business DAG Example:",
                "{mysql_business_dag_example}",
                "Feishu Sync Correct Example:",
                "{feishu_sync_correct_example}",
                "Feishu Sync Forbidden Example:",
                "{feishu_sync_forbidden_example}",
                "RAG DAG Example:",
                "{rag_dag_example}",
                "Required JSON Schema:",
                "{schema_json}",
            ]
        ),
    )


def build_default_planner_prompt_registry() -> PromptRegistry:
    registry = PromptRegistry()
    registry.register(get_task_planner_prompt_template())
    return registry
