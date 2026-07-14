from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from app.prompts.base import PromptTemplate
from app.prompts.language_policy import chinese_primary_policy_block
from app.prompts.registry import PromptRegistry

SUPERVISOR_PROMPT_NAME = "supervisor_route_prompt"
SUPERVISOR_PROMPT_VERSION = "v1"


class SupervisorPromptBundle(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    prompt_name: str = Field(..., min_length=1)
    prompt_version: str = Field(..., min_length=1)
    system_prompt: str = Field(..., min_length=1)
    user_prompt: str = Field(..., min_length=1)


def build_supervisor_prompt(*, user_input: str, context_summary: str | None = None) -> SupervisorPromptBundle:
    registry = build_default_prompt_registry()
    rendered = registry.render(
        SUPERVISOR_PROMPT_NAME,
        version=SUPERVISOR_PROMPT_VERSION,
        variables={
            "user_input": user_input.strip(),
            "context_summary": (context_summary or "None").strip() or "None",
        },
    )
    return SupervisorPromptBundle(
        prompt_name=rendered.name,
        prompt_version=rendered.version,
        system_prompt=rendered.system_prompt,
        user_prompt=rendered.user_prompt,
    )


def get_supervisor_prompt_template() -> PromptTemplate:
    return PromptTemplate(
        name=SUPERVISOR_PROMPT_NAME,
        version=SUPERVISOR_PROMPT_VERSION,
        description="Route user requests into SIMPLE_TASK or COMPLEX_TASK.",
        system_template=(
            chinese_primary_policy_block(role="你是 Agent Runtime System 的 Supervisor。")
            + "\n"
            "你的任务是把用户请求分类为 SIMPLE_TASK 或 COMPLEX_TASK。\n"
            "只有在请求可以安全地用单步骤完成、且不需要规划时，才选择 SIMPLE_TASK。\n"
            "当请求需要拆解、多工具协作、依赖关系、任务规划或检索证据时，选择 COMPLEX_TASK。\n"
            "如果可靠回答前需要检索 company knowledge、internal knowledge base、internal documents、"
            "contracts、policies、SOPs、workflows、报销制度、采购步骤、审批步骤、OA、ERP、Wiki "
            "或其他内部/外部参考资料，必须选择 COMPLEX_TASK。\n"
            "如果用户问题涉及 A/B 案例、AB 案例、A\\B 案例、A案例、B案例、奖励案例、惩罚案例、评分案例、案例打分样例、相似案例检索，"
            "或需要先查历史 A/B 案例再生成回答，必须选择 COMPLEX_TASK。\n"
            "A案例表示好事奖励，B案例表示坏事惩罚。\n"
            "不要把 knowledge-base retrieval requests 归为 SIMPLE_TASK。"
            "Do not classify knowledge-base retrieval requests as SIMPLE_TASK.\n"
            "FeishuSyncToNasTool 是会向 NAS 写文件的重副作用工具。\n"
            "只有当用户明确要求 sync、download、import、update 或 save 飞书共享文件夹到 NAS，"
            "并且提供真实 folder_url 和 nas_dir 时，才允许路由到 FeishuSyncToNasTool。\n"
            "仅仅提到 Feishu 不等于同步请求。\n"
            "摘要、问答或 RAG 类请求不要路由到 FeishuSyncToNasTool，除非用户先明确要求同步。\n"
            "如果缺少 folder_url 或 nas_dir，应路由到普通回复，让用户补充缺失参数；"
            "不要路由到 FeishuSyncToNasTool。\n"
            "永远不要把 FEISHU_FOLDER_URL 或 FEISHU_SYNC_NAS_DIR 当作真实请求参数。\n"
            "闲聊、简单改写、简单翻译或不需要检索的一步写作任务可以保持 SIMPLE_TASK。\n"
            "只返回 required function call。"
        ),
        user_template=(
            "User Input:\n"
            "{user_input}\n\n"
            "Runtime Context Summary:\n"
            "{context_summary}\n\n"
            "Routing Guidance:\n"
            "- 当答案依赖公司制度、contracts、internal knowledge base、company documents、流程、SOPs、"
            "报销规则、采购规则、审批规则、OA、ERP、Wiki 或其他内部资料时，选择 COMPLEX_TASK。\n"
            "- 当用户询问 A/B 案例、AB 案例、A\\B 案例、A案例、B案例、奖励案例、惩罚案例、评分案例、相似案例、历史案例参考、按案例打分，或判断该奖励还是惩罚时，选择 COMPLEX_TASK。A案例表示好事奖励，B案例表示坏事惩罚。\n"
            "- 运行时需要先搜索 knowledge base 再生成最终答案时，选择 COMPLEX_TASK；Choose COMPLEX_TASK when the runtime should search a knowledge base before generating the final answer。\n"
            "- FeishuSyncToNasTool 是会向 NAS 写文件的重副作用工具。\n"
            "- 只有用户明确要求 sync、download、import、update 或 save 飞书共享文件夹到 NAS，"
            "且提供真实 folder_url 和 nas_dir 时，才路由到 FeishuSyncToNasTool。\n"
            "- 不要因为用户提到 Feishu 就路由到 FeishuSyncToNasTool。\n"
            "- RAG 或摘要请求不要路由到 FeishuSyncToNasTool，除非用户明确先要求同步。\n"
            "- 如果 folder_url 或 nas_dir 缺失，请让用户补充缺失参数，不要创建 FeishuSyncToNasTool task。\n"
            "- 永远不要把 FEISHU_FOLDER_URL、$FEISHU_FOLDER_URL 或 brace form 当作真实值。\n"
            "- 永远不要把 FEISHU_SYNC_NAS_DIR、$FEISHU_SYNC_NAS_DIR 或 brace form 当作真实值。\n"
            "- Choose SIMPLE_TASK for casual chat 或不需要外部/内部检索的一步写作任务。\n\n"
            "请使用 required structured function call 决定执行路由。"
        ),
    )


def build_default_prompt_registry() -> PromptRegistry:
    registry = PromptRegistry()
    registry.register(get_supervisor_prompt_template())
    return registry
