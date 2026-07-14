from __future__ import annotations


CHINESE_PRIMARY_POLICY = "\n".join(
    [
        "语言策略：",
        "1. 业务规则、推理步骤、风险边界和面向用户的自然语言要求，优先使用中文表达。",
        "2. JSON key、schema field、tool name、function name、enum value、状态值、占位符和代码标识符必须保持英文原样。",
        "3. Section label 可保持英文，便于运行时解析和测试定位，例如 User Goal、Additional Context、Required JSON Schema。",
        "4. 如果结构字段和中文说明冲突，以结构字段、JSON Schema 和工具能力定义为准。",
    ]
)


def chinese_primary_policy_block(*, role: str | None = None) -> str:
    if not role:
        return CHINESE_PRIMARY_POLICY
    return f"{role}\n{CHINESE_PRIMARY_POLICY}"
