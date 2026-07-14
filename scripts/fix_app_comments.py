from __future__ import annotations

import ast
import re
from collections import defaultdict
from pathlib import Path


ROOT = Path("app")

FOLDER_DESC = {
    "adapters": "模型适配层",
    "agents": "智能体层",
    "api": "API 接口层",
    "context": "上下文管理层",
    "executor": "任务执行层",
    "graph": "运行图编排层",
    "observability": "可观测性层",
    "planner": "任务规划层",
    "prompts": "提示词管理层",
    "queue": "任务队列层",
    "router": "任务路由层",
    "schemas": "数据模型层",
    "state": "运行状态层",
    "tools": "工具能力层",
    "utils": "通用工具层",
}

STEM_DESC = {
    "__init__": "导出聚合",
    "base": "基础抽象",
    "bootstrap": "依赖组装",
    "service": "服务编排",
    "runtime": "运行时入口",
    "router": "路由定义",
    "schemas": "接口模型",
    "server": "服务启动",
    "logging": "日志封装",
    "supervisor": "监督代理",
    "task_router": "任务路由",
    "langgraph_state": "LangGraph 状态管理",
    "metrics": "指标采集",
    "replay": "回放支持",
    "traces": "链路追踪",
    "registry": "注册表管理",
    "llm_tools": "LLM 工具提示词",
    "task_planner": "任务规划提示词",
    "provider": "提供方抽象",
    "llm_base": "LLM 基类",
    "llm_client": "LLM 客户端",
    "llm_reason": "推理工具",
    "text_generate": "文本生成工具",
    "function_calling": "函数调用能力",
    "deepseek_client": "DeepSeek 客户端",
    "failover_client": "故障切换客户端",
    "qwen_client": "Qwen 客户端",
    "openai": "OpenAI 适配器",
    "deepseek": "DeepSeek 适配器",
    "qwen": "Qwen 适配器",
    "parser": "解析能力",
    "llm_planner": "LLM 规划器",
    "task_queue": "任务队列",
    "checkpoint": "检查点模型",
    "checkpoints": "检查点管理",
    "context": "上下文模型",
    "observability": "可观测性模型",
    "model": "模型配置",
    "llm": "LLM 数据结构",
    "graph": "运行图模型",
    "executor": "执行器模型",
    "queue": "队列模型",
    "tool_outputs": "工具输出模型",
    "tool": "工具模型",
    "task": "任务模型",
    "planner": "规划模型",
    "runtime_graph": "运行图实现",
    "utils": "辅助工具",
}

WORD_MAP = {
    "agent": "代理",
    "api": "接口",
    "attach": "附加",
    "base": "基础",
    "bind": "绑定",
    "build": "构建",
    "callback": "回调",
    "checkpoint": "检查点",
    "checkpoints": "检查点",
    "clear": "清理",
    "client": "客户端",
    "coerce": "转换",
    "collector": "采集器",
    "component": "组件",
    "components": "组件",
    "configure": "配置",
    "context": "上下文",
    "create": "创建",
    "deepseek": "DeepSeek",
    "default": "默认",
    "dependencies": "依赖",
    "detail": "详情",
    "engine": "引擎",
    "ensure": "确保",
    "error": "错误",
    "execute": "执行",
    "executor": "执行器",
    "export": "导出",
    "failover": "故障切换",
    "final": "最终",
    "from": "从",
    "generate": "生成",
    "get": "获取",
    "graph": "运行图",
    "idempotency": "幂等",
    "input": "输入",
    "invoke": "调用",
    "langgraph": "LangGraph",
    "latency": "耗时",
    "latest": "最新",
    "llm": "LLM",
    "load": "加载",
    "local": "本地",
    "log": "日志",
    "logger": "日志器",
    "manager": "管理器",
    "mark": "标记",
    "metrics": "指标",
    "mode": "模式",
    "model": "模型",
    "node": "节点",
    "normalize": "规范化",
    "openai": "OpenAI",
    "output": "输出",
    "parse": "解析",
    "parser": "解析器",
    "payload": "载荷",
    "phase": "阶段",
    "plan": "计划",
    "planner": "规划器",
    "prompt": "提示词",
    "provider": "提供方",
    "qwen": "Qwen",
    "queue": "队列",
    "reason": "推理",
    "record": "记录",
    "register": "注册",
    "registry": "注册表",
    "replay": "回放",
    "request": "请求",
    "reset": "重置",
    "resolve": "解析",
    "response": "响应",
    "result": "结果",
    "route": "路由",
    "router": "路由器",
    "run": "运行",
    "runtime": "运行时",
    "save": "保存",
    "schema": "模型",
    "session": "会话",
    "set": "设置",
    "simple": "简单",
    "snapshot": "快照",
    "start": "开始",
    "state": "状态",
    "status": "状态",
    "store": "存储",
    "supervisor": "监督器",
    "task": "任务",
    "text": "文本",
    "timeout": "超时",
    "tool": "工具",
    "trace": "追踪",
    "usage": "用量",
    "utc": "UTC",
    "validate": "校验",
}

SKIP_PARAM_NAMES = {"self", "cls"}


def split_name(name: str) -> list[str]:
    cleaned = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", name).replace("__", "_")
    return [part for part in cleaned.strip("_").lower().split("_") if part]


def phrase_for_parts(parts: list[str]) -> str:
    mapped = [WORD_MAP.get(part, part.upper() if len(part) <= 3 else part) for part in parts]
    return "".join(mapped) if mapped else "相关对象"


def phrase_for_name(name: str) -> str:
    return phrase_for_parts(split_name(name))


def module_description(path: Path) -> str:
    parts = path.parts
    folder = parts[1] if len(parts) > 1 else "app"
    folder_desc = FOLDER_DESC.get(folder, "应用层")
    stem = path.stem
    stem_desc = STEM_DESC.get(stem, f"{phrase_for_name(stem)}能力")
    if stem == "__init__":
        return f"{folder_desc}的导出模块。负责汇总并暴露该目录下对外可用的公共接口，方便上层统一导入和复用。"
    return f"{folder_desc}中的{stem_desc}模块。负责承载与{stem_desc}相关的核心逻辑，并为上层流程提供清晰、稳定的调用入口。"


def class_description(node: ast.ClassDef) -> str:
    name = node.name
    readable = phrase_for_name(name)
    if name.endswith("Error"):
        return f"表示{readable}场景下抛出的异常类型，用于让上层能够区分错误来源并执行对应的恢复或提示逻辑。"
    if any(name.endswith(suffix) for suffix in ("State", "Model", "Record", "Snapshot", "Result", "Response", "Request", "Detail", "Decision", "Plan", "Dependencies")):
        return f"定义{readable}相关的数据结构，统一约束字段语义、序列化格式以及上下游之间的交互边界。"
    if any(name.endswith(suffix) for suffix in ("Store", "Manager", "Collector", "Engine", "Builder", "Executor", "Router", "Queue", "Planner", "Parser", "Client", "Tool", "Agent")):
        return f"封装{readable}相关的行为逻辑，负责协调依赖、维护内部状态，并向外提供可复用的操作接口。"
    return f"封装{readable}相关能力，用于在当前模块内组织状态、行为以及与其他组件的协作关系。"


def describe_action(name: str) -> tuple[str, str]:
    public_name = name.lstrip("_")
    parts = split_name(public_name)
    if name == "__init__":
        return (
            "初始化实例，并保存当前对象运行所需的依赖、配置和默认状态。",
            "这样可以在对象创建完成后立即具备可用的运行上下文，减少后续方法调用时的重复准备工作。",
        )
    if public_name.startswith("build_"):
        target = phrase_for_parts(parts[1:])
        return (f"构建{target}。", "该函数会按照当前上下文或输入参数组装所需对象，并返回可直接使用的结果。")
    if public_name.startswith("create_"):
        target = phrase_for_parts(parts[1:])
        return (f"创建{target}。", "该函数会基于输入数据生成新的运行对象或状态实例。")
    if public_name.startswith("get_"):
        target = phrase_for_parts(parts[1:])
        return (f"获取{target}。", "该函数主要负责从当前上下文、缓存或配置中读取目标数据。")
    if public_name.startswith("set_"):
        target = phrase_for_parts(parts[1:])
        return (f"设置{target}。", "该函数用于更新当前对象或全局环境中的关键配置，以便后续流程读取最新值。")
    if public_name.startswith("reset_"):
        target = phrase_for_parts(parts[1:])
        return (f"重置{target}。", "该函数用于恢复默认配置或清空已有状态，避免旧数据影响新的执行流程。")
    if public_name.startswith("load_"):
        target = phrase_for_parts(parts[1:])
        return (f"加载{target}。", "该函数用于从外部存储、配置源或序列化结果中恢复目标数据。")
    if public_name.startswith("save_"):
        target = phrase_for_parts(parts[1:])
        return (f"保存{target}。", "该函数会把当前阶段产生的关键数据持久化，方便后续恢复、追踪或复用。")
    if public_name.startswith("record_"):
        target = phrase_for_parts(parts[1:])
        return (f"记录{target}。", "该函数用于沉淀运行过程中的事件、指标或中间结果，方便后续观察和诊断。")
    if public_name.startswith("attach_"):
        target = phrase_for_parts(parts[1:])
        return (f"附加{target}。", "该函数会把补充信息绑定到现有对象上，保持上下文信息的完整性。")
    if public_name.startswith("register_"):
        target = phrase_for_parts(parts[1:])
        return (f"注册{target}。", "该函数会把能力或对象写入注册表，便于后续按名称或类型查找。")
    if public_name.startswith("route_"):
        target = phrase_for_parts(parts[1:])
        return (f"决定{target}。", "该函数会根据当前状态或规则选择下一步执行路径，保证流程分支清晰可控。")
    if public_name.startswith("parse_"):
        target = phrase_for_parts(parts[1:])
        return (f"解析{target}。", "该函数负责把原始输入转换为结构化结果，方便后续模块直接消费。")
    if public_name.startswith("execute_"):
        target = phrase_for_parts(parts[1:])
        return (f"执行{target}。", "该函数会驱动核心业务步骤落地，并在需要时回收或上报执行结果。")
    if public_name.startswith("run_"):
        target = phrase_for_parts(parts[1:])
        return (f"运行{target}。", "该函数通常作为流程入口，负责串联多个子步骤并返回最终产物。")
    if public_name.startswith("resolve_"):
        target = phrase_for_parts(parts[1:])
        return (f"解析{target}。", "该函数用于根据当前条件推导出最终应使用的配置、对象或执行策略。")
    if public_name.startswith("coerce_"):
        target = phrase_for_parts(parts[1:])
        return (f"转换{target}。", "该函数用于把输入值规范化为系统内部统一的表示形式。")
    if public_name.startswith("ensure_"):
        target = phrase_for_parts(parts[1:])
        return (f"确保{target}。", "该函数用于在继续执行前补齐必要前置条件，降低后续流程出现异常的概率。")
    if public_name.startswith("clear_"):
        target = phrase_for_parts(parts[1:])
        return (f"清理{target}。", "该函数用于移除临时状态、上下文标记或缓存结果，保持运行环境整洁。")
    if public_name.startswith("bind_"):
        target = phrase_for_parts(parts[1:])
        return (f"绑定{target}。", "该函数用于把上下文信息和当前执行流程关联起来，方便日志与追踪统一定位。")
    if public_name.startswith("configure_"):
        target = phrase_for_parts(parts[1:])
        return (f"配置{target}。", "该函数用于初始化基础设施对象，使其满足当前运行环境的约束和输出需求。")
    if public_name.startswith("export_"):
        target = phrase_for_parts(parts[1:])
        return (f"导出{target}。", "该函数用于把内部累计的数据转换为适合展示、传输或持久化的格式。")
    if public_name.startswith("mark_"):
        target = phrase_for_parts(parts[1:])
        return (f"标记{target}。", "该函数用于更新任务或状态对象的阶段信息，帮助流程感知当前进度。")
    if public_name.startswith("validate_"):
        target = phrase_for_parts(parts[1:])
        return (f"校验{target}。", "该函数用于提前发现输入、配置或中间结果中的不合法情况。")
    if public_name.startswith("normalize_"):
        target = phrase_for_parts(parts[1:])
        return (f"规范化{target}。", "该函数用于统一不同来源数据的格式和字段习惯，减少分支处理。")
    if public_name.startswith("replay_"):
        target = phrase_for_parts(parts[1:])
        return (f"回放{target}。", "该函数用于复现历史执行过程，帮助调试、排查问题或观察行为轨迹。")
    target = phrase_for_name(public_name)
    prefix = "在内部处理" if name.startswith("_") else "处理"
    return (f"{prefix}{target}。", "该函数用于完成当前步骤对应的核心逻辑，并向后续流程返回可继续消费的结果。")


def param_description(name: str) -> str:
    parts = split_name(name)
    readable = phrase_for_parts(parts)
    if not parts:
        return "调用方传入的参数。"
    if parts[-1] == "id":
        return f"用于唯一标识{phrase_for_parts(parts[:-1]) or '目标对象'}。"
    if "path" in parts:
        return "目标文件或目录路径。"
    if "timeout" in parts:
        return "超时控制参数，用于限制等待时间。"
    if "request" in parts:
        return "请求上下文或请求数据对象。"
    if "response" in parts:
        return "响应对象或响应数据。"
    if "state" in parts:
        return "当前流程对应的状态对象。"
    if "context" in parts:
        return "执行过程中共享的上下文容器。"
    if "task" in parts:
        return "任务对象或任务标识，用于定位当前执行单元。"
    if "result" in parts:
        return "上一步得到的结果对象，供当前逻辑继续处理。"
    if "config" in parts:
        return "当前逻辑依赖的配置项。"
    return f"与{readable}相关的输入参数。"


def return_description(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    if node.returns is None:
        return "该函数主要通过副作用更新状态，通常不依赖返回值继续流程。"
    try:
        rendered = ast.unparse(node.returns)
    except Exception:
        rendered = ""
    lowered = rendered.lower()
    if lowered == "none":
        return "不返回额外结果，执行效果会体现在对象状态、上下文或外部副作用中。"
    if "bool" in lowered:
        return "返回布尔值，用于表示某个条件是否满足或步骤是否执行成功。"
    if "list" in lowered:
        return "返回列表结果，通常包含多个可供后续流程继续处理的元素。"
    if "dict" in lowered:
        return "返回字典结果，便于按字段读取当前步骤产出的结构化数据。"
    if "str" in lowered:
        return "返回字符串结果，通常用于提供文本内容、标识或错误说明。"
    return "返回处理后的结果对象，供调用方继续串联后续流程。"


def build_docstring(node: ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef, indent: str) -> list[str]:
    if isinstance(node, ast.ClassDef):
        return [f'{indent}"""{class_description(node)}"""\n']

    summary, details = describe_action(node.name)
    lines = [f'{indent}"""{summary}\n', "\n", f"{indent}{details}\n"]
    params = [arg.arg for arg in node.args.args + node.args.kwonlyargs if arg.arg not in SKIP_PARAM_NAMES]
    if node.args.vararg is not None:
        params.append(f"*{node.args.vararg.arg}")
    if node.args.kwarg is not None:
        params.append(f"**{node.args.kwarg.arg}")
    if params:
        lines.extend(["\n", f"{indent}参数:\n"])
        for param in params:
            raw_name = param.lstrip("*")
            lines.append(f"{indent}    {param}: {param_description(raw_name)}\n")
    lines.extend(["\n", f"{indent}返回:\n", f"{indent}    {return_description(node)}\n", f'{indent}"""\n'])
    return lines


def build_module_docstring(path: Path) -> list[str]:
    return [f'"""{module_description(path)}"""\n', "\n"]


def assign_description(name: str) -> str | None:
    cleaned = name.strip("_")
    if not cleaned:
        return None
    if cleaned.isupper() or name.startswith("_runtime_") or name.startswith("_default_") or name.startswith("_DEFAULT_"):
        readable = phrase_for_name(cleaned.lower())
        return f"# 定义与{readable}相关的模块级常量或共享对象，供当前模块的后续逻辑统一复用。\n"
    return None


def contains_chinese(text: str) -> bool:
    return any("\u4e00" <= char <= "\u9fff" for char in text)


def is_placeholder_text(text: str) -> bool:
    return "?" in text and not contains_chinese(text)


def get_docstring_expr(node: ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef) -> ast.Expr | None:
    body = getattr(node, "body", [])
    if not body:
        return None
    first = body[0]
    if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) and isinstance(first.value.value, str):
        return first
    return None


def replace_slice(lines: list[str], start: int, end: int, replacement: list[str]) -> None:
    lines[start:end] = replacement


def previous_nonempty_index(lines: list[str], index: int) -> int | None:
    cursor = index - 1
    while cursor >= 0 and lines[cursor].strip() == "":
        cursor -= 1
    return cursor if cursor >= 0 else None


def main() -> None:
    changed_files: list[str] = []

    for path in sorted(ROOT.rglob("*.py")):
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        lines = source.splitlines(keepends=True)
        replacements: list[tuple[int, int, list[str]]] = []
        insertions: dict[int, list[str]] = defaultdict(list)

        module_expr = get_docstring_expr(tree)
        if module_expr is None:
            insert_at = 0
            while insert_at < len(lines) and lines[insert_at].startswith("from __future__ import"):
                insert_at += 1
            if insert_at < len(lines) and lines[insert_at].strip() == "":
                insert_at += 1
            insertions[insert_at].extend(build_module_docstring(path))
        else:
            module_text = ast.get_docstring(tree, clean=False) or ""
            if is_placeholder_text(module_text):
                replacements.append((module_expr.lineno - 1, module_expr.end_lineno, build_module_docstring(path)))

        for node in ast.walk(tree):
            if not isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) or not node.body:
                continue

            doc_expr = get_docstring_expr(node)
            if doc_expr is None:
                body_line_index = node.body[0].lineno - 1
                indent = " " * node.body[0].col_offset
                insertions[body_line_index].extend(build_docstring(node, indent))
                continue

            node_text = ast.get_docstring(node, clean=False) or ""
            if is_placeholder_text(node_text):
                indent = " " * doc_expr.col_offset
                replacements.append((doc_expr.lineno - 1, doc_expr.end_lineno, build_docstring(node, indent)))

        for node in tree.body:
            target_name = None
            if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                target_name = node.targets[0].id
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                target_name = node.target.id
            if target_name is None:
                continue

            comment = assign_description(target_name)
            if comment is None:
                continue

            line_index = node.lineno - 1
            prev_index = previous_nonempty_index(lines, line_index)
            if prev_index is None:
                insertions[line_index].append(comment)
            else:
                prev_line = lines[prev_index]
                if prev_line.lstrip().startswith("#"):
                    if is_placeholder_text(prev_line):
                        replacements.append((prev_index, prev_index + 1, [comment]))
                else:
                    insertions[line_index].append(comment)

        if not replacements and not insertions:
            continue

        for start, end, replacement in sorted(replacements, key=lambda item: item[0], reverse=True):
            replace_slice(lines, start, end, replacement)

        new_lines: list[str] = []
        for index in range(len(lines) + 1):
            if index in insertions:
                new_lines.extend(insertions[index])
            if index < len(lines):
                new_lines.append(lines[index])

        new_source = "".join(new_lines)
        if new_source != source:
            path.write_text(new_source, encoding="utf-8")
            changed_files.append(path.as_posix())

    print(f"changed_files={len(changed_files)}")
    for item in changed_files:
        print(item)


if __name__ == "__main__":
    main()
