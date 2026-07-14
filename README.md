# Agent Runtime System

Agent Runtime System 是一个面向生产化演进的 Agent 运行时框架。它不是简单地调用一次大模型，而是把用户请求拆成可校验、可调度、可恢复、可观测的任务 DAG，通过 Supervisor、Planner、Parser、Queue、Executor 和 Aggregator 串起完整执行流程。

当前阶段：`Runtime Alpha v0.6`

## 核心能力

- HTTP API 统一入口，支持 `POST /run`
- Supervisor 判断简单任务或复杂任务
- Planner 生成结构化 `TaskPlan`
- Parser 校验 JSON、Schema、依赖关系、环路和工具契约
- Queue 按 DAG 依赖调度任务，支持并发与死锁保护
- Executor 执行工具，支持超时、重试、幂等恢复和失败分类
- Router 基于工具能力、权限和任务类型做路由
- LLM Client 支持多模型 fallback、retry、circuit breaker 和 function calling
- RAG 链路支持知识库检索、批摘要和基于证据的最终回答
- MySQL 业务工具支持周报、计划、部门自评、完成率、OPL 等结构化分析
- Feishu 到 NAS 同步工具支持明确授权的飞书文件夹同步
- Trace、metrics、checkpoint 和 replay snapshot 支持运行时观测与恢复

## 主流程

```text
API /run
  -> supervisor
  -> planner 或 simple_task
  -> parser
  -> queue
  -> executor
  -> aggregator
  -> response
```

复杂任务会被规划成多个 `TaskModel`，再由 Queue 和 Executor 按依赖关系执行。任一阶段失败都会进入聚合阶段，返回结构化错误与上下文信息。

## 项目结构

```text
app/api            FastAPI 网关与 Runtime 装配
app/graph          LangGraph 编排图
app/agents         Supervisor Agent
app/planner        LLM Planner、Parser、Repair Pipeline
app/queue          DAG 调度队列
app/executor       任务执行器、重试与状态保护
app/router         工具路由、权限、能力匹配
app/tools          RAG、LLM、MySQL、Feishu 等工具
app/adapters       DeepSeek、Qwen、OpenAI 等模型适配
app/llm            LLM 客户端、fallback、retry、function calling
app/context        Runtime Context 与 checkpoint
app/observability  trace、metrics、replay、snapshot
app/schemas        Pydantic 数据契约
tests              回归测试与流程测试
docs               架构说明、测试说明、已知债务
```

## 快速开始

建议使用 Python 3.10+。

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

如果使用 Conda：

```bash
conda create -n Agent python=3.10 -y
conda activate Agent
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

## 启动 API

```bash
uvicorn app.api.server:app --host 0.0.0.0 --port 8000
```

请求示例：

```bash
curl -X POST "http://127.0.0.1:8000/run?debug=true" \
  -H "Content-Type: application/json" \
  -d '{"user_input":"查询公司报销流程是什么"}'
```

## 常用环境变量

不同模型和工具会读取对应环境变量。常见配置包括：

```text
DEEPSEEK_API_KEY
QWEN_API_KEY
OPENAI_API_KEY
RAG_BASE_URL
MYSQL_HOST
MYSQL_PORT
MYSQL_USER
MYSQL_PASSWORD
MYSQL_DATABASE
FEISHU_APP_ID
FEISHU_APP_SECRET
```

不要把 `.env`、API Key、数据库密码或运行输出提交到 GitHub。

## 测试

运行全部测试：

```bash
python -m pytest -q
```

运行 RAG 相关关键测试：

```bash
python -m pytest -q \
  tests/test_query_parser.py \
  tests/test_search_payload_builder.py \
  tests/test_rag_search_tool.py \
  tests/test_runtime_graph_rag_flow.py \
  tests/test_rag_planning_decision.py
```

## 当前状态

当前主线已经覆盖 Runtime、RAG、MySQL 业务工具、LLM fallback、Parser Repair、Router 权限与观测能力。PPT 渲染、PPT 模板索引和 PPT workflow 相关测试目前属于非 active baseline，详见已知债务文档。

## 文档

- [项目流程与技术详解](docs/project-flow-and-tech.md)
- [测试说明](docs/testing.md)
- [Runtime Alpha v0.6 总结](docs/runtime_alpha_v0_6.md)
- [Runtime 变更记录](docs/runtime_changes.md)
- [已知债务](docs/known_debt.md)
- [MySQL Business Tools](docs/mysql_business_tools.md)

## 提交代码

```bash
git status
git add .
git commit -m "Update README"
git push
```
