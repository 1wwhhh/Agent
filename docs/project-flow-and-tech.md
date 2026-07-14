# Agent Runtime System 项目流程与技术详解

## 1. 项目定位

这个项目是一个“可编排、可恢复、可观测”的 Agent Runtime：

- 接收用户请求（HTTP API）
- 由 Supervisor 判断简单任务或复杂任务
- 复杂任务走 Planner 生成 Task DAG
- Queue + Executor 按依赖与并发执行工具
- 聚合结果并输出标准响应
- 全程记录 trace、metrics、checkpoint，支持 replay 和恢复

它不是单纯的“调用一次 LLM”，而是完整的任务执行系统。

---

## 2. 总体分层架构

- `app/api`: 网关层（FastAPI），对外统一入口
- `app/graph`: LangGraph 编排层（节点与路由）
- `app/agents`: Supervisor Agent（任务复杂度与路由决策）
- `app/planner`: LLM Planner + Parser（任务分解与 DAG 化）
- `app/queue`: DAG-aware 调度队列（并发、依赖、死锁保护）
- `app/executor`: 执行器（路由、超时、重试、幂等）
- `app/router`: 任务到工具的路由策略（static/dynamic/failover）
- `app/tools`: 工具抽象与 LLM 工具实现（function calling + schema）
- `app/adapters`: 模型供应商适配层（DeepSeek/Qwen/OpenAI）
- `app/context`: 运行时上下文与 checkpoint
- `app/observability`: trace、metrics、replay
- `app/schemas`: 全部 Pydantic 数据契约

---

## 3. 端到端主流程（请求生命周期）

### 3.1 API 接入

入口文件：

- `app/api/server.py` 创建 FastAPI 应用
- `app/api/router.py` 暴露 `POST /run`
- `app/api/service.py` 的 `run_runtime()` 是统一执行入口

`/run` 支持两个 query 参数：

- `debug=true`: 返回更详细 trace/metrics
- `replay=true`: 不跑实时执行，进入重放模式

### 3.2 Runtime 组件装配

`build_env_runtime_components()` 会：

1. 从环境变量读取 `RuntimeLLMConfig`（主模型 + fallback）
2. `ModelRouter` 构建 `LLMClient`（挂载多个 provider adapter）
3. 创建 `TaskRouter`，注册工具：
   - `llm_reason_tool`
   - `text_generate_tool`
4. 创建 `SupervisorAgent` 与 `LLMTaskPlanner`
5. 准备 `RuntimeCheckpointManager`

### 3.3 LangGraph 编排执行

图节点（`app/graph/runtime_graph.py`）：

1. `supervisor`
2. `planner`（复杂任务）或 `simple_task`（简单任务）
3. `parser`（复杂任务时）
4. `queue`
5. `executor`
6. `aggregator`

核心路由规则：

- `supervisor`:
  - `SIMPLE_TASK` -> `simple_task`
  - `COMPLEX_TASK` -> `planner`
- 任一阶段失败会转 `aggregator` 输出失败态

### 3.4 Supervisor 阶段

优先级：

1. checkpoint 恢复场景复用已有路由
2. `force_route` 元数据强制路由
3. 有 `SupervisorAgent` 时，LLM 结构化判定
4. 否则用启发式（词数+关键词）判定

输出：`SIMPLE_TASK` / `COMPLEX_TASK`，并落到 `state.supervisor_route`

### 3.5 Planner / SimpleTask 阶段

- 简单任务：直接包装为单个 `TaskModel`（默认走文本生成工具或推理工具）
- 复杂任务：
  - `LLMTaskPlanner` 通过 function calling 输出 `TaskPlan`
  - 强约束 JSON Schema（任务列表、依赖、状态、重试等字段）

### 3.6 Parser 阶段（复杂任务）

`TaskParser` 做三层校验：

1. 文本提取与 JSON 解析（支持 fenced code 提取）
2. `TaskPlan` schema 校验
3. 语义校验（依赖存在、无环、输出 key 唯一、初始状态合法）

最终转成运行时 `TaskModel` 列表。

### 3.7 Queue 阶段

`TaskQueue` 负责 DAG 调度：

- 初始化任务、建立依赖图
- 校验 missing dependency / cycle / orphan task
- 基于依赖与并发槽位选 ready tasks
- 状态流转：`PENDING/RETRY -> QUEUED -> RUNNING -> SUCCESS/FAILED/...`
- 死锁检测：永久阻塞 DAG 直接报错
- 上游失败时递归取消下游（`CANCELLED`）

### 3.8 Executor 阶段

`TaskExecutor` 负责真实执行与可靠性策略：

- 调用前先尝试幂等恢复（`idempotency_records`）
- 通过 `TaskRouter` 做工具路由：
  - `static`
  - `dynamic`
  - `failover`
- 注入依赖任务输出到 payload `context`
- 分层超时：
  - `timeout_seconds`（LLM请求）
  - `tool_timeout_seconds`（工具层）
  - `executor_timeout_seconds`（执行器层）
- `asyncio.wait_for` 控制执行上限
- 成功：写 `task_results`、写 execution result、记 checkpoint
- 失败：分类 timeout / retryable / fail-fast，决定重试或终态失败

### 3.9 Aggregator 阶段

聚合产出 `final_response`：

- request/session/supervisor 路由
- task summary（pending/completed/failed）
- `task_results` + `execution_results`
- errors + metadata
- phase/final_output_ready

API 最终响应中同时附带 `task_states` 与 `trace`。

---

## 4. 模型层与工具层技术细节

## 4.1 供应商适配（Adapter）

基类 `ModelAdapter` 提供：

- OpenAI-compatible `/chat/completions` 请求
- headers/payload/timeout 构建
- response 与 stream chunk 解析
- function call 解析
- provider 错误重试判定

具体实现：

- `DeepSeekAdapter`
  - 修正 `tool_choice` 兼容逻辑
- `QwenAdapter`
  - 明确 `supports_function_calling_with_streaming=False`
- `OpenAIAdapter`
  - 组织头与 stream options

## 4.2 LLMClient（关键可靠性层）

`LLMClient` 具备：

- 多 provider 顺序尝试（主+备）
- provider 级并发信号量（如 deepseek=5, qwen=10）
- circuit breaker（失败阈值 + reset 时间）
- retry + exponential backoff + jitter
- fail-fast timeout marker 与 retryable marker
- 统一 request/response trace 注入

## 4.3 Function Calling + Schema 强约束

`FunctionCallingAdapter.invoke_structured()`：

- 强制 `tool_choice=指定函数`
- 校验函数名与 arguments
- 用 Pydantic 输出模型校验
- 校验失败自动追加“更严格 system 提示”再重试

这使得 Supervisor/Planner/Tool 输出都可结构化消费。

## 4.4 工具体系

- `BaseTool`: 统一工具接口与超时、标准 `ToolResult`
- `BaseLLMTool`: Prompt 渲染、模板变量注入、LLM 请求构建
- `LLMReasonTool`: 偏推理分析，带 minimal retry 策略
- `TextGenerateTool`: 偏文本生成，启用 fail-fast timeout 策略

---

## 5. 数据与状态模型（Pydantic 契约）

关键模型：

- `TaskModel` / `TaskStatus`
- `TaskPlan`（Planner 输出契约）
- `LangGraphState`（图运行主状态）
- `ContextStore`（任务、结果、错误、token、tool calls、幂等记录）
- `QueueSnapshot`
- `TaskExecutionResult`
- `ToolRouteDecision`
- `RuntimeCheckpoint`
- `PersistedExecutionTrace` / `RequestMetricsSnapshot`

优点：

- 输入输出契约严格（`extra="forbid"`）
- 状态变更可审计
- replay/checkpoint/metrics 共享同一套结构化数据

---

## 6. 可观测性与可恢复性

## 6.1 Trace

`LocalExecutionTraceStore` 把请求与状态事件写入：

- `outputs/runtime_traces/{request_id}.json`

记录内容包括：

- layer/event/phase
- node 执行顺序
- task states / task graph
- tool calls
- context snapshot

## 6.2 Metrics

`RuntimeMetricsCollector` 聚合：

- task success rate
- DAG correctness rate
- retry rate
- context consistency rate
- latency 分解（node + total）

## 6.3 Replay

`RuntimeReplayEngine` 支持：

- `FULL`
- `STEP_BY_STEP`（debug）

通过 trace 还原步骤、任务状态、最终输出。

## 6.4 Checkpoint

`RuntimeCheckpointManager` 把状态快照写入：

- `outputs/runtime_checkpoints/{request_id}__{checkpoint_id}.json`

可从 latest checkpoint 恢复运行，支持中断后续跑。

---

## 7. 配置与环境变量

模型配置在 `app/schemas/model.py`，支持：

- `LLM_PROVIDER`
- `LLM_FALLBACK_PROVIDERS`
- `*_API_KEY`
- `*_BASE_URL`
- `*_MODEL_NAME`
- `*_TIMEOUT_SECONDS`
- `*_MAX_RETRIES`

Provider 默认值：

- DeepSeek: `https://api.deepseek.com`
- Qwen: `https://dashscope-intl.aliyuncs.com/compatible-mode/v1`
- OpenAI: `https://api.openai.com/v1`

---

## 8. 测试覆盖能力面

从 `tests/` 文件看，覆盖了这些重点：

- API 与系统网关流程
- 简单/复杂 DAG 执行
- 并发与死锁保护
- 工具失败、执行器失败、超时强化
- 幂等恢复与 checkpoint 恢复
- replay 与 metrics 持久化
- model router / deepseek / qwen / failover client
- prompt 上下文注入与上下文一致性/原子性

这说明你的项目已经从“功能跑通”走到“工程可靠性”层级。

---

## 9. 你这个项目的核心价值总结

你现在这套系统的技术亮点是：

- 用 LangGraph 把 Agent 过程编排成可控状态机
- 用 Pydantic + function calling 把 LLM 输出变成强契约
- 用 Queue/Executor 把 DAG 执行做成可恢复、可重试、可审计
- 用 trace/metrics/replay/checkpoint 构建可观测闭环

一句话总结：这是一个“面向生产化演进”的 Agent Runtime 骨架，不是 demo 级流水线。
