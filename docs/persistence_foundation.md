# P5：Persistence Foundation

## 设计目标

`P5 Persistence Foundation` 的目标，是把当前 Runtime 的纯内存执行状态，设计为未来可持久化的状态边界。

本阶段的目标包括：

- 将 Runtime State 从纯内存结构，整理为后续可落库的数据模型
- 支持 `request`、`task`、`trace`、`replay` 的长期保存
- 为未来接入 `PostgreSQL` 与 `Redis` 做边界准备
- 明确当前只处理 Runtime State，不处理 Memory、RAG 或 Multi-Agent

必须强调的边界：

- Persistence 不等于 Memory
- 本阶段持久化的是 Runtime State，不是长期记忆

## 当前 Runtime State 来源

当前 Runtime 的内存状态主要来自以下对象和聚合数据：

- `LangGraphState`
- `RuntimeContext` 与 `ContextStore`
- `TaskModel`
- `TaskQueue`
- `execution_history`
- `task_results`
- `metadata`
  - `parser_repair_history`
  - `task_failures`
  - `routing_history`
  - `llm_calls`
- `RuntimeTraceSnapshot`
- `ReplaySnapshot`
- `MetricsSnapshot`
- `DebugSnapshot`

这些结构对于 `Runtime Alpha v0.6` 已经足够，但它们当前仍然是“面向进程内运行”的状态，而不是“面向持久化”的稳定边界。

## 持久化模型总览

后续的 Persistence Foundation 应把数据拆成三类：

1. 关系型 source of truth
2. 面向调试和复盘的聚合快照
3. 暂时仍保留在内存中的执行态数据

推荐的 source-of-truth 表：

- `request`
- `task`
- `task_event`
- `trace_snapshot`
- `replay_snapshot`

推荐的 Projection 层：

- `RuntimeStateProjection`
- `TaskProjection`
- `ContextProjection`
- `TraceProjection`
- `ReplayProjection`

## PostgreSQL 表设计

### `request`

作用：

- 记录一次 Runtime 执行请求的顶层生命周期

主键：

- `request_id`

外键：

- 无，作为根记录存在

核心字段：

- `request_id`
- `session_id`
- `user_input`
- `status`
- `started_at`
- `finished_at`
- `latency_ms`
- `final_output`
- `error_summary`
- `metadata_json`

推荐 JSONB 字段：

- `metadata_json`

推荐索引：

- `request_id` 主键
- `session_id` 索引
- `status` 索引
- `started_at` 索引

生命周期：

- 请求进入 Runtime 时创建
- 执行过程中持续更新
- 请求进入终态后关闭

是否属于 source of truth：

- 是

说明：

- `request` 是一次 Runtime 执行的顶层记录
- 一个 `request` 可关联多条 `task` 记录和多条 `task_event`

### `task`

作用：

- 记录 Task DAG 中每个节点的当前物化状态

主键：

- `task_id`

外键：

- `request_id -> request.request_id`

核心字段：

- `task_id`
- `request_id`
- `name`
- `description`
- `tool`
- `task_type`
- `tags`
- `input_json`
- `output_key`
- `depends_on`
- `priority`
- `status`
- `retry_count`
- `max_retry`
- `timeout`
- `created_at`
- `updated_at`
- `idempotency_key`
- `attempt_key`

推荐 JSONB 字段：

- `input_json`
- `depends_on`
- `tags`

推荐索引：

- `task_id` 主键
- `request_id` 索引
- `(request_id, status)` 组合索引
- `idempotency_key` 索引
- `attempt_key` 索引

生命周期：

- 规划与 Parser 校验成功后创建
- 状态变化、重试次数变化时更新
- 请求结束后仍保留，供审计和复盘使用

是否属于 source of truth：

- 是

说明：

- `task` 表保存的是每个 DAG 节点的“当前状态”
- 当前状态不能替代事件历史，它只是事件流在当前时刻的物化结果

### `task_event`

作用：

- 以追加写方式记录任务生命周期与执行事件

主键：

- `event_id`

外键：

- `request_id -> request.request_id`
- `task_id -> task.task_id`

核心字段：

- `event_id`
- `request_id`
- `task_id`
- `event_type`
- `old_status`
- `new_status`
- `error_type`
- `error_message`
- `payload_json`
- `created_at`

推荐 JSONB 字段：

- `payload_json`

推荐索引：

- `event_id` 主键
- `request_id` 索引
- `task_id` 索引
- `(task_id, created_at)` 组合索引
- `(request_id, created_at)` 组合索引

生命周期：

- 每次任务状态变化或关键执行事件发生时追加一条记录
- 原则上不更新，只追加

是否属于 source of truth：

- 是

状态流转与 `task_event` 的映射：

- `QUEUED`：任务已进入可调度队列
- `RUNNING`：执行器已接管任务
- `SUCCESS`：工具执行完成，结果已提交
- `FAILED`：任务进入终态失败
- `RETRY`：本次执行失败，但已安排重试
- `TIMEOUT`：任务超时
- `CANCELLED`：任务被取消，或因上游失败被阻断

说明：

- `task_event` 必须保留每次 attempt 的事件历史
- 即使同一个 `task` 最终只有一个终态，重试过程中的状态变化也必须保留

### `trace_snapshot`

作用：

- 保存 Runtime Trace 的调试与审计视图

主键：

- `trace_id`

外键：

- `request_id -> request.request_id`

核心字段：

- `trace_id`
- `request_id`
- `snapshot_json`
- `metrics_json`
- `task_graph_json`
- `created_at`

推荐 JSONB 字段：

- `snapshot_json`
- `metrics_json`
- `task_graph_json`

推荐索引：

- `trace_id` 主键
- `request_id` 索引
- `created_at` 索引

生命周期：

- 在 trace 捕获点或请求结束时生成
- 后续如果支持阶段性快照，一个请求可以对应多条记录

是否属于 source of truth：

- 否

说明：

- `trace_snapshot` 保存的是 `RuntimeTraceSnapshot`
- 它属于调试与审计视图，不应作为权威执行账本

### `replay_snapshot`

作用：

- 保存供人工复盘使用的 Replay 视图

主键：

- `replay_id`

外键：

- `request_id -> request.request_id`

核心字段：

- `replay_id`
- `request_id`
- `snapshot_json`
- `raw_user_input`
- `planner_raw_output`
- `repaired_output`
- `final_output`
- `created_at`

推荐 JSONB 字段：

- `snapshot_json`

推荐索引：

- `replay_id` 主键
- `request_id` 索引
- `created_at` 索引

生命周期：

- 需要保留 replay 相关材料时生成
- 一个请求可以有一条，也可以有多条，取决于后续保留策略

是否属于 source of truth：

- 否

说明：

- `replay_snapshot` 保存的是 `ReplaySnapshot`
- 它用于人工复盘和事后分析
- 它不代表真正的 replay execution 能力

## Projection / DTO 设计

持久化层不应直接依赖 `LangGraphState`、`RuntimeContext` 或 `ContextStore`。

后续 Persistence Adapter 应只消费只读 Projection：

### `RuntimeStateProjection`

作用：

- 提供单次 Runtime 执行的稳定顶层视图

建议内容：

- request 顶层字段
- task 聚合摘要
- runtime metadata 摘要
- trace / replay 引用信息

### `TaskProjection`

作用：

- 提供单个任务的持久化安全视图

建议内容：

- 任务身份字段
- tool / routing 字段
- scheduling 字段
- retry / timeout 字段
- 当前状态
- 幂等字段

### `ContextProjection`

作用：

- 提供可持久化的上下文摘要视图

建议内容：

- task result 的引用或摘要
- execution history 摘要
- metadata 摘要
- error 与 metrics 摘要

### `TraceProjection`

作用：

- 提供与 Runtime 内部对象解耦的 trace DTO

建议内容：

- task graph
- task states
- tool call 摘要
- metrics 载荷

### `ReplayProjection`

作用：

- 提供与 Runtime 内部对象解耦的 replay DTO

建议内容：

- raw input
- planner raw output
- repaired output
- final output
- replay snapshot 本体

为什么要这样设计：

- 当前一个已知技术债是 `LangGraphState` 泄漏到了 Parser Repair 和 Observability 路径
- Projection Layer 是后续解决这类泄漏问题的方向
- 持久化层应依赖 DTO，而不是依赖 Runtime 内部对象图

## PostgreSQL / Redis 职责边界

### PostgreSQL 的未来职责

PostgreSQL 应作为 source of truth，负责：

- `request`
- `task`
- `task_event`
- `trace_snapshot`
- `replay_snapshot`

原因：

- 这些数据需要长期保存、关系完整性、可索引能力和稳定查询能力

### Redis 的未来职责

`P5 Foundation` 不接入 Redis，但 Redis 的未来边界可以先定义为：

- session cache
- distributed lock
- queue backend
- running task lease
- idempotency cache
- short-lived runtime cache

原因：

- 这些数据更适合做加速态与协调态，而不是长期权威存储

必须明确：

- `P5 Foundation` 不接 Redis
- 本阶段只设计未来边界，不做接入实现

## 暂时仍保留内存中的数据

以下数据在当前阶段仍应保留在内存中：

- 当前执行中的 `LangGraphState`
- 当前运行中的 `RuntimeContext`
- 当前 `TaskQueue`
- tool instance registry
- in-flight execution handles

原因：

- 它们属于执行时协调状态，不是当前阶段优先落库的领域实体

## 幂等设计

持久化模型中必须显式体现幂等性：

- `request_id`：标识一次请求执行链路
- `idempotency_key`：用于去重
- `attempt_key`：用于区分每次 retry attempt

建议规则：

- `request_id` 是顶层追踪键
- `task.idempotency_key` 用于识别重复提交或可安全重放的任务
- `task.attempt_key` 用于区分同一逻辑任务的不同执行尝试
- `task_event` 必须保留每次 attempt 的状态变化，而不是只保留最终状态

## 快照表与事件表的边界

Persistence 模型里必须明确区分快照和事件流。

`task_event`：

- 是事件流
- 记录状态变化与执行里程碑
- 用于审计和未来可能的回放重建

`trace_snapshot` 与 `replay_snapshot`：

- 是聚合视图
- 用于调试、审计、人工复盘
- 可由当前状态和事件历史派生得到

边界原则：

- 不要用 snapshot 替代事件流
- 不要用原始事件流替代 debug / replay 快照

## 本阶段不做的事情

本阶段明确不做以下事项：

- 不写 SQLAlchemy model
- 不写 `asyncpg` 代码
- 不接 PostgreSQL
- 不接 Redis
- 不接 Milvus
- 不做 Agent Memory
- 不做 RAG
- 不做 Multi-Agent
- 不做 distributed scheduler
- 不做真正的 replay execution

## 下一步建议

建议按以下顺序推进 P5：

### P5.1 Projection Layer

- 定义 `RuntimeStateProjection`
- 定义 task / context / trace / replay DTO
- 切断 persistence 对 `LangGraphState` 的直接依赖

### P5.2 Repository Interface

- 定义 request / task / event / trace / replay 的存储接口
- 先只定义接口，不做具体实现

### P5.3 PostgreSQL Schema Draft

- 将本文档进一步转化为具体 DDL 草案
- 校验字段命名、索引与生命周期假设

### P5.4 Persistence Adapter

- 在 repository interface 后面实现 adapter
- 保持 Runtime 内部对象与持久化层隔离

### P5.5 Runtime Integration

- 在受控边界上接入 persistence write
- 避免把存储细节泄漏回 planning、routing、queue、executor 等核心逻辑
