# Production Hardening Phase 2

## 目标

在不重写现有 LangGraph Runtime 架构的前提下，为现有系统补齐以下生产级增强能力：

- Checkpoint / Snapshot
- Resume from Checkpoint
- Task Idempotency
- Queue Deadlock Protection
- Context Concurrency Safety

## 本次增强范围

### 1. Checkpoint / Snapshot

新增本地文件型检查点能力：

- `app/context/checkpoints.py`
- `app/schemas/checkpoint.py`

检查点保存内容包括：

- `LangGraphState` 图状态快照
- `ContextStore` 完整上下文
- `task_results`
- `tool_call_chain`
- `execution_history`
- `execution_results`
- `metadata`

实现特点：

- 全部 JSON 可序列化
- 默认采用本地文件存储
- 支持按 `checkpoint_id` 加载
- 支持按 `request_id/session_id` 加载最新检查点
- 支持从失败态检查点恢复，并重置为可继续执行的 phase

### 2. 幂等保护

任务模型新增：

- `idempotency_key`
- `irreversible`

上下文新增：

- `idempotency_records`

执行器增强：

- 成功/失败/重试均写入幂等记录
- 恢复执行时优先检查幂等记录
- 若命中成功幂等记录，则直接恢复结果，不重复调用工具

适用效果：

- 避免不可逆任务重复执行
- 支持恢复后去重重放
- 工具调用结果可跨重启恢复

### 3. 队列死锁保护

队列增强了以下检测：

- 缺失依赖检测
- 循环依赖检测
- 孤儿任务检测
- 永久阻塞 DAG 检测

一旦发现永久阻塞：

- 立即 fail-fast
- 输出显式结构化错误

### 4. Context 并发安全

`ContextStore` 增加了原子写入能力：

- `set_task_result`
- `set_shared_value`
- `set_shared_mapping_value`
- `append_shared_list`
- `increment_shared_counter`

并新增冲突检测：

- 禁止共享键被静默覆盖
- 发生冲突时输出结构化日志
- 冲突会向上冒泡为真实执行失败

## 关键恢复路径

### 恢复流程

1. 从本地文件加载检查点
2. 反序列化恢复 `LangGraphState`
3. 保留已有 `ContextStore`
4. 跳过已恢复的 `Supervisor / Planner / Parser`
5. 使用 `context.tasks` 重建 Queue
6. Executor 继续执行剩余任务
7. 若命中幂等记录，则直接恢复结果而不重跑工具

### 为什么恢复时使用 `context.tasks`

`planned_tasks` 保留的是原始计划状态，不能代表真实执行进度。

恢复时必须使用：

- `context.tasks`

因为它保存了任务真实状态：

- `PENDING`
- `RUNNING`
- `SUCCESS`
- `FAILED`
- `RETRY`
- `TIMEOUT`
- `CANCELLED`

## 新增测试

新增测试文件：

- `tests/test_checkpoint_recovery.py`
- `tests/test_idempotency_recovery.py`
- `tests/test_deadlock_protection.py`
- `tests/test_context_atomicity.py`

覆盖能力：

- 检查点可序列化
- 中断后从最新检查点恢复
- 不可逆任务恢复时不重复执行
- 缺失依赖 fail-fast
- 循环依赖 fail-fast
- 永久阻塞 DAG fail-fast
- 并发共享键冲突检测

## 回归结果

在 `rag` 环境下执行：

- `python -m compileall app tests`
- `python -m pytest -q`

结果：

- `32 passed, 1 warning`

## 当前边界

本阶段仍保持以下约束：

- 使用本地文件而非外部数据库
- 不改写 Supervisor / Planner / Executor 主体架构
- 不引入额外基础设施依赖
- 恢复能力以“继续当前 Runtime”而非重建新工作流为主

## 下一步建议

后续可继续演进：

- 检查点保留策略与清理策略
- 多版本 schema 兼容恢复
- 更细粒度的任务级 replay policy
- 外部持久化存储
- 生产级审计与告警联动
