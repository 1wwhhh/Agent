# MySQL Business Tools

MySQL is used as a structured business analysis backend, not as knowledge-base search. The Agent should use these tools for weekly reports, plans, self-evaluations, completion rates, and assessment statistics. General document QA should still use `rag_search_tool` and `/search`.

## Tools

- `query_weekly_reports`: query structured weekly report items.
- `compare_weekly_plan_done`: query last week's `next_week_plan` and this week's `this_week_work` for completion analysis.
- `monthly_department_analysis`: query monthly department plans, weekly items, and self-evaluation items.

The tools never accept raw SQL from the LLM. Queries are fixed templates with parameterized values.

## Required Environment

```env
MYSQL_HOST=127.0.0.1
MYSQL_PORT=3306
MYSQL_USER=agent_readonly
MYSQL_PASSWORD=change_me
MYSQL_DATABASE=agent_business
MYSQL_CHARSET=utf8mb4
```

Optional controls:

```env
MYSQL_BUSINESS_MAX_LIMIT=2000
MYSQL_BUSINESS_MAX_CONCURRENCY=8
MYSQL_BUSINESS_MAX_CANDIDATE_MATCHES=200
MYSQL_CONNECT_TIMEOUT=5
MYSQL_READ_TIMEOUT=15
MYSQL_WRITE_TIMEOUT=15
```

## Default Table Contract

Default item table: `weekly_report_items`
Default report table: `weekly_reports`

Required columns on `weekly_report_items`:

```sql
user_name        varchar(...)
department       varchar(...)
report_date      date
item_type        varchar(...)
item_text        text
evidence_text    text
source_doc_id    varchar(...)
source_chunk_id  varchar(...)
sort_order       int
```

Required columns on `weekly_reports`:

```sql
user_name        varchar(...)
report_date      date
risk_and_help    text
source_doc_id    varchar(...)
```

Supported `item_type` values:

- `this_week_work`
- `next_week_plan`
- `self_eval`
- `department_plan`
- `dept_plan`
- `department_self_eval`

If your table or column names differ, configure them with environment variables:

```env
MYSQL_WEEKLY_ITEMS_TABLE=weekly_report_items
MYSQL_WEEKLY_REPORTS_TABLE=weekly_reports
MYSQL_WEEKLY_USER_COLUMN=user_name
MYSQL_WEEKLY_DEPARTMENT_COLUMN=department
MYSQL_WEEKLY_REPORT_DATE_COLUMN=report_date
MYSQL_WEEKLY_ITEM_TYPE_COLUMN=item_type
MYSQL_WEEKLY_ITEM_TEXT_COLUMN=item_text
MYSQL_WEEKLY_RISK_AND_HELP_COLUMN=risk_and_help
MYSQL_WEEKLY_EVIDENCE_TEXT_COLUMN=evidence_text
MYSQL_WEEKLY_SOURCE_DOC_ID_COLUMN=source_doc_id
MYSQL_WEEKLY_SOURCE_CHUNK_ID_COLUMN=source_chunk_id
MYSQL_WEEKLY_SORT_ORDER_COLUMN=sort_order
```

## Weekly Blocker Rule

对于周报卡点、风险、求助类问题：

1. 使用 `query_weekly_reports` 查询目标周，并设置 `item_type=null`、`record_level=reports`、`include_evidence_text=false`。
2. 使用 `classify_weekly_blockers` 对目标周员工自填卡点做语义分类。不要用“字段非空”或“包含无卡点”这类规则直接决定是否为卡点；“暂无卡点，但需要软件协助联调”会被分类为混合文本，并抽取具体卡点。
3. 使用 `compare_weekly_plan_done` 依赖查询和分类结果，设置 `weekly_blocker_classification_output_key`、`trace_weeks=2`、`include_historical_blockers=true`。工具只追溯 `needs_trace=true` 的人员，并拆成两个窗口：上一周计划追目标周完成，上上周计划追上一周和目标周完成。
4. `compare_weekly_plan_done` 会同时收集被追溯人员前两周的历史员工自填卡点候选，以及这些候选之后的完成项。
5. 使用 `judge_weekly_blocker_trace` 判断历史卡点是否已解决、仍持续、证据不足或不是卡点，并生成最终压缩上下文。
6. 最终 `text_generate_tool.input.context` 使用 `{{weekly_blocker_trace_judgement.weekly_blocker_context_text}}`。不要将完整 `{{weekly_reports}}`、完整 `{{weekly_plan_comparison}}` 或完整分类结果注入最终生成任务，避免大上下文导致模型超时。

标识符会经过校验，必须是简单 SQL identifier。用户输入值始终使用参数化查询。
