# Phase3 Investment Journal MVP Report (2026-02-19)

## 1. Background

根据 `docs/self/会话上下文交接-2026-02-19.md` 与 `docs/self/金融分析业务系统规划-Phase3.md`，本批进入 Phase3 第一项：Investment Journal 最小闭环。

目标不是一次性做完整社区系统，而是先让“研究 -> 决策记录 -> 复盘沉淀”形成可用链路，给后续 Community 推荐和用户画像提供结构化数据基础。

## 2. Delivered Capabilities

### 2.1 Data Layer

新增表：
- `investment_journal`
- `journal_reflection`

新增索引：
- `idx_investment_journal_user_created`
- `idx_investment_journal_stock`
- `idx_journal_reflection_journal`

作用：
- 支撑用户维度和股票维度快速检索。
- 为复盘列表提供稳定排序和查询性能。

### 2.2 Service Layer

`backend/app/web/service.py` 新增：
- `journal_create`
- `journal_list`
- `journal_reflection_add`
- `journal_reflection_list`

关键行为：
- 租户隔离与所有权校验（防止跨租户读取/写入）
- `journal_type` 白名单校验（`decision/reflection/learning`）
- `tags` 归一化（去重、截断、上限控制）
- `stock_code` 统一大写，方便过滤一致性

### 2.3 Application/API Layer

`backend/app/service.py` 新增封装：
- `journal_create`
- `journal_list`
- `journal_reflection_add`
- `journal_reflection_list`

`backend/app/http_api.py` 新增接口：
- `POST /v1/journal`
- `GET /v1/journal`
- `POST /v1/journal/{journal_id}/reflections`
- `GET /v1/journal/{journal_id}/reflections`

## 3. Business Value

本批交付后，用户在系统内可以：
- 记录每次交易或持仓调整时的决策依据；
- 在结果出现后追加复盘；
- 按股票和日志类型快速回看历史决策。

这直接补齐了“有分析、无沉淀”的缺口，为后续两类能力打基础：
- AI 复盘建议（从历史决策中提取模式）
- 社区内容推荐（基于用户真实研究偏好）

## 4. Validation

### 4.1 Targeted Tests

1. `tests/test_service.py`：
- `test_journal_lifecycle`
- 结果：`4 passed, 31 deselected`（包含 phase2 关键回归）

2. `tests/test_http_api.py`：
- `test_journal_endpoints`
- 结果：`4 passed, 21 deselected`（包含 phase2 关键回归）

### 4.2 Regression Set

- 命令：
  `.\.venv\Scripts\python.exe -m pytest -q tests -k "web or api or query or docs or portfolio or alert or backtest or journal"`
- 结果：`41 passed, 55 deselected`

## 5. Next Step

按 Phase3 规划进入下一批：
1. Journal 的 AI 辅助复盘（非阻塞异步生成洞察）
2. Community MVP（帖子/评论/点赞/关注最小闭环）
3. 打通 Journal -> Community 推荐信号复用
