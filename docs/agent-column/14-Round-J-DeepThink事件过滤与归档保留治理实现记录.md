# Round-J 实现记录：DeepThink 事件过滤与归档保留治理

日期：2026-02-15  
对应轮次：Round-J  
关联路径：`backend/app/service.py`、`frontend/app/deep-think/page.tsx`

## 1. 本轮目标
- 让 DeepThink 事件回放从“全量读取”升级为“可筛选查询”。
- 增加归档保留治理，避免单会话事件无限增长。
- 保持 Round-I 的事件存档主路径不变，仅做兼容增强。

## 2. 关键设计
- **查询扩展不换路由**：继续使用 `/v1/deep-think/sessions/{session_id}/events`，新增 `event_name` 过滤参数。
- **保留策略会话内裁剪**：每次轮次写入事件后，对该会话按“最新N条”保留；默认 `1200`。
- **请求级可配置上限**：`deep_think_run_round` 支持 `archive_max_events`，用于演练与压测。
- **前端显式筛选加载**：在控制台新增 `round/event/limit` 筛选控件，用户主动触发归档加载。

## 3. 后端落点
- `backend/app/http_api.py`
  - `/events` 增加 `event_name` 查询参数。
- `backend/app/service.py`
  - `deep_think_list_events` 透传 `event_name` 过滤。
  - `deep_think_run_round` 增加 `archive_max_events` 并传入写入逻辑。
- `backend/app/web/service.py`
  - `deep_think_list_events` 改为动态条件查询（`session_id/round_id/event_name`）。
  - `deep_think_replace_round_events` 写入后调用 `deep_think_trim_events`。
  - 新增 `deep_think_trim_events` 进行会话级归档裁剪。
- `backend/app/web/store.py`
  - 增加索引 `idx_deep_think_event_name`，提升过滤查询性能。

## 4. 前端落点
- `frontend/app/deep-think/page.tsx`
  - 新增筛选状态：`deepArchiveRoundId`、`deepArchiveEventName`、`deepArchiveLimit`。
  - 新增筛选控件：轮次选择、事件类型选择、limit 输入。
  - `loadDeepThinkEventArchive` 升级为 options 形式，支持按筛选参数请求归档。
  - 回放区显示当前筛选上下文，并通过 `deepReplayRows` 展示过滤后的事件。

## 5. 测试补强
- `tests/test_http_api.py`
  - 增加 `/events?event_name=done` 契约断言（过滤结果全部为 `done`）。
- `tests/test_service.py`
  - 增加服务层 `event_name` 过滤断言。
  - 增加 `archive_max_events` 裁剪断言（事件数不超过上限）。

## 6. 工程意义
- 归档查询从“调试能力”升级为“治理能力”：
  - 运维可按事件类型快速定位 DeepThink 异常路径。
- 存储治理从“无限累积”升级为“可控保留”：
  - 为后续多租户与审计导出提供基础约束。
- 与 Round-I 形成闭环：
  - Round-I 解决“可存档”，Round-J 解决“可筛可控”。

## 7. 本轮自测
- `.\.venv\Scripts\python -m pytest -q tests/test_service.py tests/test_http_api.py` -> `27 passed`
- `.\.venv\Scripts\python -m pytest -q` -> `63 passed`
- `cd frontend && npm run build` -> passed
- `cd frontend && npx tsc --noEmit` -> passed

## 8. 下一轮建议
- 增加时间窗口过滤（`from/to`）与按 event_seq 游标分页。
- 增加归档导出（JSONL/CSV）以接入外部审计系统。
- 增加环境级保留策略（dev短保留、prod长保留）与监控告警。
