# Round-K 实现记录：DeepThink 归档分页、时间过滤与导出

日期：2026-02-15  
对应轮次：Round-K  
关联路径：`backend/app/http_api.py`、`backend/app/service.py`、`frontend/app/deep-think/page.tsx`

## 1. 本轮目标
- 将 DeepThink 存档从“固定窗口加载”升级为“可翻页回放”。
- 增加时间窗口过滤能力（`created_from` / `created_to`）。
- 增加可导出能力（`jsonl` / `csv`），满足审计和离线分析场景。

## 2. 关键设计
- **同一路由扩展查询契约**：继续使用 `/v1/deep-think/sessions/{session_id}/events`，新增 `cursor` 与时间参数，降低改造成本。
- **分页元数据显式返回**：接口返回 `has_more/next_cursor/cursor/limit`，前端可无状态地继续翻页。
- **导出与查询统一过滤语义**：`/events/export` 复用 round/event/time/limit 过滤，避免“列表看到的”和“导出的”不一致。

## 3. 后端实现
- `backend/app/web/service.py`
  - 新增 `deep_think_list_events_page(...)`，支持分页与时间过滤。
  - 存档事件输出新增 `event_id`，供 cursor 继续查询使用。
- `backend/app/service.py`
  - `deep_think_list_events(...)` 扩展为分页元数据返回。
  - 新增 `deep_think_export_events(...)`，支持 JSONL/CSV 两种格式。
- `backend/app/http_api.py`
  - `/events` 增加 `cursor`、`created_from`、`created_to`。
  - 新增 `/events/export`，通过附件响应返回导出文件。

## 4. 前端实现
- `frontend/app/deep-think/page.tsx`
  - 新增归档控制状态：cursor、has_more、next_cursor、created_from、created_to、exporting。
  - 新增控制项：
    - 时间过滤输入框（from/to）
    - 下一页存档按钮
    - 导出 JSONL / CSV 按钮
  - 回放面板展示当前筛选上下文（round/event/limit/cursor/time）。

## 5. 测试增强
- `tests/test_service.py`
  - 新增 cursor 过滤、时间过滤、JSONL/CSV 导出断言。
- `tests/test_http_api.py`
  - 新增 `/events` 分页元数据契约断言。
  - 新增 `/events/export` 文件类型与内容断言。

## 6. 自测结果
- `.\.venv\Scripts\python -m pytest -q tests/test_service.py tests/test_http_api.py` -> `27 passed`
- `.\.venv\Scripts\python -m pytest -q` -> `63 passed`
- `cd frontend && npm run build` -> passed
- `cd frontend && npx tsc --noEmit` -> passed

## 7. 价值与后续
- 价值：
  - DeepThink 存档能力从“可看”升级为“可分页、可筛选、可导出”。
  - 形成与审计流程更贴近的数据交付链路。
- 后续建议：
  - 增加时间参数格式校验与错误提示规范化。
  - 提供大数据量异步导出任务（避免同步导出超时）。
  - 追加“上一页/定位页首”等回放导航能力。
