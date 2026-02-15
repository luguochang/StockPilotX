# Round-I 实现记录：DeepThink 跨轮差分、冲突下钻与事件存档

日期：2026-02-15  
对应轮次：Round-I  
关联路径：`frontend/app/deep-think/page.tsx`、`backend/app/service.py`

## 1. 本轮目标
- 完成 Round-H 留下的三项增强：
  - 跨轮次观点差分（opinion diff）
  - 冲突源下钻（agent + evidence 视角）
  - 会话事件存档与回放（非内存易失）
- 保持既有 DeepThink/A2A 调用路径不变，避免前后端契约漂移。

## 2. 关键设计
- **事件存档独立接口**：新增 `GET /v1/deep-think/sessions/{session_id}/events`，可按 `round_id` 与 `limit` 查询。
- **事件持久化落库**：新增 `deep_think_event` 表，按 `round_no + event_seq` 保序，支撑可重复回放。
- **前端增强优先复用字段**：观点差分与冲突下钻全部基于既有 `rounds/opinions/conflict_sources` 计算，不扩展后端 round schema。
- **兼容旧数据**：若历史轮次未存档，流接口会自动由 round snapshot 重新生成事件并补写归档。

## 3. 后端落点
- `backend/app/http_api.py`
  - 新增 `/v1/deep-think/sessions/{session_id}/events`。
- `backend/app/service.py`
  - 新增 `_build_deep_think_round_events`（统一事件生成）。
  - 新增 `deep_think_list_events`（服务层事件读取）。
  - `deep_think_run_round` 增加轮次完成后的事件归档写入。
  - `deep_think_stream_events` 优先读取归档，缺失时自动补档。
- `backend/app/web/store.py`
  - 新增 `deep_think_event` 表与索引。
- `backend/app/web/service.py`
  - 新增 `deep_think_replace_round_events`、`deep_think_list_events`。

## 4. 前端落点
- `frontend/app/deep-think/page.tsx`
  - 新增 `DeepThinkEventArchiveSnapshot` 类型。
  - 新增归档操作：`loadDeepThinkEventArchive`，并在会话创建/刷新/流回放后联动加载。
  - 新增会话存档状态：`deepArchiveLoading`、`deepArchiveCount`。
  - 新增“跨轮次观点差分”卡片：展示 signal 变化与 `delta_confidence`。
  - 新增“冲突源下钻”卡片：按 `consensus_signal + conflict_sources` 筛选冲突候选观点，展示 `evidence_ids`。

## 5. 测试补强
- `tests/test_service.py`
  - 增加 `deep_think_list_events` 快照断言（有 `round_started` / `done`）。
- `tests/test_http_api.py`
  - 增加 `/v1/deep-think/sessions/{session_id}/events` 接口契约断言。

## 6. 工程意义
- DeepThink 从“可看实时流”升级为“可审计回放”：
  - 运维与复盘不再依赖页面常驻内存。
- DeepAgent 冲突分析从单轮快照升级为“跨轮演化观察”：
  - 更容易识别某个 agent 是策略切换还是置信度漂移。
- A2A 触发路径与 DeepThink 治理面板形成闭环：
  - 可以从任务触发、到事件序列、到冲突证据一条链路复盘。

## 7. 本轮自测
- `.\.venv\Scripts\python -m pytest -q tests/test_service.py tests/test_http_api.py` -> `27 passed`
- `.\.venv\Scripts\python -m pytest -q` -> `63 passed`
- `cd frontend && npm run build` -> passed
- `cd frontend && npx tsc --noEmit` -> passed

## 8. 下一轮建议
- 支持按事件类型筛选（`arbitration_final/replan_triggered/...`）和时间窗查询。
- 为冲突下钻增加“证据详情预览”与引用链接跳转。
- 增加归档保留策略与清理机制（按会话数/天数/大小）。
