# ROUND-AO 任务稳定化 + 1Y补数基线（执行与自测记录）

## 背景
本轮目标是把“报告/高级分析能跑”提升到“任务过程可观测、异常可收敛、样本基线稳定”。
核心问题：
- 异步任务在 `running/partial_ready` 阶段缺乏可见心跳，用户容易误判为卡死。
- 报告与 DeepThink 在长周期分析场景下，历史样本门槛偏低，容易出现“结论偏短周期”的风险。

## 本轮 Checklist

### A. 报告任务链路稳定化
- [x] 后端任务状态新增运行时守卫：超时失败 + 心跳停滞失败。
- [x] 后端任务状态新增运行时字段：`deadline_at/heartbeat_at/stage_started_at/stage_elapsed_seconds/heartbeat_age_seconds`。
- [x] 后端 full 阶段新增 keepalive 心跳线程，持续更新进度与阶段文案。
- [x] 前端报告页展示阶段耗时、心跳延迟、任务超时阈值。
- [x] 前端在心跳延迟过大时给出告警提示。

### B. 1Y样本与自动补数可见化
- [x] `report/deepthink` 场景默认 `history_min` 升级到 `252`，`history_fetch_limit` 升级到 `520`。
- [x] 报告质量门控增加 `auto_refresh_failed` 原因及权重。
- [x] 报告数据包摘要新增 `refresh_action_count/refresh_failed_count`。
- [x] DeepThink `data_pack` 事件新增补数字段：`refresh_action_count/refresh_failed_count/time_horizon_coverage/refresh_actions`。

### C. 回归测试
- [x] 服务层测试更新并通过：`tests/test_service.py`
- [x] HTTP API 测试更新并通过：`tests/test_http_api.py`
- [x] 全接口 smoke 通过：`scripts/full_api_selftest.py`
- [x] 前端生产构建通过：`frontend npm run build`
- [x] 临时端口专项验证通过：报告任务新字段 + DeepThink data_pack 新字段

## 自测结果

1) 后端测试
```bash
.venv\Scripts\python -m pytest tests/test_service.py tests/test_http_api.py -q
```
结果：`83 passed`

2) 全接口 smoke
```bash
.venv\Scripts\python scripts/full_api_selftest.py
```
结果：`total=130, failed=0`

3) 前端生产构建
```bash
cd frontend
npm run build
```
结果：构建通过，静态页面 `19/19`。

4) 临时端口专项验证（`8012`）
- `/v1/report/tasks/{task_id}`：确认返回 `deadline_at/heartbeat_at/stage_elapsed_seconds/heartbeat_age_seconds`
- `/v2/deep-think/sessions/{session_id}/rounds/stream`：确认 `data_pack` 包含
  - `refresh_action_count`
  - `refresh_failed_count`
  - `time_horizon_coverage`

## 影响文件
- `backend/app/service.py`
- `frontend/app/reports/page.tsx`
- `tests/test_service.py`
- `tests/test_http_api.py`
- `docs/sources/2026-02-20/round-AO-task-stability-and-1y-autorefresh-checklist.md`

## 提交记录
- Commit: 见本轮提交（`git log --oneline`）
