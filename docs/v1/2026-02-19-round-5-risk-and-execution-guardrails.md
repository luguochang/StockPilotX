# ROUND-5 技术记录：风控与执行建议结构化落地

## 1. 本轮目标

在 Intel Card 基础上增加“可执行约束”，把结论变成可落地动作：

1. 仓位节奏建议（执行计划）  
2. 风控阈值（波动/回撤/证据数/时效）  
3. 降级策略可视化（normal/watch/degraded）

---

## 2. 后端改动

文件：`backend/app/service.py`

在 `analysis_intel_card` 中新增字段：

1. `execution_plan`
- `entry_mode`
- `cadence_hint`
- `max_single_step_pct`
- `max_position_cap`
- `stop_loss_hint_pct`
- `recheck_interval_hours`

2. `risk_thresholds`
- `volatility_20_max`
- `max_drawdown_60_max`
- `min_evidence_count`
- `max_data_staleness_minutes`

3. `degrade_status`
- `level`（normal/watch/degraded）
- `reasons`（例如 stale/insufficient_evidence/threshold breach）

新增策略逻辑：

1. 数据时效、证据数量、波动/回撤超阈会触发 `degrade_reasons`。  
2. 降级时自动下调置信度，降低“弱证据高动作”的风险。  
3. 执行计划根据 `signal + risk_profile + risk_level` 自动生成节奏建议。

---

## 3. 前端改动

文件：`frontend/app/deep-think/page.tsx`

业务卡片新增展示区：

1. 降级状态与原因标签  
2. 执行节奏建议卡（entry/cadence/单步上限/止损提示/复核间隔）  
3. 风险阈值卡（波动/回撤/最小证据/时效）

并扩展 `IntelCardSnapshot` 类型，保证前后端字段一致。

---

## 4. 测试改动

文件：
- `tests/test_service.py`
- `tests/test_http_api.py`

新增断言：
- `execution_plan`
- `risk_thresholds`
- `degrade_status`

---

## 5. 自测记录

执行时间：2026-02-19

1. Service 合约回归  
```powershell
.\.venv\Scripts\python.exe -m pytest -q tests/test_service.py -k "analysis_intel_card_contract"
```
结果：`1 passed, 39 deselected`

2. HTTP API 回归  
```powershell
.\.venv\Scripts\python.exe -m pytest -q tests/test_http_api.py -k "analysis_intel_card"
```
结果：`1 passed, 34 deselected`

3. 前端类型检查  
```powershell
cd frontend
npx tsc --noEmit
```
结果：通过（exit code 0）

---

## 6. 风险与后续

1. 当前执行建议仍是规则模板，后续可结合用户历史执行反馈自适应调整。  
2. 降级状态已可见，但还缺“降级后自动提醒与再探测机制”。  
3. 下一轮建议推进 ROUND-6（采纳反馈与偏差复盘闭环）。
