# ROUND-AM 报告链路优化（Full-First + 1Y 证据 + TradingAgents 字段）

## 1. 背景与根因

问题现象（用户侧）
- `/v1/report/tasks` 任务在 `partial_ready` 阶段停留较久，前端容易“先展示短内容 + 一直转圈”，用户误判为系统异常。
- 报告结论常出现“样本不足/仅近3个月”的提示，1Y 维度证据不够完整。
- 报告决策字段缺少 TradingAgents 常见结构化输出（target price / risk score / reward-risk / position sizing）。
- 任务接口对“当前结果是否可展示”没有显式信号，前端只能依赖 `result_level` 猜测。

根因归纳
- 异步任务是“先 partial 后 full”的两阶段生成，前端若默认渲染 partial，会造成“结果太短”的体验。
- 输入包历史上偏 30D/90D，1Y 日线、季度财报摘要、事件时间轴等信息在报告上下文中不充分。
- 报告 schema 与 TradingAgents 风格字段未对齐，前端缺少可执行指标呈现。

## 2. 本轮目标

- [x] 后端补齐 1Y 证据输入包，并显式注入 LLM context。
- [x] 后端补齐 TradingAgents 风格最终决策字段与四类角色模块（bull/bear/risky/safe）。
- [x] 后端质量门控从二值降级升级为 `pass/watch/degraded`。
- [x] 异步任务接口增加 `display_ready/partial_reason`，支持前端 Full-First 交互。
- [x] 前端报告页默认“完整报告优先展示”，临时结果改为手动开关。
- [x] 前端卡片显示新增决策字段与 1Y 数据包关键指标。

## 3. 实施清单（执行记录）

### 3.1 Backend (`backend/app/service.py`)
- [x] 新增 1Y 证据方法：
  - `_history_1y_summary`
  - `_history_monthly_summary`
  - `_quarterly_financial_summary`
  - `_event_timeline_1y`
  - `_build_evidence_uncertainty_notes`
- [x] `_build_llm_input_pack` 新增字段：
  - `history_daily_252`
  - `history_1y_summary`
  - `history_monthly_summary_12`
  - `quarterly_fundamentals_summary`
  - `event_timeline_1y`
  - `time_horizon_coverage`
  - `evidence_uncertainty`
- [x] `_build_fallback_final_decision` 新增字段：
  - `target_price`
  - `risk_score`
  - `reward_risk_ratio`
  - `position_sizing_hint`
- [x] `_build_fallback_report_modules` 新增模块：
  - `bull_case`
  - `bear_case`
  - `risky_case`
  - `safe_case`
- [x] `report_generate` 质量门控升级：
  - 原：按原因数量线性扣分
  - 现：按原因加权扣分，状态为 `pass/watch/degraded`
- [x] 报告 sections 新增：
  - `evidence_summary`
  - `uncertainty_notes`
  - `time_horizon_coverage`
- [x] LLM prompt contract 与解析逻辑扩展至新 schema 字段。
- [x] 异步任务快照/结果接口新增：
  - `display_ready`
  - `partial_reason`
- [x] `degrade` payload 新增 `severity`，并区分 `quality_watch` / `quality_degraded`。

### 3.2 Frontend (`frontend/app/reports/page.tsx`)
- [x] 类型定义扩展：`ReportTask`、`TaskResult`、`FinalDecision`。
- [x] 新增 Full-First 开关：`allowPartialPreview`（默认 false）。
- [x] 轮询闭包修正：使用 `allowPartialPreviewRef` 避免 stale state。
- [x] 任务进度区新增“临时结果预览”开关与提示文案。
- [x] 当 `result_level=partial` 且开关关闭时，默认隐藏内容并提示“正在生成完整报告”。
- [x] 决策卡片新增展示：
  - 目标价区间
  - 风险分
  - 盈亏比
  - 仓位建议
  - 执行步骤
- [x] 数据包摘要新增 1Y 指标：
  - 30/90/252 天样本
  - 1Y 覆盖状态
  - 财报/事件/不确定性计数

## 4. 自测记录

### 4.1 Python 单测（核心后端）
命令
```bash
.venv\Scripts\python -m pytest tests/test_service.py tests/test_http_api.py -q
```
结果
- 82 passed in 188.97s

说明
- Windows 控制台直接跑 pytest 时出现 stdout flush 异常，改为重定向日志后通过：`tmp_pytest_roundAM.log`。

### 4.2 前端类型检查
命令
```bash
cd frontend
npx tsc --noEmit
```
结果
- 通过（无错误输出）

### 4.3 全量 API 烟测
命令
```bash
.venv\Scripts\python scripts/full_api_selftest.py
```
结果
- total=130, failed=0
- status_hist={"200":81, "299":29, "400":4, "401":1, "404":15}

### 4.4 报告链路专项自测（临时端口加载最新代码）
说明
- 使用 `uvicorn backend.app.http_api:create_app --factory --port 8011` 启动临时服务后验证。

结果摘要
- `/v1/report/generate` 返回中已包含：
  - `report_data_pack_summary.history_252d_count`
  - `history_1y_has_full`
  - `quarterly_fundamentals_count`
  - `event_timeline_count`
  - `uncertainty_note_count`
  - `time_horizon_coverage`
  - `final_decision.target_price/risk_score/reward_risk_ratio/position_sizing_hint`
  - `report_modules` 中 `bull_case/bear_case/risky_case/safe_case`
- `/v1/report/tasks/*` 链路：
  - `partial_ready` 阶段 `display_ready=false, partial_reason=warming_up`
  - 完成后 `completed/full` 且 `display_ready=true`

## 5. 发布注意事项

- 本次验证确认：若 8000 端口为旧进程，接口不会反映本轮新字段。
- 部署/本地联调前请重启后端服务，确保加载最新 `backend/app/service.py`。

## 6. 提交记录

- Commit: 见本轮提交记录（`git log --oneline`）
- 影响文件：
  - `backend/app/service.py`
  - `frontend/app/reports/page.tsx`
  - `docs/sources/2026-02-20/round-AM-report-task-full-first-1y-evidence-checklist.md`
