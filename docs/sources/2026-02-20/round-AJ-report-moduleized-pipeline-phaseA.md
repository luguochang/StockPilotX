# Round-AJ：报告模块化迁移（Phase A）实施记录（2026-02-20）

## 1. 背景与目标
本轮落实了 TradingAgents-CN 对标方案中的 Phase A：
1. 报告输出从“单段 Markdown 主文”升级为“模块化报告包”。
2. 报告接口补充业务可解释字段：`report_modules`、`final_decision`、`committee`、`metric_snapshot`。
3. 报告任务 `partial_ready` 阶段返回最小可用模块，避免前端只能等待空态。
4. 前端报告中心增加模块化展示区，支持直接查看决策、委员会纪要与关键指标。

## 2. 代码变更
### 2.1 后端 `backend/app/service.py`
新增/强化能力：
1. 新增信号与数值标准化工具：
- `_normalize_report_signal`
- `_safe_float`
2. 新增报告业务结构化构建函数：
- `_build_report_metric_snapshot`
- `_build_fallback_final_decision`
- `_build_report_committee_notes`
- `_build_fallback_report_modules`
- `_normalize_report_modules`
3. 升级 `report_generate`：
- 先构建 deterministic fallback 模块。
- 外部 LLM 可用时按“模块化 JSON 合同”覆盖/增强。
- 回退兼容旧 JSON 合同字段（`executive_summary/core_logic/risk_matrix/...`）。
- 输出新增字段并写入缓存：
  - `report_modules`
  - `committee`
  - `final_decision`
  - `metric_snapshot`
4. 升级 `_sanitize_report_payload`：
- 增加对 `report_modules/committee/final_decision` 的清洗与标准化。
5. 升级 `_build_report_task_partial_result`：
- partial 结果也返回最小模块集与决策骨架，增强前端可读性。

### 2.2 前端 `frontend/app/reports/page.tsx`
1. 新增模块化类型定义：
- `ReportModule`
- `FinalDecision`
- `CommitteeNotes`
2. 报告结果解析增强：
- `applyReportResult` 增加模块、决策、委员会、指标快照提取。
3. 页面展示增强：
- 新增“最终决策与委员会”卡片。
- 新增“模块化报告”标签页展示。
- 新增“指标快照”卡片。

### 2.3 测试更新
1. `tests/test_service.py`
- `test_report_generate_and_get` 增加新字段断言。
- `test_report_task_lifecycle` 增加 partial/full 结果新字段断言。
2. `tests/test_http_api.py`
- `test_report_generate_and_get` 增加新字段断言。
- `test_report_task_endpoints` 增加任务结果新字段断言。

## 3. 接口输出新增字段说明
以下字段已在 `POST /v1/report/generate` 与 `GET /v1/report/{report_id}` 输出：
1. `report_modules`: 模块化报告数组（模块 ID、内容、置信度、覆盖状态、降级原因）。
2. `final_decision`: 最终决策（signal/confidence/rationale/invalidation_conditions/execution_plan）。
3. `committee`: 委员会纪要（research_note/risk_note）。
4. `metric_snapshot`: 指标快照（趋势、估值、质量、引用等关键指标）。

异步任务结果接口 `GET /v1/report/tasks/{task_id}/result`：
1. partial 结果也会返回上述核心结构（最小可用版本）。

## 4. 自测结果
执行命令：
1. `\.venv\Scripts\python.exe -m pytest tests/test_service.py -k "report_generate_and_get or report_task_lifecycle" -q`
2. `\.venv\Scripts\python.exe -m pytest tests/test_http_api.py -k "test_report_generate_and_get or test_report_task_endpoints" -q`
3. `cd frontend; npx tsc --noEmit`
4. `cd frontend; npm run build`
5. `\.venv\Scripts\python.exe scripts/full_api_selftest.py`

结果：
1. 后端服务测试：`2 passed`
2. API 测试：`2 passed`
3. 前端 TypeScript 检查：通过
4. 前端构建：通过
5. 全接口轻量 smoke：`total=129 failed=0`

## 5. 业务价值
1. 报告可读性从“原始文本堆叠”升级为“业务模块化阅读”。
2. 用户可直接看到决策信号、置信度、失效条件，减少“看了不知道如何执行”。
3. partial 阶段也有结构化反馈，减少等待焦虑和空白体验。
4. 通过 `metric_snapshot + quality_gate` 保留质量可解释性，避免“高置信度错觉”。

## 6. 下一步建议（Phase B）
1. 引入“研究汇总器 + 风险仲裁器”独立节点化执行，并写入 DeepThink 跨轮输入。
2. 增强 `report_modules` 的证据链粒度（source_id -> excerpt -> reliability -> freshness）。
3. 增加模块级质量分与可追溯审计字段（module_quality_score/module_degrade_code）。
