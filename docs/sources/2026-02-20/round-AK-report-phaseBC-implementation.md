# Round-AK：报告 Phase B/C 收口实施记录（2026-02-20）

## 1. 目标
在 Round-AJ Phase A 基础上完成剩余阶段：
1. Phase B：研究汇总器 + 风险仲裁器节点化，并接入 DeepThink 跨轮上下文。
2. Phase C：报告导出模板增强 + 报告质量看板落地 + 前端可视化。

## 2. 后端改动

### 2.1 节点化仲裁
文件：`backend/app/service.py`

新增函数：
1. `_build_report_research_summarizer_node`
2. `_build_report_risk_arbiter_node`
3. `_normalize_report_analysis_nodes`

接入点：
1. `report_generate` 在 fallback 阶段生成 `analysis_nodes`。
2. `_build_report_committee_notes` 支持读取 node 摘要，优先输出节点结论。
3. `final payload` 新增 `analysis_nodes` 字段。

### 2.2 质量看板
文件：`backend/app/service.py`

新增函数：
1. `_build_report_quality_dashboard`

指标说明：
1. `overall_score`: 综合质量分（模块质量 + 覆盖率 + 证据密度 + 一致性 + quality gate）。
2. `coverage_ratio`: 模块 full 覆盖率。
3. `evidence_density`: 每模块平均证据密度。
4. `consistency_score`: 最终决策与模块质量的一致性。
5. `low_quality_modules`: 低质量模块列表。

接入点：
1. 报告主结果：`quality_dashboard`
2. 任务快照：`report_quality_dashboard`
3. payload 清洗：`_sanitize_report_payload` 会统一重算质量看板，确保结构稳定。

### 2.3 导出格式增强
文件：`backend/app/service.py`、`backend/app/http_api.py`

变更：
1. `report_export` 支持 `format=markdown/module_markdown/json_bundle`。
2. 新增 `_render_report_module_markdown`，用于模块化 Markdown 导出。
3. `json_bundle` 导出包含：
   - `final_decision`
   - `committee`
   - `report_modules`
   - `analysis_nodes`
   - `quality_dashboard`
   - `metric_snapshot`
   - `quality_gate`
4. API `POST /v1/reports/{report_id}/export` 增加 `format` 参数与格式校验（非法格式返回 400）。

### 2.4 DeepThink 上下文桥接增强
文件：`backend/app/service.py`

变更：
1. `_latest_report_context` 增加 `research_summary/risk_summary`。
2. `deep_think_run_round_stream_events`：
   - 追加最近报告节点摘要到 round 问题上下文；
   - `report_context` 事件新增这两个字段，供前端与调试链路使用。

## 3. 前端改动
文件：`frontend/app/reports/page.tsx`

新增展示与交互：
1. 导出格式切换（Markdown / 模块化 Markdown / JSON Bundle）。
2. 节点卡片区（研究汇总器、风险仲裁器）。
3. 质量看板区（status、overall、coverage_ratio、consistency_score、evidence_ref_count）。
4. 任务轮询阶段显示 `report_quality_dashboard` 摘要。

## 4. 测试改动
文件：`tests/test_service.py`、`tests/test_http_api.py`

新增断言：
1. 报告主结果包含 `analysis_nodes`、`quality_dashboard`。
2. 模块仍包含 `module_quality_score/module_degrade_code`。
3. 报告导出 `module_markdown` 与 `json_bundle` 两种格式可用。
4. 报告任务结果（partial/full）包含节点与质量看板字段。

## 5. 自测记录
1. `.\.venv\Scripts\python.exe -m pytest tests/test_service.py tests/test_http_api.py -q`
   - 结果：`82 passed`
2. `cd frontend && npm run build`
   - 结果：通过
3. `cd frontend && npx tsc --noEmit`
   - 结果：通过（在 build 之后执行）
4. `.\.venv\Scripts\python.exe scripts/full_api_selftest.py`
   - 结果：`total=129 failed=0`
5. `$env:SMOKE_RUN_HEAVY='1'; $env:SMOKE_HEAVY_CRITICAL='1'; .\.venv\Scripts\python.exe scripts/full_api_selftest.py`
   - 结果：`total=129 failed=0`

## 6. 业务价值
1. 报告从“文本输出”升级为“可仲裁、可导出、可度量”的结构化产物。
2. 用户可直接看到研究节点与风险节点，不再只有黑盒结论。
3. 报告导出可直接用于运营归档（模块化 Markdown）与系统消费（JSON Bundle）。
4. 质量看板将“证据不足/一致性下降”显式化，降低误用高置信结论的风险。
