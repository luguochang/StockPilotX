# ROUND-R2：Report 接入多角色裁决内核 + 报告链路自测

## 1. 本轮目标
1. 报告模块复用统一多角色仲裁内核，避免 Predict/Report 结论漂移。
2. 报告结果透出 `role_opinions / judge_summary / conflict_sources / consensus_signal`。
3. 增加报告链路一键自测接口，覆盖同步生成与异步任务两条路径。

## 2. 关键改动
- `backend/app/service.py`
  - `report_generate(...)` 接入 `_predict_run_multi_role_debate(...)`。
  - 将多角色裁决结果回写到报告主结论：
    - `final_decision.signal`
    - `final_decision.confidence`（与质量门禁分数做上限约束）
    - `final_decision.invalidation_conditions`（补充冲突源）
    - `final_decision.execution_plan`（补充反方观点复核）
  - 新增报告级输出字段：
    - `multi_role_enabled`
    - `multi_role_trace_id`
    - `multi_role_decision`
    - `role_opinions`
    - `judge_summary`
    - `conflict_sources`
    - `consensus_signal`
    - `consensus_confidence`
  - 在 `report_sections` 增加 `multi_role_arbitration` 模块。
  - `_sanitize_report_payload(...)` 增加多角色字段规范化，保证持久化/导出兼容。
  - 新增 `report_self_test(...)`：同时检查同步报告与异步任务链路。
  - `_predict_run_multi_role_debate(...)` 兼容 `quality_gate.status`（Report）与 `quality_gate.overall_status`（Predict）。

- `backend/app/http_api.py`
  - 新增 `GET /v1/report/self-test`。

- `tests/test_service.py`
  - 扩展报告测试断言，覆盖多角色字段。
  - 新增 `test_report_self_test`。

- `tests/test_http_api.py`
  - 扩展 `/v1/report/generate` 响应断言，覆盖多角色字段。
  - 新增 `test_report_self_test_endpoint`。

## 3. 设计取舍
- 继续沿用规则仲裁（rule fallback）作为默认路径，避免报告时延被外部 LLM 抖动放大。
- 角色仲裁只改“结论层”与“解释层”，不破坏原有 report modules 结构，保持前端兼容。
- 保留顶层兼容字段，避免前端一次性重构。

## 4. 接口自测（2026-02-20）
执行命令：
- `python -m py_compile backend/app/service.py backend/app/http_api.py tests/test_service.py tests/test_http_api.py`
- `.venv/Scripts/python.exe -m pytest -q tests/test_service.py -k "report_generate_and_get or report_self_test or report_task_lifecycle"`
- `.venv/Scripts/python.exe -m pytest -q tests/test_http_api.py -k "report_generate_and_get or report_task_endpoints or report_self_test_endpoint"`
- `TestClient` 冒烟：`GET /v1/report/self-test` + `POST /v1/report/generate` + `GET /v1/multi-role/traces/{trace_id}`

关键结果摘要：
- `GET /v1/report/self-test`：`ok=true`，`sync.multi_role_enabled=true`，异步链路到达 `partial_ready`。
- `POST /v1/report/generate`：返回 `role_opinions.count=8`、`consensus_signal=reduce`、`consensus_confidence=0.6768`。
- `GET /v1/multi-role/traces/{trace_id}`：返回事件 `predict_multi_role_done`，包含 `consensus_signal/conflict_sources`。

## 5. 风险与后续
- 当前 Report/ Predict 多角色仲裁仍是规则优先，下一轮可在可控场景灰度启用 LLM 仲裁。
- 下一轮进入 R3：DeepThink 预裁决接入相同多角色契约，保持三模块统一口径。
