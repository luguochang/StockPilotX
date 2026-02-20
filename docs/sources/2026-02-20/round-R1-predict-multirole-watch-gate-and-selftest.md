# ROUND-R1：Predict 多角色裁决 + Watch 门禁 + 自测接口

## 1. 本轮目标
1. Predict 输出多角色裁决结果，减少“只有量化结论”的可解释性缺口。
2. 研报不足从硬降级改为 watch，避免可用结果被阻断。
3. 提供一键自测接口，便于问题定位。

## 2. 关键改动
- `backend/app/service.py`
  - `predict_run` 新增多角色裁决输出：
    - `multi_role_enabled`
    - `multi_role_trace_id`
    - `multi_role_debate`
    - `role_opinions`
    - `judge_summary`
    - `conflict_sources`
    - `consensus_signal`
    - `consensus_confidence`
  - 质量门禁改造：引入 `severity`（pass/watch/degraded）并按维度聚合。
  - `input_pack:research_insufficient` 设为 `watch`，避免硬阻断。
  - 新增 `predict_self_test()` 与 `multi_role_trace_events()`。

- `backend/app/http_api.py`
  - 新增 `GET /v1/predict/self-test`
  - 新增 `GET /v1/multi-role/traces/{trace_id}`

## 3. 设计取舍
- 复用现有 `_build_rule_based_debate_opinions` 与 `_build_llm_debate_opinions`，降低改造风险。
- 对大池子预测设置多角色辩论上限（最多 8 个标的），控制时延。
- 保留 top-level 兼容字段，避免前端一次性改造压力过大。

## 4. 技术点
1. 质量门禁从“有 reason 即 degraded”改为“按 reason severity 聚合”。
2. 多角色裁决通过 `_arbitrate_opinions` 统一冲突计算，保持 DeepThink 逻辑一致性。
3. 自测接口串联：数据刷新 -> predict_run -> predict_explain -> trace 聚合。

## 5. 自测计划
- `POST /v1/predict/run`
- `POST /v1/predict/explain`
- `GET /v1/predict/evals/latest`
- `GET /v1/predict/self-test`
- `GET /v1/multi-role/traces/{trace_id}`

## 6. 风险
- 大池子时延上升：通过 debate cap 限制。
- LLM provider 不稳定：保留 rule fallback。

## 7. 下一步
- R1 自测完成后更新 checklist 勾选并提交 commit。
- 进入 R2，将 Report 接入同一多角色决策内核。

## 8. 实际自测记录（2026-02-20）
执行命令：
- `python -m py_compile backend/app/service.py backend/app/http_api.py tests/test_service.py tests/test_http_api.py`
- `.venv/Scripts/python.exe -m pytest -q tests/test_service.py -k "predict or quality_gate"`
- `.venv/Scripts/python.exe -m pytest -q tests/test_http_api.py -k "predict"`

关键结果摘要：
- `predict_run` 返回 `quality_gate.overall_status=watch`（`research_insufficient` 不再硬降级）。
- `predict_run` 返回 `role_opinions`（8 个角色）与 `judge_summary`。
- `GET /v1/predict/self-test` 返回 `ok=true`，且包含 `multi_role_trace_id`。
- `GET /v1/multi-role/traces/{trace_id}` 返回事件 `predict_multi_role_done`，可回放仲裁摘要。
