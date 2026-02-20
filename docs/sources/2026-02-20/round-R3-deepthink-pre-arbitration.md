# ROUND-R3：DeepThink 接入多角色预裁决

## 1. 目标
1. 让 DeepThink 轮次执行在正式仲裁前先跑统一多角色预裁决（与 Predict/Report 同一内核）。
2. 通过事件流明确暴露预裁决结果，便于前端与运维诊断。
3. 避免 DeepThink 与 Predict/Report 结论口径漂移。

## 2. 关键改动
- `backend/app/service.py`
  - 在 `deep_think_run_round_stream_events(...)` 中引入预裁决调用：
    - `_predict_run_multi_role_debate(...)`
  - 新增预裁决输出对象：`multi_role_pre`，字段包括：
    - `enabled`
    - `trace_id`
    - `debate_mode`
    - `role_count`
    - `consensus_signal`
    - `consensus_confidence`
    - `disagreement_score`
    - `conflict_sources`
    - `counter_view`
    - `judge_summary`
  - 新增事件：`pre_arbitration`。
  - 预裁决输出并入 DeepThink 主仲裁流程：
    - `pre_arbitration` 角色观点优先并入最终 `opinions`
    - `pre_arbitration.conflict_sources` 合并到最终冲突源
  - 会话快照透出：
    - `snapshot.multi_role_pre`
    - `snapshot.rounds[-1].multi_role_pre`

## 3. 业务效果
- DeepThink 的“执行下一轮”不再是单独逻辑分支，而是明确复用统一多角色契约。
- 前端可直接解释“为什么进入重规划/冲突来源是什么”，避免黑箱体验。

## 4. 自测
执行命令：
- `python -m py_compile backend/app/service.py backend/app/http_api.py backend/app/config.py tests/test_service.py tests/test_http_api.py`
- `.venv/Scripts/python.exe -m pytest -q tests/test_service.py -k "deep_think_session_and_round or deep_think_v2_stream_round"`
- `.venv/Scripts/python.exe -m pytest -q tests/test_http_api.py -k "deep_think_and_a2a or deep_think_v2_round_stream"`

关键结果：
- DeepThink 会话轮次返回 `multi_role_pre.enabled=true`。
- 事件流包含 `pre_arbitration`。
- 事件流包含 `runtime_guard`（R4 同步改造影响）。
