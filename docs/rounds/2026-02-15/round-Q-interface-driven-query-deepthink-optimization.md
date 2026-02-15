# Round-Q: 接口驱动联调优化（高级分析样本上下文 + DeepThink 情报解析稳健性）

Date: 2026-02-15

## 1. 背景与目标
- 用户反馈：
  - 高级分析会输出“样本稀疏、结论不确定”，业务可读性差。
  - 期望系统自动使用最近三个月连续样本，不需要用户反复手工补数据。
  - 要求工程侧自己调用接口联调，不再把验证动作交给用户。
- 本轮目标：
  - 直接接口自测 `query/stream` + `deep-think round stream`，按返回结果逐步修复。

## 2. 接口实测发现
- `/v1/query/stream`：
  - `analysis_brief.history_sample_size=260`，但模型回答仍会把证据解释成“离散样本点”。
- `/v2/deep-think/.../rounds/stream`：
  - 初期 `intel_status=fallback`，`fallback_reason=provider_or_parse_error`。
  - 通过 `intel_trace_id` 查询日志定位到真实错误：
    - tool 注入返回 `Unsupported tool type: web_search_preview`（随后已自动无 tool 重试）
    - 二次请求成功后，`confidence_adjustment='down'` 触发 `float()` 解析异常，导致再次 fallback。

## 3. 本轮修复
- `backend/app/llm/gateway.py`
  - HTTP 4xx/5xx 现在保留响应体并抛出明确错误（不再只剩 `Bad Request`）。
  - 支持上层识别可恢复原因，如 `Unsupported tool type`。
- `backend/app/service.py`
  - `decision_adjustment.confidence_adjustment` 解析增强：
    - 兼容文本值（`down`/`up`/`neutral` 等）和文本内数字提取。
  - 历史样本刷新增强：
    - `_needs_history_refresh` 增加 `min_samples` 约束，样本不足也会强制刷新。
  - 新增三个月连续样本摘要：
    - `_history_3m_summary`
    - `_augment_question_with_history_context`
  - 在 `query` 与 `query_stream_events` 中，把“最近三个月连续样本摘要”注入模型输入上下文。
  - 在检索语料里增加 `eastmoney_history_3m_window` 证据，提升模型对连续样本的感知。
- `tests/test_service.py`
  - 新增回归用例：
    - tool 不支持时自动无 tool 重试。
    - 文本型 `confidence_adjustment` 解析。
    - 三个月样本上下文注入。

## 4. 验证结果
- 自动化测试：
  - `.\.venv\Scripts\python.exe -m pytest tests/test_service.py tests/test_http_api.py -q`
  - Result: `37 passed`
- 接口联调（本地起服务）：
  - 高级分析：
    - 不再出现“样本稀疏/离散样本点”误判文本。
    - 输出含 `eastmoney_history_3m_window` 引用，并给出三个月区间结论。
    - `analysis_brief.history_sample_size=260`。
  - DeepThink 下一轮：
    - `intel_status=external_ok`
    - `citations_count=5`
    - `fallback_reason` 为空。

## 5. 结果
- 高级分析“样本不足”问题从“用户手工补数据”转为“系统自动补三个月连续样本上下文”。
- DeepThink 情报链路从“解析异常导致降级”修复为“可用外部情报 + 可追溯引用”。
