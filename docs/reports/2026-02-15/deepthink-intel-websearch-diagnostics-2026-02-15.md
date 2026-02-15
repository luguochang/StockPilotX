# DeepThink 实时情报降级诊断报告（2026-02-15）

## 1. 现象
- 页面显示：
  - 触发条件：外部实时情报不可用，按降级策略下调置信度。
  - 失效条件：若关键风险事件落地偏负面、分歧持续扩大或预算风控触发，则信号失效。
- 用户关切：这是 prompt 问题、模型能力问题，还是系统没有真的走 websearch？

## 2. 结论
- 当前主因是运行时配置：`LLM_EXTERNAL_ENABLED=false`。
- 在该配置下，系统会直接进入本地降级情报路径，不会调用外部实时检索。
- 因此这不是“前端假流式”的问题，也不是单纯 prompt 质量问题。

## 3. 已落地的排障能力
- 新增接口：
  - `GET /v1/deep-think/intel/self-test`
  - `GET /v1/deep-think/intel/traces/{trace_id}`
- 新增稳定诊断字段：
  - `intel_status`
  - `fallback_reason`
  - `fallback_error`
  - `trace_id`
  - `websearch_tool_requested`
  - `websearch_tool_applied`
- 新增 SSE 事件：
  - `intel_status`

## 4. 本地自测证据
- `deep_think_intel_self_test` 返回：
  - `intel_status=fallback`
  - `fallback_reason=external_disabled`
  - `external_enabled=false`
  - `provider_count=1`
- trace 事件：
  - `deep_intel_fallback {"reason":"external_disabled","provider_count":1}`

## 5. 后续操作建议
- 运行环境开启：
  - `LLM_EXTERNAL_ENABLED=true`
- 确认 provider/model 支持 websearch tool。
- 通过 `/v1/deep-think/intel/self-test` 验证状态应切换为：
  - `intel_status=external_ok`
  - 且 `citation_count > 0`。
