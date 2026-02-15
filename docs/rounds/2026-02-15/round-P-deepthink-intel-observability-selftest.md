# Round-P: DeepThink 情报链路可观测性与自测接口落地

Date: 2026-02-15

## 1. Objectives
- 明确回答“为什么会出现 `external_websearch_unavailable` 降级提示”。
- 给出可复现的自测接口，直接验证外部开关、provider、websearch tool 命中与否。
- 将情报链路失败原因从“模糊报错”升级为“稳定原因码 + trace 日志”。

## 2. Root Cause Summary
- 触发“外部实时情报不可用”并不等于 prompt 写错，核心由三类条件触发：
  - 外部 LLM 总开关关闭（`LLM_EXTERNAL_ENABLED=false`）。
  - provider 未配置或不可用。
  - 模型响应无法通过结构化校验（例如 citations 为空、JSON 无法解析、tool 不支持）。
- 本次本地自检结果：
  - `external_enabled=false`
  - `provider_count=1`
  - `intel_status=fallback`
  - `fallback_reason=external_disabled`

## 3. Key Changes
- Backend `backend/app/llm/gateway.py`
  - `generate` / `stream_generate` 增加 `request_overrides`，支持按请求注入临时 body 字段。
  - 新增递归 body 合并，支持在不改 provider 静态配置的情况下挂载 `tools`。
- Backend `backend/app/service.py`
  - 情报输出增加诊断字段：
    - `intel_status`
    - `fallback_reason`
    - `fallback_error`
    - `trace_id`
    - `provider_count`
    - `provider_names`
    - `websearch_tool_requested`
    - `websearch_tool_applied`
  - `_deep_fetch_intel_via_llm_websearch`：
    - 显式尝试挂载 `web_search_preview` tool。
    - 若 tool 不支持，降级为 prompt-only 二次尝试，并记录 trace 事件。
    - 统一异常原因码映射，避免前端只看到笼统描述。
  - 新增自测/日志方法：
    - `deep_think_intel_self_test(...)`
    - `deep_think_trace_events(...)`
  - SSE 新增事件：
    - `intel_status`
- Backend API `backend/app/http_api.py`
  - 新增接口：
    - `GET /v1/deep-think/intel/self-test`
    - `GET /v1/deep-think/intel/traces/{trace_id}`
- Frontend `frontend/app/deep-think/page.tsx`
  - 分析模式新增“情报链路自检”按钮。
  - 业务结论/情报摘要卡片显示 `intel_status`、`fallback_reason`、`trace_id`、tool 命中状态。
  - 回放过滤新增 `intel_status` / `intel_self_test`。

## 4. Test Evidence
- `.\.venv\Scripts\python.exe -m pytest tests/test_service.py tests/test_http_api.py -q`
  - Result: `34 passed`
- `npm --prefix frontend run -s build`
  - Result: passed
- 手工自检脚本（服务内调用 `deep_think_intel_self_test`）
  - Result: `fallback_reason=external_disabled`
  - Trace event sample: `deep_intel_fallback {'reason': 'external_disabled', 'provider_count': 1}`

## 5. Notes
- 现在能精确区分：
  - 配置问题（开关关闭/无provider）
  - 能力问题（tool不支持）
  - 输出质量问题（JSON/citations 不合规）
- 后续若希望稳定拿到实时检索结果，需在运行环境开启：
  - `LLM_EXTERNAL_ENABLED=true`
  - 并确保所用 provider/model 支持 web search tool。
