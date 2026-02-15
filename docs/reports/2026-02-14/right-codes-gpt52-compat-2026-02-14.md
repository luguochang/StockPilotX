# right.codes GPT-5.2 接入兼容性报告（2026-02-14）

## 目标
- 主接入协议使用 OpenAI Responses API：`POST /responses`
- 主模型使用 `gpt-5.2`
- 同时支持同步查询和流式查询

## 当前配置
- 文件：`backend/config/llm_providers.local.json`
- 关键字段：
  - `api_style = openai_responses`
  - `api_base = https://www.right.codes/codex/v1`
  - `model = gpt-5.2`
  - `stream = true`

## 代码实现点
- 网关：`backend/app/llm/gateway.py`
  - 已支持 `openai_responses` 分支
  - 同步调用：`_call_openai_responses`
  - 流式调用：`_stream_openai_responses`
  - 新增兼容：`_parse_openai_responses_response` 可解析
    - 标准 JSON body
    - `stream=false` 但网关仍返回 SSE(event/data) 的场景
- 服务流式：`backend/app/service.py` -> `query_stream_events`
- HTTP 流式接口：`backend/app/http_api.py` -> `POST /v1/query/stream`

## 自测结果
- 网关同步调用：通过
  - 证据：`gw.generate(...)` 返回非空文本
- 网关流式调用：通过
  - 证据：`gw.stream_generate(...)` 持续产出 chunks
- 服务层流式链路：通过
  - 证据：`query_stream_events(...)` 出现 `stream_source = external_llm_stream`
  - 事件序列正常到 `done`
- 回归测试：通过
  - 命令：`.venv\\Scripts\\python.exe -m pytest tests/test_http_api.py tests/test_service.py tests/test_llm_fallback.py -q`
  - 结果：`21 passed`

## 结论
- 当前后端已按你给的 GPT 接口（right.codes Responses）接入，不再是 Claude-only 路径。
- 即便该网关在 `stream=false` 时返回 SSE，本项目也能正确解析并输出结果。

## 额外 HTTP 端到端验证
- 启动 `uvicorn backend.app.http_api:create_app --factory --host 127.0.0.1 --port 8011`
- 调用 `POST /v1/query`：返回 `query_ok=True`
- 调用 `POST /v1/query/stream`：返回体包含 `external_llm_stream`
