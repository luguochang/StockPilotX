# ROUND-UX-7：上传后可检索样本预览（RAG 生效可视化）

## 背景
用户反馈上传文档后缺少“是否真的能被检索到”的即时反馈，导致对 RAG 生效状态不确定。

## 本轮目标
1. 上传完成后立刻返回可验证结果，而不是只显示“上传成功”。
2. 给出“样本 query -> 命中结果”的可视化证据，帮助用户判断语料是否可用。
3. 保持现有 API 兼容，不破坏既有上传链路。

## 后端实现
1. 在 `backend/app/service.py` 新增 `rag_retrieval_preview` 能力：
   - 输入：`doc_id`、`max_queries`、`top_k`。
   - 行为：基于当前文档 chunk 生成样本 query，调用运行时 retriever 做检索。
   - 输出：每条 query 的命中情况、命中 rank、top hit 摘要、命中率。
2. 在 `backend/app/service.py` 的 `rag_workflow_upload_and_index` 中内嵌调用预览能力：
   - 上传成功后直接回传 `retrieval_preview`。
   - 预览失败时降级返回 `reason=preview_failed`，不影响上传主流程。
3. 在 `backend/app/http_api.py` 新增接口：
   - `GET /v1/rag/retrieval-preview`
   - 支持按 `doc_id` 单独重查预览结果。
4. 在 `backend/app/web/service.py` 增强内部 chunk 查询：
   - `rag_doc_chunk_list_internal` 支持 `doc_id` 过滤。
   - 查询结果附带 `effective_status`，用于判断文档是否已 active。

## 前端实现
1. 在 `frontend/app/rag-center/page.tsx` 新增 `retrievalPreview` 状态。
2. 上传流程优先使用 `upload-and-index` 返回的 `retrieval_preview`。
3. 若工作流未返回预览，则 fallback 调用 `/v1/rag/retrieval-preview`。
4. 业务模式新增“可检索样本预览”卡片，展示：
   - 当前文档命中率
   - 每条 query 是否命中当前文档
   - top 命中摘要（便于快速人工核验）

## 测试与回归
1. `tests/test_service.py`
   - 增补 `test_rag_retrieval_preview_api_wrapper`
   - 强化 `test_rag_upload_workflow_and_dashboard`，断言返回 `retrieval_preview`
2. `tests/test_http_api.py`
   - 强化 `test_rag_asset_management_endpoints`，校验：
     - 上传接口返回 `retrieval_preview`
     - 新接口 `/v1/rag/retrieval-preview` 可用

## 自测结果
1. `\.venv\Scripts\python.exe -m pytest -q tests/test_service.py -k "rag_upload_workflow_and_dashboard or rag_retrieval_preview_api_wrapper"`：Passed
2. `\.venv\Scripts\python.exe -m pytest -q tests/test_http_api.py -k "rag_asset_management_endpoints"`：Passed
3. `cd frontend && npm run build`：Passed
4. `cd frontend && npx tsc --noEmit`：Passed

## 风险与后续
1. 当前样本 query 仍是规则生成，后续可升级为“用户意图模板 + 语料类型模板”联合生成。
2. 命中只展示 top 命中摘要，后续可增加“跳转到文档片段”能力，缩短核验路径。
3. 可继续扩展为“上传后自动跑 3 类标准检索场景（事实/风险/时间）”并形成健康分。
