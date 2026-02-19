# ROUND-UX-8：命中片段可点击定位与文档页跳转

## 背景
在 ROUND-UX-7 中，用户已经能看到“可检索样本预览”，但仍缺少两项关键体验：
1. 命中片段无法直接打开对应文档 chunk 详情。
2. 跳转到文档页后无法自动定位到指定 chunk。

## 本轮目标
1. 在 RAG 预览卡片中支持“点击命中片段 -> 打开 chunk 定位视图”。
2. 支持“在文档页打开”并通过 URL 参数自动定位到 chunk。
3. 后端提供稳定的 chunk 详情接口，包含上下文片段。

## 后端实现
1. 新增 chunk 详情能力（Web Service）：
   - 文件：`backend/app/web/service.py`
   - 方法：`rag_doc_chunk_get_detail(token, chunk_id, context_window)`
   - 返回：
     - `chunk`：当前片段完整信息（含 `chunk_text_redacted/chunk_text`）
     - `context.prev` / `context.next`：上下文片段摘要
2. 新增应用服务包装：
   - 文件：`backend/app/service.py`
   - 方法：`rag_doc_chunk_detail(...)`
3. 新增 HTTP 接口：
   - 文件：`backend/app/http_api.py`
   - 路径：`GET /v1/rag/docs/chunks/{chunk_id}?context_window=1`

## 前端实现
1. `rag-center` 预览卡片增强：
   - 文件：`frontend/app/rag-center/page.tsx`
   - 每条 top hit 提供“定位查看”按钮。
   - 若命中项具备 `doc_id/chunk_id`，提供“在文档页打开”链接。
   - 新增右侧 `Drawer` 展示：当前 chunk + 上下文片段。
2. `docs-center` 参数定位增强：
   - 文件：`frontend/app/docs-center/page.tsx`
   - 读取 URL 参数 `doc_id` / `chunk_id` / `q`。
   - 自动调用 chunk 详情接口并展示“片段定位视图”。
   - 支持“重新定位 URL 片段”。

## 测试变更
1. `tests/test_service.py`
   - 在 `test_rag_doc_policy_and_chunk_management` 增加 chunk 详情断言。
2. `tests/test_http_api.py`
   - 在 `test_rag_asset_management_endpoints` 增加 `/v1/rag/docs/chunks/{chunk_id}` 详情接口断言。

## 自测结果
1. `\.venv\Scripts\python.exe -m pytest -q tests/test_service.py -k "rag_doc_policy_and_chunk_management or rag_upload_workflow_and_dashboard or rag_retrieval_preview_api_wrapper"`：Passed
2. `\.venv\Scripts\python.exe -m pytest -q tests/test_http_api.py -k "rag_asset_management_endpoints"`：Passed
3. `cd frontend && npm run build`：Passed
4. `cd frontend && npx tsc --noEmit`：Passed

## 风险与后续
1. 当前“文档页定位”基于 chunk 文本展示，后续可增加原文段落锚点与高亮。
2. 对非文档命中（缺少 `chunk_id`）目前仅展示摘要，后续可补来源分类跳转。
