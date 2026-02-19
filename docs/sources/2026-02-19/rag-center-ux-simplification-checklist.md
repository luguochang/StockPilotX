# RAG 语料中心体验优化 Checklist（2026-02-19）
> 关联轮次文档：
> 1. `docs/sources/2026-02-19/round-UX-6-rag-business-entry-simplification.md`
> 2. `docs/sources/2026-02-19/round-UX-7-rag-retrieval-preview.md`
> 3. `docs/sources/2026-02-19/round-UX-8-rag-chunk-jump-and-doc-locate.md`

## 执行要求
1. 每轮必须包含：代码实现（关键逻辑注释）+ 自测 + 技术文档 + commit。
2. 业务模式优先最小输入，默认可用；运维能力不删减，只迁移到 ops 视图。
3. 接口协议保持兼容，避免影响既有 `/v1/rag/*` 与 `/v1/ops/rag/*` 链路。

## 轮次状态
- [x] ROUND-UX-6：RAG 业务入口简化（预设驱动 + 高级项折叠 + 一键上传生效）
- [x] ROUND-UX-7：上传后可检索样本预览（后端预览能力 + 前端结果可视化）
- [x] ROUND-UX-8：命中片段可点击定位 + 文档页跳转定位

## 任务清单
- [x] 新增 `business | ops` 双模式视图切换。
- [x] 业务模式引入上传预设（财报/公告/研报/会议纪要/自定义）。
- [x] 预设自动回填 `source` 与 `tags`，减少手动输入。
- [x] 默认主路径收敛为「选择类型 -> 选择文件 -> 上传并生效」。
- [x] 高级参数折叠在「高级设置（可选）」中，默认不干扰。
- [x] 运维治理能力保留（source/chunk/memory/trace）。
- [x] 新增 `/v1/rag/retrieval-preview` 接口，按 `doc_id` 输出样本 query 与命中结果。
- [x] `upload-and-index` 工作流回传 `retrieval_preview`，上传后立即可见验证反馈。
- [x] 业务页新增“可检索样本预览”卡片，展示命中率、命中 rank 和 top 命中片段。
- [x] 新增 `/v1/rag/docs/chunks/{chunk_id}` 详情接口，返回当前 chunk 与上下文片段。
- [x] 预览命中项支持“定位查看”抽屉和“在文档页打开”。
- [x] docs-center 支持 `doc_id/chunk_id` URL 参数自动定位展示。

## 自测记录
1. `\.venv\Scripts\python.exe -m pytest -q tests/test_service.py -k "rag_doc_policy_and_chunk_management or rag_upload_workflow_and_dashboard or rag_retrieval_preview_api_wrapper"` -> Passed
2. `\.venv\Scripts\python.exe -m pytest -q tests/test_http_api.py -k "rag_asset_management_endpoints"` -> Passed
3. `cd frontend && npm run build` -> Passed
4. `cd frontend && npx tsc --noEmit` -> Passed

## 关键变更文件
1. `backend/app/web/service.py`
2. `backend/app/service.py`
3. `backend/app/http_api.py`
4. `frontend/app/rag-center/page.tsx`
5. `frontend/app/docs-center/page.tsx`
6. `tests/test_service.py`
7. `tests/test_http_api.py`
8. `docs/sources/2026-02-19/round-UX-8-rag-chunk-jump-and-doc-locate.md`
9. `docs/sources/2026-02-19/rag-center-ux-simplification-checklist.md`

## Commit 记录
1. `223e7b1` | `feat(ux): simplify rag-center business upload flow with preset-first entry`
2. `0678a1c` | `feat(rag): add upload-time retrieval preview and verification UI`
3. 待本轮提交后补充（ROUND-UX-8）
