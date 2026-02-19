# RAG 语料中心体验优化 Checklist（2026-02-19）
> 关联轮次文档：`docs/sources/2026-02-19/round-UX-6-rag-business-entry-simplification.md`

## 执行要求
1. 每轮必须包含：代码实现（关键逻辑注释）+ 自测 + 技术文档 + commit。
2. 业务模式优先最小输入，默认可用；运维能力不删减，只迁移到 ops 视图。
3. 接口协议保持兼容，避免影响既有 `/v1/rag/*` 与 `/v1/ops/rag/*` 链路。

## 轮次状态
- [x] ROUND-UX-6：RAG 业务入口简化（预设驱动 + 高级项折叠 + 一键上传生效）

## 任务清单
- [x] 新增 `business | ops` 双模式视图切换。
- [x] 业务模式引入上传预设（财报/公告/研报/会议纪要/自定义）。
- [x] 预设自动回填 `source` 与 `tags`，减少手动输入。
- [x] 默认主路径收敛为「选择类型 -> 选择文件 -> 上传并生效」。
- [x] 高级参数折叠在「高级设置（可选）」中，默认不干扰。
- [x] 运维治理能力保留（source/chunk/memory/trace）。

## 自测记录
1. `cd frontend && npm run build` -> Passed
2. `cd frontend && npx tsc --noEmit` -> Passed

## 关键变更文件
1. `frontend/app/rag-center/page.tsx`
2. `docs/sources/2026-02-19/rag-center-ux-simplification-checklist.md`
3. `docs/sources/2026-02-19/round-UX-6-rag-business-entry-simplification.md`

## Commit 记录
1. 待本轮提交后补充（与本次改动同一 commit）
