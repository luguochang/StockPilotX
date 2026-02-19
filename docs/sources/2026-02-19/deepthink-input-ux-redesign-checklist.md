# DeepThink 输入体验重构 Checklist（2026-02-19）
> 计划：`docs/sources/2026-02-19/deepthink-input-ux-redesign-master-plan.md`

## 执行要求
1. 每轮必须包含：代码实现（含注释）+ 自测 + 技术文档 + commit。
2. 业务页优先可用性，工程能力保持可达但不打扰业务流。
3. 不破坏既有接口协议，优先前端交互重构。

## 轮次状态
- [x] ROUND-UX-1：双页分工（业务页/工程控制台）
- [x] ROUND-UX-2：模板优先输入（静态模板 + 槽位）
- [x] ROUND-UX-3：输入质量守卫（质量分 + 风险提示）
- [x] ROUND-UX-4：业务聚焦（业务页默认分析视角，工程能力迁移）
- [x] ROUND-UX-5：Analysis Studio 重构（不再嵌套 DeepThink 全页）

## 自测记录
1. `cd frontend && npm run build` -> Passed
2. `cd frontend && npx tsc --noEmit` -> Passed（本地先补齐 `.next/types` 占位后执行）

## 关键变更文件
1. `frontend/app/deep-think/page.tsx`
2. `frontend/app/deep-think/console/page.tsx`
3. `frontend/app/analysis-studio/page.tsx`
4. `frontend/app/lib/analysis/template-config.ts`
5. `frontend/app/lib/analysis/template-compose.ts`
6. `frontend/app/lib/analysis/guardrails.ts`

## Commit 记录
1. `ad5df2d` | `feat(ux): implement deepthink input redesign and console split (UX-1~UX-5)`
