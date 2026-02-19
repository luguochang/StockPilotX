# ROUND-UX-3 记录：输入质量守卫

## 改动目标
1. 在提交前识别低质量输入，减少无效分析。
2. 给出可修复建议，而不是静默失败。

## 关键改动
1. `frontend/app/lib/analysis/guardrails.ts`
   - 新增质量评分、错误/警告/建议生成逻辑。
2. `frontend/app/deep-think/page.tsx`
   - 提交前守卫校验。
   - 显示输入质量分与守卫提示标签。

## 自测
1. `npm run build` 通过。
2. `npx tsc --noEmit` 通过。
