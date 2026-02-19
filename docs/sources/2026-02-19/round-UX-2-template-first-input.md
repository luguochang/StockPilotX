# ROUND-UX-2 记录：模板优先输入

## 改动目标
1. 默认提供结构化模板，减少用户从空白输入开始。
2. 通过固定槽位约束问题语义，提升可执行性。

## 关键改动
1. `frontend/app/lib/analysis/template-config.ts`
   - 定义模板与槽位类型、模板目录。
2. `frontend/app/lib/analysis/template-compose.ts`
   - 将模板 + 槽位组合成可执行问题文本。
3. `frontend/app/deep-think/page.tsx`
   - 新增模板模式与自由输入模式切换。
   - 新增模板选择、周期/风险/持仓槽位输入。

## 自测
1. `npm run build` 通过。
2. `npx tsc --noEmit` 通过。
