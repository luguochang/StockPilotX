# ROUND-UX-1 记录：双页分工与控制台路由拆分

## 改动目标
1. `/deep-think` 聚焦业务分析。
2. `/deep-think/console` 作为工程控制台入口。

## 关键改动
1. `frontend/app/deep-think/page.tsx`
   - 新增 workspace 识别逻辑（通过 URL query）。
   - console 工作区隐藏业务大面板，仅保留控制台区域。
2. `frontend/app/deep-think/console/page.tsx`
   - 新增路由，重定向到 `/deep-think?workspace=console`。

## 自测
1. `npm run build` 通过。
2. `npx tsc --noEmit` 通过。
