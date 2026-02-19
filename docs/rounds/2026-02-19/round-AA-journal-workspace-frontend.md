# Round AA - Journal Workspace Frontend

日期：2026-02-19  
目标：提供 Journal 一体化前端工作台，覆盖创建、复盘、AI复盘和洞察查看。

## Implementation

1. 新增页面
- `frontend/app/journal/page.tsx`
  - 日志创建表单（`POST /v1/journal`）
  - 日志筛选与列表（`GET /v1/journal`）
  - 手工复盘记录（`POST/GET /v1/journal/{journal_id}/reflections`）
  - AI复盘生成与查看（`POST/GET /v1/journal/{journal_id}/ai-reflection*`）
  - 聚合洞察看板（`GET /v1/journal/insights`）

2. 导航与入口
- `frontend/app/layout.tsx`
  - 顶部导航新增：`/journal`（投资日志）
- `frontend/app/page.tsx`
  - 首页功能卡新增“投资日志工作台”

3. 交互策略
- 使用单页闭环布局：左侧“创建+列表”，右侧“洞察”，底部“详情 Tabs”。
- 通过提示消息统一回传 API 操作结果，减少无反馈点击。
- 默认参数简化，避免页面出现过多输入项。

## Verification

1. `cd frontend && npm run build`
- 结果：`Compiled successfully`，`/journal` 路由构建通过。

2. `cd frontend && npx tsc --noEmit`
- 结果：通过（无类型错误）。
