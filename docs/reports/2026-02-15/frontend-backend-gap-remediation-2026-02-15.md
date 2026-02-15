# 前后端缺口整改清单（2026-02-15）

> 目标：把“后端已实现接口”尽量在前端提供可操作入口，并给出验收路径。

## A. 本轮已完成
- [x] DOC-UI-001 文档中心补齐上传与索引
  - 接口：`POST /v1/docs/upload`、`POST /v1/docs/{doc_id}/index`
  - 页面：`frontend/app/docs-center/page.tsx`
- [x] DOC-UI-002 文档复核动作补齐
  - 接口：`POST /v1/docs/{doc_id}/review/approve`、`POST /v1/docs/{doc_id}/review/reject`
  - 页面：`frontend/app/docs-center/page.tsx`
- [x] OPS-UI-001 调度状态与停启补齐
  - 接口：`GET /v1/scheduler/status`、`POST /v1/scheduler/pause`、`POST /v1/scheduler/resume`
  - 页面：`frontend/app/ops/scheduler/page.tsx`
- [x] REP-UI-001 报告生成与详情补齐
  - 接口：`POST /v1/report/generate`、`GET /v1/report/{report_id}`
  - 页面：`frontend/app/reports/page.tsx`
- [x] REP-UI-002 报告版本历史补齐
  - 接口：`GET /v1/reports/{report_id}/versions`
  - 页面：`frontend/app/reports/page.tsx`
- [x] AUTH-UI-001 登录页补齐 me/refresh 调试能力
  - 接口：`GET /v1/auth/me`、`POST /v1/auth/refresh`
  - 页面：`frontend/app/login/page.tsx`

## B. 待继续补齐
- [x] DATA-UI-001 数据摄取运维入口
  - 接口：`POST /v1/ingest/market-daily`、`POST /v1/ingest/announcements`
  - 页面：`frontend/app/ops/health/page.tsx` 已增加“手动补抓 + 健康刷新 + 结果回显”
- [x] OPS-UI-002 告警确认后的局部刷新与筛选
  - 接口：`GET /v1/alerts`、`POST /v1/alerts/{alert_id}/ack`
  - 页面：`frontend/app/ops/alerts/page.tsx` 已支持级别/状态/关键词筛选与 ack 后自动刷新
- [x] REP-UI-003 报告导出格式扩展
  - 当前：markdown 文本展示
  - 已实现：支持 `下载 .md`、`复制 Markdown`、`复制分享信息`

## C. 验收命令
1. 前端构建：`cd frontend; npm run build`
2. 后端自测：`\.venv\Scripts\python -m pytest -q`
3. 手工联调：
   - 登录页拿 token -> 文档中心上传/索引/复核
   - 调度页运行任务 + pause/resume
   - 报告页生成报告 -> 查详情 -> 查版本 -> 导出
