# ROUND-AI 执行 Checklist（全接口自测 + 输入补数统一）

## 执行要求
1. 每完成一个子项，必须先自测再勾选。
2. 每轮必须补充技术文档记录。
3. 每轮结束必须提交 commit（禁止把无关改动混入）。
4. 代码改动需带必要注释，方便后续维护。

## Checklist
- [x] 后端统一输入数据包：Query / Query Stream / DeepThink / Predict / Report 全链路接入
- [x] 报告任务快照新增 `data_pack_status`、`data_pack_missing`、`quality_gate_detail`
- [x] 报告页展示任务级数据缺口与质量状态
- [x] Journal 最小输入可创建（后端自动补全 + 前端防 `trim` 空值报错）
- [x] 新增并增强全接口自测脚本 `scripts/full_api_selftest.py`
- [x] 单测通过：`tests/test_service.py` + `tests/test_http_api.py`（82 passed）
- [x] 前端类型检查通过：`npx tsc --noEmit`
- [x] 前端生产构建通过：`npm run build`（2026-02-20 复验通过，Next.js 构建成功）

## 本轮结果摘要
- API 路由 smoke：`129` 路由，`0` 个 5xx。
- 轻量模式跳过重型路由 `29` 个（状态码 `299` 标识），用于提高本地回归稳定性。
- 前端生产构建复验完成：`19/19` 静态页面生成成功，`/reports` 等核心页面可产物化。
