# Round-AL：报告版本差异与证据时效评分（2026-02-20）

## 1. 本轮目标
在 Round-AK 基础上完成 Phase D：
1. 让报告版本具备可比较的结构化快照（不仅是 markdown）。
2. 提供版本差异接口，输出模块/节点/质量/决策层面的变化。
3. 引入证据 freshness 评分，并纳入质量看板。
4. 为导出 JSON Bundle 增加 schema version，便于后续程序化消费。

## 2. 主要改动

### 2.1 报告版本持久化结构升级
文件：`backend/app/web/store.py`、`backend/app/web/service.py`

1. `report_version` 表新增 `payload_json` 字段（自动迁移，历史库兼容）。
2. `save_report_index` 新增 `payload_json` 入参，保存每个版本的结构化报告快照。
3. 新增 `report_version_rows`，用于读取版本对比所需完整字段（version/created_at/markdown/payload_json）。

### 2.2 报告版本差异接口
文件：`backend/app/service.py`、`backend/app/http_api.py`

新增能力：
1. `report_versions`：返回增强版本列表（signal、quality_score、schema_version、delta_vs_prev）。
2. `report_versions_diff`：返回结构化差异结果：
   - `quality_delta`
   - `decision_delta`
   - `module_deltas`
   - `node_deltas`
   - `summary`
3. API 路由：
   - `GET /v1/reports/{report_id}/versions/diff`
   - 支持 query: `base_version`、`candidate_version`

### 2.3 证据 freshness 评分
文件：`backend/app/service.py`

新增函数：
1. `_safe_parse_datetime`
2. `_evidence_freshness_profile`
3. `_normalize_report_evidence_refs`
4. `_build_report_evidence_ref`

落地效果：
1. `evidence_refs` 新增字段：
   - `freshness_score`
   - `freshness_tier`
   - `age_hours`
2. 报告生成阶段会对 query/intel 证据统一计算 freshness。

### 2.4 质量看板融合 freshness
文件：`backend/app/service.py`

`_build_report_quality_dashboard` 新增指标：
1. `evidence_freshness_score`
2. `stale_evidence_ratio`

并将 freshness 纳入 overall_score 计算，使“信息时效”对结论质量有可解释影响。

### 2.5 导出 schema version
文件：`backend/app/service.py`

1. 报告导出 `json_bundle` 增加 `schema_version`。
2. 报告主结果增加 `schema_version`。
3. `report_export` 兼容从内存/持久化 payload 取结构化数据。

### 2.6 前端版本差异入口
文件：`frontend/app/reports/page.tsx`

新增：
1. 报告列表操作按钮“版本差异”。
2. 调用 `/v1/reports/{report_id}/versions/diff` 并展示返回 JSON。
3. 质量看板卡片显示 `freshness_score`。

## 3. 测试与回归

1. 后端回归：
```bash
.\.venv\Scripts\python.exe -m pytest tests/test_service.py tests/test_http_api.py -q
```
结果：`82 passed`

2. 前端构建：
```bash
cd frontend
npm run build
```
结果：通过

3. 前端类型检查：
```bash
cd frontend
npx tsc --noEmit
```
结果：通过（在 `build` 后执行）

4. 全接口轻量 smoke：
```bash
.\.venv\Scripts\python.exe scripts/full_api_selftest.py
```
结果：`total=130 failed=0`

5. heavy-critical smoke：
```bash
$env:SMOKE_RUN_HEAVY='1'
$env:SMOKE_HEAVY_CRITICAL='1'
.\.venv\Scripts\python.exe scripts/full_api_selftest.py
```
结果：`total=130 failed=0`

## 4. 业务收益
1. 版本演进可解释：不再只能看 markdown 文本差异。
2. 报告质量可量化：新增“证据时效”维度，能更贴合实时金融分析场景。
3. 对外集成更稳定：`json_bundle + schema_version` 支持后续自动化解析和回放。
