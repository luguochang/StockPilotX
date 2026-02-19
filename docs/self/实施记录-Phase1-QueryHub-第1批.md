# Phase1 实施记录 - Query Hub 第1批（缓存/对比/历史）

日期：2026-02-19
阶段：金融分析业务系统规划 Phase1 -> 模块1 Query Hub 完善

## 本批目标

- 增加查询缓存，减少重复请求的计算开销。
- 增加多标的对比查询接口。
- 增加查询历史落库与查询能力。
- 保持现有 `/v1/query`、`/v1/query/stream` 行为兼容。

## 变更清单

1. 新增 Query 域模块
- `backend/app/query/__init__.py`
- `backend/app/query/optimizer.py`
- `backend/app/query/comparator.py`

2. Query 主链路接入缓存与历史
- `backend/app/service.py`

3. Web 数据库新增 Query History 表
- `backend/app/web/store.py`

4. Web Service 新增 Query History 读写能力
- `backend/app/web/service.py`

5. 新增 HTTP API
- `POST /v1/query/compare`
- `GET /v1/query/history`
- `DELETE /v1/query/history`
- 文件：`backend/app/http_api.py`

6. 测试补充
- `tests/test_http_api.py` 新增 `test_query_cache_compare_and_history`

## 接口说明

### 1) `POST /v1/query/compare`
请求示例：
```json
{
  "user_id": "u1",
  "question": "compare SH600000 and SZ000001",
  "stock_codes": ["SH600000", "SZ000001"]
}
```
返回核心字段：
- `question`
- `count`
- `best_stock_code`
- `items[]`（含 `signal/confidence/expected_excess_return/risk_flag_count/citation_count/cache_hit`）

### 2) `GET /v1/query/history?limit=50`
返回当前用户最近查询记录（含 `question/stock_codes/trace_id/intent/cache_hit/latency_ms/created_at`）。

### 3) `DELETE /v1/query/history`
清空当前用户查询历史。

## 自测结果

执行命令：
```bash
.\.venv\Scripts\python -m pytest -q tests/test_http_api.py -k "query_cache_compare_and_history or test_query"
.\.venv\Scripts\python -m pytest -q tests/test_service.py -k "test_query_basic or test_query_repeated_calls_do_not_hit_global_model_limit"
.\.venv\Scripts\python -m pytest -q tests -k "web or api or watchlist"
```
结果：
- 3 passed, 15 deselected
- 2 passed, 24 deselected
- 19 passed, 61 deselected

## Checklist

- [x] Query Hub 查询缓存接入
- [x] Query Hub 多标的对比接口
- [x] Query Hub 查询历史表与接口
- [x] 回归测试通过
- [x] 阶段文档补齐

## 下一批（Phase1 按文档顺序）

- Query Hub：补充“查询失败可观测性字段（错误分类/降级原因）”
- Query Hub：补充“历史筛选（按股票/时间）”
- Knowledge Hub：开始文档处理 Pipeline 第1批优化

## 第2批增量（Knowledge Hub 可观测性）

### 新增能力：文档质量报告

新增接口：`GET /v1/docs/{doc_id}/quality-report`

输出包含：
- `quality_score` / `quality_level`
- `chunk_stats`（分块数、活跃分块数、平均分块长度、短分块占比）
- `recommendations`（可执行建议）

实现文件：
- `backend/app/service.py`（`docs_quality_report`）
- `backend/app/http_api.py`（新增路由）
- `tests/test_http_api.py`（补充接口测试）

### 第2批自测

执行命令：
```bash
.\.venv\Scripts\python -m pytest -q tests/test_http_api.py
.\.venv\Scripts\python -m pytest -q tests -k "web or api or watchlist"
```
结果：
- 18 passed
- 19 passed, 61 deselected

### 第2批 Checklist

- [x] 文档质量报告能力
- [x] 质量建议规则
- [x] API 路由接入
- [x] 回归测试通过

## 第3批增量（Query Hub 稳定性与历史筛选）

### 新增能力

1. 查询异常降级响应（避免原始 500）
- `backend/app/service.py`
  - `query()` 增加 `TimeoutError/Exception` 捕获
  - 新增 `_build_query_degraded_response()`，统一返回结构化降级结果：
    - `degraded: true`
    - `error_code`（`query_timeout` / `query_runtime_error`）
    - `error_message`
    - 兼容原响应关键字段（`trace_id/answer/citations/risk_flags/analysis_brief`）
- 降级结果写入 `query_history.error`，便于后续追溯错误分布。

2. 查询历史筛选能力（按股票 + 时间范围）
- `backend/app/web/service.py`
  - `query_history_list()` 新增参数：
    - `stock_code`
    - `created_from`
    - `created_to`
  - 新增时间标准化方法，支持：
    - `YYYY-MM-DD`
    - `YYYY-MM-DD HH:MM:SS`
  - 增加时间区间校验（`created_from <= created_to`）。
- `backend/app/service.py`
  - `query_history_list()` 同步透传筛选参数。
- `backend/app/http_api.py`
  - `GET /v1/query/history` 新增查询参数：
    - `stock_code`
    - `created_from`
    - `created_to`
  - 参数非法时返回 `400`。

3. Query 接口参数校验错误映射
- `backend/app/http_api.py`
  - `POST /v1/query` 新增 `ValueError` 捕获（含 Pydantic 校验异常）并返回 `400`。

### 第3批自测

执行命令：
```bash
.\.venv\Scripts\python -m pytest -q tests/test_service.py -k "query_timeout_returns_degraded_payload or query_history_filter_by_stock_and_time"
.\.venv\Scripts\python -m pytest -q tests/test_http_api.py -k "query_cache_compare_and_history or query_validation_returns_400"
.\.venv\Scripts\python -m pytest -q tests -k "query or web or api"
```

结果：
- 2 passed, 26 deselected
- 2 passed, 17 deselected
- 27 passed, 56 deselected

### 第3批 Checklist

- [x] Query 超时/异常结构化降级
- [x] 降级错误写入 query_history
- [x] Query history 支持股票筛选
- [x] Query history 支持时间范围筛选
- [x] `/v1/query` 校验异常返回 400
- [x] 回归测试通过并记录结果
