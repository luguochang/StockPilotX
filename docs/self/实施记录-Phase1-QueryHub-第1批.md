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
