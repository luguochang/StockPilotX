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

## 第4批增量（Knowledge Hub - 文档 Pipeline 可追溯）

### 新增能力

1. 文档处理运行记录（pipeline run）
- `backend/app/web/store.py`
  - 新增表：`doc_pipeline_run`
  - 新增索引：
    - `idx_doc_pipeline_run_doc_created`
    - `idx_doc_pipeline_run_stage_status`
- `backend/app/web/service.py`
  - 新增写入方法：`doc_pipeline_run_add()`
  - 新增查询方法：`doc_pipeline_runs()`

2. 文档版本列表（基于成功索引序列）
- `backend/app/web/service.py`
  - 新增：`doc_versions()`
  - 规则：对 `stage=index 且 status=ok` 的运行按时间顺序编号为版本号，接口按最新版本倒序返回。
- `backend/app/service.py`
  - 新增应用层透传：
    - `docs_versions()`
    - `docs_pipeline_runs()`

3. Upload/Index 流程接入运行追踪
- `backend/app/service.py`
  - `docs_upload()` 写入 `stage=upload,status=ok` 记录
  - `docs_index()` 写入：
    - `stage=index,status=ok`（索引成功）
    - `stage=index,status=not_found`（索引对象不存在）

4. 新增 API
- `backend/app/http_api.py`
  - `GET /v1/docs/{doc_id}/versions?limit=20`
  - `GET /v1/docs/{doc_id}/pipeline-runs?limit=30`

### 第4批自测

执行命令：
```bash
.\.venv\Scripts\python -m pytest -q tests/test_service.py -k "test_doc_upload_and_index"
.\.venv\Scripts\python -m pytest -q tests/test_http_api.py -k "test_docs_upload_and_index"
.\.venv\Scripts\python -m pytest -q tests -k "docs or query or web or api"
```

结果：
- 1 passed, 27 deselected
- 1 passed, 18 deselected
- 28 passed, 55 deselected

### 第4批 Checklist

- [x] 文档 pipeline 运行记录落库
- [x] 文档版本列表接口
- [x] Upload/Index 流程接入可追溯日志
- [x] API 路由接入
- [x] 相关回归测试通过

## 第5批增量（Knowledge Hub - 知识图谱基础查询）

### 新增能力

1. 知识图谱邻域查询服务
- `backend/app/service.py`
  - 新增 `knowledge_graph_view(entity_id, limit)`：
    - 从现有 GraphRAG store 读取关系
    - 输出一跳邻域 `nodes + relations`
    - 返回 `entity_type/node_count/relation_count`
  - 新增 `_infer_graph_entity_type()`，用于实体类型归一化（`stock/concept`）。

2. 新增图谱查询 API
- `backend/app/http_api.py`
  - 新增：`GET /v1/knowledge/graph/{entity_id}?limit=20`
  - 参数非法返回 `400`。

### 第5批自测

执行命令：
```bash
.\.venv\Scripts\python -m pytest -q tests/test_service.py -k "test_knowledge_graph_view"
.\.venv\Scripts\python -m pytest -q tests/test_http_api.py -k "test_knowledge_graph_view"
.\.venv\Scripts\python -m pytest -q tests -k "docs or query or web or api or knowledge_graph"
```

结果：
- 1 passed, 28 deselected
- 1 passed, 19 deselected
- 30 passed, 55 deselected

### 第5批 Checklist

- [x] 知识图谱一跳邻域服务
- [x] 知识图谱 API 路由
- [x] 服务层/API 层测试
- [x] 回归测试通过

## 第6批增量（Knowledge Hub - 智能推荐）

### 新增能力

1. 文档推荐引擎（history + context + graph）
- 新增文件：`backend/app/knowledge/recommender.py`
  - `DocumentRecommender.recommend()`：
    - 历史偏好信号：基于 query history 的股票频次
    - 上下文信号：`stock_code` 与问题关键词匹配
    - 图谱信号：知识图谱邻域概念词匹配
    - 质量信号：`quality_score` 加权
  - 输出文档级推荐结果（去重、排序、原因）。

2. 服务层推荐能力
- `backend/app/service.py`
  - 注入 `DocumentRecommender`
  - 新增 `docs_recommend(token, payload)`：
    - 聚合 `query_history + rag_doc_chunk + knowledge_graph`
    - 返回推荐列表（含 `score/reasons/stock_codes/source`）

3. 新增 API
- `backend/app/http_api.py`
  - `POST /v1/docs/recommend`

### 第6批自测

执行命令：
```bash
.\.venv\Scripts\python -m pytest -q tests/test_service.py -k "test_docs_recommend"
.\.venv\Scripts\python -m pytest -q tests/test_http_api.py -k "test_docs_upload_and_index"
.\.venv\Scripts\python -m pytest -q tests -k "docs or query or web or api or knowledge_graph or recommend"
```

结果：
- 1 passed, 29 deselected
- 1 passed, 19 deselected
- 31 passed, 55 deselected

### 第6批 Checklist

- [x] 文档推荐引擎
- [x] 推荐服务层聚合逻辑
- [x] 推荐 API
- [x] 新增测试与回归通过

## 第7批增量（DeepThink - 报告导出）

### 新增能力

1. DeepThink 报告导出器
- 新增文件：`backend/app/deepthink_exporter.py`
  - `export_markdown(session)`：输出结构化 Markdown 报告
  - `export_pdf_bytes(session)`：输出 PDF 二进制（内置最小 PDF 生成器，无三方依赖）

2. 服务层导出能力
- `backend/app/service.py`
  - 注入 `DeepThinkReportExporter`
  - 新增 `deep_think_export_report(session_id, format)`，支持：
    - `markdown`
    - `pdf`

3. 新增 API
- `backend/app/http_api.py`
  - `GET /v1/deep-think/sessions/{session_id}/report-export?format=markdown|pdf`
  - 通过 `Response` 直接下载文件。

### 第7批自测

执行命令：
```bash
.\.venv\Scripts\python -m pytest -q tests/test_service.py -k "test_deep_think_report_export_markdown_and_pdf"
.\.venv\Scripts\python -m pytest -q tests/test_http_api.py -k "test_deep_think_report_export"
.\.venv\Scripts\python -m pytest -q tests -k "deep_think or api or docs or query"
```

结果：
- 1 passed, 30 deselected
- 1 passed, 20 deselected
- 38 passed, 50 deselected

### 第7批 Checklist

- [x] DeepThink Markdown 报告导出
- [x] DeepThink PDF 报告导出
- [x] 导出 API 路由接入
- [x] 服务层/API 层测试
- [x] 回归测试通过

## 第8批增量（DeepThink - 分析维度结构化）

### 新增能力

1. Business Summary 增加结构化分析维度
- `backend/app/service.py`
  - `_deep_build_business_summary()` 新增参数 `opinions`
  - 新增 `_deep_build_analysis_dimensions()`，输出维度清单：
    - `industry`
    - `competition`
    - `supply_chain`
    - `risk`
    - `macro`
    - `execution`
  - 在 `business_summary` 中新增字段：`analysis_dimensions`

2. 轮次流程接入维度输出
- `backend/app/service.py`
  - `deep_think_run_round_stream_events()` 在生成 `business_summary` 时传入 `opinions`

### 第8批自测

执行命令：
```bash
.\.venv\Scripts\python -m pytest -q tests/test_service.py -k "deep_think_session_and_round or deep_think_v2_stream_round"
.\.venv\Scripts\python -m pytest -q tests/test_http_api.py -k "test_deep_think_v2_round_stream"
.\.venv\Scripts\python -m pytest -q tests -k "deep_think or api or docs or query"
```

结果：
- 3 passed, 28 deselected
- 1 passed, 20 deselected
- 38 passed, 50 deselected

### 第8批 Checklist

- [x] DeepThink 分析维度结构化输出
- [x] 流式事件携带维度面板
- [x] 服务层/API 层测试
- [x] 回归测试通过

## 第9批增量（Phase2 - Portfolio Manager 第1批）

### 新增能力

1. 组合/持仓/交易表结构
- `backend/app/web/store.py`
  - 新增：
    - `portfolio`
    - `portfolio_position`
    - `portfolio_transaction`
  - 新增索引：
    - `idx_portfolio_user_created`
    - `idx_portfolio_position_pid`
    - `idx_portfolio_tx_pid_date`

2. Portfolio 领域服务
- `backend/app/web/service.py`
  - 新增：
    - `portfolio_create`
    - `portfolio_list`
    - `portfolio_add_transaction`
    - `portfolio_positions`
    - `portfolio_transactions`
    - `portfolio_revalue`
    - `portfolio_summary`

3. 应用服务层接入
- `backend/app/service.py`
  - 新增：
    - `portfolio_create`
    - `portfolio_list`
    - `portfolio_add_transaction`
    - `portfolio_summary`
    - `portfolio_transactions`
  - 新增 `_portfolio_price_map`：组合估值时自动刷新行情并回填价格。

4. API 路由
- `backend/app/http_api.py`
  - `POST /v1/portfolio`
  - `GET /v1/portfolio`
  - `POST /v1/portfolio/{portfolio_id}/transactions`
  - `GET /v1/portfolio/{portfolio_id}`
  - `GET /v1/portfolio/{portfolio_id}/transactions`

### 第9批自测

执行命令：
```bash
.\.venv\Scripts\python -m pytest -q tests/test_service.py -k "test_portfolio_lifecycle"
.\.venv\Scripts\python -m pytest -q tests/test_http_api.py -k "test_portfolio_endpoints"
.\.venv\Scripts\python -m pytest -q tests -k "web or api or query or docs or portfolio"
```

结果：
- 1 passed, 31 deselected
- 1 passed, 21 deselected
- 33 passed, 57 deselected

### 第9批 Checklist

- [x] Portfolio 表结构
- [x] 交易驱动持仓更新
- [x] 组合估值与收益汇总
- [x] Portfolio API 路由
- [x] 服务层/API 层测试
- [x] 回归测试通过
