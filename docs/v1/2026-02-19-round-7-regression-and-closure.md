# ROUND-7 技术记录：全链路回归与交付收口

## 1. 本轮目标

完成本阶段计划收口：

1. 全链路回归  
2. 风险清单更新  
3. 最终交付文档说明

---

## 2. 回归范围

本轮回归覆盖以下链路：

1. Query 主流程（含 citation 归因字段）  
2. Datasource 管理链路（sources/health/fetch/logs）  
3. Intel Card 主流程（分析 + 风控执行建议）  
4. Intel 反馈复盘链路（feedback/review）  
5. RAG 检索器（HybridRetrieverV2 两阶段重排）  
6. DeepThink 前端编译与类型一致性

---

## 3. 自测命令与结果

1. Service 层回归  
```powershell
.\.venv\Scripts\python.exe -m pytest -q tests/test_service.py -k "query_basic or query_persists_rag_qa_memory_and_trace or ingest_endpoints or datasource_ops_catalog_health_fetch_logs or analysis_intel_card_contract or analysis_intel_feedback_and_review"
```
结果：`6 passed, 35 deselected`

2. HTTP API 回归  
```powershell
.\.venv\Scripts\python.exe -m pytest -q tests/test_http_api.py -k "query_stream or datasource_management_endpoints or analysis_intel_card or analysis_intel_feedback_and_review"
```
结果：`4 passed, 32 deselected`

3. RAG 检索回归  
```powershell
.\.venv\Scripts\python.exe -m pytest -q tests/test_rag_retrieval.py
```
结果：`3 passed`

4. 前端类型检查  
```powershell
cd frontend
npx tsc --noEmit
```
结果：通过（exit code 0）

---

## 4. 风险清单（更新）

1. **规则型重排风险**  
当前 rerank 仍是规则特征融合，复杂语义关系场景下可能逊于专用 reranker 模型。

2. **复盘样本冷启动风险**  
T+1/T+5/T+20 统计依赖足够历史样本与反馈数量；早期样本少时统计显著性有限。

3. **跨源时效差异风险**  
不同数据源更新节奏差异较大，降级策略已标识但仍可能出现短时不一致。

4. **前端信息密度风险**  
业务卡片已结构化，但专业信息量仍高，后续可按用户层级做分层展示。

---

## 5. 本阶段交付总结

已按计划完成 ROUND-1 ~ ROUND-7 的核心交付：

1. 数据源可见性与运维一致性  
2. Intel Card 业务聚合接口  
3. RAG 两阶段检索与归因一致性  
4. DeepThink 业务视图重构  
5. 风控阈值与执行建议结构化  
6. 采纳反馈与偏差复盘闭环  
7. 回归与文档收口

对应文档目录：`docs/v1/`

---

## 6. 运维建议（短期）

1. 每日检查 `/v1/datasources/health` 的 `staleness_minutes` 与失败率。  
2. 每周观察 `/v1/analysis/intel-card/review` 的样本数与命中率趋势。  
3. 若 `degrade_status=degraded` 高频出现，优先排查数据源时效与证据覆盖。
