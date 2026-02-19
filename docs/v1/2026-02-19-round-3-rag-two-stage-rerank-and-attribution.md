# ROUND-3 技术记录：RAG 两阶段检索与引用归因一致性

## 1. 本轮目标

落地计划中的 ROUND-3：

1. 将检索链路明确为两阶段：粗排召回 -> 精排重排。  
2. 给每条证据和引用补齐归因字段，支持“这条结论来自哪条检索轨道”的排障。  
3. 在服务层统一规范 citations，避免不同调用链输出结构不一致。

---

## 2. 关键实现

## 2.1 检索器升级（粗排 + 精排）

文件：`backend/app/rag/hybrid_retriever_v2.py`

主要改动：

1. **粗排候选池扩容**  
- 由 `rerank_top_n` 推导 `coarse_pool_size`，提升候选覆盖，避免早期截断。  
- 词法粗排和语义召回都使用更大候选规模。

2. **精排特征化打分**  
精排综合以下特征：
- lexical rank score  
- semantic rank score  
- query token overlap score  
- reliability score  
- freshness score

3. **多样性惩罚**  
- 对同一 `source_id` 的重复高位结果施加小幅 penalty，减少 Top-N 被单源垄断。

4. **可解释元数据**  
输出 metadata 包括：
- `retrieval_stage=coarse_to_rerank_v2`
- `lex_rank_score`
- `sem_rank_score`
- `query_overlap_score`
- `freshness_score`
- `source_diversity_penalty`
- `rerank_score`

## 2.2 Workflow 证据链补齐

文件：`backend/app/agents/workflow.py`

主要改动：

1. `state.evidence_pack` 增加：
- `retrieval_track`
- `metadata`
- `rerank_score`

2. `_build_citations` 输出增加：
- `retrieval_track`
- `rerank_score`

这样 `query/query_stream` 都能拿到一致的引用归因字段。

## 2.3 服务层 citation 统一规范与轨迹标识

文件：`backend/app/service.py`

主要改动：

1. 新增 `_normalize_citations_for_output`  
- 去重  
- 字段补齐（source_id/source_url/event_time/reliability/excerpt）  
- 补齐 `retrieval_track`（优先 citation，自 fallback 到 evidence metadata）  
- 保留 `rerank_score`

2. 新增 `_trace_source_identity`  
- 轨迹日志统一标识为 `source_id|retrieval_track`，便于定位“召回与选用不一致”。

3. `_record_rag_retrieval_trace` 改造  
- `retrieved_ids/selected_ids` 采用上述统一标识。

4. `_build_evidence_rich_answer` 追加的行情/历史引用补齐 `retrieval_track`，避免落入 `unknown_track`。

## 2.4 引用模型扩展

文件：`backend/app/models.py`

`Citation` 新增可选字段：
- `retrieval_track: str | None`
- `rerank_score: float | None`

保持向后兼容（可选字段）。

---

## 3. 测试改动

## 3.1 新增/增强测试

1. `tests/test_rag_retrieval.py`
- 新增 `HybridRetrieverV2` 用例，校验二阶段重排元数据是否输出完整。

2. `tests/test_service.py`
- `query_basic` 增加 `retrieval_track` 非空断言。  
- `query_persists_rag_qa_memory_and_trace` 增加 `selected_ids` 含 `source|track` 断言。

3. `tests/test_http_api.py`
- `test_query` 增加 API 返回 citations 含 `retrieval_track` 的断言。

## 3.2 自测命令与结果

1. RAG 检索单测  
```powershell
.\.venv\Scripts\python.exe -m pytest -q tests/test_rag_retrieval.py
```
结果：`3 passed`

2. 服务层回归  
```powershell
.\.venv\Scripts\python.exe -m pytest -q tests/test_service.py -k "query_basic or query_persists_rag_qa_memory_and_trace or datasource_ops_catalog_health_fetch_logs or analysis_intel_card_contract"
```
结果：`4 passed, 36 deselected`

3. API 回归  
```powershell
.\.venv\Scripts\python.exe -m pytest -q tests/test_http_api.py -k "query or datasource_management_endpoints or analysis_intel_card"
```
结果：`6 passed, 29 deselected`

---

## 4. 风险与后续

1. 当前重排仍是规则特征融合，后续可接入模型级 reranker 提升语义辨识。  
2. `source_id|track` 轨迹编码提高了排障可读性，但若未来要做统计看板建议拆成结构化列。  
3. 下一轮建议执行 ROUND-4，将 intel-card 与新归因字段在 DeepThink 页面中产品化展示。
