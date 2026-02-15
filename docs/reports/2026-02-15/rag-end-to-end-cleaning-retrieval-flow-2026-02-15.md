# StockPilotX RAG 完整流程说明（清洗、入库、粗排精排、回填）

## 1. 文档目标
- 说明当前 StockPilotX 中 RAG 的真实实现链路，不讲抽象方案，只讲代码里已落地的流程。
- 覆盖你关心的核心问题：
  - `pdf/doc/docx` 等附件如何清洗并入库。
  - 粗排和精排怎么做，评分怎么合并。
  - 除检索外，RAG 还包含哪些关键步骤（门禁、沉淀、观测、重建）。

## 2. 端到端总览（从上传到被检索）
1. 前端上传附件（`/v1/rag/workflow/upload-and-index` 或 `/v1/rag/uploads`）。
2. 后端解码、抽取文本、去重（`file_sha256`）、写 `doc_index`。
3. 执行文档索引：清洗文本、切 chunk、抽取结构（表格/代码）。
4. 将 chunk 持久化到 `rag_doc_chunk`（当前策略：上传后直接 `active`）。
5. 查询时构建运行时语料池（行情 + 公告 + 历史K线 + 文档chunk + QA记忆）。
6. 执行检索：
  - 词法粗排（BM25 + n-gram）
  - 语义召回（向量索引，summary-first）
  - 原文回填（origin backfill）
  - 统一精排（融合词法/语义/可靠度/时效）
7. 生成答案并写回共享记忆池（`rag_qa_memory`），供后续用户复用。
8. 记录检索轨迹和指标（`rag_retrieval_trace`、看板统计、可重建索引）。

---

## 3. 上传与文本清洗（重点：pdf/doc/docx）

### 3.1 上传入口与数据形态
- 入口：
  - `POST /v1/rag/uploads`
  - `POST /v1/rag/workflow/upload-and-index`
- 处理函数：`backend/app/service.py` -> `rag_upload_from_payload(...)`
- 支持两种输入：
  - 直接传 `content`（纯文本）
  - 传 `content_base64`（附件字节）

### 3.2 去重策略
- 对原始字节计算 `sha256`，在 `rag_upload_asset` 中查重。
- 默认命中同哈希则走 `deduplicated` 返回，不重复入库。
- 传 `force_reupload=true` 时可强制重传，生成后缀化 `doc_id`。

### 3.3 文本抽取策略（按后缀）
实现位置：`backend/app/service.py` -> `_extract_text_from_upload_bytes(...)`

1. `txt/md/csv/json/log/html/htm/ts/js/py`
- 走 `_decode_text_bytes`，按 `utf-8 -> gbk -> utf-16 -> latin1` 依次尝试解码。

2. `docx`
- 解压 zip，读取 `word/document.xml`，遍历 XML 文本节点拼接。
- 失败时回退通用字节解码。

3. `xlsx/xlsm`
- 使用 `openpyxl` 读首个工作表，按行列拼接（限制 `max_rows=1500`, `max_cols=32`）。
- 失败时回退通用字节解码。

4. `pdf`
- 优先 `pypdf` 提取页面文本。
- 若解析器不可用或提取失败，回退 ASCII 可见串抽取（正则抽取可打印段），保证最低可检索性。

5. 其他扩展名（包括 `.doc`）
- 当前没有 `.doc` 专用解析器分支，会走 `generic_decode` 通用解码路径。
- 这意味着 `.doc` 在复杂排版/二进制内容下抽取质量不稳定，建议业务上转为 `.docx` 或 `.pdf`。

### 3.4 入库前清洗与切块
有两层清洗逻辑：

1. 文档处理管线（`backend/app/docs/pipeline.py`）
- `_clean`：压缩连续空白、归并多余空行。
- `_split`：固定窗口切块（`chunk_size=900`, `overlap=120`）。
- `_extract_tables`：简单 `|` 表格行抽取。

2. RAG 持久化时脱敏（`backend/app/service.py` -> `_redact_text`）
- 对每个 chunk 生成 `chunk_text_redacted`：
  - 邮箱 -> `[REDACTED_EMAIL]`
  - 手机号 -> `[REDACTED_PHONE]`
  - 长证件号 -> `[REDACTED_ID]`

### 3.5 入库结构
- 文档元数据：`doc_index`
- 上传资产：`rag_upload_asset`
- 检索chunk：`rag_doc_chunk`
- 关键字段：
  - 原文：`chunk_text`
  - 脱敏文本：`chunk_text_redacted`
  - 状态：`effective_status`
  - 质量：`quality_score`
  - 标的关联：`stock_codes_json`

### 3.6 当前激活策略（你最近要求改完）
- 上传与索引后不再经过审核门禁，直接可检索：
  - `doc_index.needs_review = false`
  - `rag_doc_chunk.effective_status = active`

---

## 4. 查询期语料构建（RAG 不是只查文档）
实现：`backend/app/service.py` -> `_build_runtime_corpus(...)`

每次查询动态拼语料，主要包含：
1. 实时行情（quotes）
2. 公告（announcements）
3. 历史K线（history bars）
4. 最近三个月连续窗口摘要（防“样本稀疏误判”）
5. 内存态已索引文档 chunk（旧链路）
6. 持久化文档 chunk（`rag_doc_chunk`，只取 `active`）
7. QA共享记忆摘要（`rag_qa_memory` 且 `retrieval_enabled=1`）

每条语料统一成 `RetrievalItem`，带：
- `text/source_id/source_url`
- `event_time`（给时效打分）
- `reliability_score`（给可信度打分）
- `metadata`（track/chunk_id/memory_id 等）

---

## 5. 粗排与精排（核心）

## 5.1 粗排：词法召回（HybridRetriever）
实现：`backend/app/rag/retriever.py`

1. BM25 打分（词项相关度）
2. 字符 n-gram Jaccard 打分（近似语义）
3. 候选集合合并（BM25 topK + n-gram topK）
4. 粗排分数：
- `0.55 * bm25 + 0.35 * vector_ngram + 0.10 * reliability`

这一步的目标是“召回尽量全”，不是最终排序。

## 5.2 语义召回：summary-first
实现：
- 构建索引记录：`_build_summary_vector_records(...)`
- 向量检索：`LocalSummaryVectorStore.search(...)`

索引内容不是全量原文，而是“摘要优先”：
1. 文档 chunk：`summary_text = chunk_text_redacted`，`parent_text = chunk_text`
2. QA记忆：`summary_text = summary_text`，`parent_text = answer_redacted/answer_text`

向量后端：
1. 优先 FAISS（`IndexFlatIP` 内积）
2. 无 FAISS 时降级 JSON 向量 + Python 内积扫描
3. Embedding 优先远程 provider，失败可回退本地 `local_hash` 向量

## 5.3 原文回填：origin backfill
实现：`_semantic_summary_origin_hits(...)`

流程：
1. 先命中摘要（更稳、更短、噪音低）
2. 对每个摘要命中追加一条原文回填项（`origin_backfill=true`）
3. 回填项分数略降（`score * 0.92`）

这样可以兼顾：
- 召回稳定性（摘要）
- 可读证据完整性（原文）

## 5.4 精排：词法 + 语义统一融合（HybridRetrieverV2）
实现：`backend/app/rag/hybrid_retriever_v2.py`

融合输入：
1. 词法命中列表（lexical hits）
2. 语义命中列表（semantic + backfill hits）

融合特征：
- 词法排名得分 `lex_rank_score`
- 语义排名得分 `sem_rank_score`
- 时效得分 `freshness_score`（180天线性衰减）
- 数据可靠度 `reliability_score`

最终分数：
- `0.45 * lex + 0.35 * sem + 0.10 * reliability + 0.10 * freshness`

输出 topN 作为最终证据包。

---

## 6. QA知识沉淀（把答案变成下一轮语料）
实现：`_persist_query_knowledge_memory(...)`

每次 `query` / `query_stream` 完成后：
1. 计算质量分 `quality_score`
2. 命中门禁才开启复用：
  - 引用数 >= 2
  - 质量 >= 0.65
  - 无高风险标志（如 `missing_citation/compliance_block`）
3. 生成双轨内容：
  - `answer_text`（原文）
  - `answer_redacted`（脱敏）
  - `summary_text`（可检索摘要）
4. 落库 `rag_qa_memory`，默认 `share_scope=global`

这一步构成“问答飞轮”：越问越有可复用知识。

---

## 7. 观测、治理与运维动作

### 7.1 检索轨迹
- 表：`rag_retrieval_trace`
- 记录：
  - `retrieved_ids`（召回）
  - `selected_ids`（最终引用）
  - `latency_ms`

用于定位“召回了但没被答案使用”的问题。

### 7.2 上传资产治理
- 表：`rag_upload_asset`
- 记录上传来源、哈希、大小、标签、状态、解析备注（`parse_note`）。

### 7.3 看板摘要
- 接口：`GET /v1/rag/dashboard`
- 指标：
  - `doc_total`
  - `active_chunks`
  - `review_pending`
  - `qa_memory_total`
  - `retrieval_hit_rate_7d`
  - `last_reindex_at`

### 7.4 重建向量索引
- 接口：`POST /v1/ops/rag/reindex`
- 动作：
  - 重新抽取 summary records
  - rebuild 向量索引
  - 更新 `last_reindex_at`

---

## 8. 降级与容错（生产上最关键）
1. 文档解析降级：
- `pdf` 解析失败 -> ASCII fallback
- 未识别格式（含 `.doc`）-> generic decode

2. 向量检索降级：
- 无 FAISS -> JSON fallback
- 远程 embedding 失败 -> local_hash fallback（若配置允许）

3. 数据刷新降级：
- 查询前刷新行情/公告/历史失败不阻塞主问答，继续用已有语料。

4. 流式体验保障：
- `query_stream` 在模型首 token 前持续发 `model_wait` 心跳，避免前端“假死”。

---

## 9. 当前实现边界与后续建议
1. `.doc` 专用解析
- 当前 `.doc` 仅通用解码，建议增加 `antiword/libreoffice` 转换链路，统一转 `.docx` 后解析。

2. 质量治理
- 已有 `quality_score + retrieval_enabled` 门禁。
- 后续可引入“用户反馈闭环（thumb up/down）”反向调权。

3. 检索效果
- 已有 summary-first + origin-backfill。
- 后续可增加 cross-encoder reranker 做二次精排，提升长文档问答稳定性。

---

## 10. 关键代码索引（便于你快速跳转）
- 上传/解析/持久化：
  - `backend/app/service.py` (`rag_upload_from_payload`, `_extract_text_from_upload_bytes`, `_persist_doc_chunks_to_rag`)
- 文档基础处理：
  - `backend/app/docs/pipeline.py` (`process`, `_clean`, `_split`)
  - `backend/app/data/ingestion.py` (`upload_doc`, `index_doc`)
- 检索与排序：
  - `backend/app/rag/retriever.py` (`HybridRetriever`)
  - `backend/app/rag/hybrid_retriever_v2.py` (`HybridRetrieverV2`)
  - `backend/app/rag/vector_store.py` (`LocalSummaryVectorStore`)
  - `backend/app/rag/embedding_provider.py` (`EmbeddingProvider`)
- 运行时语料与记忆沉淀：
  - `backend/app/service.py` (`_build_runtime_corpus`, `_persist_query_knowledge_memory`, `_record_rag_retrieval_trace`)
- 持久层：
  - `backend/app/web/store.py`（RAG 表结构）
  - `backend/app/web/service.py`（RAG CRUD 与 dashboard）

