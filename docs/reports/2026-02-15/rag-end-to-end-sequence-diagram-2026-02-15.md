# StockPilotX RAG 时序图版说明（接口调用 + 入库顺序）

## 1. 文档目的
- 给研发/产品/运维一个“看图即懂”的版本。
- 重点回答三件事：
  - 附件上传后如何清洗并入库。
  - 查询时粗排/精排如何串起来。
  - 每一步落哪些表、依赖哪些状态字段。

## 2. 上传与索引时序（Upload -> Parse -> Index -> Active）

```mermaid
sequenceDiagram
    autonumber
    participant FE as Frontend
    participant API as FastAPI (/v1/rag/workflow/upload-and-index)
    participant SVC as AShareAgentService
    participant ING as IngestionService
    participant DOC as DocumentPipeline
    participant WEB as WebAppService
    participant DB as SQLite(web.db)

    FE->>API: POST /v1/rag/workflow/upload-and-index
    API->>SVC: rag_workflow_upload_and_index(payload)
    SVC->>SVC: rag_upload_from_payload(payload)
    SVC->>SVC: decode base64 / 解析文本(_extract_text_from_upload_bytes)
    SVC->>WEB: rag_upload_asset_get_by_hash(file_sha256)
    WEB->>DB: SELECT rag_upload_asset by hash
    alt 命中去重且非force_reupload
        SVC-->>API: deduplicated
        API-->>FE: status=deduplicated
    else 新上传或强制重传
        SVC->>ING: docs_upload(doc_id, filename, extracted, source)
        ING->>DOC: process(parse/clean/split)
        ING->>ING: store.docs[doc_id]=uploaded
        SVC->>WEB: doc_upsert(needs_review=false)
        WEB->>DB: UPSERT doc_index

        SVC->>ING: docs_index(doc_id)
        ING->>DOC: process(parse/clean/split)
        ING->>ING: store.docs[doc_id].indexed=true + chunks
        SVC->>WEB: rag_doc_chunk_replace(effective_status=active)
        WEB->>DB: DELETE/INSERT rag_doc_chunk by doc_id

        SVC->>WEB: rag_upload_asset_upsert(status=active/indexed)
        WEB->>DB: UPSERT rag_upload_asset
        SVC-->>API: status=ok + timeline
        API-->>FE: 上传并索引完成
    end
```

## 3. 文档清洗细节（按类型）

```mermaid
flowchart TD
    A[收到文件字节] --> B{文件后缀}
    B -->|txt/md/csv/json/log/html| C[多编码解码: utf-8/gbk/utf-16/latin1]
    B -->|docx| D[读取word/document.xml并抽取文本]
    B -->|xlsx/xlsm| E[openpyxl提取行列文本]
    B -->|pdf| F[pypdf提取页面文本]
    F -->|失败| G[ASCII可见串fallback]
    B -->|doc/其他| H[generic decode fallback]
    C --> I[DocumentPipeline.clean]
    D --> I
    E --> I
    G --> I
    H --> I
    I --> J[DocumentPipeline.split chunk_size=900 overlap=120]
    J --> K[脱敏_redact_text]
    K --> L[rag_doc_chunk: chunk_text + chunk_text_redacted]
```

说明：
- `.doc` 当前没有专用解析器，走通用解码降级路径。
- `pdf` 有 `pypdf` 则用结构化提取；无依赖/失败时最少保留可检索 ASCII 文本。

## 4. 查询链路时序（Retrieval + Rerank + Memory Writeback）

```mermaid
sequenceDiagram
    autonumber
    participant FE as Frontend
    participant API as FastAPI (/v1/query or /v1/query/stream)
    participant SVC as AShareAgentService
    participant RET as Retriever(HybridRetrieverV2)
    participant VEC as LocalSummaryVectorStore
    participant WEB as WebAppService
    participant DB as SQLite(web.db)

    FE->>API: POST /v1/query(/stream)
    API->>SVC: query(payload)
    SVC->>SVC: 数据刷新(行情/公告/历史)可降级
    SVC->>SVC: _build_runtime_corpus()
    SVC->>SVC: _refresh_summary_vector_index()
    SVC->>RET: retrieve(query)
    RET->>RET: 词法粗排(BM25 + n-gram)
    RET->>VEC: search(summary vectors)
    VEC-->>RET: 语义召回(summary hits)
    RET->>RET: origin backfill(补parent_text)
    RET->>RET: 融合精排(lex+sem+reliability+freshness)
    RET-->>SVC: evidence_pack topN
    SVC->>SVC: 生成答案 + citations
    SVC->>WEB: rag_retrieval_trace_add(retrieved_ids, selected_ids)
    WEB->>DB: INSERT rag_retrieval_trace
    SVC->>WEB: rag_qa_memory_add(global + retrieval_enabled gate)
    WEB->>DB: UPSERT rag_qa_memory
    API-->>FE: answer / stream events(含knowledge_persisted)
```

## 5. 粗排与精排公式（实现口径）

## 5.1 粗排（词法）
- 组件：`HybridRetriever`
- 候选：
  - BM25 topK
  - n-gram 相似度 topK
- 粗排分：
  - `score = 0.55*bm25 + 0.35*vector_ngram + 0.10*reliability`

## 5.2 精排（融合）
- 组件：`HybridRetrieverV2`
- 融合特征：
  - `lex_rank_score`
  - `sem_rank_score`
  - `freshness_score`
  - `reliability_score`
- 精排分：
  - `final = 0.45*lex + 0.35*sem + 0.10*reliability + 0.10*freshness`

## 5.3 summary-first + origin-backfill
- 先用摘要做语义召回，降低噪声。
- 再补原文（`origin_backfill=true`）保证可读证据完整。
- 回填分数轻微折扣（`*0.92`）避免原文挤占摘要排序。

## 6. 关键表写入顺序（Checklist）
1. `doc_index`：上传/索引后的文档元信息（当前 `needs_review=false`）。
2. `rag_doc_chunk`：可检索文档分块（当前默认 `effective_status=active`）。
3. `rag_upload_asset`：上传资产与去重信息（hash、tags、parse_note、status）。
4. `rag_retrieval_trace`：每次查询的召回/入选轨迹与时延。
5. `rag_qa_memory`：答案沉淀（raw + redacted + summary + 质量门禁）。
6. `rag_ops_meta`：如 `last_reindex_at` 运维元数据。

## 7. 状态机（业务视角）

```mermaid
stateDiagram-v2
    [*] --> Uploaded
    Uploaded --> Indexed: docs_index成功
    Indexed --> Active: chunk持久化(effective_status=active)
    Active --> Archived: ops手工治理
    Active --> Rejected: 审计/治理下线
    Archived --> Active: 重新激活
```

备注：
- 当前产品策略是“上传即生效”，即 `Indexed` 后直接进入 `Active`。
- 仍保留运维治理能力（按 chunk/doc 调整状态）。

## 8. 代码入口速查
- 上传解析链路：`backend/app/service.py`
  - `rag_upload_from_payload`
  - `_extract_text_from_upload_bytes`
  - `_persist_doc_chunks_to_rag`
- 文档清洗：`backend/app/docs/pipeline.py`
- 查询检索：`backend/app/service.py`
  - `_build_runtime_corpus`
  - `_build_runtime_retriever`
  - `_semantic_summary_origin_hits`
- 粗排：`backend/app/rag/retriever.py`
- 精排：`backend/app/rag/hybrid_retriever_v2.py`
- 向量索引：`backend/app/rag/vector_store.py`
- 持久层：`backend/app/web/store.py`、`backend/app/web/service.py`

