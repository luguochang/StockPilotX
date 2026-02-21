from __future__ import annotations

from .shared import *

class RagMixin:
    def docs_upload(self, doc_id: str, filename: str, content: str, source: str) -> dict[str, Any]:
        """Upload raw document content."""
        result = self.ingestion.upload_doc(doc_id, filename, content, source)
        doc = self.ingestion.store.docs.get(doc_id, {})
        self.web.doc_upsert(
            doc_id=doc_id,
            filename=filename,
            parse_confidence=float(doc.get("parse_confidence", 0.0)),
            # Product requirement: upload should be effective immediately.
            # Keep parse_confidence for observability, but do not block by review gate.
            needs_review=False,
        )
        self.web.doc_pipeline_run_add(
            doc_id=doc_id,
            stage="upload",
            status="ok",
            filename=filename,
            parse_confidence=float(doc.get("parse_confidence", 0.0) or 0.0),
            chunk_count=0,
            table_count=0,
            parse_notes="upload_received",
            metadata={
                "source": source,
                "doc_hash": str(doc.get("doc_hash", "")),
                "pipeline_version": int(doc.get("version", 1) or 1),
            },
        )
        return result

    def docs_index(self, doc_id: str) -> dict[str, Any]:
        """Index one uploaded document into chunks and write to retrieval stores."""
        result = self.ingestion.index_doc(doc_id)
        doc = self.ingestion.store.docs.get(doc_id, {})
        if str(result.get("status", "")) != "indexed":
            self.web.doc_pipeline_run_add(
                doc_id=doc_id,
                stage="index",
                status="not_found",
                filename=str(doc.get("filename", "")),
                parse_confidence=float(doc.get("parse_confidence", 0.0) or 0.0),
                parse_notes="doc_not_found",
                metadata={"result_status": str(result.get("status", ""))},
            )
            return result
        if doc:
            self.web.doc_upsert(
                doc_id=doc_id,
                filename=doc.get("filename", ""),
                parse_confidence=float(doc.get("parse_confidence", 0.0)),
                # Product requirement: index completion should not create review gate.
                needs_review=False,
            )
            # 鏂囨。绱㈠紩瀹屾垚鍚庯紝鎶?chunk 鎸佷箙鍖栧埌 RAG 璧勪骇搴擄紝渚涘悗缁绱笌娌荤悊澶嶇敤銆?
            self._persist_doc_chunks_to_rag(doc_id, doc)
            self.web.doc_pipeline_run_add(
                doc_id=doc_id,
                stage="index",
                status="ok",
                filename=str(doc.get("filename", "")),
                parse_confidence=float(doc.get("parse_confidence", 0.0) or 0.0),
                chunk_count=int(result.get("chunk_count", 0) or 0),
                table_count=int(result.get("table_count", 0) or 0),
                parse_notes="index_completed",
                metadata={
                    "source": str(doc.get("source", "")),
                    "doc_hash": str(doc.get("doc_hash", "")),
                    "pipeline_version": int(doc.get("version", 1) or 1),
                },
            )
        return result

    def docs_versions(self, token: str, doc_id: str, *, limit: int = 20) -> list[dict[str, Any]]:
        return self.web.doc_versions(token, doc_id, limit=limit)

    def docs_pipeline_runs(self, token: str, doc_id: str, *, limit: int = 30) -> list[dict[str, Any]]:
        return self.web.doc_pipeline_runs(token, doc_id, limit=limit)

    def docs_recommend(self, token: str, payload: dict[str, Any]) -> dict[str, Any]:
        """Recommend docs using history + context + graph signals."""
        _ = self.web.auth_me(token)
        context = payload if isinstance(payload, dict) else {}
        safe_top_k = max(1, min(30, int(context.get("top_k", 5) or 5)))
        stock_code = str(context.get("stock_code", "")).strip().upper()

        history_rows = self.web.query_history_list(token, limit=120)
        chunks = self.web.rag_doc_chunk_list_internal(status="active", limit=2500)
        graph_terms: list[str] = []
        if stock_code:
            graph = self.knowledge_graph_view(stock_code, limit=30)
            # Pull neighbor concept words as ranking hints.
            graph_terms = [
                str(x.get("target", "")).strip().upper()
                for x in graph.get("relations", [])
                if str(x.get("target", "")).strip()
            ][:60]

        candidate_top_k = max(60, safe_top_k * 12)
        ranked = self.doc_recommender.recommend(
            chunks=chunks,
            query_history_rows=history_rows,
            context=context,
            graph_terms=graph_terms,
            top_k=candidate_top_k,
        )
        if stock_code:
            # Stabilize recommendation ordering under a large shared corpus:
            # prioritize docs explicitly tagged with the requested stock.
            matched = [x for x in ranked if stock_code in [str(s).upper() for s in list(x.get("stock_codes", []))]]
            unmatched = [x for x in ranked if x not in matched]
            ranked = (matched + unmatched)[:safe_top_k]

        items: list[dict[str, Any]] = []
        for row in ranked:
            doc_meta = self.web.store.query_one(
                """
                SELECT doc_id, filename, status, parse_confidence, created_at
                FROM doc_index
                WHERE doc_id = ?
                """,
                (str(row.get("doc_id", "")),),
            ) or {}
            items.append(
                {
                    "doc_id": row.get("doc_id", ""),
                    "filename": str(doc_meta.get("filename", "")),
                    "status": str(doc_meta.get("status", "")),
                    "parse_confidence": float(doc_meta.get("parse_confidence", 0.0) or 0.0),
                    "score": float(row.get("score", 0.0) or 0.0),
                    "reasons": list(row.get("reasons", [])),
                    "stock_codes": list(row.get("stock_codes", [])),
                    "source": str(row.get("source", "")),
                    "updated_at": str(row.get("updated_at", "")),
                    "created_at": str(doc_meta.get("created_at", "")),
                }
            )

        return {
            "top_k": safe_top_k,
            "count": len(items),
            "context": {
                "stock_code": stock_code,
                "question": str(context.get("question", "")),
            },
            "items": items,
        }

    def knowledge_graph_view(self, entity_id: str, *, limit: int = 20) -> dict[str, Any]:
        """Return one-hop graph neighborhood for a given entity."""
        normalized = str(entity_id or "").strip().upper()
        if not normalized:
            raise ValueError("entity_id is required")
        safe_limit = max(1, min(200, int(limit)))
        raw_relations = self.workflow.graph_rag.store.find_relations([], limit=safe_limit * 3)

        rows: list[dict[str, str]] = []
        for row in raw_relations:
            src = str(getattr(row, "src", "") or "")
            dst = str(getattr(row, "dst", "") or "")
            rel_type = str(getattr(row, "rel_type", "") or "")
            source_id = str(getattr(row, "source_id", "") or "graph")
            source_url = str(getattr(row, "source_url", "") or "")
            if normalized not in (src.upper(), dst.upper()):
                continue
            rows.append(
                {
                    "source": src,
                    "target": dst,
                    "relation_type": rel_type,
                    "source_id": source_id,
                    "source_url": source_url,
                }
            )
            if len(rows) >= safe_limit:
                break

        node_index: dict[str, dict[str, Any]] = {}
        for relation in rows:
            src = str(relation.get("source", ""))
            dst = str(relation.get("target", ""))
            if src and src not in node_index:
                node_index[src] = {"entity_id": src, "entity_type": self._infer_graph_entity_type(src)}
            if dst and dst not in node_index:
                node_index[dst] = {"entity_id": dst, "entity_type": self._infer_graph_entity_type(dst)}

        return {
            "entity_id": normalized,
            "entity_type": self._infer_graph_entity_type(normalized),
            "node_count": len(node_index),
            "relation_count": len(rows),
            "nodes": list(node_index.values()),
            "relations": rows,
        }

    @staticmethod
    def _looks_like_pdf_binary_stream_text(text: str) -> bool:
        normalized = str(text or "")
        if not normalized:
            return False
        markers = [
            "Filter/FlateDecode",
            "endstream",
            "stream",
            "/Length",
            "Subtype/Type1C",
            "obj",
            "endobj",
        ]
        hit_count = sum(normalized.count(marker) for marker in markers)
        return hit_count >= 8 or (hit_count >= 4 and len(normalized) >= 300)

    def docs_quality_report(self, doc_id: str) -> dict[str, Any]:
        """Build a quality report for one indexed document.

        This is a lightweight quality dashboard used by Knowledge Hub Phase 1:
        - Parse confidence from `doc_index`
        - Chunk distribution from `rag_doc_chunk`
        - Actionable recommendations for low quality inputs
        """
        doc = self.web.store.query_one(
            """
            SELECT doc_id, filename, status, parse_confidence, needs_review, created_at
            FROM doc_index
            WHERE doc_id = ?
            """,
            (doc_id,),
        )
        if not doc:
            return {"error": "not_found", "doc_id": doc_id}

        chunks = self.web.store.query_all(
            """
            SELECT chunk_id, chunk_no, chunk_text_redacted, quality_score, effective_status, updated_at
            FROM rag_doc_chunk
            WHERE doc_id = ?
            ORDER BY chunk_no ASC
            """,
            (doc_id,),
        )
        chunk_lengths = [len(str(row.get("chunk_text_redacted", ""))) for row in chunks]
        chunk_count = len(chunks)
        avg_chunk_len = (sum(chunk_lengths) / chunk_count) if chunk_count else 0.0
        short_chunk_count = sum(1 for x in chunk_lengths if x < 60)
        short_chunk_ratio = (short_chunk_count / chunk_count) if chunk_count else 0.0
        avg_quality = (
            sum(float(row.get("quality_score", 0.0) or 0.0) for row in chunks) / chunk_count if chunk_count else 0.0
        )
        active_chunk_count = sum(1 for row in chunks if str(row.get("effective_status", "")) == "active")

        parse_confidence = float(doc.get("parse_confidence", 0.0) or 0.0)
        quality_score = max(
            0.0,
            min(
                1.0,
                parse_confidence * 0.5 + avg_quality * 0.3 + (1.0 - short_chunk_ratio) * 0.2,
            ),
        )
        quality_level = "high" if quality_score >= 0.8 else ("medium" if quality_score >= 0.6 else "low")

        recommendations: list[str] = []
        if parse_confidence < 0.7:
            recommendations.append("parse_confidence ????????????? OCR ????????")
        if chunk_count < 3:
            recommendations.append("??????????????????")
        if short_chunk_ratio > 0.4:
            recommendations.append("????????????????????????")
        if active_chunk_count == 0 and chunk_count > 0:
            recommendations.append("??? active ????????????")
        if not recommendations:
            recommendations.append("??????????????????")

        return {
            "doc_id": doc_id,
            "filename": str(doc.get("filename", "")),
            "status": str(doc.get("status", "")),
            "parse_confidence": round(parse_confidence, 4),
            "needs_review": bool(int(doc.get("needs_review", 0) or 0)),
            "quality_score": round(quality_score, 4),
            "quality_level": quality_level,
            "chunk_stats": {
                "chunk_count": chunk_count,
                "active_chunk_count": active_chunk_count,
                "avg_chunk_len": round(avg_chunk_len, 2),
                "short_chunk_count": short_chunk_count,
                "short_chunk_ratio": round(short_chunk_ratio, 4),
                "avg_chunk_quality": round(avg_quality, 4),
            },
            "recommendations": recommendations,
            "created_at": str(doc.get("created_at", "")),
        }

    def rag_upload_from_payload(self, token: str, payload: dict[str, Any]) -> dict[str, Any]:
        """Unified RAG upload flow: parse -> de-duplicate -> docs_upload -> optional index -> persist asset."""
        filename = str(payload.get("filename", "")).strip()
        if not filename:
            raise ValueError("filename is required")
        source = str(payload.get("source", "user_upload")).strip().lower() or "user_upload"
        source_url = str(payload.get("source_url", "")).strip()
        content_type = str(payload.get("content_type", "")).strip()
        stock_codes = [str(x).strip().upper() for x in payload.get("stock_codes", []) if str(x).strip()]
        tags = [str(x).strip() for x in payload.get("tags", []) if str(x).strip()]
        auto_index = bool(payload.get("auto_index", True))
        force_reupload = bool(payload.get("force_reupload", False))
        user_hint = str(payload.get("user_id", "frontend-user")).strip() or "frontend-user"

        text_content = str(payload.get("content", "") or "")
        raw_bytes: bytes
        if text_content.strip():
            raw_bytes = text_content.encode("utf-8", errors="ignore")
        else:
            encoded = str(payload.get("content_base64", "")).strip()
            if not encoded:
                raise ValueError("content or content_base64 is required")
            try:
                raw_bytes = base64.b64decode(encoded, validate=True)
            except Exception as ex:  # noqa: BLE001
                raise ValueError("content_base64 is invalid") from ex

        file_sha256 = hashlib.sha256(raw_bytes).hexdigest().lower()
        doc_id = str(payload.get("doc_id", "")).strip()
        if not doc_id:
            # Stable doc id for the same file hash so status and retrieval trace are easy to correlate.
            doc_id = f"ragdoc-{file_sha256[:12]}"
            if force_reupload:
                doc_id = f"{doc_id}-{uuid.uuid4().hex[:4]}"
        upload_id = str(payload.get("upload_id", "")).strip() or f"ragu-{uuid.uuid4().hex[:12]}"

        if not force_reupload:
            existing = self.web.rag_upload_asset_get_by_hash(file_sha256)
            if existing:
                return {
                    "status": "deduplicated",
                    "dedupe_hit": True,
                    "existing": existing,
                    "doc_id": str(existing.get("doc_id", "")),
                    "upload_id": str(existing.get("upload_id", "")),
                }

        self.web.rag_upload_asset_upsert(
            token,
            upload_id=upload_id,
            doc_id=doc_id,
            filename=filename,
            source=source,
            source_url=source_url,
            file_sha256=file_sha256,
            file_size=len(raw_bytes),
            content_type=content_type,
            stock_codes=stock_codes,
            tags=tags,
            parse_note="upload_received",
            status="uploaded",
            job_status="running",
            current_stage="upload_received",
            created_by=user_hint,
        )
        self.web.rag_upload_stage_log_add(
            upload_id=upload_id,
            doc_id=doc_id,
            stage="upload_received",
            status="done",
            detail={
                "filename": filename,
                "content_type": content_type,
                "file_size": len(raw_bytes),
                "source": source,
            },
        )
        try:
            if text_content.strip():
                extracted = text_content
                parse_note = "text_payload"
                normalized = re.sub(r"\s+", " ", extracted).strip()
                coverage = min(1.0, len(normalized) / 3000.0)
                parse_trace = {
                    "parser_name": "text_payload",
                    "parser_version": "1",
                    "ocr_used": False,
                    "duration_ms": 0,
                    "notes": ["text_payload"],
                }
                parse_quality = {
                    "text_coverage_ratio": coverage,
                    "garbled_ratio": 0.0,
                    "ocr_confidence_avg": 1.0,
                    "quality_score": max(0.0, min(1.0, coverage * 0.7 + 0.3 if normalized else 0.0)),
                }
            else:
                parsed = self.document_parser.parse(
                    filename=filename,
                    raw_bytes=raw_bytes,
                    content_type=content_type,
                )
                extracted = str(parsed.plain_text or "")
                parse_note = str(parsed.parse_note or "")
                parse_trace = parsed.trace.to_dict()
                parse_quality = parsed.quality.to_dict()
            parse_error = self._validate_rag_parsed_text(
                extracted_text=extracted,
                parse_note=parse_note,
                parse_quality=parse_quality,
            )
            if parse_error:
                raise ValueError(parse_error)

            self.web.rag_upload_asset_set_runtime(
                upload_id=upload_id,
                current_stage="parsing",
                parser_name=str(parse_trace.get("parser_name", "")),
                ocr_used=bool(parse_trace.get("ocr_used", False)),
                quality_score=float(parse_quality.get("quality_score", 0.0) or 0.0),
                parse_note=parse_note,
            )
            self.web.rag_upload_stage_log_add(
                upload_id=upload_id,
                doc_id=doc_id,
                stage="parsing",
                status="done",
                detail={
                    "parse_note": parse_note,
                    "trace": parse_trace,
                    "quality": parse_quality,
                },
            )

            _ = self.docs_upload(doc_id, filename, extracted, source)
            self.web.rag_upload_stage_log_add(
                upload_id=upload_id,
                doc_id=doc_id,
                stage="asset_uploaded",
                status="done",
                detail={"doc_id": doc_id, "source": source},
            )

            indexed = {}
            status = "uploaded"
            vector_ready = False
            if auto_index:
                indexed = self.docs_index(doc_id)
                index_status = str(indexed.get("status", "indexed"))
                self.web.rag_upload_stage_log_add(
                    upload_id=upload_id,
                    doc_id=doc_id,
                    stage="indexing",
                    status="done" if index_status == "indexed" else "failed",
                    detail={"index_result": indexed},
                )
                status = "indexed"
                chunk_rows = self.rag_doc_chunks_list(token, doc_id=doc_id, limit=1)
                if chunk_rows:
                    status = str(chunk_rows[0].get("effective_status", "indexed"))
                    vector_ready = status == "active"
            else:
                self.web.rag_upload_stage_log_add(
                    upload_id=upload_id,
                    doc_id=doc_id,
                    stage="indexing",
                    status="skipped",
                    detail={"reason": "auto_index_disabled"},
                )

            asset = self.web.rag_upload_asset_upsert(
                token,
                upload_id=upload_id,
                doc_id=doc_id,
                filename=filename,
                source=source,
                source_url=source_url,
                file_sha256=file_sha256,
                file_size=len(raw_bytes),
                content_type=content_type,
                stock_codes=stock_codes,
                tags=tags,
                parse_note=parse_note,
                status=status,
                job_status="completed",
                current_stage="asset_recorded",
                error_code="",
                error_message="",
                parser_name=str(parse_trace.get("parser_name", "")),
                ocr_used=bool(parse_trace.get("ocr_used", False)),
                quality_score=float(parse_quality.get("quality_score", 0.0) or 0.0),
                vector_ready=vector_ready,
                verification_passed=False,
                created_by=user_hint,
            )
            self.web.rag_upload_stage_log_add(
                upload_id=upload_id,
                doc_id=doc_id,
                stage="asset_recorded",
                status="done",
                detail={"status": status, "vector_ready": vector_ready},
            )
            return {
                "status": "ok",
                "dedupe_hit": False,
                "upload_id": upload_id,
                "doc_id": doc_id,
                "source": source,
                "auto_index": auto_index,
                "index_result": indexed,
                "asset": asset,
                "parse_note": parse_note,
                "parse_trace": parse_trace,
                "parse_quality": parse_quality,
            }
        except Exception as ex:  # noqa: BLE001
            self.web.rag_upload_asset_set_runtime(
                upload_id=upload_id,
                job_status="failed",
                current_stage="failed",
                error_code="upload_failed",
                error_message=str(ex)[:240],
            )
            self.web.rag_upload_stage_log_add(
                upload_id=upload_id,
                doc_id=doc_id,
                stage="failed",
                status="failed",
                detail={"error": str(ex)[:240]},
            )
            raise

    def rag_retrieval_preview(
        self,
        token: str,
        *,
        doc_id: str,
        max_queries: int = 3,
        top_k: int = 5,
        hint_stock_codes: list[str] | None = None,
        hint_tags: list[str] | None = None,
    ) -> dict[str, Any]:
        """Build retrieval verification preview for one uploaded document."""
        _ = self.web.require_role(token, {"admin", "ops"})
        doc_id_clean = str(doc_id or "").strip()
        if not doc_id_clean:
            raise ValueError("doc_id is required")
        safe_max_queries = max(1, min(6, int(max_queries)))
        safe_top_k = max(1, min(8, int(top_k)))

        all_chunks = self.web.rag_doc_chunk_list_internal(doc_id=doc_id_clean, limit=180)
        if not all_chunks:
            return {
                "doc_id": doc_id_clean,
                "ready": False,
                "passed": False,
                "reason": "doc_not_found",
                "total_chunk_count": 0,
                "active_chunk_count": 0,
                "query_count": 0,
                "matched_query_count": 0,
                "target_hit_rate": 0.0,
                "items": [],
            }

        active_chunks = [row for row in all_chunks if str(row.get("effective_status", "")).strip().lower() == "active"]
        if not active_chunks:
            return {
                "doc_id": doc_id_clean,
                "ready": False,
                "passed": False,
                "reason": "doc_not_active",
                "total_chunk_count": len(all_chunks),
                "active_chunk_count": 0,
                "query_count": 0,
                "matched_query_count": 0,
                "target_hit_rate": 0.0,
                "items": [],
            }

        stock_codes: list[str] = []
        for raw in hint_stock_codes or []:
            code = str(raw or "").strip().upper()
            if code and code not in stock_codes:
                stock_codes.append(code)
        for row in active_chunks:
            for raw in row.get("stock_codes", []) or []:
                code = str(raw or "").strip().upper()
                if code and code not in stock_codes:
                    stock_codes.append(code)

        tags: list[str] = []
        for raw in hint_tags or []:
            tag = str(raw or "").strip()
            if tag and tag not in tags:
                tags.append(tag)

        preview_queries = self._build_rag_preview_queries(
            chunks=active_chunks,
            stock_codes=stock_codes,
            tags=tags,
            max_queries=safe_max_queries,
        )
        retriever = self._build_runtime_retriever(stock_codes)

        items: list[dict[str, Any]] = []
        matched_queries = 0
        for query in preview_queries:
            started = time.perf_counter()
            hits = retriever.retrieve(
                query,
                top_k_vector=max(8, safe_top_k * 2),
                top_k_bm25=max(12, safe_top_k * 2),
                rerank_top_n=safe_top_k,
            )
            latency_ms = int((time.perf_counter() - started) * 1000)
            hit_rows: list[dict[str, Any]] = []
            target_hit_rank: int | None = None
            for rank, hit in enumerate(hits, start=1):
                meta = dict(hit.metadata or {})
                hit_doc_id = str(meta.get("doc_id", "")).strip()
                hit_source_url = str(hit.source_url or "")
                # Some lexical retrieval paths may not preserve doc_id metadata, so keep source_url fallback.
                is_target = bool(
                    (hit_doc_id and hit_doc_id == doc_id_clean)
                    or (f"/docs/{doc_id_clean}" in hit_source_url)
                    or hit_source_url.endswith(f"://docs/{doc_id_clean}")
                )
                if is_target and target_hit_rank is None:
                    target_hit_rank = rank
                hit_rows.append(
                    {
                        "rank": rank,
                        "score": round(float(hit.score or 0.0), 4),
                        "source_id": str(hit.source_id or ""),
                        "source_url": hit_source_url,
                        "retrieval_track": str(meta.get("retrieval_track", "")),
                        "doc_id": hit_doc_id,
                        "chunk_id": str(meta.get("chunk_id", "")),
                        "is_target_doc": is_target,
                        "excerpt": self._rag_preview_trim(str(hit.text or ""), max_len=120),
                    }
                )
            if target_hit_rank is not None:
                matched_queries += 1
            items.append(
                {
                    "query": query,
                    "latency_ms": latency_ms,
                    "target_hit": target_hit_rank is not None,
                    "target_hit_rank": target_hit_rank,
                    "top_hits": hit_rows,
                }
            )

        query_count = len(items)
        hit_rate = round((matched_queries / query_count), 4) if query_count > 0 else 0.0
        return {
            "doc_id": doc_id_clean,
            "ready": True,
            "passed": matched_queries > 0,
            "reason": "",
            "total_chunk_count": len(all_chunks),
            "active_chunk_count": len(active_chunks),
            "query_count": query_count,
            "matched_query_count": matched_queries,
            "target_hit_rate": hit_rate,
            "items": items,
        }

    def rag_workflow_upload_and_index(self, token: str, payload: dict[str, Any]) -> dict[str, Any]:
        """Business entrypoint: upload then index immediately and return timeline for frontend progress."""
        req = dict(payload or {})
        req["auto_index"] = True
        result = self.rag_upload_from_payload(token, req)
        upload_id = str(result.get("upload_id", "")).strip()
        timeline: list[dict[str, Any]] = []
        if upload_id:
            stage_rows = self.web.rag_upload_stage_log_list(token, upload_id=upload_id, limit=120)
            timeline = [
                {
                    "phase": str(row.get("stage", "")),
                    "status": str(row.get("status", "")),
                    "at": str(row.get("created_at", "")),
                    "detail": row.get("detail", {}),
                }
                for row in stage_rows
            ]
        preview_doc_id = str(result.get("doc_id", "")).strip()
        retrieval_preview: dict[str, Any] = {
            "doc_id": preview_doc_id,
            "ready": False,
            "passed": False,
            "reason": "doc_id_missing",
            "query_count": 0,
            "matched_query_count": 0,
            "target_hit_rate": 0.0,
            "items": [],
        }
        if preview_doc_id and bool(result.get("dedupe_hit", False)) and upload_id:
            retrieval_preview = self.rag_upload_verification(token, upload_id=upload_id)
        elif preview_doc_id:
            try:
                asset = result.get("asset", {}) if isinstance(result.get("asset"), dict) else {}
                hint_codes = [str(x).strip().upper() for x in asset.get("stock_codes", []) if str(x).strip()]
                hint_tags = [str(x).strip() for x in asset.get("tags", []) if str(x).strip()]
                if upload_id:
                    self.web.rag_upload_stage_log_add(
                        upload_id=upload_id,
                        doc_id=preview_doc_id,
                        stage="verifying",
                        status="running",
                        detail={"max_queries": 2, "top_k": 4},
                    )
                retrieval_preview = self.rag_retrieval_preview(
                    token,
                    doc_id=preview_doc_id,
                    max_queries=2,
                    top_k=4,
                    hint_stock_codes=hint_codes,
                    hint_tags=hint_tags,
                )
                if upload_id:
                    self.web.rag_upload_verification_replace(
                        upload_id=upload_id,
                        doc_id=preview_doc_id,
                        items=list(retrieval_preview.get("items", [])),
                    )
                    self.web.rag_upload_asset_set_runtime(
                        upload_id=upload_id,
                        job_status="completed",
                        current_stage="completed",
                        verification_passed=bool(retrieval_preview.get("passed", False)),
                        vector_ready=bool(retrieval_preview.get("ready", False)),
                        error_code="",
                        error_message="",
                    )
                    self.web.rag_upload_stage_log_add(
                        upload_id=upload_id,
                        doc_id=preview_doc_id,
                        stage="verifying",
                        status="done",
                        detail={
                            "query_count": int(retrieval_preview.get("query_count", 0) or 0),
                            "matched_query_count": int(retrieval_preview.get("matched_query_count", 0) or 0),
                            "target_hit_rate": float(retrieval_preview.get("target_hit_rate", 0.0) or 0.0),
                        },
                    )
            except Exception as ex:  # noqa: BLE001
                retrieval_preview = {
                    "doc_id": preview_doc_id,
                    "ready": False,
                    "passed": False,
                    "reason": "preview_failed",
                    "error": str(ex)[:240],
                    "query_count": 0,
                    "matched_query_count": 0,
                    "target_hit_rate": 0.0,
                    "items": [],
                }
                if upload_id:
                    self.web.rag_upload_asset_set_runtime(
                        upload_id=upload_id,
                        job_status="partial",
                        current_stage="verifying",
                        error_code="preview_failed",
                        error_message=str(ex)[:240],
                        verification_passed=False,
                    )
                    self.web.rag_upload_stage_log_add(
                        upload_id=upload_id,
                        doc_id=preview_doc_id,
                        stage="verifying",
                        status="failed",
                        detail={"error": str(ex)[:240]},
                    )
        if upload_id:
            stage_rows = self.web.rag_upload_stage_log_list(token, upload_id=upload_id, limit=120)
            timeline = [
                {
                    "phase": str(row.get("stage", "")),
                    "status": str(row.get("status", "")),
                    "at": str(row.get("created_at", "")),
                    "detail": row.get("detail", {}),
                }
                for row in stage_rows
            ]
        return {
            "status": "ok",
            "result": result,
            "timeline": timeline,
            "retrieval_preview": retrieval_preview,
        }

    def rag_retrieval_preview_api(
        self,
        token: str,
        *,
        doc_id: str,
        max_queries: int = 3,
        top_k: int = 5,
    ) -> dict[str, Any]:
        """Public API wrapper for upload-time retrieval preview."""
        return self.rag_retrieval_preview(
            token,
            doc_id=doc_id,
            max_queries=max_queries,
            top_k=top_k,
        )

    def rag_uploads_list(
        self,
        token: str,
        *,
        status: str = "",
        source: str = "",
        limit: int = 40,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        return self.web.rag_upload_asset_list(
            token,
            status=status,
            source=source,
            limit=limit,
            offset=offset,
        )

    def rag_upload_status(self, token: str, *, upload_id: str) -> dict[str, Any]:
        asset = self.web.rag_upload_asset_get(token, upload_id=upload_id)
        if "error" in asset:
            return asset
        stage_rows = self.web.rag_upload_stage_log_list(token, upload_id=upload_id, limit=240)
        verification_rows = self.web.rag_upload_verification_list(token, upload_id=upload_id, limit=240)
        query_count = len(verification_rows)
        matched_query_count = sum(1 for row in verification_rows if bool(row.get("target_hit", False)))
        target_hit_rate = round((matched_query_count / query_count), 4) if query_count > 0 else 0.0
        return {
            "upload_id": str(upload_id),
            "doc_id": str(asset.get("doc_id", "")),
            "asset": asset,
            "timeline": [
                {
                    "phase": str(row.get("stage", "")),
                    "status": str(row.get("status", "")),
                    "at": str(row.get("created_at", "")),
                    "detail": row.get("detail", {}),
                }
                for row in stage_rows
            ],
            "verification_summary": {
                "query_count": query_count,
                "matched_query_count": matched_query_count,
                "target_hit_rate": target_hit_rate,
            },
        }

    def rag_upload_verification(self, token: str, *, upload_id: str) -> dict[str, Any]:
        asset = self.web.rag_upload_asset_get(token, upload_id=upload_id)
        if "error" in asset:
            return asset
        rows = self.web.rag_upload_verification_list(token, upload_id=upload_id, limit=240)
        query_count = len(rows)
        matched_query_count = sum(1 for row in rows if bool(row.get("target_hit", False)))
        target_hit_rate = round((matched_query_count / query_count), 4) if query_count > 0 else 0.0
        return {
            "upload_id": str(upload_id),
            "doc_id": str(asset.get("doc_id", "")),
            "ready": bool(asset.get("vector_ready", False)),
            "passed": bool(asset.get("verification_passed", False) or matched_query_count > 0),
            "reason": "",
            "query_count": query_count,
            "matched_query_count": matched_query_count,
            "target_hit_rate": target_hit_rate,
            "items": [
                {
                    "query": str(row.get("query_text", "")),
                    "latency_ms": int(row.get("latency_ms", 0) or 0),
                    "target_hit": bool(row.get("target_hit", False)),
                    "target_hit_rank": (
                        int(row.get("target_hit_rank")) if row.get("target_hit_rank") is not None else None
                    ),
                    "top_hits": list(row.get("top_hits", [])),
                }
                for row in rows
            ],
        }

    def rag_upload_delete(self, token: str, *, upload_id: str) -> dict[str, Any]:
        deleted = self.web.rag_upload_asset_delete_hard(token, upload_id=upload_id)
        if "error" in deleted:
            return deleted
        doc_id = str(deleted.get("doc_id", "")).strip()
        if doc_id and doc_id in self.ingestion.store.docs:
            self.ingestion.store.docs.pop(doc_id, None)
        reindex_result: dict[str, Any]
        try:
            reindex_result = self._refresh_summary_vector_index([], force=True)
        except Exception as ex:  # noqa: BLE001
            reindex_result = {"status": "failed", "error": str(ex)[:240]}
        return {
            **deleted,
            "vector_reindex": reindex_result,
        }

    def rag_doc_preview(self, token: str, *, doc_id: str, page: int = 1) -> dict[str, Any]:
        _ = self.web.require_role(token, {"admin", "ops"})
        normalized_doc_id = str(doc_id or "").strip()
        if not normalized_doc_id:
            raise ValueError("doc_id is required")
        quality = self.docs_quality_report(normalized_doc_id)
        if "error" in quality:
            return quality
        safe_page = max(1, int(page))
        page_size = 6
        chunks = self.web.rag_doc_chunk_list_internal(doc_id=normalized_doc_id, limit=600)
        total_chunks = len(chunks)
        total_pages = max(1, (total_chunks + page_size - 1) // page_size) if total_chunks else 1
        safe_page = min(safe_page, total_pages)
        start = (safe_page - 1) * page_size
        selected = chunks[start : start + page_size]
        stage_logs: list[dict[str, Any]] = []
        parse_stage_detail: dict[str, Any] = {}
        upload_row = self.web.store.query_one(
            """
            SELECT upload_id, parser_name, ocr_used, quality_score, parse_note, job_status, current_stage,
                   verification_passed, vector_ready, updated_at
            FROM rag_upload_asset
            WHERE doc_id = ?
            ORDER BY updated_at DESC
            LIMIT 1
            """,
            (normalized_doc_id,),
        )
        if upload_row and str(upload_row.get("upload_id", "")).strip():
            upload_id = str(upload_row.get("upload_id", ""))
            stage_logs = self.web.rag_upload_stage_log_list(token, upload_id=upload_id, limit=120)
            for row in reversed(stage_logs):
                if str(row.get("stage", "")) == "parsing":
                    parse_stage_detail = row.get("detail", {}) if isinstance(row.get("detail"), dict) else {}
                    break
            upload_meta = {
                "upload_id": upload_id,
                "parser_name": str(upload_row.get("parser_name", "")),
                "ocr_used": bool(int(upload_row.get("ocr_used", 0) or 0)),
                "quality_score": float(upload_row.get("quality_score", 0.0) or 0.0),
                "parse_note": str(upload_row.get("parse_note", "")),
                "job_status": str(upload_row.get("job_status", "")),
                "current_stage": str(upload_row.get("current_stage", "")),
                "verification_passed": bool(int(upload_row.get("verification_passed", 0) or 0)),
                "vector_ready": bool(int(upload_row.get("vector_ready", 0) or 0)),
            }
        else:
            upload_meta = {}

        preview_items = [
            {
                "chunk_id": str(row.get("chunk_id", "")),
                "chunk_no": int(row.get("chunk_no", 0) or 0),
                "source": str(row.get("source", "")),
                "status": str(row.get("effective_status", "")),
                "quality_score": round(float(row.get("quality_score", 0.0) or 0.0), 4),
                "excerpt": self._rag_preview_trim(
                    str(row.get("chunk_text_redacted") or row.get("chunk_text") or ""),
                    max_len=260,
                ),
            }
            for row in selected
        ]
        parse_verdict = {"status": "ok", "message": "Parsed text looks readable"}
        parse_note_lower = str(upload_meta.get("parse_note", "")).lower()
        if "pdf_binary_stream_detected" in parse_note_lower:
            parse_verdict = {
                "status": "failed",
                "message": "PDF parse failed: detected compressed binary stream instead of document text",
            }
        elif "pdf_parser_unavailable" in parse_note_lower:
            parse_verdict = {
                "status": "failed",
                "message": "PDF parse failed: parser unavailable in runtime, please install parser and re-upload",
            }
        elif "pdf_ascii_fallback" in parse_note_lower and (
            ("pdf_parser_unavailable" in parse_note_lower) or not parse_stage_detail
        ):
            parse_verdict = {
                "status": "warning",
                "message": "PDF parser unavailable, fallback text may be unreliable",
            }
        elif preview_items and sum(1 for x in preview_items if "Filter/FlateDecode" in str(x.get("excerpt", ""))) >= max(
            1,
            len(preview_items) // 2,
        ):
            parse_verdict = {
                "status": "failed",
                "message": "Preview shows PDF internals, not human-readable content",
            }
        return {
            "doc_id": normalized_doc_id,
            "page": safe_page,
            "page_size": page_size,
            "total_chunks": total_chunks,
            "total_pages": total_pages,
            "items": preview_items,
            "quality_report": quality,
            "upload": upload_meta,
            "parse_verdict": parse_verdict,
            "parse_trace": parse_stage_detail.get("trace", {}),
            "parse_quality": parse_stage_detail.get("quality", {}),
            "stage_logs": [
                {
                    "phase": str(row.get("stage", "")),
                    "status": str(row.get("status", "")),
                    "at": str(row.get("created_at", "")),
                    "detail": row.get("detail", {}),
                }
                for row in stage_logs
            ],
        }

    def rag_dashboard(self, token: str) -> dict[str, Any]:
        return self.web.rag_dashboard_summary(token)

    def docs_list(self, token: str) -> list[dict[str, Any]]:
        return self.web.docs_list(token)

    def docs_review_queue(self, token: str) -> list[dict[str, Any]]:
        return self.web.docs_review_queue(token)

    def docs_review_action(self, token: str, doc_id: str, action: str, comment: str = "") -> dict[str, Any]:
        result = self.web.docs_review_action(token, doc_id, action, comment)
        # 瀹℃牳鍔ㄤ綔闇€瑕佸悓姝ュ埌 chunk 鐢熸晥鐘舵€侊紝閬垮厤鈥滄枃妗ｇ姸鎬佸凡鏀逛絾妫€绱粛鍛戒腑鏃х墖娈碘€濄€?
        if action == "approve":
            self.web.rag_doc_chunk_set_status_by_doc(doc_id=doc_id, status="active")
            self.web.rag_upload_asset_set_status(doc_id=doc_id, status="active", parse_note="review_approved")
        elif action == "reject":
            self.web.rag_doc_chunk_set_status_by_doc(doc_id=doc_id, status="rejected")
            self.web.rag_upload_asset_set_status(doc_id=doc_id, status="rejected", parse_note="review_rejected")
        return result

    # ----------------- RAG Asset APIs -----------------

    def rag_source_policy_list(self, token: str) -> list[dict[str, Any]]:
        return self.web.rag_source_policy_list(token)

    def rag_source_policy_set(self, token: str, source: str, payload: dict[str, Any]) -> dict[str, Any]:
        return self.web.rag_source_policy_upsert(
            token,
            source=source,
            auto_approve=bool(payload.get("auto_approve", False)),
            trust_score=float(payload.get("trust_score", 0.7)),
            enabled=bool(payload.get("enabled", True)),
        )

    def rag_doc_chunks_list(
        self,
        token: str,
        *,
        doc_id: str = "",
        status: str = "",
        source: str = "",
        stock_code: str = "",
        limit: int = 60,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        return self.web.rag_doc_chunk_list(
            token,
            doc_id=doc_id,
            status=status,
            source=source,
            stock_code=stock_code,
            limit=limit,
            offset=offset,
        )

    def rag_doc_chunk_detail(
        self,
        token: str,
        chunk_id: str,
        *,
        context_window: int = 1,
    ) -> dict[str, Any]:
        """Return chunk detail with nearby context for瀹氫綅鏌ョ湅."""
        return self.web.rag_doc_chunk_get_detail(
            token,
            chunk_id=chunk_id,
            context_window=context_window,
        )

    def rag_doc_chunk_status_set(self, token: str, chunk_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        return self.web.rag_doc_chunk_set_status(token, chunk_id=chunk_id, status=str(payload.get("status", "review")))

    def rag_qa_memory_list(
        self,
        token: str,
        *,
        stock_code: str = "",
        retrieval_enabled: int = -1,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        return self.web.rag_qa_memory_list(
            token,
            stock_code=stock_code,
            retrieval_enabled=retrieval_enabled,
            limit=limit,
            offset=offset,
        )

    def rag_qa_memory_toggle(self, token: str, memory_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        return self.web.rag_qa_memory_toggle(
            token,
            memory_id=memory_id,
            retrieval_enabled=bool(payload.get("retrieval_enabled", False)),
        )

