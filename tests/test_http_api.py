from __future__ import annotations

import base64
import json
import subprocess
import time
import unittest
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path


class HttpApiTestCase(unittest.TestCase):
    """API 契约测试：通过真实 uvicorn 进程覆盖全部 `/v1/*` 接口。"""
    REQUEST_TIMEOUT = 15

    @classmethod
    def setUpClass(cls) -> None:
        root = Path(__file__).resolve().parents[1]
        cls.base_url = "http://127.0.0.1:8011"
        py = root / ".venv" / "Scripts" / "python.exe"
        # 启动真实 HTTP 服务，避免依赖 TestClient/httpx。
        cls.proc = subprocess.Popen(
            [
                str(py),
                "-m",
                "uvicorn",
                "backend.app.http_api:create_app",
                "--factory",
                "--host",
                "127.0.0.1",
                "--port",
                "8011",
                "--log-level",
                "warning",
            ],
            cwd=str(root),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        cls._wait_until_ready(cls.base_url + "/docs")

    @classmethod
    def tearDownClass(cls) -> None:
        cls.proc.terminate()
        try:
            cls.proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            cls.proc.kill()

    @classmethod
    def _wait_until_ready(cls, url: str, timeout: float = 8.0) -> None:
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                with urllib.request.urlopen(url, timeout=2) as resp:  # noqa: S310 - local smoke url
                    if resp.status == 200:
                        return
            except Exception:
                time.sleep(0.2)
        raise RuntimeError("uvicorn service did not become ready in time")

    def _post(self, path: str, payload: dict) -> tuple[int, dict]:
        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            self.base_url + path,
            data=body,
            method="POST",
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=self.REQUEST_TIMEOUT) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))

    def _post_error(self, path: str, payload: dict) -> tuple[int, dict]:
        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            self.base_url + path,
            data=body,
            method="POST",
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=self.REQUEST_TIMEOUT) as resp:
                return resp.status, json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as ex:
            return ex.code, json.loads(ex.read().decode("utf-8"))

    def _patch(self, path: str, payload: dict) -> tuple[int, dict]:
        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            self.base_url + path,
            data=body,
            method="PATCH",
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=self.REQUEST_TIMEOUT) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))

    def _get(self, path: str) -> tuple[int, dict]:
        with urllib.request.urlopen(self.base_url + path, timeout=self.REQUEST_TIMEOUT) as resp:  # noqa: S310 - local smoke url
            return resp.status, json.loads(resp.read().decode("utf-8"))

    def test_query(self) -> None:
        code, body = self._post(
            "/v1/query",
            {
                "user_id": "api-u1",
                "question": "请分析SH600000近期风险与机会",
                "stock_codes": ["SH600000"],
            },
        )
        self.assertEqual(code, 200)
        self.assertIn("trace_id", body)
        self.assertIn("citations", body)
        self.assertIn("workflow_runtime", body)
        self.assertIn("analysis_brief", body)
        self.assertIn("market_regime", body["analysis_brief"])
        self.assertIn("regime_confidence", body["analysis_brief"])
        self.assertIn("signal_guard_applied", body["analysis_brief"])
        self.assertIn("data_packs", body)
        self.assertTrue(isinstance(body.get("data_packs", []), list))
        self.assertTrue(all(str(x.get("retrieval_track", "")).strip() for x in body.get("citations", [])))
        self.assertIn(body["workflow_runtime"], ("langgraph", "direct"))

        code_mem, memory_rows = self._get("/v1/rag/qa-memory?stock_code=SH600000&limit=20")
        self.assertEqual(code_mem, 200)
        self.assertTrue(isinstance(memory_rows, list))
        self.assertGreater(len(memory_rows), 0)

        # 支持请求级运行时覆盖，便于对比验证。
        code2, body2 = self._post(
            "/v1/query",
            {
                "user_id": "api-u1",
                "question": "请分析SH600000近期风险与机会",
                "stock_codes": ["SH600000"],
                "workflow_runtime": "direct",
            },
        )
        self.assertEqual(code2, 200)
        self.assertEqual(body2["workflow_runtime"], "direct")

    def test_query_cache_compare_and_history(self) -> None:
        req = {
            "user_id": "api-u-cache",
            "question": "query cache smoke for SH600000",
            "stock_codes": ["SH600000"],
        }
        c1, body1 = self._post("/v1/query", req)
        self.assertEqual(c1, 200)
        self.assertFalse(bool(body1.get("cache_hit", True)))

        c2, body2 = self._post("/v1/query", req)
        self.assertEqual(c2, 200)
        self.assertTrue(bool(body2.get("cache_hit", False)))

        c3, compared = self._post(
            "/v1/query/compare",
            {
                "user_id": "api-u-compare",
                "question": "compare SH600000 vs SZ000001",
                "stock_codes": ["SH600000", "SZ000001"],
            },
        )
        self.assertEqual(c3, 200)
        self.assertEqual(int(compared.get("count", 0)), 2)
        self.assertGreaterEqual(len(compared.get("items", [])), 2)

        c4, history_rows = self._get("/v1/query/history?limit=20")
        self.assertEqual(c4, 200)
        self.assertTrue(isinstance(history_rows, list))
        self.assertGreaterEqual(len(history_rows), 1)
        created_at = urllib.parse.quote(str(history_rows[0].get("created_at", "")))

        c4b, history_by_stock = self._get("/v1/query/history?limit=20&stock_code=SH600000")
        self.assertEqual(c4b, 200)
        self.assertGreaterEqual(len(history_by_stock), 1)
        self.assertTrue(all("SH600000" in list(map(str, x.get("stock_codes", []))) for x in history_by_stock))

        c4c, history_by_time = self._get(
            f"/v1/query/history?limit=20&created_from={created_at}&created_to={created_at}"
        )
        self.assertEqual(c4c, 200)
        self.assertGreaterEqual(len(history_by_time), 1)

        with self.assertRaises(urllib.error.HTTPError) as bad_history_time:
            urllib.request.urlopen(  # noqa: S310 - local endpoint
                self.base_url + "/v1/query/history?created_from=2026-02-20%2000:00:00&created_to=2026-02-19%2000:00:00",
                timeout=8,
            )
        self.assertEqual(bad_history_time.exception.code, 400)

        clear_req = urllib.request.Request(self.base_url + "/v1/query/history", method="DELETE")
        with urllib.request.urlopen(clear_req, timeout=8) as resp:
            self.assertEqual(resp.status, 200)

        c5, history_after = self._get("/v1/query/history?limit=20")
        self.assertEqual(c5, 200)
        self.assertEqual(len(history_after), 0)

    def test_query_validation_returns_400(self) -> None:
        code, body = self._post_error(
            "/v1/query",
            {
                "user_id": "api-u-invalid",
                "question": "x",
                "stock_codes": ["SH600000"],
            },
        )
        self.assertEqual(code, 400)
        self.assertIn("detail", body)

    def test_query_stream(self) -> None:
        body = json.dumps(
            {
                "user_id": "api-u-stream",
                "question": "请分析SH600000近期风险与机会",
                "stock_codes": ["SH600000"],
            }
        ).encode("utf-8")
        req = urllib.request.Request(
            self.base_url + "/v1/query/stream",
            data=body,
            method="POST",
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            self.assertEqual(resp.status, 200)
            text = resp.read().decode("utf-8", errors="ignore")
        self.assertIn("event: start", text)
        self.assertIn("event: market_regime", text)
        self.assertIn("event: stream_runtime", text)
        self.assertIn("event: answer_delta", text)
        self.assertIn("event: stream_source", text)
        self.assertIn("event: knowledge_persisted", text)
        self.assertIn("event: analysis_brief", text)
        self.assertIn("event: done", text)

        # 流式同样支持覆盖 direct runtime。
        body2 = json.dumps(
            {
                "user_id": "api-u-stream2",
                "question": "请分析SH600000近期风险与机会",
                "stock_codes": ["SH600000"],
                "workflow_runtime": "direct",
            }
        ).encode("utf-8")
        req2 = urllib.request.Request(
            self.base_url + "/v1/query/stream",
            data=body2,
            method="POST",
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req2, timeout=15) as resp2:
            text2 = resp2.read().decode("utf-8", errors="ignore")
        self.assertIn('"runtime": "direct"', text2)

    def test_report_generate_and_get(self) -> None:
        code, report = self._post(
            "/v1/report/generate",
            {
                "user_id": "api-u2",
                "stock_code": "SH600000",
                "period": "1y",
                "report_type": "fact",
            },
        )
        self.assertEqual(code, 200)
        report_id = report["report_id"]

        get_code, loaded = self._get(f"/v1/report/{report_id}")
        self.assertEqual(get_code, 200)
        self.assertIn("markdown", loaded)
        self.assertIn("quality_gate", report)
        self.assertIn("report_data_pack_summary", report)
        self.assertIn("generation_mode", report)
        self.assertIn("confidence_attribution", report)
        self.assertIn("llm_input_pack", report)
        self.assertIn("report_modules", report)
        self.assertIn("final_decision", report)
        self.assertIn("committee", report)
        self.assertIn("metric_snapshot", report)
        self.assertIn("analysis_nodes", report)
        self.assertIn("quality_dashboard", report)
        self.assertIn("multi_role_enabled", report)
        self.assertIn("multi_role_trace_id", report)
        self.assertIn("multi_role_decision", report)
        self.assertIn("role_opinions", report)
        self.assertIn("judge_summary", report)
        self.assertIn("conflict_sources", report)
        self.assertIn("consensus_signal", report)
        self.assertIn("consensus_confidence", report)
        self.assertIn("schema_version", report)
        modules = report.get("report_modules", [])
        if isinstance(modules, list) and modules:
            self.assertIn("module_quality_score", modules[0])
            self.assertIn("module_degrade_code", modules[0])
        evidence_refs = report.get("evidence_refs", [])
        if isinstance(evidence_refs, list) and evidence_refs:
            self.assertIn("freshness_score", evidence_refs[0])
            self.assertIn("freshness_tier", evidence_refs[0])

        export_code, export_body = self._post(f"/v1/reports/{report_id}/export?format=module_markdown", {})
        self.assertEqual(export_code, 200)
        self.assertEqual(str(export_body.get("format", "")), "module_markdown")
        self.assertIn("模块化报告导出", str(export_body.get("content", "")))
        export_json_code, export_json = self._post(f"/v1/reports/{report_id}/export?format=json_bundle", {})
        self.assertEqual(export_json_code, 200)
        self.assertIn("schema_version", (export_json.get("json_bundle", {}) if isinstance(export_json.get("json_bundle"), dict) else {}))

        diff_code, diff_body = self._get(f"/v1/reports/{report_id}/versions/diff")
        self.assertEqual(diff_code, 200)
        self.assertIn("diff", diff_body)

    def test_report_self_test_endpoint(self) -> None:
        code, body = self._get("/v1/report/self-test?stock_code=SH600000&report_type=research&period=1y")
        self.assertEqual(code, 200)
        self.assertTrue(bool(body.get("ok", False)))
        self.assertEqual(str(body.get("stock_code", "")), "SH600000")
        self.assertIn("sync", body)
        self.assertIn("async", body)
        sync = body.get("sync", {})
        if isinstance(sync, dict):
            self.assertIn("multi_role_enabled", sync)
            self.assertIn("multi_role_trace_id", sync)
        async_part = body.get("async", {})
        if isinstance(async_part, dict):
            self.assertIn(str(async_part.get("final_status", "")), {"completed", "partial_ready", "failed"})

    def test_report_task_endpoints(self) -> None:
        code, created = self._post(
            "/v1/report/tasks",
            {
                "user_id": "api-u-report-task",
                "stock_code": "SH600000",
                "period": "1y",
                "report_type": "research",
            },
        )
        self.assertEqual(code, 200)
        task_id = str(created.get("task_id", ""))
        self.assertTrue(task_id)

        final_status = ""
        for _ in range(120):
            c_get, snapshot = self._get(f"/v1/report/tasks/{task_id}")
            self.assertEqual(c_get, 200)
            self.assertIn("report_quality_dashboard", snapshot)
            self.assertIn("deadline_at", snapshot)
            self.assertIn("heartbeat_at", snapshot)
            self.assertIn("stage_elapsed_seconds", snapshot)
            self.assertIn("heartbeat_age_seconds", snapshot)
            final_status = str(snapshot.get("status", ""))
            if final_status in {"completed", "failed"}:
                break
            time.sleep(0.05)
        self.assertIn(final_status, {"completed", "failed"})

        c_result, result = self._get(f"/v1/report/tasks/{task_id}/result")
        self.assertEqual(c_result, 200)
        self.assertIn("result_level", result)
        self.assertIn("status", result)
        self.assertIn("deadline_at", result)
        self.assertIn("heartbeat_at", result)
        payload = result.get("result")
        if isinstance(payload, dict):
            self.assertIn("report_modules", payload)
            self.assertIn("final_decision", payload)
            self.assertIn("committee", payload)
            self.assertIn("analysis_nodes", payload)
            self.assertIn("quality_dashboard", payload)

    def test_ingest_market_daily(self) -> None:
        code, body = self._post("/v1/ingest/market-daily", {"stock_codes": ["SH600000", "SZ000001"]})
        self.assertEqual(code, 200)
        self.assertEqual(body["task_name"], "market-daily")

    def test_ingest_announcements(self) -> None:
        code, body = self._post("/v1/ingest/announcements", {"stock_codes": ["SH600000"]})
        self.assertEqual(code, 200)
        self.assertEqual(body["task_name"], "announcements")

    def test_ingest_financials(self) -> None:
        code, body = self._post("/v1/ingest/financials", {"stock_codes": ["SH600000"]})
        self.assertEqual(code, 200)
        self.assertEqual(body["task_name"], "financial-snapshot")

    def test_ingest_news(self) -> None:
        code, body = self._post("/v1/ingest/news", {"stock_codes": ["SH600000"], "limit": 3})
        self.assertEqual(code, 200)
        self.assertEqual(body["task_name"], "news-ingest")

    def test_ingest_research(self) -> None:
        code, body = self._post("/v1/ingest/research", {"stock_codes": ["SH600000"], "limit": 3})
        self.assertEqual(code, 200)
        self.assertEqual(body["task_name"], "research-ingest")

    def test_ingest_macro(self) -> None:
        code, body = self._post("/v1/ingest/macro", {"limit": 3})
        self.assertEqual(code, 200)
        self.assertEqual(body["task_name"], "macro-ingest")

    def test_ingest_fund(self) -> None:
        code, body = self._post("/v1/ingest/fund", {"stock_codes": ["SH600000"]})
        self.assertEqual(code, 200)
        self.assertEqual(body["task_name"], "fund-ingest")

    def test_datasource_management_endpoints(self) -> None:
        c1, sources = self._get("/v1/datasources/sources")
        self.assertEqual(c1, 200)
        self.assertGreater(int(sources.get("count", 0)), 0)
        self.assertTrue(isinstance(sources.get("items", []), list))
        self.assertTrue(all(isinstance(x.get("used_in_ui_modules", []), list) for x in sources.get("items", [])))
        target = next((x for x in sources.get("items", []) if str(x.get("category", "")) == "news"), sources["items"][0])
        source_id = str(target.get("source_id", ""))
        self.assertTrue(source_id)

        c2, fetched = self._post(
            "/v1/datasources/fetch",
            {"source_id": source_id, "stock_codes": ["SH600000"], "limit": 2},
        )
        self.assertEqual(c2, 200)
        self.assertEqual(str(fetched.get("source_id", "")), source_id)
        self.assertIn(str(fetched.get("status", "")), {"ok", "partial", "failed"})

        encoded_source = urllib.parse.quote(source_id)
        c3, logs = self._get(f"/v1/datasources/logs?source_id={encoded_source}&limit=20")
        self.assertEqual(c3, 200)
        self.assertGreaterEqual(int(logs.get("count", 0)), 1)
        self.assertTrue(isinstance(logs.get("items", []), list))

        c4, health = self._get("/v1/datasources/health?limit=200")
        self.assertEqual(c4, 200)
        self.assertGreater(int(health.get("count", 0)), 0)
        target_health = next((x for x in health.get("items", []) if str(x.get("source_id", "")) == source_id), None)
        self.assertIsNotNone(target_health)
        self.assertTrue(isinstance((target_health or {}).get("used_in_ui_modules", []), list))
        self.assertIn("last_used_at", target_health or {})
        self.assertIn("staleness_minutes", target_health or {})

        c5, business = self._get("/v1/business/data-health?stock_code=SH600000&limit=200")
        self.assertEqual(c5, 200)
        self.assertIn("status", business)
        self.assertIn("module_health", business)
        self.assertIn("stock_snapshot", business)

    def test_docs_upload_and_index(self) -> None:
        upload_code, uploaded = self._post(
            "/v1/docs/upload",
            {"doc_id": "api-doc-1", "filename": "demo.pdf", "content": "财报内容" * 500, "source": "api"},
        )
        self.assertEqual(upload_code, 200)
        self.assertEqual(uploaded["status"], "uploaded")

        index_code, indexed = self._post("/v1/docs/api-doc-1/index", {})
        self.assertEqual(index_code, 200)
        self.assertEqual(indexed["status"], "indexed")

        v_code, versions = self._get("/v1/docs/api-doc-1/versions?limit=20")
        self.assertEqual(v_code, 200)
        self.assertTrue(isinstance(versions, list))
        self.assertGreaterEqual(len(versions), 1)
        self.assertGreaterEqual(int(versions[0].get("version", 0)), 1)

        p_code, runs = self._get("/v1/docs/api-doc-1/pipeline-runs?limit=20")
        self.assertEqual(p_code, 200)
        self.assertTrue(isinstance(runs, list))
        self.assertGreaterEqual(len(runs), 2)
        stages = {str(x.get("stage", "")) for x in runs}
        self.assertIn("upload", stages)
        self.assertIn("index", stages)

        c_rec, rec = self._post(
            "/v1/docs/recommend",
            {"stock_code": "SH600000", "question": "请推荐和银行业相关的材料", "top_k": 5},
        )
        self.assertEqual(c_rec, 200)
        self.assertIn("items", rec)
        self.assertTrue(isinstance(rec.get("items", []), list))

        qr_code, quality = self._get("/v1/docs/api-doc-1/quality-report")
        self.assertEqual(qr_code, 200)
        self.assertIn("quality_score", quality)
        self.assertIn("chunk_stats", quality)
        self.assertIn("recommendations", quality)

    def test_rag_asset_management_endpoints(self) -> None:
        c1, policies = self._get("/v1/rag/source-policy")
        self.assertEqual(c1, 200)
        self.assertTrue(isinstance(policies, list))
        self.assertTrue(any(str(x.get("source")) == "cninfo" for x in policies))

        c2, policy_updated = self._post(
            "/v1/rag/source-policy/user_upload",
            {"auto_approve": True, "trust_score": 0.8, "enabled": True},
        )
        self.assertEqual(c2, 200)
        self.assertEqual(str(policy_updated.get("source", "")), "user_upload")

        _ = self._post(
            "/v1/docs/upload",
            {"doc_id": "api-rag-doc-1", "filename": "rag-demo.pdf", "content": "SH600000 纪要" * 380, "source": "cninfo"},
        )
        _ = self._post("/v1/docs/api-rag-doc-1/index", {})

        c3, chunks = self._get("/v1/rag/docs/chunks?doc_id=api-rag-doc-1&limit=20")
        self.assertEqual(c3, 200)
        self.assertTrue(isinstance(chunks, list))
        self.assertGreater(len(chunks), 0)
        chunk_id = str(chunks[0].get("chunk_id", ""))
        self.assertTrue(chunk_id)

        c4, updated = self._post(f"/v1/rag/docs/chunks/{chunk_id}/status", {"status": "review"})
        self.assertEqual(c4, 200)
        self.assertEqual(str(updated.get("effective_status", "")), "review")
        c4b, detail = self._get(f"/v1/rag/docs/chunks/{chunk_id}?context_window=1")
        self.assertEqual(c4b, 200)
        self.assertEqual(str((detail.get("chunk") or {}).get("chunk_id", "")), chunk_id)
        self.assertIn("context", detail)

        c5, qa_pool = self._get("/v1/rag/qa-memory?limit=10")
        self.assertEqual(c5, 200)
        self.assertTrue(isinstance(qa_pool, list))

        c6, trace = self._get("/v1/ops/rag/retrieval-trace?limit=10")
        self.assertEqual(c6, 200)
        self.assertIn("count", trace)

        c7, reindex = self._post("/v1/ops/rag/reindex", {"limit": 2000})
        self.assertEqual(c7, 200)
        self.assertEqual(str(reindex.get("status", "")), "ok")

        encoded = base64.b64encode("SH600000 新上传附件样本".encode("utf-8")).decode("ascii")
        c8, uploaded = self._post(
            "/v1/rag/workflow/upload-and-index",
            {
                "filename": "api-upload.txt",
                "content_type": "text/plain",
                "content_base64": encoded,
                "source": "user_upload",
                "stock_codes": ["SH600000"],
                "force_reupload": True,
                "tags": ["api"],
            },
        )
        self.assertEqual(c8, 200)
        self.assertEqual(str(uploaded.get("status", "")), "ok")
        self.assertIn("timeline", uploaded)
        self.assertEqual(str(((uploaded.get("result") or {}).get("asset") or {}).get("status", "")), "active")
        self.assertIn("retrieval_preview", uploaded)
        preview = uploaded.get("retrieval_preview", {})
        self.assertTrue(bool(preview.get("ready")))
        self.assertGreaterEqual(int(preview.get("query_count", 0)), 1)

        upload_id = str((uploaded.get("result") or {}).get("upload_id", ""))
        doc_id = str((uploaded.get("result") or {}).get("doc_id", ""))
        self.assertTrue(upload_id)
        self.assertTrue(doc_id)
        c8b, preview_api = self._get(f"/v1/rag/retrieval-preview?doc_id={urllib.parse.quote(doc_id)}&max_queries=2&top_k=4")
        self.assertEqual(c8b, 200)
        self.assertEqual(str(preview_api.get("doc_id", "")), doc_id)
        self.assertIn("items", preview_api)

        c8c, status_payload = self._get(f"/v1/rag/uploads/{urllib.parse.quote(upload_id)}/status")
        self.assertEqual(c8c, 200)
        self.assertEqual(str(status_payload.get("upload_id", "")), upload_id)
        self.assertIn("asset", status_payload)
        self.assertTrue(isinstance(status_payload.get("timeline", []), list))

        c8d, verification_payload = self._get(f"/v1/rag/uploads/{urllib.parse.quote(upload_id)}/verification")
        self.assertEqual(c8d, 200)
        self.assertEqual(str(verification_payload.get("upload_id", "")), upload_id)
        self.assertGreaterEqual(int(verification_payload.get("query_count", 0)), 1)
        self.assertTrue(isinstance(verification_payload.get("items", []), list))

        c8e, doc_preview = self._get(f"/v1/rag/docs/{urllib.parse.quote(doc_id)}/preview?page=1")
        self.assertEqual(c8e, 200)
        self.assertEqual(str(doc_preview.get("doc_id", "")), doc_id)
        self.assertIn("quality_report", doc_preview)
        self.assertIn("parse_verdict", doc_preview)
        self.assertIn(str((doc_preview.get("parse_verdict") or {}).get("status", "")), {"ok", "warning", "failed"})
        self.assertTrue(isinstance(doc_preview.get("items", []), list))

        c9, dashboard = self._get("/v1/rag/dashboard")
        self.assertEqual(c9, 200)
        self.assertIn("doc_total", dashboard)
        self.assertIn("active_chunks", dashboard)

        c10, upload_rows = self._get("/v1/rag/uploads?limit=10")
        self.assertEqual(c10, 200)
        self.assertTrue(isinstance(upload_rows, list))
        self.assertTrue(any(str(x.get("filename", "")) == "api-upload.txt" for x in upload_rows))

        req = urllib.request.Request(
            self.base_url + f"/v1/rag/uploads/{urllib.parse.quote(upload_id)}",
            method="DELETE",
        )
        with urllib.request.urlopen(req, timeout=8) as resp:
            self.assertEqual(resp.status, 200)
            body = json.loads(resp.read().decode("utf-8"))
            self.assertEqual(str(body.get("status", "")), "ok")
            self.assertEqual(str(body.get("upload_id", "")), upload_id)

        c10b, upload_rows_after = self._get("/v1/rag/uploads?limit=10")
        self.assertEqual(c10b, 200)
        self.assertFalse(any(str(x.get("upload_id", "")) == upload_id for x in upload_rows_after))

    def test_rag_upload_rejects_pdf_binary_stream_payload(self) -> None:
        fake_pdf_body = (
            "%PDF-1.4\n"
            + ("Filter/FlateDecode /Length 2484 stream endstream obj endobj Subtype/Type1C " * 60)
            + "\n%%EOF"
        ).encode("latin1", errors="ignore")
        encoded = base64.b64encode(fake_pdf_body).decode("ascii")
        upload_id = f"api-rag-bad-{int(time.time() * 1000)}"

        code, body = self._post_error(
            "/v1/rag/uploads",
            {
                "upload_id": upload_id,
                "filename": "api-broken-preview.pdf",
                "content_type": "application/pdf",
                "content_base64": encoded,
                "source": "user_upload",
                "force_reupload": True,
            },
        )
        self.assertEqual(code, 400)
        self.assertIn("detail", body)
        self.assertIn("PDF parsing failed: binary stream detected", str(body.get("detail", "")))

        c2, status_payload = self._get(f"/v1/rag/uploads/{urllib.parse.quote(upload_id)}/status")
        self.assertEqual(c2, 200)
        self.assertEqual(str(status_payload.get("upload_id", "")), upload_id)
        asset = status_payload.get("asset", {})
        self.assertEqual(str(asset.get("job_status", "")), "failed")
        self.assertEqual(str(asset.get("current_stage", "")), "failed")
        self.assertEqual(str(asset.get("error_code", "")), "upload_failed")

    def test_evals_run_and_get(self) -> None:
        code, run = self._post(
            "/v1/evals/run",
            {
                "samples": [
                    {"fact_correct": True, "has_citation": True, "hallucination": False, "violation": False},
                    {"fact_correct": True, "has_citation": True, "hallucination": False, "violation": False},
                ]
            },
        )
        self.assertEqual(code, 200)
        self.assertTrue(run["pass_gate"])

        get_code, loaded = self._get(f"/v1/evals/{run['eval_run_id']}")
        self.assertEqual(get_code, 200)
        self.assertIn("status", loaded)

    def test_scheduler_run_and_status(self) -> None:
        code, status = self._get("/v1/scheduler/status")
        self.assertEqual(code, 200)
        self.assertIn("intraday_quote_ingest", status)

        run_code, run_result = self._post("/v1/scheduler/run", {"job_name": "intraday_quote_ingest"})
        self.assertEqual(run_code, 200)
        self.assertIn(run_result["status"], ("ok", "failed", "circuit_open"))

    def test_predict_endpoints(self) -> None:
        code, run = self._post(
            "/v1/predict/run",
            {
                "stock_codes": ["SH600000", "SZ000001"],
                "horizons": ["5d", "20d"],
            },
        )
        self.assertEqual(code, 200)
        self.assertIn("run_id", run)
        self.assertEqual(len(run["results"]), 2)
        self.assertIn("data_quality", run)
        self.assertIn("degrade_reasons", run)
        self.assertIn("source_coverage", run)
        self.assertIn("metric_mode", run)
        self.assertIn("metrics_backtest", run)
        self.assertIn("metrics_simulated", run)
        self.assertIn("eval_provenance", run)
        self.assertIn("quality_gate", run)
        self.assertIn("engine_profile", run)
        self.assertIn(str((run.get("quality_gate") or {}).get("overall_status", "")), {"pass", "watch", "degraded"})

        get_code, loaded = self._get(f"/v1/predict/{run['run_id']}")
        self.assertEqual(get_code, 200)
        self.assertEqual(loaded["run_id"], run["run_id"])

        explain_code, explain = self._post(
            "/v1/predict/explain",
            {
                "run_id": run["run_id"],
                "stock_code": run["results"][0]["stock_code"],
                "horizon": "20d",
            },
        )
        self.assertEqual(explain_code, 200)
        self.assertEqual(str(explain.get("run_id", "")), str(run["run_id"]))
        self.assertTrue(str(explain.get("summary", "")).strip())
        self.assertTrue(isinstance(explain.get("drivers", []), list))
        self.assertTrue(isinstance(explain.get("risks", []), list))
        self.assertTrue(isinstance(explain.get("actions", []), list))
        self.assertIn("llm_used", explain)

        factor_code, factor = self._get("/v1/factors/SH600000")
        self.assertEqual(factor_code, 200)
        self.assertIn("factors", factor)
        self.assertIn("risk_score", factor["factors"])
        self.assertIn("history_data_mode", factor["source"])

        eval_code, latest = self._get("/v1/predict/evals/latest")
        self.assertEqual(eval_code, 200)
        self.assertIn("metrics", latest)
        self.assertIn(str(latest.get("metric_mode", "")), {"simulated", "backtest_proxy"})

    def test_market_overview(self) -> None:
        code, body = self._get("/v1/market/overview/SH600000")
        self.assertEqual(code, 200)
        self.assertIn("realtime", body)
        self.assertIn("history", body)

    def test_analysis_intel_card(self) -> None:
        code, body = self._get("/v1/analysis/intel-card?stock_code=SH600000&horizon=30d&risk_profile=neutral")
        self.assertEqual(code, 200)
        self.assertEqual(str(body.get("stock_code", "")), "SH600000")
        self.assertIn(str(body.get("overall_signal", "")), {"buy", "hold", "reduce"})
        self.assertIn("evidence", body)
        self.assertIn("event_calendar", body)
        self.assertIn("data_freshness", body)
        self.assertIn("execution_plan", body)
        self.assertIn("risk_thresholds", body)
        self.assertIn("degrade_status", body)

        with self.assertRaises(urllib.error.HTTPError) as bad_horizon:
            urllib.request.urlopen(  # noqa: S310 - local endpoint
                self.base_url + "/v1/analysis/intel-card?stock_code=SH600000&horizon=15d",
                timeout=8,
            )
        self.assertEqual(bad_horizon.exception.code, 400)

    def test_analysis_intel_feedback_and_review(self) -> None:
        c1, card = self._get("/v1/analysis/intel-card?stock_code=SH600000&horizon=30d&risk_profile=neutral")
        self.assertEqual(c1, 200)
        c2, feedback = self._post(
            "/v1/analysis/intel-card/feedback",
            {
                "stock_code": "SH600000",
                "trace_id": "api-intel-feedback-1",
                "signal": str(card.get("overall_signal", "hold")),
                "confidence": float(card.get("confidence", 0.0) or 0.0),
                "position_hint": str(card.get("position_hint", "")),
                "feedback": "adopt",
            },
        )
        self.assertEqual(c2, 200)
        self.assertEqual(str(feedback.get("status", "")), "ok")
        c3, review = self._get("/v1/analysis/intel-card/review?stock_code=SH600000&limit=20")
        self.assertEqual(c3, 200)
        self.assertIn("stats", review)
        self.assertIn("t5", review.get("stats", {}))

    def test_portfolio_endpoints(self) -> None:
        c1, created = self._post(
            "/v1/portfolio",
            {"portfolio_name": "api-core", "initial_capital": 80000, "description": "api-test"},
        )
        self.assertEqual(c1, 200)
        pid = int(created["portfolio_id"])
        self.assertGreater(pid, 0)

        c2, _tx = self._post(
            f"/v1/portfolio/{pid}/transactions",
            {"stock_code": "SH600000", "transaction_type": "buy", "quantity": 100, "price": 9.8, "fee": 1},
        )
        self.assertEqual(c2, 200)

        c3, summary = self._get(f"/v1/portfolio/{pid}")
        self.assertEqual(c3, 200)
        self.assertEqual(int(summary.get("portfolio_id", 0)), pid)
        self.assertGreaterEqual(int(summary.get("position_count", 0)), 1)

        c4, tx_rows = self._get(f"/v1/portfolio/{pid}/transactions?limit=20")
        self.assertEqual(c4, 200)
        self.assertTrue(isinstance(tx_rows, list))
        self.assertGreaterEqual(len(tx_rows), 1)

        c5, rows = self._get("/v1/portfolio")
        self.assertEqual(c5, 200)
        self.assertTrue(any(int(x.get("portfolio_id", 0)) == pid for x in rows))

    def test_alert_rule_endpoints(self) -> None:
        c1, created = self._post(
            "/v1/alerts/rules",
            {
                "rule_name": "api-price-rule",
                "rule_type": "price",
                "stock_code": "SH600000",
                "operator": ">",
                "target_value": 0,
            },
        )
        self.assertEqual(c1, 200)
        rid = int(created["rule_id"])
        self.assertGreater(rid, 0)

        c2, rules = self._get("/v1/alerts/rules")
        self.assertEqual(c2, 200)
        self.assertTrue(any(int(x.get("rule_id", 0)) == rid for x in rules))

        c3, checked = self._post("/v1/alerts/check", {})
        self.assertEqual(c3, 200)
        self.assertGreaterEqual(int(checked.get("checked_rules", 0)), 1)

        c4, logs = self._get("/v1/alerts/logs?limit=20")
        self.assertEqual(c4, 200)
        self.assertTrue(isinstance(logs, list))
        self.assertGreaterEqual(len(logs), 1)

        req = urllib.request.Request(self.base_url + f"/v1/alerts/rules/{rid}", method="DELETE")
        with urllib.request.urlopen(req, timeout=8) as resp:
            self.assertEqual(resp.status, 200)

    def test_backtest_endpoints(self) -> None:
        c1, run = self._post(
            "/v1/backtest/run",
            {
                "stock_code": "SH600000",
                "start_date": "2024-01-01",
                "end_date": "2026-02-15",
                "initial_capital": 100000,
                "ma_window": 20,
            },
        )
        self.assertEqual(c1, 200)
        self.assertIn("run_id", run)
        run_id = str(run["run_id"])
        c2, loaded = self._get(f"/v1/backtest/{run_id}")
        self.assertEqual(c2, 200)
        self.assertEqual(str(loaded.get("run_id", "")), run_id)

    def test_journal_endpoints(self) -> None:
        c1, created = self._post(
            "/v1/journal",
            {
                "journal_type": "decision",
                "title": "API 决策记录",
                "content": "阶段性仓位控制，等待量价确认。",
                "stock_code": "SH600000",
                "decision_type": "hold",
                "tags": ["仓位", "复盘"],
            },
        )
        self.assertEqual(c1, 200)
        journal_id = int(created.get("journal_id", 0))
        self.assertGreater(journal_id, 0)

        c2, rows = self._get("/v1/journal?limit=20")
        self.assertEqual(c2, 200)
        self.assertTrue(any(int(x.get("journal_id", 0)) == journal_id for x in rows))

        c3, filtered = self._get("/v1/journal?journal_type=decision&stock_code=SH600000&limit=20")
        self.assertEqual(c3, 200)
        self.assertTrue(any(int(x.get("journal_id", 0)) == journal_id for x in filtered))

        c4, reflection = self._post(
            f"/v1/journal/{journal_id}/reflections",
            {
                "reflection_content": "复盘确认信号分歧较大，下一轮降低仓位。",
                "ai_insights": "宏观扰动导致估值扩张受限。",
            },
        )
        self.assertEqual(c4, 200)
        self.assertGreater(int(reflection.get("reflection_id", 0)), 0)

        c5, reflections = self._get(f"/v1/journal/{journal_id}/reflections?limit=20")
        self.assertEqual(c5, 200)
        self.assertTrue(any("降低仓位" in str(x.get("reflection_content", "")) for x in reflections))

        bad_code, _ = self._post_error(
            "/v1/journal",
            {
                "journal_type": "invalid_type",
                "title": "bad",
                "content": "bad",
            },
        )
        self.assertEqual(bad_code, 400)

    def test_journal_ai_reflection_endpoints(self) -> None:
        c1, created = self._post(
            "/v1/journal",
            {
                "journal_type": "decision",
                "title": "API AI复盘",
                "content": "计划基于估值修复做分批加仓，并在两周后复核。",
                "stock_code": "SH600000",
                "decision_type": "buy",
            },
        )
        self.assertEqual(c1, 200)
        journal_id = int(created.get("journal_id", 0))
        self.assertGreater(journal_id, 0)

        c2, generated = self._post(
            f"/v1/journal/{journal_id}/ai-reflection/generate",
            {"focus": "优先分析触发条件和失效条件"},
        )
        self.assertEqual(c2, 200)
        self.assertIn(str(generated.get("status", "")), {"ready", "fallback"})
        self.assertTrue(str(generated.get("summary", "")).strip())
        self.assertTrue(isinstance(generated.get("insights", []), list))
        self.assertTrue(isinstance(generated.get("lessons", []), list))

        c3, loaded = self._get(f"/v1/journal/{journal_id}/ai-reflection")
        self.assertEqual(c3, 200)
        self.assertEqual(int(loaded.get("journal_id", 0)), journal_id)
        self.assertTrue(str(loaded.get("summary", "")).strip())

    def test_journal_create_minimal_payload_auto_fill(self) -> None:
        c1, created = self._post(
            "/v1/journal",
            {
                "journal_type": "decision",
                "stock_code": "SH600000",
                "decision_type": "hold",
            },
        )
        self.assertEqual(c1, 200)
        self.assertGreater(int(created.get("journal_id", 0)), 0)
        self.assertTrue(str(created.get("title", "")).strip())
        self.assertTrue(str(created.get("content", "")).strip())

    def test_journal_quick_review_outcome_and_execution_board_endpoints(self) -> None:
        c1, quick = self._post(
            "/v1/journal/quick",
            {"stock_code": "SH600000", "event_type": "buy", "review_days": 3, "thesis": "api quick thesis"},
        )
        self.assertEqual(c1, 200)
        journal_id = int(quick.get("journal_id", 0))
        self.assertGreater(journal_id, 0)
        self.assertEqual(str(quick.get("status", "")), "open")
        self.assertEqual(str(quick.get("source_type", "")), "manual")

        c2, due = self._post(
            "/v1/journal",
            {
                "journal_type": "decision",
                "title": "api due queue item",
                "content": "api due queue item content",
                "stock_code": "SH600000",
                "review_due_at": "2000-01-01 00:00:00",
                "status": "open",
            },
        )
        self.assertEqual(c2, 200)
        due_id = int(due.get("journal_id", 0))
        self.assertGreater(due_id, 0)

        c3, queue = self._get("/v1/journal/review-queue?status=review_due&stock_code=SH600000&limit=100")
        self.assertEqual(c3, 200)
        self.assertTrue(isinstance(queue, list))
        due_item = next((x for x in queue if int(x.get("journal_id", 0)) == due_id), {})
        self.assertTrue(due_item)
        self.assertEqual(str(due_item.get("status", "")), "review_due")
        self.assertTrue(bool(due_item.get("is_overdue", False)))

        c4, closed = self._patch(
            f"/v1/journal/{journal_id}/outcome",
            {
                "executed_as_planned": True,
                "outcome_rating": "good",
                "outcome_note": "api close",
                "close": True,
            },
        )
        self.assertEqual(c4, 200)
        self.assertEqual(str(closed.get("status", "")), "closed")
        self.assertTrue(str(closed.get("closed_at", "")).strip())

        c5, board = self._get("/v1/journal/execution-board?window_days=365")
        self.assertEqual(c5, 200)
        self.assertEqual(str(board.get("status", "")), "ok")
        self.assertGreaterEqual(int(board.get("closed_count_30d", 0)), 1)
        self.assertGreaterEqual(int(board.get("review_due_count", 0)), 1)
        self.assertTrue(isinstance(board.get("top_deviation_reasons", []), list))

    def test_journal_from_transaction_endpoint_idempotent(self) -> None:
        c1, portfolio = self._post(
            "/v1/portfolio",
            {"portfolio_name": "api-journal-tx", "initial_capital": 60000, "description": "journal tx"},
        )
        self.assertEqual(c1, 200)
        portfolio_id = int(portfolio.get("portfolio_id", 0))
        self.assertGreater(portfolio_id, 0)

        c2, tx = self._post(
            f"/v1/portfolio/{portfolio_id}/transactions",
            {"stock_code": "SH600000", "transaction_type": "buy", "quantity": 100, "price": 10.0, "fee": 1.0},
        )
        self.assertEqual(c2, 200)
        transaction_id = int(tx.get("transaction_id", 0))
        self.assertGreater(transaction_id, 0)

        c3, first = self._post(
            "/v1/journal/from-transaction",
            {"portfolio_id": portfolio_id, "transaction_id": transaction_id, "review_days": 5},
        )
        self.assertEqual(c3, 200)
        first_id = int(first.get("journal_id", 0))
        self.assertGreater(first_id, 0)
        self.assertEqual(str(first.get("action", "")), "created")
        self.assertEqual(str(first.get("source_type", "")), "transaction")
        self.assertEqual(str(first.get("source_ref_id", "")), f"{portfolio_id}:{transaction_id}")

        c4, second = self._post(
            "/v1/journal/from-transaction",
            {"portfolio_id": portfolio_id, "transaction_id": transaction_id, "review_days": 5},
        )
        self.assertEqual(c4, 200)
        self.assertEqual(str(second.get("action", "")), "reused")
        self.assertEqual(int(second.get("journal_id", 0)), first_id)

    def test_journal_insights_endpoint(self) -> None:
        c1, created = self._post(
            "/v1/journal",
            {
                "journal_type": "decision",
                "title": "API 洞察样本",
                "content": "等待盈利预期修复，先小仓位试错并跟踪成交量。",
                "stock_code": "SH600000",
                "decision_type": "hold",
                "tags": ["盈利", "成交量"],
            },
        )
        self.assertEqual(c1, 200)
        journal_id = int(created.get("journal_id", 0))
        self.assertGreater(journal_id, 0)

        c2, _ = self._post(
            f"/v1/journal/{journal_id}/reflections",
            {"reflection_content": "入场节奏控制较好，但风控阈值还需量化。"},
        )
        self.assertEqual(c2, 200)

        c3, _ = self._post(
            f"/v1/journal/{journal_id}/ai-reflection/generate",
            {"focus": "关注可执行改进与风险边界"},
        )
        self.assertEqual(c3, 200)

        c4, insights = self._get("/v1/journal/insights?window_days=365&timeline_days=90&limit=600")
        self.assertEqual(c4, 200)
        self.assertEqual(str(insights.get("status", "")), "ok")
        self.assertGreaterEqual(int(insights.get("total_journals", 0)), 1)
        self.assertTrue(isinstance(insights.get("type_distribution", []), list))
        self.assertTrue(isinstance(insights.get("decision_distribution", []), list))
        self.assertTrue(isinstance(insights.get("stock_activity", []), list))
        self.assertTrue(isinstance(insights.get("keyword_profile", []), list))
        self.assertTrue(isinstance(insights.get("timeline", []), list))

    def test_ops_journal_health_endpoint(self) -> None:
        c1, created = self._post(
            "/v1/journal",
            {
                "journal_type": "decision",
                "title": "Ops 健康样本",
                "content": "用于验证 journal 健康看板接口。",
                "stock_code": "SH600000",
                "decision_type": "hold",
            },
        )
        self.assertEqual(c1, 200)
        journal_id = int(created.get("journal_id", 0))
        self.assertGreater(journal_id, 0)

        c2, _ = self._post(
            f"/v1/journal/{journal_id}/ai-reflection/generate",
            {"focus": "验证健康看板统计"},
        )
        self.assertEqual(c2, 200)

        c3, health = self._get("/v1/ops/journal/health?window_hours=720&limit=300")
        self.assertEqual(c3, 200)
        self.assertEqual(str(health.get("status", "")), "ok")
        attempts = health.get("attempts", {})
        self.assertGreaterEqual(int(attempts.get("total", 0)), 1)
        self.assertTrue(isinstance(health.get("provider_breakdown", []), list))
        self.assertTrue(isinstance(health.get("recent_failures", []), list))

    def test_ops_capabilities(self) -> None:
        code, body = self._get("/v1/ops/capabilities")
        self.assertEqual(code, 200)
        self.assertIn("runtime", body)
        self.assertIn("capabilities", body)
        self.assertTrue(any(x.get("key") == "langgraph" for x in body["capabilities"]))

    def test_ops_debate_rag_and_prompt_compare(self) -> None:
        c1, debate = self._get("/v1/ops/agent/debate?stock_code=SH600000")
        self.assertEqual(c1, 200)
        self.assertIn("opinions", debate)
        self.assertIn("disagreement_score", debate)
        self.assertIn("debate_mode", debate)

        c2, rag = self._get("/v1/ops/rag/quality")
        self.assertEqual(c2, 200)
        self.assertIn("metrics", rag)
        self.assertIn("offline", rag)
        self.assertIn("online", rag)

        c3, comp = self._post(
            "/v1/ops/prompts/compare",
            {
                "prompt_id": "fact_qa",
                "base_version": "1.0.0",
                "candidate_version": "1.1.0",
                "variables": {"question": "test", "stock_codes": ["SH600000"], "evidence": "source:cninfo"},
            },
        )
        self.assertEqual(c3, 200)
        self.assertIn("diff_summary", comp)

        c4, versions = self._get("/v1/ops/prompts/fact_qa/versions")
        self.assertEqual(c4, 200)
        self.assertTrue(any(v.get("version") == "1.0.0" for v in versions))
        self.assertTrue(any(v.get("version") == "1.1.0" for v in versions))

    def test_knowledge_graph_view(self) -> None:
        c1, graph = self._get("/v1/knowledge/graph/SH600000?limit=20")
        self.assertEqual(c1, 200)
        self.assertEqual(str(graph.get("entity_id", "")), "SH600000")
        self.assertGreaterEqual(int(graph.get("relation_count", 0)), 1)
        self.assertTrue(isinstance(graph.get("nodes", []), list))
        self.assertTrue(isinstance(graph.get("relations", []), list))

    def test_deep_think_and_a2a(self) -> None:
        c1, session = self._post(
            "/v1/deep-think/sessions",
            {
                "user_id": "api-deep-1",
                "question": "请多agent深度分析SH600000",
                "stock_codes": ["SH600000"],
                "max_rounds": 2,
            },
        )
        self.assertEqual(c1, 200)
        session_id = session["session_id"]

        c2, round_snapshot = self._post(f"/v1/deep-think/sessions/{session_id}/rounds", {})
        self.assertEqual(c2, 200)
        self.assertEqual(round_snapshot["current_round"], 1)
        self.assertTrue(round_snapshot["rounds"])
        self.assertIn("consensus_signal", round_snapshot["rounds"][-1])
        self.assertIn("task_graph", round_snapshot["rounds"][-1])
        self.assertIn("budget_usage", round_snapshot["rounds"][-1])
        self.assertIn("multi_role_pre", round_snapshot["rounds"][-1])
        self.assertIn("runtime_guard", round_snapshot["rounds"][-1])

        c3, loaded = self._get(f"/v1/deep-think/sessions/{session_id}")
        self.assertEqual(c3, 200)
        self.assertEqual(loaded["session_id"], session_id)

        with urllib.request.urlopen(self.base_url + f"/v1/deep-think/sessions/{session_id}/stream", timeout=15) as resp:  # noqa: S310 - local
            self.assertEqual(resp.status, 200)
            stream_text = resp.read().decode("utf-8", errors="ignore")
        self.assertIn("event: round_started", stream_text)
        self.assertIn("event: market_regime", stream_text)
        self.assertIn("event: intel_snapshot", stream_text)
        self.assertIn("event: intel_status", stream_text)
        self.assertIn("event: pre_arbitration", stream_text)
        self.assertIn("event: runtime_guard", stream_text)
        self.assertIn("event: agent_opinion_final", stream_text)
        self.assertIn("event: arbitration_final", stream_text)
        self.assertIn("event: journal_linked", stream_text)
        latest = round_snapshot["rounds"][-1]
        if latest.get("replan_triggered") or latest.get("budget_usage", {}).get("warn"):
            self.assertTrue("event: budget_warning" in stream_text or "event: replan_triggered" in stream_text)
        self.assertIn("event: done", stream_text)

        c3b, events_snapshot = self._get(f"/v1/deep-think/sessions/{session_id}/events?limit=120")
        self.assertEqual(c3b, 200)
        self.assertGreater(events_snapshot["count"], 0)
        self.assertTrue(any(str(x.get("event")) == "round_started" for x in events_snapshot["events"]))
        self.assertTrue(any(str(x.get("event")) == "journal_linked" for x in events_snapshot["events"]))
        c3c, done_events = self._get(f"/v1/deep-think/sessions/{session_id}/events?limit=120&event_name=done")
        self.assertEqual(c3c, 200)
        self.assertEqual(done_events["event_name"], "done")
        self.assertGreater(done_events["count"], 0)
        self.assertTrue(all(str(x.get("event")) == "done" for x in done_events["events"]))
        first_event = events_snapshot["events"][0]
        first_event_id = int(first_event.get("event_id", 0))
        c3d, cursor_page = self._get(f"/v1/deep-think/sessions/{session_id}/events?limit=120&cursor={first_event_id}")
        self.assertEqual(c3d, 200)
        self.assertIn("has_more", cursor_page)
        self.assertIn("next_cursor", cursor_page)
        self.assertTrue(all(int(x.get("event_id", 0)) > first_event_id for x in cursor_page["events"]))
        created_from = urllib.parse.quote(str(first_event.get("created_at", "")))
        c3e, created_page = self._get(
            f"/v1/deep-think/sessions/{session_id}/events?limit=120&created_from={created_from}"
        )
        self.assertEqual(c3e, 200)
        self.assertGreater(created_page["count"], 0)
        with urllib.request.urlopen(
            self.base_url + f"/v1/deep-think/sessions/{session_id}/events/export?format=jsonl&limit=120&event_name=done",
            timeout=8,
        ) as resp:
            self.assertEqual(resp.status, 200)
            self.assertIn("application/x-ndjson", resp.headers.get("Content-Type", ""))
            self.assertIn(".jsonl", resp.headers.get("Content-Disposition", ""))
            exported_jsonl = resp.read().decode("utf-8")
        jsonl_lines = [line for line in exported_jsonl.splitlines() if line.strip()]
        self.assertGreater(len(jsonl_lines), 0)
        self.assertTrue(all(str(json.loads(line).get("event")) == "done" for line in jsonl_lines))
        with urllib.request.urlopen(
            self.base_url + f"/v1/deep-think/sessions/{session_id}/events/export?format=csv&limit=120",
            timeout=8,
        ) as resp:
            self.assertEqual(resp.status, 200)
            self.assertIn("text/csv", resp.headers.get("Content-Type", ""))
            self.assertIn(".csv", resp.headers.get("Content-Disposition", ""))
            exported_csv = resp.read().decode("utf-8")
        self.assertIn("event_id,session_id,round_id,round_no,event_seq,event,created_at,data_json", exported_csv.splitlines()[0].lstrip("\ufeff"))
        with urllib.request.urlopen(
            self.base_url + f"/v1/deep-think/sessions/{session_id}/business-export?format=csv&limit=120",
            timeout=8,
        ) as resp:
            self.assertEqual(resp.status, 200)
            self.assertIn("text/csv", resp.headers.get("Content-Type", ""))
            self.assertIn("deepthink-business", resp.headers.get("Content-Disposition", ""))
            business_csv = resp.read().decode("utf-8")
        self.assertIn("session_id,round_id,round_no,stock_code,signal,confidence", business_csv.splitlines()[0].lstrip("\ufeff"))
        with self.assertRaises(urllib.error.HTTPError) as bad_time:
            urllib.request.urlopen(  # noqa: S310 - local endpoint
                self.base_url + f"/v1/deep-think/sessions/{session_id}/events?created_from=2026-02-15T00:00:00",
                timeout=8,
            )
        self.assertEqual(bad_time.exception.code, 400)

        c3f, export_task = self._post(
            f"/v1/deep-think/sessions/{session_id}/events/export-tasks",
            {"format": "jsonl", "limit": 120, "event_name": "done"},
        )
        self.assertEqual(c3f, 200)
        self.assertGreaterEqual(int(export_task.get("max_attempts", 0)), 1)
        task_id = export_task["task_id"]
        task_snapshot = export_task
        for _ in range(50):
            c3g, task_snapshot = self._get(f"/v1/deep-think/sessions/{session_id}/events/export-tasks/{task_id}")
            self.assertEqual(c3g, 200)
            if task_snapshot["status"] in {"completed", "failed"}:
                break
            time.sleep(0.05)
        self.assertEqual(task_snapshot["status"], "completed")
        self.assertGreaterEqual(int(task_snapshot.get("attempt_count", 0)), 1)
        with urllib.request.urlopen(
            self.base_url + f"/v1/deep-think/sessions/{session_id}/events/export-tasks/{task_id}/download",
            timeout=8,
        ) as resp:
            self.assertEqual(resp.status, 200)
            self.assertIn("application/x-ndjson", resp.headers.get("Content-Type", ""))
            task_jsonl = resp.read().decode("utf-8")
        task_lines = [line for line in task_jsonl.splitlines() if line.strip()]
        self.assertGreater(len(task_lines), 0)
        self.assertTrue(all(str(json.loads(line).get("event")) == "done" for line in task_lines))

        c3h, archive_metrics = self._get("/v1/ops/deep-think/archive-metrics?window_hours=24")
        self.assertEqual(c3h, 200)
        self.assertGreaterEqual(int(archive_metrics.get("total_calls", 0)), 1)
        self.assertIn("by_action", archive_metrics)
        self.assertIn("p95_latency_ms", archive_metrics)
        self.assertIn("p99_latency_ms", archive_metrics)
        self.assertIn("top_sessions", archive_metrics)
        self.assertIn("by_action_status", archive_metrics)

        c4, cards = self._get("/v1/a2a/agent-cards")
        self.assertEqual(c4, 200)
        self.assertTrue(any(card.get("agent_id") == "supervisor_agent" for card in cards))

        c5, task = self._post(
            "/v1/a2a/tasks",
            {
                "agent_id": "supervisor_agent",
                "session_id": session_id,
                "task_type": "deep_round",
                "question": "继续做下一轮",
            },
        )
        self.assertEqual(c5, 200)
        self.assertEqual(task["status"], "completed")
        task_id = task["task_id"]

        c6, loaded_task = self._get(f"/v1/a2a/tasks/{task_id}")
        self.assertEqual(c6, 200)
        self.assertEqual(loaded_task["task_id"], task_id)

    def test_deep_think_v2_round_stream(self) -> None:
        c1, session = self._post(
            "/v1/deep-think/sessions",
            {
                "user_id": "api-deep-v2-1",
                "question": "请流式输出多角色研判过程",
                "stock_codes": ["SH600000"],
                "max_rounds": 2,
            },
        )
        self.assertEqual(c1, 200)
        session_id = session["session_id"]

        body = json.dumps(
            {
                "question": "请执行下一轮并输出过程事件",
                "stock_codes": ["SH600000"],
                "archive_max_events": 220,
            }
        ).encode("utf-8")
        req = urllib.request.Request(
            self.base_url + f"/v2/deep-think/sessions/{session_id}/rounds/stream",
            data=body,
            method="POST",
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=20) as resp:
            self.assertEqual(resp.status, 200)
            stream_text = resp.read().decode("utf-8", errors="ignore")
        self.assertIn("event: round_started", stream_text)
        self.assertIn("event: market_regime", stream_text)
        self.assertIn("event: intel_snapshot", stream_text)
        self.assertIn("event: intel_status", stream_text)
        self.assertIn("event: pre_arbitration", stream_text)
        self.assertIn("event: runtime_guard", stream_text)
        self.assertIn("event: agent_opinion_final", stream_text)
        self.assertIn("event: arbitration_final", stream_text)
        self.assertIn("event: business_summary", stream_text)
        self.assertIn("event: journal_linked", stream_text)
        self.assertIn("event: round_persisted", stream_text)
        self.assertIn("event: done", stream_text)
        self.assertIn('"ok": true', stream_text)
        self.assertIn('"analysis_dimensions"', stream_text)

        c2, loaded = self._get(f"/v1/deep-think/sessions/{session_id}")
        self.assertEqual(c2, 200)
        self.assertEqual(int(loaded.get("current_round", 0)), 1)

    def test_deep_think_report_export(self) -> None:
        c1, session = self._post(
            "/v1/deep-think/sessions",
            {
                "user_id": "api-deep-report",
                "question": "请导出报告",
                "stock_codes": ["SH600000"],
                "max_rounds": 2,
            },
        )
        self.assertEqual(c1, 200)
        session_id = session["session_id"]
        c2, _snapshot = self._post(f"/v1/deep-think/sessions/{session_id}/rounds", {})
        self.assertEqual(c2, 200)

        with urllib.request.urlopen(
            self.base_url + f"/v1/deep-think/sessions/{session_id}/report-export?format=markdown",
            timeout=8,
        ) as resp_md:
            self.assertEqual(resp_md.status, 200)
            self.assertIn("text/markdown", resp_md.headers.get("Content-Type", ""))
            md_text = resp_md.read().decode("utf-8")
        self.assertIn("DeepThink Report", md_text)

        with urllib.request.urlopen(
            self.base_url + f"/v1/deep-think/sessions/{session_id}/report-export?format=pdf",
            timeout=8,
        ) as resp_pdf:
            self.assertEqual(resp_pdf.status, 200)
            self.assertIn("application/pdf", resp_pdf.headers.get("Content-Type", ""))
            pdf_bytes = resp_pdf.read()
        self.assertTrue(pdf_bytes.startswith(b"%PDF-1.4"))

    def test_deep_think_budget_exceeded(self) -> None:
        c1, session = self._post(
            "/v1/deep-think/sessions",
            {
                "user_id": "api-deep-budget",
                "question": "请深度分析SH600000",
                "stock_codes": ["SH600000"],
                "max_rounds": 3,
                "budget": {"token_budget": 10, "time_budget_ms": 10, "tool_call_budget": 1},
            },
        )
        self.assertEqual(c1, 200)
        session_id = session["session_id"]
        c2, snapshot = self._post(f"/v1/deep-think/sessions/{session_id}/rounds", {})
        self.assertEqual(c2, 200)
        latest = snapshot["rounds"][-1]
        self.assertEqual(latest["stop_reason"], "DEEP_BUDGET_EXCEEDED")

    def test_deep_think_runtime_timeout_guard(self) -> None:
        c1, session = self._post(
            "/v1/deep-think/sessions",
            {
                "user_id": "api-deep-timeout",
                "question": "璇锋祴璇昏疆娆¤秴鏃堕檷绾ч€昏緫",
                "stock_codes": ["SH600000"],
                "max_rounds": 3,
            },
        )
        self.assertEqual(c1, 200)
        session_id = session["session_id"]
        c2, snapshot = self._post(
            f"/v1/deep-think/sessions/{session_id}/rounds",
            {"round_timeout_seconds": 0.1, "stage_soft_timeout_seconds": 0.05},
        )
        self.assertEqual(c2, 200)
        latest = snapshot["rounds"][-1]
        self.assertEqual(str(latest.get("stop_reason", "")), "DEEP_ROUND_TIMEOUT")
        self.assertEqual(str(snapshot.get("status", "")), "in_progress")
        runtime_guard = dict((latest.get("budget_usage", {}) or {}).get("runtime_guard", {}) or {})
        self.assertTrue(bool(runtime_guard.get("timed_out", False)))
        c3, events_snapshot = self._get(f"/v1/deep-think/sessions/{session_id}/events?limit=200")
        self.assertEqual(c3, 200)
        self.assertTrue(any(str(x.get("event", "")) == "runtime_timeout" for x in events_snapshot.get("events", [])))

    def test_deep_think_intel_self_test_and_trace(self) -> None:
        c1, probe = self._get("/v1/deep-think/intel/self-test?stock_code=SH600000&question=%E8%87%AA%E6%A3%80")
        self.assertEqual(c1, 200)
        self.assertIn("intel_status", probe)
        self.assertIn("fallback_reason", probe)
        self.assertIn("trace_id", probe)
        self.assertIn("trace_events", probe)
        trace_id = urllib.parse.quote(str(probe.get("trace_id", "")))
        c2, trace_rows = self._get(f"/v1/deep-think/intel/traces/{trace_id}?limit=80")
        self.assertEqual(c2, 200)
        self.assertIn("events", trace_rows)
        self.assertTrue(isinstance(trace_rows["events"], list))


if __name__ == "__main__":
    unittest.main()
