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
        with urllib.request.urlopen(req, timeout=8) as resp:
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
            with urllib.request.urlopen(req, timeout=8) as resp:
                return resp.status, json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as ex:
            return ex.code, json.loads(ex.read().decode("utf-8"))

    def _get(self, path: str) -> tuple[int, dict]:
        with urllib.request.urlopen(self.base_url + path, timeout=8) as resp:  # noqa: S310 - local smoke url
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

    def test_ingest_market_daily(self) -> None:
        code, body = self._post("/v1/ingest/market-daily", {"stock_codes": ["SH600000", "SZ000001"]})
        self.assertEqual(code, 200)
        self.assertEqual(body["task_name"], "market-daily")

    def test_ingest_announcements(self) -> None:
        code, body = self._post("/v1/ingest/announcements", {"stock_codes": ["SH600000"]})
        self.assertEqual(code, 200)
        self.assertEqual(body["task_name"], "announcements")

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

        c9, dashboard = self._get("/v1/rag/dashboard")
        self.assertEqual(c9, 200)
        self.assertIn("doc_total", dashboard)
        self.assertIn("active_chunks", dashboard)

        c10, upload_rows = self._get("/v1/rag/uploads?limit=10")
        self.assertEqual(c10, 200)
        self.assertTrue(isinstance(upload_rows, list))
        self.assertTrue(any(str(x.get("filename", "")) == "api-upload.txt" for x in upload_rows))

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

        get_code, loaded = self._get(f"/v1/predict/{run['run_id']}")
        self.assertEqual(get_code, 200)
        self.assertEqual(loaded["run_id"], run["run_id"])

        factor_code, factor = self._get("/v1/factors/SH600000")
        self.assertEqual(factor_code, 200)
        self.assertIn("factors", factor)
        self.assertIn("risk_score", factor["factors"])
        self.assertIn("history_data_mode", factor["source"])

        eval_code, latest = self._get("/v1/predict/evals/latest")
        self.assertEqual(eval_code, 200)
        self.assertIn("metrics", latest)

    def test_market_overview(self) -> None:
        code, body = self._get("/v1/market/overview/SH600000")
        self.assertEqual(code, 200)
        self.assertIn("realtime", body)
        self.assertIn("history", body)

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
        self.assertIn("event: agent_opinion_final", stream_text)
        self.assertIn("event: arbitration_final", stream_text)
        latest = round_snapshot["rounds"][-1]
        if latest.get("replan_triggered") or latest.get("budget_usage", {}).get("warn"):
            self.assertTrue("event: budget_warning" in stream_text or "event: replan_triggered" in stream_text)
        self.assertIn("event: done", stream_text)

        c3b, events_snapshot = self._get(f"/v1/deep-think/sessions/{session_id}/events?limit=120")
        self.assertEqual(c3b, 200)
        self.assertGreater(events_snapshot["count"], 0)
        self.assertTrue(any(str(x.get("event")) == "round_started" for x in events_snapshot["events"]))
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
        self.assertIn("event: agent_opinion_final", stream_text)
        self.assertIn("event: arbitration_final", stream_text)
        self.assertIn("event: business_summary", stream_text)
        self.assertIn("event: round_persisted", stream_text)
        self.assertIn("event: done", stream_text)
        self.assertIn('"ok": true', stream_text)

        c2, loaded = self._get(f"/v1/deep-think/sessions/{session_id}")
        self.assertEqual(c2, 200)
        self.assertEqual(int(loaded.get("current_round", 0)), 1)

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
