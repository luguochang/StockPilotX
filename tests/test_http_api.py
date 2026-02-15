from __future__ import annotations

import json
import subprocess
import time
import unittest
import urllib.error
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
        self.assertIn(body["workflow_runtime"], ("langgraph", "direct"))

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
        self.assertIn("event: stream_runtime", text)
        self.assertIn("event: answer_delta", text)
        self.assertIn("event: stream_source", text)
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


if __name__ == "__main__":
    unittest.main()
