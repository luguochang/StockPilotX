from __future__ import annotations

import json
import subprocess
import time
import unittest
import urllib.request
from pathlib import Path


class WebEndpointsE2ETestCase(unittest.TestCase):
    """WEB-009 关键路径 E2E：鉴权->业务->运营。"""

    @classmethod
    def setUpClass(cls) -> None:
        root = Path(__file__).resolve().parents[1]
        cls.base_url = "http://127.0.0.1:8013"
        py = root / ".venv" / "Scripts" / "python.exe"
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
                "8013",
                "--log-level",
                "warning",
            ],
            cwd=str(root),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        cls._wait_ready()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.proc.terminate()
        try:
            cls.proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            cls.proc.kill()

    @classmethod
    def _wait_ready(cls) -> None:
        deadline = time.time() + 10
        while time.time() < deadline:
            try:
                with urllib.request.urlopen(cls.base_url + "/docs", timeout=2) as r:  # noqa: S310
                    if r.status == 200:
                        return
            except Exception:
                time.sleep(0.2)
        raise RuntimeError("server not ready")

    def _post(self, path: str, payload: dict, token: str | None = None) -> tuple[int, dict]:
        data = json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if token:
            headers["Authorization"] = f"Bearer {token}"
        req = urllib.request.Request(self.base_url + path, data=data, method="POST", headers=headers)
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))

    def _get(self, path: str, token: str | None = None) -> tuple[int, dict | list]:
        headers = {}
        if token:
            headers["Authorization"] = f"Bearer {token}"
        req = urllib.request.Request(self.base_url + path, method="GET", headers=headers)
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))

    def _delete(self, path: str, token: str | None = None) -> tuple[int, dict]:
        headers = {}
        if token:
            headers["Authorization"] = f"Bearer {token}"
        req = urllib.request.Request(self.base_url + path, method="DELETE", headers=headers)
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))

    def test_e2e_main_paths(self) -> None:
        uname = f"web_user_{int(time.time() * 1000)}"
        # auth
        _, _ = self._post("/v1/auth/register", {"username": uname, "password": "pw123456", "tenant_name": "teamA"})
        _, login = self._post("/v1/auth/login", {"username": uname, "password": "pw123456"})
        token = login["access_token"]
        code, me = self._get("/v1/auth/me", token=token)
        self.assertEqual(code, 200)
        self.assertEqual(me["username"], uname)

        # watchlist + dashboard
        self._post("/v1/watchlist", {"stock_code": "SH600000"}, token=token)
        _, wl = self._get("/v1/watchlist", token=token)
        self.assertGreaterEqual(len(wl), 1)
        _, dash = self._get("/v1/dashboard/overview", token=token)
        self.assertIn("watchlist_count", dash)
        self._delete("/v1/watchlist/SH600000", token=token)

        # reports center
        _, rep = self._post(
            "/v1/report/generate",
            {"user_id": uname, "stock_code": "SH600000", "period": "1y", "report_type": "fact", "token": token},
        )
        report_id = rep["report_id"]
        _, reports = self._get("/v1/reports", token=token)
        self.assertGreaterEqual(len(reports), 1)
        _, versions = self._get(f"/v1/reports/{report_id}/versions", token=token)
        self.assertGreaterEqual(len(versions), 1)
        _, exported = self._post(f"/v1/reports/{report_id}/export", {}, token=token)
        self.assertIn("markdown", exported)

        # docs center
        self._post("/v1/docs/upload", {"doc_id": "web-doc-1", "filename": "x.pdf", "content": "PDF内容", "source": "web"})
        self._post("/v1/docs/web-doc-1/index", {})
        _, docs = self._get("/v1/docs", token=token)
        self.assertGreaterEqual(len(docs), 1)
        _, q = self._get("/v1/docs/review-queue", token=token)
        self.assertIsInstance(q, list)
        self._post("/v1/docs/web-doc-1/review/approve", {"comment": "ok"}, token=token)

        # ops center
        _, health = self._get("/v1/ops/data-sources/health", token=token)
        self.assertIsInstance(health, list)
        _, _ = self._post("/v1/evals/run", {"samples": [{"fact_correct": True, "has_citation": True, "hallucination": False, "violation": False}]})
        _, evals = self._get("/v1/ops/evals/history", token=token)
        self.assertIsInstance(evals, list)
        _, releases = self._get("/v1/ops/prompts/releases", token=token)
        self.assertIsInstance(releases, list)
        _, _ = self._post("/v1/scheduler/pause", {"job_name": "intraday_quote_ingest"}, token=token)
        _, _ = self._post("/v1/scheduler/resume", {"job_name": "intraday_quote_ingest"}, token=token)
        _, alerts = self._get("/v1/alerts", token=token)
        self.assertIsInstance(alerts, list)
        self._post("/v1/alerts/1/ack", {}, token=token)

        # prediction center
        _, pr = self._post("/v1/predict/run", {"stock_codes": ["SH600000"], "horizons": ["5d", "20d"]})
        self.assertIn("run_id", pr)
        _, pf = self._get("/v1/factors/SH600000")
        self.assertIn("factors", pf)
        _, pe = self._get("/v1/predict/evals/latest")
        self.assertIn("metrics", pe)


if __name__ == "__main__":
    unittest.main()
