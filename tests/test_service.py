from __future__ import annotations

import unittest

from backend.app.data.sources import NeteaseAdapter, QuoteService, SinaAdapter, TencentAdapter
from backend.app.service import AShareAgentService


class ServiceTestCase(unittest.TestCase):
    """系统主流程自测。"""

    def setUp(self) -> None:
        self.svc = AShareAgentService()

    def test_query_basic(self) -> None:
        result = self.svc.query(
            {
                "user_id": "u1",
                "question": "请分析SH600000的最新风险和机会",
                "stock_codes": ["SH600000"],
            }
        )
        self.assertIn("trace_id", result)
        self.assertTrue(result["citations"])
        self.assertIn("analysis_brief", result)
        self.assertIn("仅供研究参考", result["answer"])

    def test_query_graphrag_mode(self) -> None:
        result = self.svc.query(
            {
                "user_id": "u2",
                "question": "请分析SH600000与行业上下游关系演化",
                "stock_codes": ["SH600000"],
            }
        )
        self.assertEqual(result["mode"], "graph_rag")

    def test_report_generate_and_get(self) -> None:
        generated = self.svc.report_generate(
            {"user_id": "u3", "stock_code": "SH600000", "period": "1y", "report_type": "fact"}
        )
        loaded = self.svc.report_get(generated["report_id"])
        self.assertIn("# SH600000", loaded["markdown"])

    def test_ingest_endpoints(self) -> None:
        daily = self.svc.ingest_market_daily(["SH600000", "SZ000001"])
        ann = self.svc.ingest_announcements(["SH600000"])
        self.assertEqual(daily["failed_count"], 0)
        self.assertEqual(ann["success_count"], 1)

    def test_doc_upload_and_index(self) -> None:
        up = self.svc.docs_upload("d1", "demo.pdf", "财报正文" * 600, "user")
        idx = self.svc.docs_index("d1")
        self.assertEqual(up["status"], "uploaded")
        self.assertEqual(idx["status"], "indexed")
        self.assertGreater(idx["chunk_count"], 1)

    def test_eval_gate(self) -> None:
        run = self.svc.evals_run(
            [
                {"fact_correct": True, "has_citation": True, "hallucination": False, "violation": False},
                {"fact_correct": True, "has_citation": True, "hallucination": False, "violation": False},
            ]
        )
        self.assertTrue(run["pass_gate"])

    def test_prediction_run_and_eval(self) -> None:
        run = self.svc.predict_run({"stock_codes": ["SH600000", "SZ000001"], "horizons": ["5d", "20d"]})
        self.assertIn("run_id", run)
        self.assertEqual(len(run["results"]), 2)
        self.assertIn("history_data_mode", run["results"][0]["source"])
        latest = self.svc.predict_eval_latest()
        self.assertEqual(latest["status"], "ok")
        self.assertIn("ic", latest["metrics"])

    def test_market_overview_contains_realtime_and_history(self) -> None:
        data = self.svc.market_overview("SH600000")
        self.assertIn("realtime", data)
        self.assertIn("history", data)
        self.assertGreater(len(data["history"]), 30)

    def test_query_repeated_calls_do_not_hit_global_model_limit(self) -> None:
        # 回归：预算计数应按请求重置，连续请求不应在第9次后失败。
        for idx in range(12):
            result = self.svc.query(
                {
                    "user_id": f"u-repeat-{idx}",
                    "question": "请分析SH600000近期风险与机会",
                    "stock_codes": ["SH600000"],
                }
            )
            self.assertIn("answer", result)


class QuoteFallbackTestCase(unittest.TestCase):
    """验证免费源回退顺序。"""

    def test_fallback_order(self) -> None:
        t = TencentAdapter(fail_codes={"SH600000"})
        n = NeteaseAdapter()
        s = SinaAdapter()
        svc = QuoteService([t, n, s])
        quote = svc.get_quote("SH600000")
        self.assertEqual(quote.source_id, "netease")


if __name__ == "__main__":
    unittest.main()
