from __future__ import annotations

import json
import time
import unittest
import uuid

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

    def test_deep_think_session_and_round(self) -> None:
        created = self.svc.deep_think_create_session(
            {
                "user_id": "deep-u1",
                "question": "请对SH600000做深度多角色研判",
                "stock_codes": ["SH600000"],
                "max_rounds": 2,
            }
        )
        self.assertIn("session_id", created)
        session_id = created["session_id"]
        self.assertEqual(created["current_round"], 0)

        updated = self.svc.deep_think_run_round(session_id, {})
        self.assertEqual(updated["current_round"], 1)
        self.assertTrue(updated["rounds"])
        latest = updated["rounds"][-1]
        self.assertIn("consensus_signal", latest)
        self.assertGreaterEqual(len(latest["opinions"]), 8)
        self.assertIn("task_graph", latest)
        self.assertTrue(latest["task_graph"])
        self.assertIn("budget_usage", latest)

        stream_events = list(self.svc.deep_think_stream_events(session_id))
        names = [x["event"] for x in stream_events]
        self.assertIn("round_started", names)
        self.assertIn("agent_opinion_final", names)
        self.assertIn("arbitration_final", names)
        self.assertEqual(names[-1], "done")
        events_snapshot = self.svc.deep_think_list_events(session_id)
        self.assertGreater(events_snapshot["count"], 0)
        stored_names = [str(x.get("event", "")) for x in events_snapshot["events"]]
        self.assertIn("round_started", stored_names)
        self.assertIn("done", stored_names)
        done_only = self.svc.deep_think_list_events(session_id, event_name="done")
        self.assertGreater(done_only["count"], 0)
        self.assertTrue(all(str(x.get("event", "")) == "done" for x in done_only["events"]))
        paged = self.svc.deep_think_list_events(session_id, limit=2)
        self.assertEqual(paged["limit"], 2)
        self.assertIsInstance(paged["has_more"], bool)
        self.assertGreater(paged["count"], 0)
        if paged["has_more"]:
            self.assertIsNotNone(paged["next_cursor"])
        first_event_id = int(paged["events"][0]["event_id"])
        cursor_filtered = self.svc.deep_think_list_events(session_id, limit=200, cursor=first_event_id)
        self.assertTrue(all(int(x.get("event_id", 0)) > first_event_id for x in cursor_filtered["events"]))
        first_created_at = str(paged["events"][0].get("created_at", ""))
        created_filtered = self.svc.deep_think_list_events(session_id, limit=200, created_from=first_created_at)
        self.assertGreater(created_filtered["count"], 0)
        self.assertTrue(all(str(x.get("created_at", "")) >= first_created_at for x in created_filtered["events"]))
        exported_done = self.svc.deep_think_export_events(session_id, event_name="done", format="jsonl", limit=50)
        self.assertEqual(exported_done["format"], "jsonl")
        jsonl_lines = [line for line in str(exported_done["content"]).splitlines() if line.strip()]
        self.assertGreater(len(jsonl_lines), 0)
        self.assertTrue(all(str(json.loads(line).get("event")) == "done" for line in jsonl_lines))
        exported_csv = self.svc.deep_think_export_events(session_id, format="csv", limit=50)
        self.assertEqual(exported_csv["format"], "csv")
        self.assertIn(
            "event_id,session_id,round_id,round_no,event_seq,event,created_at,data_json",
            str(exported_csv["content"]).splitlines()[0],
        )
        updated2 = self.svc.deep_think_run_round(session_id, {"archive_max_events": 4})
        self.assertEqual(updated2["current_round"], 2)
        trimmed = self.svc.deep_think_list_events(session_id, limit=2000)
        self.assertLessEqual(trimmed["count"], 4)
        latest_round = updated["rounds"][-1]
        has_optional_event = any(name in names for name in ("budget_warning", "replan_triggered"))
        if latest_round.get("replan_triggered") or latest_round.get("budget_usage", {}).get("warn"):
            self.assertTrue(has_optional_event)

    def test_deep_think_budget_exceeded_stop(self) -> None:
        created = self.svc.deep_think_create_session(
            {
                "user_id": "deep-u3",
                "question": "请做深度多角色分析",
                "stock_codes": ["SH600000"],
                "budget": {"token_budget": 10, "time_budget_ms": 10, "tool_call_budget": 1},
                "max_rounds": 3,
            }
        )
        snapshot = self.svc.deep_think_run_round(created["session_id"], {})
        latest = snapshot["rounds"][-1]
        self.assertEqual(latest["stop_reason"], "DEEP_BUDGET_EXCEEDED")
        self.assertEqual(snapshot["status"], "completed")

    def test_deep_think_export_task_and_archive_metrics(self) -> None:
        created = self.svc.deep_think_create_session(
            {
                "user_id": "deep-export-user",
                "question": "请导出deep-think归档",
                "stock_codes": ["SH600000"],
                "max_rounds": 2,
            }
        )
        session_id = created["session_id"]
        updated = self.svc.deep_think_run_round(session_id, {})
        self.assertEqual(updated["current_round"], 1)
        with self.assertRaises(ValueError):
            _ = self.svc.deep_think_list_events(session_id, created_from="2026-02-15T00:00:00")
        with self.assertRaises(ValueError):
            _ = self.svc.deep_think_list_events(
                session_id,
                created_from="2026-02-15 12:00:00",
                created_to="2026-02-15 01:00:00",
            )
        task = self.svc.deep_think_create_export_task(session_id, format="jsonl", limit=80)
        self.assertEqual(task["session_id"], session_id)
        task_id = task["task_id"]
        final_task = task
        for _ in range(50):
            final_task = self.svc.deep_think_get_export_task(session_id, task_id)
            if final_task.get("status") in {"completed", "failed"}:
                break
            time.sleep(0.05)
        self.assertEqual(final_task["status"], "completed")
        self.assertGreaterEqual(int(final_task.get("attempt_count", 0)), 1)
        self.assertGreaterEqual(int(final_task.get("max_attempts", 0)), int(final_task.get("attempt_count", 0)))
        exported = self.svc.deep_think_download_export_task(session_id, task_id)
        self.assertEqual(exported["status"], "completed")
        self.assertTrue(str(exported["content"]).strip())
        self.assertEqual(exported["format"], "jsonl")
        metrics = self.svc.ops_deep_think_archive_metrics("", window_hours=24)
        self.assertGreater(metrics["total_calls"], 0)
        self.assertIn("p95_latency_ms", metrics)
        self.assertIn("p99_latency_ms", metrics)
        self.assertIn("by_action_status", metrics)
        self.assertIn("top_sessions", metrics)
        actions = {str(x.get("action", "")) for x in metrics.get("by_action", [])}
        self.assertIn("archive_query", actions)
        self.assertIn("archive_export_task_create", actions)

    def test_deep_think_export_task_retry_attempts(self) -> None:
        created = self.svc.deep_think_create_session(
            {
                "user_id": "deep-retry-user",
                "question": "test deep-think archive export retry path",
                "stock_codes": ["SH600000"],
                "max_rounds": 2,
            }
        )
        session_id = created["session_id"]
        self.svc.deep_think_run_round(session_id, {})
        original_export = self.svc.deep_think_export_events
        fail_once = {"done": False}

        def flaky_export(*args, **kwargs):
            if not fail_once["done"]:
                fail_once["done"] = True
                raise RuntimeError("transient_export_error")
            return original_export(*args, **kwargs)

        self.svc.deep_think_export_events = flaky_export  # type: ignore[assignment]
        try:
            filters = self.svc._build_deep_archive_query_options(limit=80)  # type: ignore[attr-defined]
            task_id = f"dtexp-retry-{uuid.uuid4().hex[:10]}"
            self.svc.web.deep_think_export_task_create(
                task_id=task_id,
                session_id=session_id,
                status="queued",
                format="jsonl",
                filters=filters,
                max_attempts=2,
            )
            self.svc._run_deep_archive_export_task(task_id, session_id, "jsonl", filters)  # type: ignore[attr-defined]
            final_task = self.svc.deep_think_get_export_task(session_id, task_id)
            self.assertEqual(final_task["status"], "completed")
            self.assertGreaterEqual(int(final_task.get("attempt_count", 0)), 2)
            self.assertGreaterEqual(int(final_task.get("max_attempts", 0)), int(final_task.get("attempt_count", 0)))
        finally:
            self.svc.deep_think_export_events = original_export  # type: ignore[assignment]

    def test_a2a_task_lifecycle(self) -> None:
        created = self.svc.deep_think_create_session(
            {
                "user_id": "deep-u2",
                "question": "请多角色评估SH600000风险",
                "stock_codes": ["SH600000"],
            }
        )
        session_id = created["session_id"]
        cards = self.svc.a2a_agent_cards()
        self.assertTrue(any(x.get("agent_id") == "supervisor_agent" for x in cards))
        task = self.svc.a2a_create_task(
            {
                "agent_id": "supervisor_agent",
                "session_id": session_id,
                "task_type": "deep_round",
            }
        )
        self.assertEqual(task["status"], "completed")
        self.assertEqual(task["agent_id"], "supervisor_agent")


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

