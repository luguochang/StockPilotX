from __future__ import annotations

import base64
import json
import time
import unittest
import uuid
from datetime import datetime, timedelta
from types import SimpleNamespace

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
        self.assertIn("market_regime", result["analysis_brief"])
        self.assertIn("regime_confidence", result["analysis_brief"])
        self.assertIn("signal_guard_applied", result["analysis_brief"])
        self.assertIn("signal_guard_detail", result["analysis_brief"])
        self.assertIn("仅供研究参考", result["answer"])

    def test_query_persists_rag_qa_memory_and_trace(self) -> None:
        _ = self.svc.query(
            {
                "user_id": "u-rag-memory-1",
                "question": "请分析SH600000近期风险与机会并给出证据",
                "stock_codes": ["SH600000"],
            }
        )
        pool = self.svc.rag_qa_memory_list("", stock_code="SH600000", limit=30)
        self.assertGreater(len(pool), 0)
        self.assertIn("summary_text", pool[0])
        self.assertIn("retrieval_enabled", pool[0])

        traces = self.svc.ops_rag_retrieval_trace("", limit=30)
        self.assertGreaterEqual(int(traces.get("count", 0)), 1)

    def test_query_graphrag_mode(self) -> None:
        result = self.svc.query(
            {
                "user_id": "u2",
                "question": "请分析SH600000与行业上下游关系演化",
                "stock_codes": ["SH600000"],
            }
        )
        self.assertEqual(result["mode"], "graph_rag")

    def test_knowledge_graph_view(self) -> None:
        graph = self.svc.knowledge_graph_view("SH600000", limit=20)
        self.assertEqual(str(graph.get("entity_id", "")), "SH600000")
        self.assertGreaterEqual(int(graph.get("relation_count", 0)), 1)
        self.assertTrue(isinstance(graph.get("nodes", []), list))
        self.assertTrue(isinstance(graph.get("relations", []), list))

    def test_docs_recommend(self) -> None:
        _ = self.svc.docs_upload("doc-rec-1", "rec-a.pdf", "SH600000 银行业 利率波动 风险提示 " * 120, "cninfo")
        _ = self.svc.docs_index("doc-rec-1")
        _ = self.svc.docs_upload("doc-rec-2", "rec-b.pdf", "SZ000001 消费行业 复苏 " * 120, "cninfo")
        _ = self.svc.docs_index("doc-rec-2")
        # Build history preference signal.
        self.svc.web.query_history_add(
            "",
            question="请分析SH600000风险",
            stock_codes=["SH600000"],
            trace_id="t-rec-1",
            intent="fact",
            cache_hit=False,
            latency_ms=123,
            summary="history seed",
        )
        result = self.svc.docs_recommend("", {"stock_code": "SH600000", "question": "请分析银行业和利率影响", "top_k": 5})
        self.assertGreaterEqual(int(result.get("count", 0)), 1)
        items = result.get("items", [])
        self.assertTrue(isinstance(items, list))
        self.assertIn("doc-rec-1", [str(x.get("doc_id", "")) for x in items])

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
        versions = self.svc.docs_versions("", "d1", limit=20)
        self.assertGreaterEqual(len(versions), 1)
        self.assertGreaterEqual(int(versions[0].get("version", 0)), 1)
        runs = self.svc.docs_pipeline_runs("", "d1", limit=20)
        self.assertGreaterEqual(len(runs), 2)
        stages = {str(x.get("stage", "")) for x in runs}
        self.assertIn("upload", stages)
        self.assertIn("index", stages)
        # Regression guard: upload/index should not enter review queue anymore.
        docs = self.svc.docs_list("")
        row = next((x for x in docs if str(x.get("doc_id", "")) == "d1"), None)
        self.assertIsNotNone(row)
        self.assertFalse(bool((row or {}).get("needs_review")))
        queue = self.svc.docs_review_queue("")
        self.assertFalse(any(str(x.get("doc_id", "")) == "d1" for x in queue))
        chunks = self.svc.rag_doc_chunks_list("", doc_id="d1", limit=5)
        self.assertTrue(chunks)
        self.assertEqual(str(chunks[0].get("effective_status", "")), "active")

    def test_rag_doc_policy_and_chunk_management(self) -> None:
        policies = self.svc.rag_source_policy_list("")
        self.assertTrue(any(str(x.get("source")) == "cninfo" for x in policies))

        _ = self.svc.docs_upload("rag-doc-1", "cninfo-demo.pdf", "SH600000 业绩说明会纪要" * 120, "cninfo")
        _ = self.svc.docs_index("rag-doc-1")
        chunks = self.svc.rag_doc_chunks_list("", doc_id="rag-doc-1", status="active", limit=30)
        self.assertGreater(len(chunks), 0)
        chunk_id = str(chunks[0]["chunk_id"])

        updated = self.svc.rag_doc_chunk_status_set("", chunk_id, {"status": "review"})
        self.assertEqual(str(updated.get("effective_status", "")), "review")

        _ = self.svc.rag_source_policy_set("", "user_upload", {"auto_approve": True, "trust_score": 0.8, "enabled": True})
        policy_after = self.svc.web.rag_source_policy_get("user_upload")
        self.assertTrue(bool(policy_after.get("auto_approve")))

    def test_runtime_corpus_contains_persisted_docs_and_qa_memory(self) -> None:
        _ = self.svc.docs_upload("rag-doc-2", "cninfo-rag.pdf", "SH600000 经营数据纪要" * 220, "cninfo")
        _ = self.svc.docs_index("rag-doc-2")
        _ = self.svc.query(
            {
                "user_id": "u-rag-memory-2",
                "question": "请分析SH600000近期变化并引用历史经验",
                "stock_codes": ["SH600000"],
            }
        )
        corpus = self.svc._build_runtime_corpus(["SH600000"])  # type: ignore[attr-defined]
        source_ids = [item.source_id for item in corpus]
        self.assertTrue(any(s.startswith("doc::") for s in source_ids))
        self.assertIn("qa_memory_summary", source_ids)

    def test_rag_upload_workflow_and_dashboard(self) -> None:
        encoded = base64.b64encode("SH600000 附件上传测试文本".encode("utf-8")).decode("ascii")
        workflow = self.svc.rag_workflow_upload_and_index(
            "",
            {
                "filename": "upload-demo.txt",
                "content_type": "text/plain",
                "content_base64": encoded,
                "source": "user_upload",
                "stock_codes": ["SH600000"],
                "force_reupload": True,
                "tags": ["测试"],
            },
        )
        self.assertEqual(workflow["status"], "ok")
        self.assertIn("timeline", workflow)
        result = workflow.get("result", {})
        self.assertTrue(str(result.get("doc_id", "")).startswith("ragdoc-"))
        self.assertEqual(str((result.get("asset") or {}).get("status", "")), "active")

        uploads = self.svc.rag_uploads_list("", limit=20)
        self.assertGreater(len(uploads), 0)
        self.assertTrue(any(str(x.get("filename", "")) == "upload-demo.txt" for x in uploads))

        dashboard = self.svc.rag_dashboard("")
        self.assertIn("doc_total", dashboard)
        self.assertIn("active_chunks", dashboard)
        self.assertIn("retrieval_hit_rate_7d", dashboard)

    def test_semantic_summary_then_origin_backfill(self) -> None:
        _ = self.svc.docs_upload("rag-doc-3", "semantic-rag.pdf", "SH600000 纪要 提及现金流改善与订单增长" * 200, "cninfo")
        _ = self.svc.docs_index("rag-doc-3")
        _ = self.svc.query(
            {
                "user_id": "u-rag-memory-3",
                "question": "请分析SH600000现金流改善是否可持续",
                "stock_codes": ["SH600000"],
            }
        )
        reindex = self.svc.ops_rag_reindex("", limit=2000)
        self.assertEqual(reindex["status"], "ok")
        hits = self.svc._semantic_summary_origin_hits("现金流改善 纪要", top_k=5)  # type: ignore[attr-defined]
        self.assertGreater(len(hits), 0)
        self.assertTrue(any(bool((x.metadata or {}).get("origin_backfill")) for x in hits))

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

    def test_portfolio_lifecycle(self) -> None:
        created = self.svc.portfolio_create(
            "",
            {"portfolio_name": "core", "initial_capital": 100000, "description": "phase2 test"},
        )
        pid = int(created["portfolio_id"])
        self.assertGreater(pid, 0)
        _ = self.svc.portfolio_add_transaction(
            "",
            pid,
            {"stock_code": "SH600000", "transaction_type": "buy", "quantity": 100, "price": 10.0, "fee": 1.0},
        )
        summary = self.svc.portfolio_summary("", pid)
        self.assertEqual(int(summary.get("portfolio_id", 0)), pid)
        self.assertGreaterEqual(int(summary.get("position_count", 0)), 1)
        tx = self.svc.portfolio_transactions("", pid, limit=20)
        self.assertGreaterEqual(len(tx), 1)

    def test_alert_rule_lifecycle_and_check(self) -> None:
        rule = self.svc.alert_rule_create(
            "",
            {
                "rule_name": "price > 0",
                "rule_type": "price",
                "stock_code": "SH600000",
                "operator": ">",
                "target_value": 0,
            },
        )
        rid = int(rule["rule_id"])
        self.assertGreater(rid, 0)
        rules = self.svc.alert_rule_list("")
        self.assertTrue(any(int(x.get("rule_id", 0)) == rid for x in rules))
        checked = self.svc.alert_rule_check("")
        self.assertGreaterEqual(int(checked.get("checked_rules", 0)), 1)
        self.assertGreaterEqual(int(checked.get("triggered_count", 0)), 1)
        logs = self.svc.alert_trigger_logs("", limit=20)
        self.assertGreaterEqual(len(logs), 1)
        _ = self.svc.alert_rule_delete("", rid)

    def test_backtest_run_and_get(self) -> None:
        run = self.svc.backtest_run(
            {
                "stock_code": "SH600000",
                "start_date": "2024-01-01",
                "end_date": "2026-02-15",
                "initial_capital": 100000,
                "ma_window": 20,
            }
        )
        self.assertIn("run_id", run)
        self.assertIn("metrics", run)
        loaded = self.svc.backtest_get(run["run_id"])
        self.assertEqual(str(loaded.get("run_id", "")), str(run["run_id"]))

    def test_journal_lifecycle(self) -> None:
        created = self.svc.journal_create(
            "",
            {
                "journal_type": "decision",
                "title": "加仓决策记录",
                "content": "基于现金流改善和估值回落，计划分批加仓。",
                "stock_code": "SH600000",
                "decision_type": "buy",
                "tags": ["银行", "估值", "现金流"],
                "sentiment": "neutral",
            },
        )
        journal_id = int(created.get("journal_id", 0))
        self.assertGreater(journal_id, 0)
        self.assertEqual(str(created.get("stock_code", "")), "SH600000")
        self.assertIn("银行", list(created.get("tags", [])))

        rows = self.svc.journal_list("", limit=20)
        self.assertTrue(any(int(x.get("journal_id", 0)) == journal_id for x in rows))

        filtered = self.svc.journal_list("", journal_type="decision", stock_code="SH600000", limit=20)
        self.assertTrue(any(int(x.get("journal_id", 0)) == journal_id for x in filtered))

        reflection = self.svc.journal_reflection_add(
            "",
            journal_id,
            {
                "reflection_content": "复盘后发现择时偏早，需要增加成交量确认。",
                "ai_insights": "触发条件主要受大盘波动影响。",
                "lessons_learned": "后续加入趋势确认条件。",
            },
        )
        self.assertGreater(int(reflection.get("reflection_id", 0)), 0)

        reflections = self.svc.journal_reflection_list("", journal_id, limit=20)
        self.assertGreaterEqual(len(reflections), 1)
        self.assertTrue(any("择时偏早" in str(x.get("reflection_content", "")) for x in reflections))

        with self.assertRaises(ValueError):
            _ = self.svc.journal_create(
                "",
                {
                    "journal_type": "unknown",
                    "title": "bad",
                    "content": "bad",
                },
            )

    def test_journal_ai_reflection_generate_and_get(self) -> None:
        created = self.svc.journal_create(
            "",
            {
                "journal_type": "decision",
                "title": "AI复盘样本",
                "content": "买入理由是估值回落+现金流改善，计划两周复核。",
                "stock_code": "SH600000",
                "decision_type": "buy",
                "tags": ["复盘", "现金流"],
            },
        )
        journal_id = int(created.get("journal_id", 0))
        self.assertGreater(journal_id, 0)

        _ = self.svc.journal_reflection_add(
            "",
            journal_id,
            {
                "reflection_content": "实际执行偏慢，触发条件定义不够清晰。",
            },
        )
        generated = self.svc.journal_ai_reflection_generate(
            "",
            journal_id,
            {"focus": "强调执行偏差和后续可验证改进"},
        )
        self.assertGreater(int(generated.get("journal_id", 0)), 0)
        self.assertIn(str(generated.get("status", "")), {"ready", "fallback"})
        self.assertTrue(str(generated.get("summary", "")).strip())
        self.assertTrue(isinstance(generated.get("insights", []), list))
        self.assertTrue(isinstance(generated.get("lessons", []), list))
        self.assertGreaterEqual(int(generated.get("latency_ms", 0)), 0)

        loaded = self.svc.journal_ai_reflection_get("", journal_id)
        self.assertEqual(int(loaded.get("journal_id", 0)), journal_id)
        self.assertTrue(str(loaded.get("summary", "")).strip())

    def test_journal_insights(self) -> None:
        created = self.svc.journal_create(
            "",
            {
                "journal_type": "decision",
                "title": "洞察样本A",
                "content": "估值回落后分批建仓，重点观察现金流与量能。",
                "stock_code": "SH600000",
                "decision_type": "buy",
                "tags": ["估值", "现金流", "量能"],
            },
        )
        journal_id = int(created.get("journal_id", 0))
        self.assertGreater(journal_id, 0)
        _ = self.svc.journal_create(
            "",
            {
                "journal_type": "learning",
                "title": "洞察样本B",
                "content": "行业景气度恢复偏慢，优先跟踪盈利兑现速度。",
                "stock_code": "SZ000001",
                "decision_type": "hold",
                "tags": ["行业", "盈利", "景气度"],
            },
        )
        _ = self.svc.journal_reflection_add(
            "",
            journal_id,
            {
                "reflection_content": "执行节奏偏慢，触发条件定义还不够量化。",
            },
        )
        _ = self.svc.journal_ai_reflection_generate(
            "",
            journal_id,
            {"focus": "检查触发条件是否可验证"},
        )

        result = self.svc.journal_insights("", window_days=365, limit=600, timeline_days=60)
        self.assertEqual(str(result.get("status", "")), "ok")
        self.assertGreaterEqual(int(result.get("total_journals", 0)), 2)
        self.assertTrue(isinstance(result.get("type_distribution", []), list))
        self.assertTrue(isinstance(result.get("decision_distribution", []), list))
        self.assertTrue(isinstance(result.get("stock_activity", []), list))
        self.assertTrue(isinstance(result.get("keyword_profile", []), list))
        self.assertTrue(isinstance(result.get("timeline", []), list))

        coverage = result.get("reflection_coverage", {})
        self.assertTrue(isinstance(coverage, dict))
        self.assertGreaterEqual(int(coverage.get("with_reflection", 0)), 1)
        self.assertGreaterEqual(float(coverage.get("reflection_coverage_rate", 0.0)), 0.0)
        self.assertLessEqual(float(coverage.get("reflection_coverage_rate", 0.0)), 1.0)

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

    def test_query_timeout_returns_degraded_payload(self) -> None:
        original = self.svc.query_optimizer.run_with_timeout
        self.svc.query_optimizer.run_with_timeout = (  # type: ignore[assignment]
            lambda fn: (_ for _ in ()).throw(TimeoutError("forced timeout"))
        )
        try:
            result = self.svc.query(
                {
                    "user_id": "u-timeout",
                    "question": "请分析SH600000超时兜底",
                    "stock_codes": ["SH600000"],
                }
            )
        finally:
            self.svc.query_optimizer.run_with_timeout = original  # type: ignore[assignment]

        self.assertTrue(bool(result.get("degraded", False)))
        self.assertEqual(str(result.get("error_code", "")), "query_timeout")
        self.assertIn("query_timeout", result.get("risk_flags", []))
        rows = self.svc.query_history_list("", limit=20, stock_code="SH600000")
        self.assertTrue(any("query_timeout" in str(x.get("error", "")) for x in rows))

    def test_query_history_filter_by_stock_and_time(self) -> None:
        _ = self.svc.query(
            {
                "user_id": "u-history-filter-1",
                "question": "请分析SH600000近况",
                "stock_codes": ["SH600000"],
            }
        )
        _ = self.svc.query(
            {
                "user_id": "u-history-filter-2",
                "question": "请分析SZ000001近况",
                "stock_codes": ["SZ000001"],
            }
        )
        all_rows = self.svc.query_history_list("", limit=80)
        self.assertGreaterEqual(len(all_rows), 2)
        target_ts = str(all_rows[0].get("created_at", ""))

        sh_rows = self.svc.query_history_list("", limit=80, stock_code="SH600000")
        self.assertGreaterEqual(len(sh_rows), 1)
        self.assertTrue(all("SH600000" in list(map(str, x.get("stock_codes", []))) for x in sh_rows))

        empty_rows = self.svc.query_history_list("", limit=80, stock_code="BJ999999")
        self.assertEqual(len(empty_rows), 0)

        same_time_rows = self.svc.query_history_list("", limit=80, created_from=target_ts, created_to=target_ts)
        self.assertGreaterEqual(len(same_time_rows), 1)

        with self.assertRaises(ValueError):
            _ = self.svc.query_history_list(
                "",
                limit=20,
                created_from="2026-02-20 00:00:00",
                created_to="2026-02-19 00:00:00",
            )

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
        self.assertIn("market_regime", names)
        self.assertIn("intel_snapshot", names)
        self.assertIn("intel_status", names)
        self.assertIn("agent_opinion_final", names)
        self.assertIn("arbitration_final", names)
        self.assertIn("business_summary", names)
        self.assertEqual(names[-1], "done")
        business_payload = next((x.get("data", {}) for x in stream_events if str(x.get("event", "")) == "business_summary"), {})
        self.assertIn("market_regime", business_payload)
        self.assertIn("regime_confidence", business_payload)
        self.assertIn("signal_guard_applied", business_payload)
        self.assertIn("confidence_adjustment_detail", business_payload)
        self.assertIn("analysis_dimensions", business_payload)
        self.assertTrue(isinstance(business_payload.get("analysis_dimensions", []), list))
        self.assertGreaterEqual(len(business_payload.get("analysis_dimensions", [])), 5)
        events_snapshot = self.svc.deep_think_list_events(session_id)
        self.assertGreater(events_snapshot["count"], 0)
        stored_names = [str(x.get("event", "")) for x in events_snapshot["events"]]
        self.assertIn("round_started", stored_names)
        self.assertIn("done", stored_names)
        self.assertIn("business_summary", stored_names)
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
        csv_header = str(exported_csv["content"]).splitlines()[0].lstrip("\ufeff")
        self.assertIn(
            "event_id,session_id,round_id,round_no,event_seq,event,created_at,data_json",
            csv_header,
        )
        business_csv = self.svc.deep_think_export_business(session_id, format="csv", limit=80)
        self.assertEqual(business_csv["format"], "csv")
        business_header = str(business_csv["content"]).splitlines()[0].lstrip("\ufeff")
        self.assertIn("session_id,round_id,round_no,stock_code,signal,confidence", business_header)
        self.assertGreater(int(business_csv.get("count", 0)), 0)
        business_json = self.svc.deep_think_export_business(session_id, format="json", limit=80)
        self.assertEqual(business_json["format"], "json")
        parsed_rows = json.loads(str(business_json["content"]))
        self.assertTrue(isinstance(parsed_rows, list))
        self.assertGreater(len(parsed_rows), 0)
        updated2 = self.svc.deep_think_run_round(session_id, {"archive_max_events": 4})
        self.assertEqual(updated2["current_round"], 2)
        trimmed = self.svc.deep_think_list_events(session_id, limit=2000)
        self.assertLessEqual(trimmed["count"], 4)
        latest_round = updated["rounds"][-1]
        has_optional_event = any(name in names for name in ("budget_warning", "replan_triggered"))
        if latest_round.get("replan_triggered") or latest_round.get("budget_usage", {}).get("warn"):
            self.assertTrue(has_optional_event)

    def test_deep_think_v2_stream_round(self) -> None:
        created = self.svc.deep_think_create_session(
            {
                "user_id": "deep-v2-u1",
                "question": "请对SH600000做流式多角色研判",
                "stock_codes": ["SH600000"],
                "max_rounds": 2,
            }
        )
        session_id = created["session_id"]

        stream = self.svc.deep_think_run_round_stream_events(session_id, {"question": "请输出可追踪过程"})
        events: list[dict] = []
        snapshot: dict | None = None
        while True:
            try:
                events.append(next(stream))
            except StopIteration as stop:
                snapshot = stop.value if isinstance(stop.value, dict) else None
                break

        self.assertIsNotNone(snapshot)
        assert snapshot is not None
        self.assertEqual(snapshot["current_round"], 1)
        names = [str(x.get("event", "")) for x in events]
        self.assertIn("round_started", names)
        self.assertIn("market_regime", names)
        self.assertIn("intel_snapshot", names)
        self.assertIn("intel_status", names)
        self.assertIn("agent_opinion_final", names)
        self.assertIn("arbitration_final", names)
        self.assertIn("business_summary", names)
        self.assertIn("round_persisted", names)
        self.assertEqual(names[-1], "done")
        self.assertTrue(bool(events[-1].get("data", {}).get("ok")))
        business_payload = next((x.get("data", {}) for x in events if str(x.get("event", "")) == "business_summary"), {})
        self.assertIn("market_regime", business_payload)
        self.assertIn("regime_confidence", business_payload)
        self.assertIn("signal_guard_applied", business_payload)
        self.assertIn("confidence_adjustment_detail", business_payload)
        self.assertIn("analysis_dimensions", business_payload)
        self.assertGreaterEqual(len(business_payload.get("analysis_dimensions", [])), 5)

        round_ids = {str(x.get("data", {}).get("round_id", "")) for x in events if "data" in x}
        self.assertEqual(len(round_ids), 1)
        for idx, item in enumerate(events, start=1):
            data = item.get("data", {})
            self.assertEqual(int(data.get("event_seq", 0)), idx)
            self.assertEqual(str(data.get("session_id", "")), session_id)
            self.assertTrue(str(data.get("round_id", "")).startswith("dtr-"))

        stored = self.svc.deep_think_list_events(session_id, limit=200)
        stored_names = [str(x.get("event", "")) for x in stored["events"]]
        self.assertIn("round_persisted", stored_names)
        self.assertIn("done", stored_names)

    def test_deep_think_v2_stream_round_mutex_conflict(self) -> None:
        created = self.svc.deep_think_create_session(
            {
                "user_id": "deep-v2-u2",
                "question": "请模拟并发冲突",
                "stock_codes": ["SH600000"],
                "max_rounds": 2,
            }
        )
        session_id = created["session_id"]
        self.assertTrue(self.svc._deep_round_try_acquire(session_id))
        try:
            events = list(self.svc.deep_think_run_round_stream_events(session_id, {}))
        finally:
            self.svc._deep_round_release(session_id)
        self.assertTrue(events)
        self.assertEqual(str(events[-1].get("event", "")), "done")
        self.assertEqual(str(events[-1].get("data", {}).get("error", "")), "round_in_progress")

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

    def test_deep_think_report_export_markdown_and_pdf(self) -> None:
        created = self.svc.deep_think_create_session(
            {
                "user_id": "deep-export-report",
                "question": "请导出深度研判报告",
                "stock_codes": ["SH600000"],
                "max_rounds": 2,
            }
        )
        session_id = created["session_id"]
        _ = self.svc.deep_think_run_round(session_id, {})

        md = self.svc.deep_think_export_report(session_id, format="markdown")
        self.assertEqual(str(md.get("format", "")), "markdown")
        self.assertIn("# DeepThink Report", str(md.get("content", "")))

        pdf = self.svc.deep_think_export_report(session_id, format="pdf")
        self.assertEqual(str(pdf.get("format", "")), "pdf")
        payload = pdf.get("content", b"")
        self.assertTrue(isinstance(payload, (bytes, bytearray)))
        self.assertTrue(bytes(payload).startswith(b"%PDF-1.4"))

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

    def test_deep_think_intel_self_test_and_trace(self) -> None:
        result = self.svc.deep_think_intel_self_test(stock_code="SH600000", question="测试实时情报链路")
        self.assertIn("external_enabled", result)
        self.assertIn("provider_count", result)
        self.assertIn("intel_status", result)
        self.assertIn("fallback_reason", result)
        self.assertIn("trace_id", result)
        self.assertIn("trace_events", result)
        self.assertTrue(isinstance(result["trace_events"], list))
        self.assertIn(result["intel_status"], {"external_ok", "fallback"})
        trace_rows = self.svc.deep_think_trace_events(str(result.get("trace_id", "")), limit=60)
        self.assertIn("events", trace_rows)
        self.assertTrue(isinstance(trace_rows["events"], list))

    def test_deep_fetch_intel_retry_without_tool_when_tool_unsupported(self) -> None:
        # 回归：当 provider 不支持 web-search tool 参数时，应自动回退到无 tools 再尝试一次。
        self.svc.settings.llm_external_enabled = True
        self.svc.llm_gateway.providers = [
            SimpleNamespace(enabled=True, name="mock-provider", api_style="openai_responses", model="mock-model")
        ]
        calls: list[dict[str, object]] = []

        def fake_generate(state, prompt, request_overrides=None):  # noqa: ANN001
            calls.append({"request_overrides": request_overrides})
            if isinstance(request_overrides, dict) and request_overrides.get("tools"):
                raise RuntimeError('HTTP 400: {"detail":"Unsupported tool type: web_search_preview"}')
            return json.dumps(
                {
                    "as_of": "2026-02-15T10:00:00Z",
                    "macro_signals": [
                        {
                            "title": "政策窗口观察",
                            "summary": "测试数据：验证无 tool 回退后仍可返回结构化情报。",
                            "impact_direction": "mixed",
                            "impact_horizon": "1w",
                            "why_relevant_to_stock": "影响风险偏好",
                            "url": "https://example.com/macro",
                            "published_at": "2026-02-15T09:00:00Z",
                            "source_type": "media",
                        }
                    ],
                    "industry_forward_events": [],
                    "stock_specific_catalysts": [],
                    "calendar_watchlist": [],
                    "impact_chain": [],
                    "decision_adjustment": {
                        "signal_bias": "hold",
                        "confidence_adjustment": -0.05,
                        "rationale": "测试回退链路",
                    },
                    "citations": [
                        {
                            "title": "示例来源",
                            "url": "https://example.com/source",
                            "published_at": "2026-02-15T09:00:00Z",
                            "source_type": "media",
                        }
                    ],
                },
                ensure_ascii=False,
            )

        original_generate = self.svc.llm_gateway.generate
        self.svc.llm_gateway.generate = fake_generate  # type: ignore[assignment]
        try:
            result = self.svc._deep_fetch_intel_via_llm_websearch(
                stock_code="SH600000",
                question="请做实时情报自检",
                quote={"price": 10.5, "pct_change": 0.8},
                trend={"momentum_20": 0.03, "max_drawdown_60": 0.12},
                quant_20={"signal": "hold"},
            )
        finally:
            self.svc.llm_gateway.generate = original_generate  # type: ignore[assignment]

        self.assertEqual(len(calls), 2)
        self.assertTrue(bool(calls[0]["request_overrides"]))
        self.assertFalse(bool(calls[1]["request_overrides"]))
        self.assertEqual(result["intel_status"], "external_ok")
        self.assertEqual(result["websearch_tool_requested"], True)
        self.assertEqual(result["websearch_tool_applied"], False)
        self.assertGreaterEqual(len(result.get("citations", [])), 1)

    def test_deep_validate_intel_payload_accepts_text_confidence_adjustment(self) -> None:
        payload = {
            "as_of": "2026-02-15T00:00:00Z",
            "macro_signals": [
                {
                    "title": "测试",
                    "summary": "测试文本置信度",
                    "impact_direction": "mixed",
                    "impact_horizon": "1w",
                    "why_relevant_to_stock": "测试",
                    "url": "https://example.com",
                    "published_at": "2026-02-15T00:00:00Z",
                    "source_type": "media",
                }
            ],
            "decision_adjustment": {
                "signal_bias": "hold",
                "confidence_adjustment": "down",
                "rationale": "文本置信度表达",
            },
            "citations": [{"title": "x", "url": "https://example.com/x", "published_at": "", "source_type": "media"}],
        }
        normalized = self.svc._deep_validate_intel_payload(payload)  # type: ignore[attr-defined]
        self.assertIn("decision_adjustment", normalized)
        self.assertAlmostEqual(float(normalized["decision_adjustment"]["confidence_adjustment"]), -0.12, places=3)

    def test_augment_question_with_history_context_includes_3m_summary(self) -> None:
        code = "SH600000"
        base = datetime(2025, 10, 1)
        rows = []
        for i in range(95):
            day = (base + timedelta(days=i)).strftime("%Y-%m-%d")
            close = 10.0 + i * 0.03
            rows.append(
                {
                    "stock_code": code,
                    "trade_date": day,
                    "open": close - 0.05,
                    "close": close,
                    "high": close + 0.08,
                    "low": close - 0.1,
                    "volume": 100000 + i,
                    "source_id": "eastmoney_history",
                    "source_url": "https://example.com/kline",
                    "reliability_score": 0.9,
                }
            )
        self.svc.ingestion_store.history_bars = rows
        enriched = self.svc._augment_question_with_history_context("请做高级分析", [code])  # type: ignore[attr-defined]
        self.assertIn("系统补充", enriched)
        self.assertIn("最近三个月连续日线样本", enriched)
        self.assertIn(code, enriched)

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

