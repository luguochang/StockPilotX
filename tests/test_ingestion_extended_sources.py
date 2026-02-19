from __future__ import annotations

import unittest

from backend.app.data.ingestion import IngestionService, IngestionStore
from backend.app.data.sources import AnnouncementService, QuoteService, TencentAdapter


class _FixedNewsService:
    def fetch_news(self, stock_code: str, limit: int = 20) -> list[dict]:
        _ = limit
        return [
            {
                "stock_code": stock_code,
                "title": f"{stock_code} 新闻标题",
                "content": "新闻正文",
                "event_time": "2026-02-19T10:00:00+00:00",
                "source_id": "news_fixed",
                "source_url": "https://example.com/news",
                "reliability_score": 0.8,
            }
        ]


class _FixedResearchService:
    def fetch_reports(self, stock_code: str, limit: int = 20) -> list[dict]:
        _ = limit
        return [
            {
                "stock_code": stock_code,
                "title": f"{stock_code} 研报标题",
                "content": "研报正文",
                "published_at": "2026-02-18T10:00:00+00:00",
                "author": "analyst",
                "org_name": "org",
                "source_id": "research_fixed",
                "source_url": "https://example.com/research",
                "reliability_score": 0.82,
            }
        ]


class _FixedMacroService:
    def fetch_macro_indicators(self, limit: int = 20) -> list[dict]:
        _ = limit
        return [
            {
                "metric_name": "CPI",
                "metric_value": "2.1",
                "report_date": "2026-01-31",
                "event_time": "2026-01-31T00:00:00+00:00",
                "source_id": "macro_fixed",
                "source_url": "https://example.com/macro",
                "reliability_score": 0.8,
            }
        ]


class _FixedFundService:
    def fetch_fund_snapshot(self, stock_code: str) -> dict:
        return {
            "stock_code": stock_code,
            "trade_date": "2026-02-19",
            "main_inflow": 100.0,
            "small_inflow": 20.0,
            "middle_inflow": 30.0,
            "large_inflow": 40.0,
            "ts": "2026-02-19T10:00:00+00:00",
            "source_id": "fund_fixed",
            "source_url": "https://example.com/fund",
            "reliability_score": 0.8,
        }


class IngestionExtendedSourcesTestCase(unittest.TestCase):
    def test_ingest_extended_sources_success(self) -> None:
        store = IngestionStore()
        svc = IngestionService(
            quote_service=QuoteService([TencentAdapter()]),
            announcement_service=AnnouncementService(adapters=[]),
            news_service=_FixedNewsService(),
            research_service=_FixedResearchService(),
            macro_service=_FixedMacroService(),
            fund_service=_FixedFundService(),
            store=store,
        )

        news_result = svc.ingest_news(["600000"], limit=3)
        research_result = svc.ingest_research_reports(["600000"], limit=3)
        macro_result = svc.ingest_macro_indicators(limit=3)
        fund_result = svc.ingest_fund_snapshots(["600000"])

        self.assertEqual(news_result["success_count"], 1)
        self.assertEqual(research_result["success_count"], 1)
        self.assertEqual(macro_result["success_count"], 1)
        self.assertEqual(fund_result["success_count"], 1)
        self.assertEqual(store.news_items[0]["stock_code"], "SH600000")
        self.assertEqual(store.research_reports[0]["stock_code"], "SH600000")
        self.assertEqual(store.macro_indicators[0]["metric_name"], "CPI")
        self.assertEqual(store.fund_snapshots[0]["stock_code"], "SH600000")

    def test_ingest_news_not_configured(self) -> None:
        store = IngestionStore()
        svc = IngestionService(
            quote_service=QuoteService([TencentAdapter()]),
            announcement_service=AnnouncementService(adapters=[]),
            store=store,
        )
        result = svc.ingest_news(["SH600000"])
        self.assertEqual(result["success_count"], 0)
        self.assertEqual(result["failed_count"], 1)


if __name__ == "__main__":
    unittest.main()
