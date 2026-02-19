from __future__ import annotations

import unittest

from backend.app.data.ingestion import IngestionService, IngestionStore
from backend.app.data.sources import AnnouncementService, QuoteService, TencentAdapter


class _FixedFinancialService:
    def get_financial_snapshot(self, stock_code: str) -> dict:
        return {
            "stock_code": stock_code,
            "report_period": "20241231",
            "roe": 13.2,
            "gross_margin": 40.1,
            "revenue_yoy": 8.3,
            "net_profit_yoy": 10.7,
            "asset_liability_ratio": 52.4,
            "pe_ttm": 15.6,
            "pb_mrq": 1.8,
            "ts": "2026-02-19T00:00:00+00:00",
            "source_id": "financial_mock_service",
            "source_url": "https://example.com/financial",
            "reliability_score": 0.9,
            "source_note": "unit_test",
        }


class IngestionFinancialTestCase(unittest.TestCase):
    def test_ingest_financials_success(self) -> None:
        store = IngestionStore()
        svc = IngestionService(
            quote_service=QuoteService([TencentAdapter()]),
            announcement_service=AnnouncementService(adapters=[]),
            financial_service=_FixedFinancialService(),
            store=store,
        )
        result = svc.ingest_financials(["600000"])
        self.assertEqual(result["failed_count"], 0)
        self.assertEqual(result["success_count"], 1)
        self.assertEqual(len(store.financial_snapshots), 1)
        self.assertEqual(store.financial_snapshots[0]["stock_code"], "SH600000")

    def test_ingest_financials_not_configured(self) -> None:
        store = IngestionStore()
        svc = IngestionService(
            quote_service=QuoteService([TencentAdapter()]),
            announcement_service=AnnouncementService(adapters=[]),
            store=store,
        )
        result = svc.ingest_financials(["SH600000"])
        self.assertEqual(result["success_count"], 0)
        self.assertEqual(result["failed_count"], 1)


if __name__ == "__main__":
    unittest.main()

