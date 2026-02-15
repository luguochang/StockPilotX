from __future__ import annotations

import unittest

from backend.app.data.ingestion import IngestionService, IngestionStore
from backend.app.data.sources import AnnouncementService, QuoteService, TencentAdapter


class _FixedAnnouncementService(AnnouncementService):
    def __init__(self) -> None:
        pass

    def fetch_announcements(self, stock_code: str) -> list[dict]:
        return [
            {
                "stock_code": stock_code,
                "event_type": "announcement",
                "title": "同名公告",
                "content": "版本A",
                "event_time": "2026-02-13T00:00:00+00:00",
                "source_id": "cninfo",
                "source_url": "https://www.cninfo.com.cn/",
                "reliability_score": 0.98,
            }
        ]


class _FixedAnnouncementServiceB(AnnouncementService):
    def __init__(self) -> None:
        pass

    def fetch_announcements(self, stock_code: str) -> list[dict]:
        return [
            {
                "stock_code": stock_code,
                "event_type": "announcement",
                "title": "同名公告",
                "content": "版本B",
                "event_time": "2026-02-13T00:00:00+00:00",
                "source_id": "sse",
                "source_url": "https://www.sse.com.cn/",
                "reliability_score": 0.97,
            }
        ]


class IngestionQualityTestCase(unittest.TestCase):
    """DATA-003：标准化、质量标记与冲突标记测试。"""

    def test_quote_normalization(self) -> None:
        store = IngestionStore()
        svc = IngestionService(
            quote_service=QuoteService([TencentAdapter()]),
            announcement_service=_FixedAnnouncementService(),
            store=store,
        )
        result = svc.ingest_market_daily(["600000"])
        self.assertEqual(result["failed_count"], 0)
        row = store.quotes[0]
        self.assertEqual(row["stock_code"], "SH600000")
        self.assertIn("quality_flags", row)
        self.assertIn("conflict_flag", row)

    def test_announcement_conflict_flag(self) -> None:
        store = IngestionStore()
        svc_a = IngestionService(
            quote_service=QuoteService([TencentAdapter()]),
            announcement_service=_FixedAnnouncementService(),
            store=store,
        )
        svc_b = IngestionService(
            quote_service=QuoteService([TencentAdapter()]),
            announcement_service=_FixedAnnouncementServiceB(),
            store=store,
        )
        svc_a.ingest_announcements(["SH600000"])
        svc_b.ingest_announcements(["SH600000"])
        self.assertFalse(store.announcements[0]["conflict_flag"])
        self.assertTrue(store.announcements[1]["conflict_flag"])


if __name__ == "__main__":
    unittest.main()

