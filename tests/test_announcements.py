from __future__ import annotations

import unittest

from backend.app.data.sources import AnnouncementService


class _OkAdapter:
    source_id = "ok"

    def fetch_announcements(self, stock_code: str) -> list[dict]:
        return [
            {
                "stock_code": stock_code,
                "event_type": "announcement_snapshot",
                "title": "公告列表页面",
                "content": "ok",
                "event_time": "2026-02-13T00:00:00+00:00",
                "source_id": self.source_id,
                "source_url": "https://example.com",
                "reliability_score": 0.95,
            }
        ]


class _FailAdapter:
    source_id = "fail"

    def fetch_announcements(self, stock_code: str) -> list[dict]:
        _ = stock_code
        raise RuntimeError("source unavailable")


class AnnouncementServiceTestCase(unittest.TestCase):
    """公告源测试：验证真实源适配与失败回退。"""

    def test_collects_from_live_adapter(self) -> None:
        svc = AnnouncementService(adapters=[_OkAdapter()])
        items = svc.fetch_announcements("SH600000")
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0]["source_id"], "ok")

    def test_fallback_to_mock_when_all_failed(self) -> None:
        svc = AnnouncementService(adapters=[_FailAdapter(), _FailAdapter()])
        items = svc.fetch_announcements("SH600000")
        self.assertGreaterEqual(len(items), 1)
        self.assertIn("source_id", items[0])
        self.assertTrue(items[0]["source_id"].endswith("_mock"))


if __name__ == "__main__":
    unittest.main()

