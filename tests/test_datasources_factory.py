from __future__ import annotations

import unittest

from backend.app.config import Settings
from backend.app.datasources import (
    build_default_announcement_service,
    build_default_history_service,
    build_default_quote_service,
)


class DatasourceFactoryTestCase(unittest.TestCase):
    def test_quote_service_uses_cookie_from_settings(self) -> None:
        settings = Settings(datasource_xueqiu_cookie="xq-cookie-demo")
        svc = build_default_quote_service(settings)
        adapter = next((x for x in svc.adapters if getattr(x, "source_id", "") == "xueqiu"), None)
        self.assertIsNotNone(adapter)
        self.assertEqual(getattr(adapter, "cookie", None), "xq-cookie-demo")

    def test_announcement_service_is_constructed(self) -> None:
        svc = build_default_announcement_service(Settings())
        items = svc.fetch_announcements("SH600000")
        self.assertGreaterEqual(len(items), 1)

    def test_history_service_is_constructed(self) -> None:
        svc = build_default_history_service(Settings())
        # Assert interface contract without making a real HTTP call.
        self.assertTrue(hasattr(svc, "fetch_daily_bars"))


if __name__ == "__main__":
    unittest.main()

