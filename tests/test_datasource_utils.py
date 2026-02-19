from __future__ import annotations

import unittest

from backend.app.datasources.base.utils import decode_response, normalize_stock_code


class DatasourceUtilsTestCase(unittest.TestCase):
    def test_normalize_stock_code_with_prefix(self) -> None:
        self.assertEqual(normalize_stock_code("sh600000"), "SH600000")
        self.assertEqual(normalize_stock_code("SZ000001"), "SZ000001")

    def test_normalize_stock_code_without_prefix(self) -> None:
        self.assertEqual(normalize_stock_code("600000"), "SH600000")
        self.assertEqual(normalize_stock_code("000001"), "SZ000001")
        self.assertEqual(normalize_stock_code("300750"), "SZ300750")

    def test_decode_response_prefers_utf8(self) -> None:
        payload = "行情正常".encode("utf-8")
        self.assertEqual(decode_response(payload), "行情正常")

    def test_decode_response_fallback_to_gbk(self) -> None:
        payload = "腾讯行情".encode("gbk")
        self.assertEqual(decode_response(payload), "腾讯行情")


if __name__ == "__main__":
    unittest.main()

