from __future__ import annotations

import unittest

from backend.app.datasources.base.adapter import DataSourceConfig
from backend.app.datasources.quote.netease import NeteaseQuoteAdapter
from backend.app.datasources.quote.service import QuoteService
from backend.app.datasources.quote.sina import SinaQuoteAdapter
from backend.app.datasources.quote.tencent import TencentQuoteAdapter
from backend.app.datasources.quote.xueqiu import XueqiuQuoteAdapter


class _StubClient:
    def __init__(self, payloads: dict[str, bytes]) -> None:
        self.payloads = payloads

    def get_bytes(self, url: str, headers: dict[str, str] | None = None) -> bytes:
        _ = headers
        for key, value in self.payloads.items():
            if key in url:
                return value
        raise RuntimeError(f"stub payload not found for url={url}")


class _FailAdapter:
    source_id = "fail"

    def fetch_quote(self, stock_code: str):
        _ = stock_code
        raise RuntimeError("upstream down")


class DatasourceQuoteAdapterTestCase(unittest.TestCase):
    def test_tencent_adapter_parse(self) -> None:
        text = (
            'v_sh600000="51~浦发银行~600000~10.25~10.20~10.30~10.10~0~0~0~0~0~0~0~0~0~0~0~0~0~0~0~0~0~0~0~0~0~0~0~0~1.23~0~0~0~123456~789012~0";'
        )
        adapter = TencentQuoteAdapter(
            DataSourceConfig(source_id="tencent", reliability_score=0.85),
            client=_StubClient({"qt.gtimg.cn": text.encode("utf-8")}),
        )
        quote = adapter.fetch_quote("SH600000")
        self.assertEqual(quote.source_id, "tencent")
        self.assertAlmostEqual(quote.price, 10.25, places=2)

    def test_netease_adapter_parse(self) -> None:
        text = '_ntes_quote_callback({"0600000":{"price":10.26,"percent":1.11,"volume":12345,"turnover":998877}});'
        adapter = NeteaseQuoteAdapter(
            DataSourceConfig(source_id="netease", reliability_score=0.82),
            client=_StubClient({"api.money.126.net": text.encode("utf-8")}),
        )
        quote = adapter.fetch_quote("SH600000")
        self.assertEqual(quote.source_id, "netease")
        self.assertAlmostEqual(quote.price, 10.26, places=2)

    def test_sina_adapter_parse(self) -> None:
        text = 'var hq_str_sh600000="浦发银行,10.20,10.10,10.30,10.40,10.00,10.29,10.30,23456,123456.7";'
        adapter = SinaQuoteAdapter(
            DataSourceConfig(source_id="sina", reliability_score=0.80),
            client=_StubClient({"hq.sinajs.cn": text.encode("gbk")}),
        )
        quote = adapter.fetch_quote("SH600000")
        self.assertEqual(quote.source_id, "sina")
        self.assertAlmostEqual(quote.price, 10.30, places=2)

    def test_xueqiu_requires_cookie(self) -> None:
        text = '{"data":[{"current":10.55,"percent":2.1,"volume":2222,"amount":3333}]}'
        adapter = XueqiuQuoteAdapter(
            DataSourceConfig(source_id="xueqiu", reliability_score=0.78),
            cookie="",
            client=_StubClient({"stock.xueqiu.com": text.encode("utf-8")}),
        )
        with self.assertRaises(RuntimeError):
            adapter.fetch_quote("SH600000")

    def test_quote_service_fallback(self) -> None:
        text = '_ntes_quote_callback({"0600000":{"price":10.28,"percent":0.88,"volume":12345,"turnover":998877}});'
        netease = NeteaseQuoteAdapter(
            DataSourceConfig(source_id="netease", reliability_score=0.82),
            client=_StubClient({"api.money.126.net": text.encode("utf-8")}),
        )
        svc = QuoteService([_FailAdapter(), netease])
        quote = svc.get_quote("SH600000")
        self.assertEqual(quote.source_id, "netease")


if __name__ == "__main__":
    unittest.main()

