from __future__ import annotations

import unittest

from backend.app.data.sources import NeteaseLiveAdapter, QuoteService, SinaLiveAdapter, TencentLiveAdapter


class _TencentStub(TencentLiveAdapter):
    def __init__(self, text: str, fail: bool = False) -> None:
        self._text = text
        self._fail = fail

    def _fetch_text(self, url: str, headers: dict[str, str] | None = None) -> str:
        _ = url, headers
        if self._fail:
            raise RuntimeError("tencent down")
        return self._text


class _NeteaseStub(NeteaseLiveAdapter):
    def __init__(self, text: str) -> None:
        self._text = text

    def _fetch_text(self, url: str, headers: dict[str, str] | None = None) -> str:
        _ = url, headers
        return self._text


class _SinaStub(SinaLiveAdapter):
    def __init__(self, text: str) -> None:
        self._text = text

    def _fetch_text(self, url: str, headers: dict[str, str] | None = None) -> str:
        _ = url, headers
        return self._text


class LiveAdapterParseTestCase(unittest.TestCase):
    """验证真实源解析逻辑与回退顺序。"""

    def test_tencent_parse(self) -> None:
        text = 'v_sh600000="51~浦发银行~600000~10.25~10.20~10.30~10.10~0~0~0~0~0~0~0~0~0~0~0~0~0~0~0~0~0~0~0~0~0~0~0~0~0~1.23~0~0~0~123456~789012~0";'
        q = _TencentStub(text).fetch_quote("SH600000")
        self.assertEqual(q.source_id, "tencent")
        self.assertAlmostEqual(q.price, 10.25, places=2)

    def test_netease_parse(self) -> None:
        text = '_ntes_quote_callback({"0600000":{"price":10.26,"percent":1.11,"volume":12345,"turnover":998877}});'
        q = _NeteaseStub(text).fetch_quote("SH600000")
        self.assertEqual(q.source_id, "netease")
        self.assertAlmostEqual(q.price, 10.26, places=2)

    def test_sina_parse(self) -> None:
        text = 'var hq_str_sh600000="浦发银行,10.20,10.10,10.30,10.40,10.00,10.29,10.30,23456,123456.7";'
        q = _SinaStub(text).fetch_quote("SH600000")
        self.assertEqual(q.source_id, "sina")
        self.assertAlmostEqual(q.price, 10.3, places=2)

    def test_fallback_from_tencent_to_netease(self) -> None:
        t_text = 'v_sh600000="bad";'
        n_text = '_ntes_quote_callback({"0600000":{"price":10.28,"percent":0.88,"volume":12345,"turnover":998877}});'
        svc = QuoteService([_TencentStub(t_text, fail=True), _NeteaseStub(n_text)])
        quote = svc.get_quote("SH600000")
        self.assertEqual(quote.source_id, "netease")


if __name__ == "__main__":
    unittest.main()

