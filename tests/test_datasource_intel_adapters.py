from __future__ import annotations

import json
import unittest

from backend.app.datasources.base.adapter import DataSourceConfig
from backend.app.datasources.fund.service import FundService, MockFundAdapter
from backend.app.datasources.fund.ttjj import TTJJFundAdapter
from backend.app.datasources.macro.eastmoney_macro import EastmoneyMacroAdapter
from backend.app.datasources.macro.service import MacroService, MockMacroAdapter
from backend.app.datasources.news.cls import CLSNewsAdapter
from backend.app.datasources.news.service import MockNewsAdapter, NewsService
from backend.app.datasources.news.tradingview import TradingViewNewsAdapter
from backend.app.datasources.news.xueqiu_news import XueqiuNewsAdapter
from backend.app.datasources.research.eastmoney_research import EastmoneyResearchAdapter
from backend.app.datasources.research.service import MockResearchAdapter, ResearchService


class _StubClient:
    def __init__(self, payloads: dict[str, bytes]) -> None:
        self.payloads = payloads

    def get_bytes(self, url: str, headers: dict | None = None) -> bytes:
        _ = headers
        for key, payload in self.payloads.items():
            if key in url:
                return payload
        raise RuntimeError(f"stub payload not found for url={url}")


class _FailNewsAdapter:
    source_id = "fail_news"

    def fetch_news(self, stock_code: str, limit: int = 20) -> list[dict]:
        _ = stock_code, limit
        raise RuntimeError("adapter down")


class _FailResearchAdapter:
    source_id = "fail_research"

    def fetch_reports(self, stock_code: str, limit: int = 20) -> list[dict]:
        _ = stock_code, limit
        raise RuntimeError("adapter down")


class _FailMacroAdapter:
    source_id = "fail_macro"

    def fetch_macro_indicators(self, limit: int = 20) -> list[dict]:
        _ = limit
        raise RuntimeError("adapter down")


class _FailFundAdapter:
    source_id = "fail_fund"

    def fetch_fund_snapshot(self, stock_code: str) -> dict:
        _ = stock_code
        raise RuntimeError("adapter down")


class DatasourceIntelAdapterTestCase(unittest.TestCase):
    def test_cls_news_adapter_parse(self) -> None:
        payload = {
            "data": {
                "roll_data": [
                    {
                        "title": "银行行业政策跟踪",
                        "content": "监管发布最新指引",
                        "shareurl": "https://cls.cn/news/1",
                    }
                ]
            }
        }
        adapter = CLSNewsAdapter(
            DataSourceConfig(source_id="cls_news", reliability_score=0.78),
            client=_StubClient({"cls.cn/nodeapi/telegraphList": json.dumps(payload).encode("utf-8")}),
        )
        rows = adapter.fetch_news("SH600000", limit=5)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["stock_code"], "SH600000")

    def test_tradingview_news_adapter_parse(self) -> None:
        payload = {"items": [{"title": "海外银行板块异动", "description": "摘要", "url": "https://tv.com/n1"}]}
        adapter = TradingViewNewsAdapter(
            DataSourceConfig(source_id="tradingview_news", reliability_score=0.72),
            client=_StubClient({"news-headlines.tradingview.com": json.dumps(payload).encode("utf-8")}),
        )
        rows = adapter.fetch_news("SZ000001", limit=3)
        self.assertEqual(rows[0]["stock_code"], "SZ000001")
        self.assertEqual(rows[0]["source_id"], "tradingview_news")

    def test_xueqiu_news_requires_cookie(self) -> None:
        adapter = XueqiuNewsAdapter(
            DataSourceConfig(source_id="xueqiu_news", reliability_score=0.70),
            cookie="",
            client=_StubClient({}),
        )
        with self.assertRaises(RuntimeError):
            adapter.fetch_news("SH600000")

    def test_xueqiu_news_parse(self) -> None:
        payload = {
            "data": {
                "items": [
                    {
                        "title": "雪球新闻",
                        "description": "内容",
                        "target": "https://xueqiu.com/n1",
                    }
                ]
            }
        }
        adapter = XueqiuNewsAdapter(
            DataSourceConfig(source_id="xueqiu_news", reliability_score=0.70),
            cookie="xq_cookie",
            client=_StubClient({"stock.xueqiu.com": json.dumps(payload).encode("utf-8")}),
        )
        rows = adapter.fetch_news("SH600000", limit=2)
        self.assertEqual(rows[0]["source_id"], "xueqiu_news")

    def test_news_service_fallback(self) -> None:
        svc = NewsService([_FailNewsAdapter(), MockNewsAdapter()])
        rows = svc.fetch_news("SH600000", limit=1)
        self.assertEqual(rows[0]["source_id"], "news_mock")

    def test_research_adapter_parse(self) -> None:
        payload = {
            "result": {
                "data": [
                    {
                        "TITLE": "银行板块估值研究",
                        "AUTHOR_NAME": "分析师A",
                        "ORG_NAME": "机构B",
                        "PUBLISH_DATE": "2026-02-19",
                        "REPORT_URL": "https://eastmoney.com/r1",
                    }
                ]
            }
        }
        adapter = EastmoneyResearchAdapter(
            DataSourceConfig(source_id="eastmoney_research", reliability_score=0.82),
            client=_StubClient({"datacenter-web.eastmoney.com": json.dumps(payload).encode("utf-8")}),
        )
        rows = adapter.fetch_reports("SH600000", limit=3)
        self.assertEqual(rows[0]["stock_code"], "SH600000")
        self.assertEqual(rows[0]["source_id"], "eastmoney_research")

    def test_research_service_fallback(self) -> None:
        svc = ResearchService([_FailResearchAdapter(), MockResearchAdapter()])
        rows = svc.fetch_reports("SZ000001", limit=1)
        self.assertEqual(rows[0]["source_id"], "research_mock")

    def test_macro_adapter_parse(self) -> None:
        payload = {"result": {"data": [{"REPORT_DATE": "2026-01-31", "VALUE": "2.1"}]}}
        adapter = EastmoneyMacroAdapter(
            DataSourceConfig(source_id="eastmoney_macro", reliability_score=0.8),
            client=_StubClient({"datacenter-web.eastmoney.com": json.dumps(payload).encode("utf-8")}),
        )
        rows = adapter.fetch_macro_indicators(limit=1)
        self.assertGreaterEqual(len(rows), 3)
        self.assertEqual(rows[0]["source_id"], "eastmoney_macro")

    def test_macro_service_fallback(self) -> None:
        svc = MacroService([_FailMacroAdapter(), MockMacroAdapter()])
        rows = svc.fetch_macro_indicators(limit=2)
        self.assertGreaterEqual(len(rows), 1)
        self.assertEqual(rows[0]["source_id"], "macro_mock")

    def test_fund_adapter_parse(self) -> None:
        payload = {"data": {"klines": ["2026-02-19,100,20,30,40"]}}
        adapter = TTJJFundAdapter(
            DataSourceConfig(source_id="ttjj_fund", reliability_score=0.76),
            client=_StubClient({"push2.eastmoney.com": json.dumps(payload).encode("utf-8")}),
        )
        row = adapter.fetch_fund_snapshot("SH600000")
        self.assertEqual(row["stock_code"], "SH600000")
        self.assertAlmostEqual(float(row["main_inflow"]), 100.0, places=2)

    def test_fund_service_fallback(self) -> None:
        svc = FundService([_FailFundAdapter(), MockFundAdapter()])
        row = svc.fetch_fund_snapshot("SZ000001")
        self.assertEqual(row["source_id"], "fund_mock")


if __name__ == "__main__":
    unittest.main()
