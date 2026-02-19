from __future__ import annotations

import json
import unittest

from backend.app.datasources.base.adapter import DataSourceConfig
from backend.app.datasources.financial.eastmoney import EastmoneyFinancialAdapter
from backend.app.datasources.financial.service import FinancialService, MockFinancialAdapter
from backend.app.datasources.financial.tushare import TushareFinancialAdapter


class _StubClient:
    def __init__(self, *, post_payload: bytes = b"", get_payload: bytes = b"") -> None:
        self._post_payload = post_payload
        self._get_payload = get_payload

    def post_json_bytes(self, url: str, payload: dict, headers: dict | None = None) -> bytes:
        _ = url, payload, headers
        return self._post_payload

    def get_bytes(self, url: str, headers: dict | None = None) -> bytes:
        _ = url, headers
        return self._get_payload


class _FailFinancialAdapter:
    source_id = "fail_fin"

    def fetch_financial_snapshot(self, stock_code: str) -> dict:
        _ = stock_code
        raise RuntimeError("adapter down")


class DatasourceFinancialAdapterTestCase(unittest.TestCase):
    def test_tushare_adapter_parse(self) -> None:
        response = {
            "code": 0,
            "data": {
                "fields": [
                    "ts_code",
                    "end_date",
                    "roe",
                    "grossprofit_margin",
                    "tr_yoy",
                    "netprofit_yoy",
                    "debt_to_assets",
                    "current_ratio",
                    "quick_ratio",
                ],
                "items": [["600000.SH", "20241231", 12.3, 39.2, 8.1, 10.7, 55.4, 1.3, 1.1]],
            },
        }
        adapter = TushareFinancialAdapter(
            DataSourceConfig(source_id="tushare_financial", reliability_score=0.86),
            token="token-demo",
            client=_StubClient(post_payload=json.dumps(response).encode("utf-8")),
        )
        item = adapter.fetch_financial_snapshot("SH600000")
        self.assertEqual(item["stock_code"], "SH600000")
        self.assertEqual(item["report_period"], "20241231")
        self.assertAlmostEqual(float(item["roe"]), 12.3, places=2)

    def test_eastmoney_adapter_parse(self) -> None:
        response = {"data": {"f162": 13.2, "f167": 1.56, "f127": 9.8, "f128": 11.1, "f130": 54.4}}
        adapter = EastmoneyFinancialAdapter(
            DataSourceConfig(source_id="eastmoney_financial", reliability_score=0.84),
            client=_StubClient(get_payload=json.dumps(response).encode("utf-8")),
        )
        item = adapter.fetch_financial_snapshot("SH600000")
        self.assertEqual(item["stock_code"], "SH600000")
        self.assertAlmostEqual(float(item["pe_ttm"]), 13.2, places=2)
        self.assertAlmostEqual(float(item["pb_mrq"]), 1.56, places=2)

    def test_financial_service_fallback(self) -> None:
        svc = FinancialService([_FailFinancialAdapter(), MockFinancialAdapter()])
        item = svc.get_financial_snapshot("SZ000001")
        self.assertEqual(item["stock_code"], "SZ000001")
        self.assertIn("source_id", item)


if __name__ == "__main__":
    unittest.main()

