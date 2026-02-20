from __future__ import annotations

import unittest

from backend.app.query.sql_guard import SQLSafetyValidator


class SQLGuardTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.tables = {"daily_quote", "financial_snapshot"}
        self.columns = {"stock_code", "trade_date", "close", "pe_ttm"}

    def test_allow_valid_select(self) -> None:
        result = SQLSafetyValidator.validate_select_sql(
            "SELECT stock_code, close FROM daily_quote WHERE stock_code='SH600000' LIMIT 100",
            allowed_tables=self.tables,
            allowed_columns=self.columns,
            max_limit=500,
        )
        self.assertTrue(result["ok"])

    def test_reject_non_select(self) -> None:
        result = SQLSafetyValidator.validate_select_sql(
            "DELETE FROM daily_quote WHERE stock_code='SH600000'",
            allowed_tables=self.tables,
            allowed_columns=self.columns,
        )
        self.assertFalse(result["ok"])
        self.assertEqual(result["reason"], "only_select_allowed")

    def test_reject_table_outside_whitelist(self) -> None:
        result = SQLSafetyValidator.validate_select_sql(
            "SELECT stock_code FROM secret_table LIMIT 10",
            allowed_tables=self.tables,
            allowed_columns=self.columns,
        )
        self.assertFalse(result["ok"])
        self.assertEqual(result["reason"], "table_not_allowed")

    def test_reject_limit_exceeded(self) -> None:
        result = SQLSafetyValidator.validate_select_sql(
            "SELECT stock_code FROM daily_quote LIMIT 1000",
            allowed_tables=self.tables,
            allowed_columns=self.columns,
            max_limit=500,
        )
        self.assertFalse(result["ok"])
        self.assertEqual(result["reason"], "limit_exceeded")

    def test_reject_forbidden_pattern(self) -> None:
        result = SQLSafetyValidator.validate_select_sql(
            "SELECT stock_code FROM daily_quote LIMIT 10; DROP TABLE daily_quote",
            allowed_tables=self.tables,
            allowed_columns=self.columns,
        )
        self.assertFalse(result["ok"])
        self.assertEqual(result["reason"], "forbidden_sql_pattern")


if __name__ == "__main__":
    unittest.main()
