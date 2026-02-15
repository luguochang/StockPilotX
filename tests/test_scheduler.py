from __future__ import annotations

import unittest

from backend.app.data.scheduler import JobConfig, LocalJobScheduler
from backend.app.service import AShareAgentService


class SchedulerTestCase(unittest.TestCase):
    """DATA-004：调度重试与熔断测试。"""

    def test_retry_then_success(self) -> None:
        scheduler = LocalJobScheduler()
        count = {"n": 0}

        def flaky() -> dict:
            count["n"] += 1
            if count["n"] < 2:
                raise RuntimeError("temporary failure")
            return {"ok": True}

        scheduler.register(JobConfig(name="flaky", cadence="daily", fn=flaky, max_retries=3, cooldown_seconds=5))
        result = scheduler.run_once("flaky")
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["attempt"], 2)

    def test_circuit_breaker_open(self) -> None:
        scheduler = LocalJobScheduler()

        def always_fail() -> dict:
            raise RuntimeError("boom")

        scheduler.register(
            JobConfig(name="always_fail", cadence="daily", fn=always_fail, max_retries=2, cooldown_seconds=60)
        )
        first = scheduler.run_once("always_fail")
        self.assertEqual(first["status"], "failed")
        second = scheduler.run_once("always_fail")
        self.assertEqual(second["status"], "circuit_open")

    def test_service_default_jobs_registered(self) -> None:
        svc = AShareAgentService()
        status = svc.scheduler_status()
        self.assertIn("intraday_quote_ingest", status)
        self.assertIn("daily_announcement_ingest", status)
        self.assertIn("weekly_rebuild", status)


if __name__ == "__main__":
    unittest.main()

