from __future__ import annotations

import unittest

from backend.app.evals.service import EvalService


class EvalGateDecisionTestCase(unittest.TestCase):
    def test_gate_a_hold_when_intent_f1_below_threshold(self) -> None:
        svc = EvalService()
        gate_a = svc.assess_gate_a(0.82)
        self.assertEqual(gate_a["status"], "hold")
        self.assertEqual(gate_a["decision"], "introduce_lightweight_intent_model")

    def test_gate_a_go_when_intent_f1_meets_threshold(self) -> None:
        svc = EvalService()
        gate_a = svc.assess_gate_a(0.9)
        self.assertEqual(gate_a["status"], "go")
        self.assertEqual(gate_a["decision"], "keep_rule_first_and_optimize_dataset")

    def test_gate_b_requires_all_conditions(self) -> None:
        svc = EvalService()
        gate_b = svc.assess_gate_b(
            sql_sample_pass_rate=0.9,
            sql_high_risk_count=0,
            observability_ready=True,
        )
        self.assertEqual(gate_b["status"], "go")
        gate_b_hold = svc.assess_gate_b(
            sql_sample_pass_rate=0.9,
            sql_high_risk_count=1,
            observability_ready=True,
        )
        self.assertEqual(gate_b_hold["status"], "hold")

    def test_run_eval_contains_gate_assessment(self) -> None:
        svc = EvalService()
        run = svc.run_eval(
            [
                {"fact_correct": True, "has_citation": True, "hallucination": False, "violation": False},
                {"fact_correct": True, "has_citation": True, "hallucination": False, "violation": False},
            ]
        )
        self.assertIn("gate_assessment", run)
        self.assertIn("gate_a", run["gate_assessment"])
        self.assertIn("gate_b", run["gate_assessment"])


if __name__ == "__main__":
    unittest.main()
