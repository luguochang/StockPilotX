from __future__ import annotations

import uuid
from typing import Any

from backend.app.prompt.evaluator import PromptRegressionRunner, default_prompt_generate_fn
from backend.app.rag.evaluation import RetrievalEvaluator, default_retrieval_dataset
from backend.app.rag.retriever import HybridRetriever


class EvalService:
    """评测服务（MVP）。"""

    def run_eval(self, samples: list[dict[str, Any]] | None = None) -> dict[str, Any]:
        """计算评测指标并给出是否通过门禁。"""
        if not samples:
            # 默认样本：用于离线快速自测
            samples = [
                {"fact_correct": True, "has_citation": True, "hallucination": False, "violation": False},
                {"fact_correct": True, "has_citation": True, "hallucination": False, "violation": False},
                {"fact_correct": False, "has_citation": True, "hallucination": True, "violation": False},
            ]

        n = max(1, len(samples))
        fact_accuracy = sum(1 for s in samples if s["fact_correct"]) / n
        citation_coverage = sum(1 for s in samples if s["has_citation"]) / n
        hallucination_rate = sum(1 for s in samples if s["hallucination"]) / n
        violation_rate = sum(1 for s in samples if s["violation"]) / n

        metrics = {
            "fact_accuracy": round(fact_accuracy, 4),
            "citation_coverage": round(citation_coverage, 4),
            "hallucination_rate": round(hallucination_rate, 4),
            "violation_rate": round(violation_rate, 4),
        }
        retrieval_metrics = RetrievalEvaluator(HybridRetriever()).run(default_retrieval_dataset(), k=5)
        metrics.update(retrieval_metrics)
        prompt_metrics = PromptRegressionRunner().run(default_prompt_generate_fn)
        metrics.update(prompt_metrics)

        pass_gate = (
            fact_accuracy >= 0.85
            and citation_coverage >= 0.95
            and hallucination_rate <= 0.05
            and violation_rate == 0
            and retrieval_metrics["recall_at_k"] >= 0.8
            and bool(prompt_metrics["prompt_pass_gate"])
        )
        gate_assessment = self.assess_gate_readiness(
            intent_f1=float(metrics.get("intent_f1_proxy", fact_accuracy)),
            sql_sample_pass_rate=0.0,
            sql_high_risk_count=0,
            observability_ready=False,
        )
        return {
            "eval_run_id": str(uuid.uuid4()),
            "metrics": metrics,
            "pass_gate": pass_gate,
            "gate_assessment": gate_assessment,
        }

    @staticmethod
    def assess_gate_a(intent_f1: float) -> dict[str, Any]:
        """Gate A rule from v1.1 plan: only introduce intent model when F1 < 0.85."""
        safe_f1 = max(0.0, min(1.0, float(intent_f1)))
        if safe_f1 < 0.85:
            return {
                "status": "hold",
                "reason": "intent_f1_below_threshold",
                "decision": "introduce_lightweight_intent_model",
                "intent_f1": round(safe_f1, 4),
            }
        return {
            "status": "go",
            "reason": "intent_f1_meets_threshold",
            "decision": "keep_rule_first_and_optimize_dataset",
            "intent_f1": round(safe_f1, 4),
        }

    @staticmethod
    def assess_gate_b(
        *,
        sql_sample_pass_rate: float,
        sql_high_risk_count: int,
        observability_ready: bool,
    ) -> dict[str, Any]:
        """Gate B rule from v1.1 plan for SQL Agent rollout decision."""
        safe_pass_rate = max(0.0, min(1.0, float(sql_sample_pass_rate)))
        safe_high_risk = max(0, int(sql_high_risk_count))
        safe_observability = bool(observability_ready)
        can_go = safe_pass_rate >= 0.85 and safe_high_risk == 0 and safe_observability
        return {
            "status": "go" if can_go else "hold",
            "reason": "all_conditions_met" if can_go else "conditions_not_met",
            "decision": "enable_sql_agent_gradual_rollout" if can_go else "keep_sql_agent_poc_only",
            "sql_sample_pass_rate": round(safe_pass_rate, 4),
            "sql_high_risk_count": safe_high_risk,
            "observability_ready": safe_observability,
        }

    def assess_gate_readiness(
        self,
        *,
        intent_f1: float,
        sql_sample_pass_rate: float,
        sql_high_risk_count: int,
        observability_ready: bool,
    ) -> dict[str, Any]:
        """Unified Gate A/B assessment payload used by scripts and ops reporting."""
        return {
            "gate_a": self.assess_gate_a(intent_f1),
            "gate_b": self.assess_gate_b(
                sql_sample_pass_rate=sql_sample_pass_rate,
                sql_high_risk_count=sql_high_risk_count,
                observability_ready=observability_ready,
            ),
        }
