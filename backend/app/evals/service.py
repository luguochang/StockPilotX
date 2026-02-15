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
        return {"eval_run_id": str(uuid.uuid4()), "metrics": metrics, "pass_gate": pass_gate}
