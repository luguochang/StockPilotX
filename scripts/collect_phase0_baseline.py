from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics
import time
from typing import Any

from backend.app.agents.workflow import route_intent_with_confidence
from backend.app.service import AShareAgentService


def _intent_eval_samples() -> list[dict[str, str]]:
    return [
        {"question": "对比 SH600000 和 SZ000001 的盈利质量", "expected": "compare"},
        {"question": "解读这份年报PDF的关键风险", "expected": "doc_qa"},
        {"question": "做一个深度归因分析，看行业景气变化", "expected": "deep"},
        {"question": "浦发银行今天走势如何", "expected": "fact"},
        {"question": "compare two stocks by valuation and growth", "expected": "compare"},
        {"question": "帮我看下这个doc里的财务摘要", "expected": "doc_qa"},
    ]


def _query_eval_samples() -> list[dict[str, Any]]:
    return [
        {"mode": "fact", "question": "SH600000 今天的基本面结论是什么", "stock_codes": ["SH600000"]},
        {"mode": "deep", "question": "请做深度归因和风险拆解", "stock_codes": ["SH600000"]},
        {"mode": "doc_qa", "question": "请按报告口径总结这个文档内容", "stock_codes": ["SZ000001"]},
        {"mode": "compare", "question": "对比 SH600000 与 SZ000001 的趋势", "stock_codes": ["SH600000", "SZ000001"]},
    ]


def _safe_p95(values: list[int]) -> int:
    if not values:
        return 0
    if len(values) == 1:
        return values[0]
    return int(statistics.quantiles(values, n=100, method="inclusive")[94])


def run_baseline() -> dict[str, Any]:
    intent_samples = _intent_eval_samples()
    intent_correct = 0
    intent_rows: list[dict[str, Any]] = []
    for row in intent_samples:
        result = route_intent_with_confidence(row["question"])
        ok = result.intent == row["expected"]
        intent_correct += 1 if ok else 0
        intent_rows.append(
            {
                "question": row["question"],
                "expected": row["expected"],
                "predicted": result.intent,
                "confidence": result.confidence,
                "ok": ok,
            }
        )
    svc = AShareAgentService()
    latencies: list[int] = []
    failures = 0
    model_calls: list[int] = []
    timeout_reason_count = 0
    by_mode: dict[str, list[int]] = {"fact": [], "deep": [], "doc_qa": [], "compare": []}
    query_rows: list[dict[str, Any]] = []
    for idx, sample in enumerate(_query_eval_samples(), start=1):
        payload = {
            "user_id": f"baseline-u{idx}",
            "question": sample["question"],
            "stock_codes": sample["stock_codes"],
        }
        started = time.perf_counter()
        result = svc.query(payload)
        latency_ms = int((time.perf_counter() - started) * 1000)
        latencies.append(latency_ms)
        mode = str(sample["mode"])
        by_mode.setdefault(mode, []).append(latency_ms)
        risk_flags = [str(x) for x in result.get("risk_flags", [])]
        analysis_brief = result.get("analysis_brief", {}) if isinstance(result, dict) else {}
        timeout_reason = str(analysis_brief.get("timeout_reason", "")).strip()
        failed = bool(result.get("degraded")) or bool(timeout_reason)
        failures += 1 if failed else 0
        model_calls.append(int(analysis_brief.get("model_call_count", 0) or 0))
        if timeout_reason:
            timeout_reason_count += 1
        query_rows.append(
            {
                "question": sample["question"],
                "mode": mode,
                "latency_ms": latency_ms,
                "intent": result.get("intent", ""),
                "intent_confidence": float(analysis_brief.get("intent_confidence", 0.0) or 0.0),
                "model_call_count": int(analysis_brief.get("model_call_count", 0) or 0),
                "retrieval_track": dict(analysis_brief.get("retrieval_track", {}) or {}),
                "timeout_reason": timeout_reason,
                "failed": failed,
            }
        )
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "intent_baseline": {
            "sample_size": len(intent_samples),
            "accuracy": round(intent_correct / max(1, len(intent_samples)), 4),
            "rows": intent_rows,
        },
        "query_baseline": {
            "sample_size": len(query_rows),
            "failure_rate": round(failures / max(1, len(query_rows)), 4),
            "latency_p95_ms": _safe_p95(latencies),
            "latency_avg_ms": int(sum(latencies) / max(1, len(latencies))),
            "model_call_count_avg": round(sum(model_calls) / max(1, len(model_calls)), 4),
            "timeout_reason_count": timeout_reason_count,
            "by_mode_latency_ms": {k: (int(sum(v) / len(v)) if v else 0) for k, v in by_mode.items()},
            "rows": query_rows,
        },
        "memory_stats": svc.memory.stats(),
    }


def build_markdown_report(data: dict[str, Any]) -> str:
    intent = data["intent_baseline"]
    query = data["query_baseline"]
    lines = [
        "# Phase 0 Baseline Report",
        "",
        f"- generated_at: `{data['generated_at']}`",
        f"- intent_accuracy: `{intent['accuracy']}`",
        f"- query_failure_rate: `{query['failure_rate']}`",
        f"- query_latency_p95_ms: `{query['latency_p95_ms']}`",
        f"- model_call_count_avg: `{query['model_call_count_avg']}`",
        "",
        "## Query Baseline (fact/deep/doc_qa/compare)",
        "",
        "| mode | avg_latency_ms |",
        "| --- | ---: |",
    ]
    for mode, val in query["by_mode_latency_ms"].items():
        lines.append(f"| {mode} | {val} |")
    lines.extend(
        [
            "",
            "## Observability Coverage",
            "",
            "- fields: `intent_confidence`, `retrieval_track`, `model_call_count`, `timeout_reason`",
            f"- timeout_reason_count: `{query['timeout_reason_count']}`",
            "",
            "## Memory Stats",
            "",
            f"- similarity_hit_rate: `{data['memory_stats'].get('similarity_hit_rate', 0.0)}`",
            f"- cleanup_deleted_rows: `{data['memory_stats'].get('cleanup_deleted_rows', 0)}`",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect StockPilotX Phase0 baseline metrics.")
    parser.add_argument(
        "--out-dir",
        default="docs/v1/baseline",
        help="output directory for baseline json/markdown artifacts",
    )
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data = run_baseline()
    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    json_path = out_dir / f"phase0-baseline-{ts}.json"
    md_path = out_dir / f"phase0-baseline-{ts}.md"
    json_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(build_markdown_report(data), encoding="utf-8")
    print(str(json_path))
    print(str(md_path))


if __name__ == "__main__":
    main()
