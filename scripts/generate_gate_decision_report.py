from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

from backend.app.evals.service import EvalService


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"invalid baseline json: {path}")
    return payload


def _build_report(
    baseline_path: Path,
    baseline: dict[str, Any],
    gate: dict[str, Any],
    sql_sample_pass_rate: float,
    sql_high_risk_count: int,
    observability_ready: bool,
) -> str:
    intent_f1 = float((baseline.get("intent_baseline") or {}).get("accuracy", 0.0) or 0.0)
    q = baseline.get("query_baseline", {}) if isinstance(baseline.get("query_baseline"), dict) else {}
    lines = [
        "# Gate Decision Report",
        "",
        f"- generated_at: `{datetime.now(timezone.utc).isoformat()}`",
        f"- baseline_file: `{baseline_path.as_posix()}`",
        f"- intent_f1_proxy: `{intent_f1:.4f}`",
        f"- query_latency_p95_ms: `{int(q.get('latency_p95_ms', 0) or 0)}`",
        f"- query_failure_rate: `{float(q.get('failure_rate', 0.0) or 0.0):.4f}`",
        "",
        "## Gate A",
        "",
        f"- status: `{gate['gate_a']['status']}`",
        f"- decision: `{gate['gate_a']['decision']}`",
        f"- reason: `{gate['gate_a']['reason']}`",
        "",
        "## Gate B",
        "",
        f"- status: `{gate['gate_b']['status']}`",
        f"- decision: `{gate['gate_b']['decision']}`",
        f"- reason: `{gate['gate_b']['reason']}`",
        f"- sql_sample_pass_rate: `{sql_sample_pass_rate:.4f}`",
        f"- sql_high_risk_count: `{sql_high_risk_count}`",
        f"- observability_ready: `{observability_ready}`",
        "",
        "## Checklist",
        "",
        "- [ ] Gate A sample set expanded to production-like workload",
        "- [ ] Gate B security regression with 0 high-risk findings",
        "- [ ] Rollback threshold and owner confirmed",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Gate A/B decision report from baseline metrics.")
    parser.add_argument("--baseline-json", required=True, help="path to phase0 baseline json")
    parser.add_argument("--out-file", default="", help="optional markdown output path")
    parser.add_argument("--sql-pass-rate", type=float, default=0.0, help="SQL sample pass rate [0,1]")
    parser.add_argument("--sql-high-risk", type=int, default=0, help="SQL high-risk finding count")
    parser.add_argument(
        "--observability-ready",
        action="store_true",
        help="mark observability/rollback strategy as ready for Gate B",
    )
    args = parser.parse_args()

    baseline_path = Path(args.baseline_json).resolve()
    baseline = _load_json(baseline_path)
    intent_f1 = float((baseline.get("intent_baseline") or {}).get("accuracy", 0.0) or 0.0)
    svc = EvalService()
    gate = svc.assess_gate_readiness(
        intent_f1=intent_f1,
        sql_sample_pass_rate=float(args.sql_pass_rate),
        sql_high_risk_count=int(args.sql_high_risk),
        observability_ready=bool(args.observability_ready),
    )
    report = _build_report(
        baseline_path=baseline_path,
        baseline=baseline,
        gate=gate,
        sql_sample_pass_rate=float(args.sql_pass_rate),
        sql_high_risk_count=int(args.sql_high_risk),
        observability_ready=bool(args.observability_ready),
    )
    if args.out_file:
        out_path = Path(args.out_file)
    else:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        out_path = baseline_path.parent / f"gate-decision-{stamp}.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report, encoding="utf-8")
    print(str(out_path))


if __name__ == "__main__":
    main()
