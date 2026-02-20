from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fastapi.routing import APIRoute
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.app.http_api import create_app  # noqa: E402


@dataclass
class SmokeResult:
    method: str
    route: str
    status_code: int
    ok: bool
    detail: str


def _replace_path_params(route: str, ctx: dict[str, str]) -> str:
    path = route
    for key, value in ctx.items():
        path = path.replace("{" + key + "}", value)
    return path


def _should_skip_light_route(method: str, route: str) -> bool:
    """Skip routes that are expected to be long-running or externally IO-bound."""
    if route.endswith("/stream"):
        return True
    if route.endswith("/rounds") or route.endswith("/rounds/stream"):
        return True
    exact_skip = {
        "/v1/query",
        "/v1/query/compare",
        "/v1/query/stream",
        "/v1/deep-think/sessions",
        "/v1/report/generate",
        "/v1/report/tasks",
        "/v1/predict/run",
        "/v1/backtest/run",
        "/v1/datasources/fetch",
        "/v1/stocks/sync",
        "/v1/ops/rag/reindex",
        "/v1/rag/uploads",
        "/v1/rag/workflow/upload-and-index",
        "/v1/scheduler/run",
        "/v1/market/overview/{stock_code}",
        "/v1/analysis/intel-card",
        "/v1/analysis/intel-card/feedback",
    }
    if route in exact_skip:
        return True
    if route.startswith("/v1/ingest/"):
        return True
    # Some POST actions can still be expensive (reindex/job control); keep light mode deterministic.
    if method == "POST" and route.startswith("/v1/ops/"):
        return True
    return False


def _is_heavy_critical_route(method: str, route: str) -> bool:
    """Limit heavy mode to critical user-facing chains when requested."""
    critical_exact = {
        "/v1/query",
        "/v1/query/stream",
        "/v1/deep-think/sessions",
        "/v1/deep-think/sessions/{session_id}/rounds",
        "/v2/deep-think/sessions/{session_id}/rounds/stream",
        "/v1/report/generate",
        "/v1/report/tasks",
        "/v1/report/tasks/{task_id}",
        "/v1/report/tasks/{task_id}/result",
        "/v1/predict/run",
        "/v1/analysis/intel-card",
        "/v1/market/overview/{stock_code}",
    }
    if route in critical_exact:
        return True
    # Keep a small read-only surface for baseline liveness around critical flows.
    if method == "GET" and route in {"/v1/auth/me", "/v1/reports"}:
        return True
    return False


def _payload_for(route: str, ctx: dict[str, str]) -> dict[str, Any]:
    # Default smoke mode is lightweight: validate route liveness and input handling
    # without forcing expensive model/data workflows on every run.
    run_heavy = os.environ.get("SMOKE_RUN_HEAVY", "").strip() in {"1", "true", "TRUE"}
    if not run_heavy:
        if route in {
            "/v1/query",
            "/v1/query/compare",
            "/v1/query/stream",
            "/v1/deep-think/sessions",
            "/v1/report/generate",
            "/v1/report/tasks",
            "/v1/predict/run",
            "/v1/backtest/run",
        }:
            return {}
        if route.endswith("/rounds") or route.endswith("/rounds/stream"):
            return {}

    # Keep payloads minimal but valid enough to exercise route logic.
    if route == "/v1/query":
        return {"user_id": "smoke-user", "question": "请分析 SH600000 的风险与机会", "stock_codes": ["SH600000"]}
    if route == "/v1/query/compare":
        return {"user_id": "smoke-user", "question": "对比 SH600000 与 SZ000001", "stock_codes": ["SH600000", "SZ000001"]}
    if route == "/v1/query/stream":
        return {"user_id": "smoke-user", "question": "流式分析 SH600000", "stock_codes": ["SH600000"]}
    if route == "/v1/deep-think/sessions":
        return {
            "user_id": "smoke-user",
            "question": "请给出 SH600000 的深度分析",
            "stock_codes": ["SH600000"],
            "max_rounds": 2,
        }
    if route.endswith("/rounds"):
        return {"question": "继续执行下一轮", "stock_codes": ["SH600000"]}
    if route.endswith("/rounds/stream"):
        return {"question": "继续执行下一轮", "stock_codes": ["SH600000"], "auto_journal": True}
    if route == "/v1/report/generate":
        return {"user_id": "smoke-user", "stock_code": "SH600000", "period": "1y", "report_type": "research"}
    if route == "/v1/report/tasks":
        return {"user_id": "smoke-user", "stock_code": "SH600000", "period": "1y", "report_type": "fact"}
    if route.endswith("/cancel"):
        return {}
    if route.startswith("/v1/ingest/"):
        if route.endswith("/macro"):
            return {"limit": 4}
        return {"stock_codes": ["SH600000"], "limit": 4}
    if route == "/v1/datasources/fetch":
        return {"source_id": "eastmoney_history", "stock_codes": ["SH600000"], "limit": 3}
    if route == "/v1/docs/upload":
        return {"doc_id": "smoke-doc-1", "filename": "smoke.txt", "content": "SH600000 测试文档 " * 40, "source": "upload"}
    if route.endswith("/index"):
        return {}
    if route == "/v1/evals/run":
        return {"samples": [{"fact_correct": True, "has_citation": True, "hallucination": False, "violation": False}]}
    if route == "/v1/scheduler/run":
        return {"job_name": "intraday_quote_ingest"}
    if route == "/v1/predict/run":
        return {"stock_codes": ["SH600000"], "horizons": ["5d", "20d"]}
    if route == "/v1/analysis/intel-card/feedback":
        return {
            "stock_code": "SH600000",
            "signal": "hold",
            "confidence": 0.6,
            "position_hint": "20-30%",
            "trace_id": "smoke-trace",
        }
    if route == "/v1/backtest/run":
        return {
            "stock_code": "SH600000",
            "start_date": "2024-01-01",
            "end_date": "2026-02-15",
            "initial_capital": 100000,
            "ma_window": 20,
        }
    if route == "/v1/auth/register":
        return {"username": f"smoke_{int(time.time())}", "password": "Passw0rd!"}
    if route == "/v1/auth/login":
        return {"username": "admin", "password": "admin123"}
    if route == "/v1/auth/refresh":
        return {}
    if route == "/v1/watchlist":
        return {"stock_code": "SH600000"}
    if route == "/v1/watchlist/pools":
        return {"pool_name": "smoke-pool", "description": "smoke"}
    if route.endswith("/stocks") and "/watchlist/pools/" in route:
        return {"stock_code": "SH600000", "source_filters": {}}
    if route == "/v1/portfolio":
        return {"portfolio_name": "smoke-portfolio", "initial_capital": 100000, "description": "smoke"}
    if route.endswith("/transactions") and route.startswith("/v1/portfolio/"):
        return {
            "stock_code": "SH600000",
            "transaction_type": "buy",
            "quantity": 100,
            "price": 10.5,
            "fee": 1.0,
            "transaction_date": "2026-02-19",
        }
    if route == "/v1/journal":
        return {"journal_type": "decision", "stock_code": "SH600000", "decision_type": "hold"}
    if route.endswith("/reflections"):
        return {"reflection_content": "smoke reflection"}
    if route.endswith("/ai-reflection/generate"):
        return {"focus": "检查执行偏差与可操作改进"}
    if route.endswith("/export") and route.startswith("/v1/reports/"):
        return {}
    if route.endswith("/review/approve"):
        return {"comment": "smoke approve"}
    if route.endswith("/review/reject"):
        return {"comment": "smoke reject"}
    if route.startswith("/v1/rag/source-policy/"):
        return {"auto_approve": True, "trust_score": 0.8, "enabled": True}
    if route.endswith("/status") and "/v1/rag/docs/chunks/" in route:
        return {"status": "active"}
    if route.endswith("/toggle") and "/v1/rag/qa-memory/" in route:
        return {"retrieval_enabled": True}
    if route == "/v1/rag/uploads":
        content = "RAG smoke content " * 200
        encoded = content.encode("utf-8")
        import base64

        return {
            "upload_id": "smoke-upload-1",
            "doc_id": "smoke-rag-doc-1",
            "filename": "smoke-rag.txt",
            "source": "user_upload",
            "source_url": "",
            "content_b64": base64.b64encode(encoded).decode("ascii"),
            "content_type": "text/plain",
            "stock_codes": ["SH600000"],
            "tags": ["smoke"],
        }
    if route == "/v1/rag/workflow/upload-and-index":
        return {
            "doc_id": "smoke-workflow-doc",
            "filename": "workflow.txt",
            "content": "workflow content " * 80,
            "source": "upload",
        }
    if route == "/v1/ops/rag/reindex":
        return {"force": False}
    if route == "/v1/ops/prompts/compare":
        prompt_id = str(ctx.get("prompt_id", "fact_qa") or "fact_qa")
        prompt_version = str(ctx.get("prompt_version", "")).strip()
        return {
            "prompt_id": prompt_id,
            # Use discovered prompt version to avoid synthetic-not-found failures.
            "base_version": prompt_version or "stable",
            "candidate_version": prompt_version or "stable",
            # Include common template variables to avoid false negatives from strict prompt contracts.
            "variables": {"question": "smoke", "stock_codes": ["SH600000"], "evidence": "source:smoke"},
        }
    if route == "/v1/a2a/tasks":
        return {"agent_id": "supervisor_agent", "task_type": "smoke", "payload": {"question": "smoke"}}
    if route == "/v1/scheduler/pause" or route == "/v1/scheduler/resume":
        return {"job_name": "intraday_quote_ingest"}
    if route.startswith("/v1/alerts/rules"):
        return {
            "rule_name": "smoke-rule",
            "rule_type": "price",
            "stock_code": "SH600000",
            "operator": ">",
            "target_value": 1,
            "event_type": "quote",
            "is_active": True,
        }
    if route == "/v1/alerts/check":
        return {"stock_code": "SH600000"}
    if route == "/v1/stocks/sync":
        return {}
    return {}


def _params_for(route: str, ctx: dict[str, str]) -> dict[str, Any]:
    if route == "/v1/deep-think/intel/traces/{trace_id}":
        return {}
    if route == "/v1/analysis/intel-card":
        return {"stock_code": "SH600000", "horizon": "30d", "risk_profile": "neutral"}
    if route == "/v1/analysis/intel-card/review":
        return {"stock_code": "SH600000", "limit": 20}
    if route == "/v1/docs/{doc_id}/quality-report":
        return {}
    if route == "/v1/docs/{doc_id}/pipeline-runs":
        return {"limit": 10}
    if route == "/v1/docs/{doc_id}/versions":
        return {"limit": 10}
    if route == "/v1/docs/recommend":
        return {}
    if route == "/v1/rag/docs/chunks":
        return {"limit": 20}
    if route == "/v1/rag/docs/chunks/{chunk_id}":
        return {"context_window": 1}
    if route == "/v1/rag/qa-memory":
        return {"limit": 20}
    if route == "/v1/rag/retrieval-preview":
        return {"doc_id": ctx.get("doc_id", "smoke-doc-1"), "max_queries": 2, "top_k": 4}
    if route == "/v1/market/overview/{stock_code}":
        return {}
    if route == "/v1/predict/evals/latest":
        return {}
    if route == "/v1/predict/{run_id}":
        return {}
    if route == "/v1/factors/{stock_code}":
        return {}
    if route == "/v1/backtest/{run_id}":
        return {}
    if route == "/v1/query/history":
        return {"limit": 20}
    if route == "/v1/deep-think/sessions/{session_id}/events":
        return {"limit": 30}
    if route == "/v1/deep-think/sessions/{session_id}/events/export":
        return {"limit": 30}
    if route == "/v1/deep-think/sessions/{session_id}/business-export":
        return {"format": "markdown"}
    if route == "/v1/deep-think/sessions/{session_id}/report-export":
        return {"format": "markdown"}
    if route == "/v1/deep-think/sessions/{session_id}/events/export-tasks/{task_id}":
        return {}
    if route == "/v1/deep-think/sessions/{session_id}/events/export-tasks/{task_id}/download":
        return {}
    if route == "/v1/business/data-health":
        return {"stock_code": "SH600000", "limit": 200}
    if route == "/v1/datasources/health":
        return {"limit": 200}
    if route == "/v1/datasources/logs":
        return {"limit": 50}
    if route == "/v1/ops/data-sources/health":
        return {"limit": 200}
    if route == "/v1/ops/journal/health":
        return {"limit": 200}
    if route == "/v1/ops/deep-think/archive-metrics":
        return {"limit": 200}
    if route == "/v1/ops/rag/retrieval-trace":
        return {"limit": 20}
    if route == "/v1/ops/agent/debate":
        return {"stock_code": "SH600000"}
    if route == "/v1/stocks/search":
        return {"keyword": "银行", "limit": 10}
    return {}


def _seed_context(client: TestClient, headers: dict[str, str]) -> dict[str, str]:
    ctx: dict[str, str] = {
        "session_id": "missing-session",
        "task_id": "missing-task",
        "report_id": "missing-report",
        "doc_id": "smoke-doc-1",
        "journal_id": "1",
        "portfolio_id": "1",
        "pool_id": "missing-pool",
        "alert_id": "1",
        "rule_id": "1",
        "chunk_id": "missing-chunk",
        "memory_id": "missing-memory",
        "run_id": "missing-run",
        "eval_run_id": "missing-eval",
        "prompt_id": "fact_qa",
        "prompt_version": "",
        "trace_id": "missing-trace",
        "source": "user_upload",
        "stock_code": "SH600000",
    }

    # Default to lightweight seed to keep smoke runtime stable in CI/local dev.
    # Set `SMOKE_SEED_FULL=1` only when you intentionally want deeper fixture setup.
    full_seed = os.environ.get("SMOKE_SEED_FULL", "").strip() in {"1", "true", "TRUE"}
    if not full_seed:
        prompt_versions = client.get(f"/v1/ops/prompts/{ctx['prompt_id']}/versions", headers=headers)
        if prompt_versions.status_code == 200 and isinstance(prompt_versions.json(), list) and prompt_versions.json():
            first = prompt_versions.json()[0]
            if isinstance(first, dict):
                version = str(first.get("version", "")).strip()
                if version:
                    ctx["prompt_version"] = version
        return ctx

    # Query + history trace seed.
    q = client.post("/v1/query", json={"user_id": "smoke-user", "question": "seed", "stock_codes": ["SH600000"]}, headers=headers)
    if q.status_code == 200:
        body = q.json()
        if str(body.get("trace_id", "")).strip():
            ctx["trace_id"] = str(body.get("trace_id"))

    # Eval run id.
    ev = client.post("/v1/evals/run", json={"samples": [{"fact_correct": True, "has_citation": True, "hallucination": False, "violation": False}]}, headers=headers)
    if ev.status_code == 200:
        eval_run_id = str(ev.json().get("eval_run_id", "")).strip()
        if eval_run_id:
            ctx["eval_run_id"] = eval_run_id

    # DeepThink session/task ids.
    s = client.post(
        "/v1/deep-think/sessions",
        json={"user_id": "smoke-user", "question": "seed deep", "stock_codes": ["SH600000"], "max_rounds": 2},
        headers=headers,
    )
    if s.status_code == 200:
        session_id = str(s.json().get("session_id", "")).strip()
        if session_id:
            ctx["session_id"] = session_id
            r = client.post(f"/v1/deep-think/sessions/{session_id}/rounds", json={"question": "seed round"}, headers=headers)
            if r.status_code == 200:
                round_id = str((r.json().get("latest_round") or {}).get("round_id", "")).strip()
                if round_id:
                    ctx["round_id"] = round_id

    # Report ids.
    rep = client.post("/v1/report/generate", json={"user_id": "smoke-user", "stock_code": "SH600000", "period": "1y", "report_type": "fact"}, headers=headers)
    if rep.status_code == 200:
        report_id = str(rep.json().get("report_id", "")).strip()
        if report_id:
            ctx["report_id"] = report_id

    task = client.post("/v1/report/tasks", json={"user_id": "smoke-user", "stock_code": "SH600000", "period": "1y", "report_type": "fact"}, headers=headers)
    if task.status_code == 200:
        task_id = str(task.json().get("task_id", "")).strip()
        if task_id:
            ctx["task_id"] = task_id

    # Docs / rag chunks.
    up = client.post(
        "/v1/docs/upload",
        json={"doc_id": "smoke-doc-1", "filename": "smoke.txt", "content": "SH600000 smoke " * 100, "source": "upload"},
        headers=headers,
    )
    if up.status_code == 200:
        ctx["doc_id"] = "smoke-doc-1"
        _ = client.post("/v1/docs/smoke-doc-1/index", json={}, headers=headers)
        chunks = client.get("/v1/rag/docs/chunks?doc_id=smoke-doc-1&limit=5", headers=headers)
        if chunks.status_code == 200 and isinstance(chunks.json(), list) and chunks.json():
            ctx["chunk_id"] = str(chunks.json()[0].get("chunk_id", "missing-chunk"))

    # Journal id.
    j = client.post("/v1/journal", json={"journal_type": "decision", "stock_code": "SH600000", "decision_type": "hold"}, headers=headers)
    if j.status_code == 200:
        jid = str(j.json().get("journal_id", "")).strip()
        if jid:
            ctx["journal_id"] = jid

    # Watchlist pool id.
    p = client.post("/v1/watchlist/pools", json={"pool_name": "smoke-pool", "description": "smoke"}, headers=headers)
    if p.status_code == 200:
        pool_id = str(p.json().get("pool_id", "")).strip()
        if pool_id:
            ctx["pool_id"] = pool_id

    # Portfolio id + backtest run id.
    port = client.post("/v1/portfolio", json={"portfolio_name": "smoke-portfolio", "initial_capital": 100000, "description": "smoke"}, headers=headers)
    if port.status_code == 200:
        pid = str(port.json().get("portfolio_id", "")).strip()
        if pid:
            ctx["portfolio_id"] = pid

    back = client.post(
        "/v1/backtest/run",
        json={"stock_code": "SH600000", "start_date": "2024-01-01", "end_date": "2026-02-15", "initial_capital": 100000, "ma_window": 20},
        headers=headers,
    )
    if back.status_code == 200:
        rid = str(back.json().get("run_id", "")).strip()
        if rid:
            ctx["run_id"] = rid

    # RAG memory id.
    mem = client.get("/v1/rag/qa-memory?limit=5", headers=headers)
    if mem.status_code == 200 and isinstance(mem.json(), list) and mem.json():
        ctx["memory_id"] = str(mem.json()[0].get("memory_id", "missing-memory"))

    # DeepThink intel trace id.
    intel = client.get("/v1/deep-think/intel/self-test?stock_code=SH600000", headers=headers)
    if intel.status_code == 200:
        tid = str(intel.json().get("trace_id", "")).strip()
        if tid:
            ctx["trace_id"] = tid

    # Alert rule id.
    rule = client.post(
        "/v1/alerts/rules",
        json={"rule_name": "smoke-rule", "rule_type": "price", "stock_code": "SH600000", "operator": ">", "target_value": 1, "event_type": "quote", "is_active": True},
        headers=headers,
    )
    if rule.status_code == 200:
        rule_id = str(rule.json().get("rule_id", "")).strip()
        if rule_id:
            ctx["rule_id"] = rule_id

    # Prompt compare context: detect an existing version to avoid false negatives.
    prompt_versions = client.get(f"/v1/ops/prompts/{ctx['prompt_id']}/versions", headers=headers)
    if prompt_versions.status_code == 200 and isinstance(prompt_versions.json(), list) and prompt_versions.json():
        first = prompt_versions.json()[0]
        if isinstance(first, dict):
            version = str(first.get("version", "")).strip()
            if version:
                ctx["prompt_version"] = version

    return ctx


def _call_route(
    client: TestClient,
    *,
    method: str,
    template: str,
    path: str,
    params: dict[str, Any],
    payload: dict[str, Any] | None,
    headers: dict[str, str],
) -> SmokeResult:
    """
    Execute a single route probe.

    For SSE routes we only read the first available chunk then close the stream,
    so the full-suite smoke test does not block on long-running event streams.
    """
    request_kwargs: dict[str, Any] = {"params": params, "headers": headers}
    if payload is not None:
        request_kwargs["json"] = payload
    run_heavy = os.environ.get("SMOKE_RUN_HEAVY", "").strip() in {"1", "true", "TRUE"}

    try:
        if template.endswith("/stream"):
            with client.stream(method, path, timeout=12.0, **request_kwargs) as resp:
                first_chunk = ""
                if run_heavy:
                    # Only read one event chunk for liveness assertion.
                    for chunk in resp.iter_text():
                        text = str(chunk or "").strip()
                        if text:
                            first_chunk = text[:200]
                            break
                return SmokeResult(
                    method=method,
                    route=template,
                    status_code=resp.status_code,
                    ok=resp.status_code < 500,
                    detail=first_chunk if resp.status_code >= 500 else "",
                )

        resp = client.request(method, path, timeout=12.0, **request_kwargs)
        return SmokeResult(
            method=method,
            route=template,
            status_code=resp.status_code,
            ok=resp.status_code < 500,
            detail=resp.text[:300] if resp.status_code >= 500 else "",
        )
    except Exception as ex:  # noqa: BLE001
        return SmokeResult(method=method, route=template, status_code=599, ok=False, detail=str(ex)[:300])


def run_full_smoke() -> int:
    app = create_app()
    with TestClient(app) as client:
        headers = {"Authorization": "Bearer smoke-dev-token"}
        ctx = _seed_context(client, headers)
        verbose = os.environ.get("SMOKE_VERBOSE", "").strip() in {"1", "true", "TRUE"}
        run_heavy = os.environ.get("SMOKE_RUN_HEAVY", "").strip() in {"1", "true", "TRUE"}
        heavy_critical_only = os.environ.get("SMOKE_HEAVY_CRITICAL", "").strip() in {"1", "true", "TRUE"}

        routes: list[tuple[str, str]] = []
        for route in app.routes:
            if not isinstance(route, APIRoute):
                continue
            path = str(route.path)
            if not (path.startswith("/v1/") or path.startswith("/v2/")):
                continue
            for method in sorted(m for m in route.methods if m not in {"HEAD", "OPTIONS"}):
                routes.append((method, path))

        results: list[SmokeResult] = []
        dedup_routes = sorted(set(routes))
        total = len(dedup_routes)
        for idx, (method, template) in enumerate(dedup_routes, start=1):
            path = _replace_path_params(template, ctx)
            params = _params_for(template, ctx)
            payload = _payload_for(template, ctx) if method in {"POST", "PUT", "PATCH", "DELETE"} else None
            if verbose:
                print(f"[api-full-selftest] probing {idx}/{total}: {method} {template}", flush=True)

            # Lightweight mode intentionally skips SSE endpoints that can block forever
            # without a live producer session.
            if (not run_heavy) and _should_skip_light_route(method, template):
                results.append(
                    SmokeResult(
                        method=method,
                        route=template,
                        status_code=299,
                        ok=True,
                        detail="skipped_light_mode",
                    )
                )
                continue
            if run_heavy and heavy_critical_only and (not _is_heavy_critical_route(method, template)):
                results.append(
                    SmokeResult(
                        method=method,
                        route=template,
                        status_code=298,
                        ok=True,
                        detail="skipped_heavy_critical_only",
                    )
                )
                continue

            results.append(
                _call_route(
                    client,
                    method=method,
                    template=template,
                    path=path,
                    params=params,
                    payload=payload,
                    headers=headers,
                )
            )

        failed = [r for r in results if not r.ok]
        print(f"[api-full-selftest] total={len(results)} failed={len(failed)}")
        for row in failed:
            print(f"[FAIL] {row.method} {row.route} -> {row.status_code} {row.detail}")

        # Emit quick status histogram for easier regression tracking.
        status_hist: dict[int, int] = {}
        for row in results:
            status_hist[row.status_code] = status_hist.get(row.status_code, 0) + 1
        print("[api-full-selftest] status_hist=", json.dumps(status_hist, ensure_ascii=False, sort_keys=True))

        return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(run_full_smoke())
