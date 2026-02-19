from __future__ import annotations

import json

from backend.app.service import AShareAgentService

try:
    from fastapi import FastAPI, Header, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import Response, StreamingResponse
except Exception as ex:  # pragma: no cover
    raise RuntimeError("FastAPI is not installed. Please install fastapi and uvicorn.") from ex


def _extract_bearer_token(authorization: str | None) -> str:
    if not authorization:
        return ""
    prefix = "Bearer "
    if not authorization.startswith(prefix):
        return ""
    return authorization[len(prefix) :].strip()


def _extract_optional_bearer_token(authorization: str | None) -> str:
    if not authorization:
        return ""
    prefix = "Bearer "
    if not authorization.startswith(prefix):
        return ""
    return authorization[len(prefix) :].strip()


def _raise_auth_http_error(ex: Exception) -> None:
    if isinstance(ex, PermissionError):
        raise HTTPException(status_code=403, detail=str(ex)) from ex
    raise HTTPException(status_code=400, detail=str(ex)) from ex


def _error_code(result: dict | None) -> str:
    if not isinstance(result, dict):
        return ""
    return str(result.get("error", "") or "").strip()


def create_app() -> FastAPI:
    app = FastAPI(title="A-Share Agent System API")
    # 允许前端本地开发跨域访问后端接口
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://127.0.0.1:3000",
            "http://localhost:3000",
        ],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    svc = AShareAgentService()

    # ---------------- Existing Core APIs ----------------
    @app.post("/v1/query")
    def query(payload: dict):
        try:
            return svc.query(payload)
        except ValueError as ex:
            raise HTTPException(status_code=400, detail=str(ex)) from ex

    @app.post("/v1/query/compare")
    def query_compare(payload: dict):
        try:
            return svc.query_compare(payload)
        except ValueError as ex:
            raise HTTPException(status_code=400, detail=str(ex)) from ex

    @app.get("/v1/query/history")
    def query_history(
        limit: int = 50,
        stock_code: str = "",
        created_from: str = "",
        created_to: str = "",
        authorization: str | None = Header(default=None),
    ):
        token = _extract_optional_bearer_token(authorization)
        try:
            return svc.query_history_list(
                token,
                limit=limit,
                stock_code=stock_code,
                created_from=created_from,
                created_to=created_to,
            )
        except ValueError as ex:
            raise HTTPException(status_code=400, detail=str(ex)) from ex
        except Exception as ex:  # noqa: BLE001
            _raise_auth_http_error(ex)

    @app.delete("/v1/query/history")
    def query_history_clear(authorization: str | None = Header(default=None)):
        token = _extract_optional_bearer_token(authorization)
        try:
            return svc.query_history_clear(token)
        except Exception as ex:  # noqa: BLE001
            _raise_auth_http_error(ex)

    @app.post("/v1/query/stream")
    def query_stream(payload: dict):
        """SSE 流式接口：按阶段推送问答内容，前端可实时渲染。"""

        def event_gen():
            for event in svc.query_stream_events(payload):
                event_name = str(event.get("event", "message"))
                data = event.get("data", {})
                yield f"event: {event_name}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"

        return StreamingResponse(
            event_gen(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    @app.post("/v1/deep-think/sessions")
    def deep_think_create_session(payload: dict):
        return svc.deep_think_create_session(payload)

    @app.post("/v1/deep-think/sessions/{session_id}/rounds")
    def deep_think_run_round(session_id: str, payload: dict):
        result = svc.deep_think_run_round(session_id, payload)
        if "error" in result:
            raise HTTPException(status_code=404, detail=result)
        return result

    @app.post("/v2/deep-think/sessions/{session_id}/rounds/stream")
    def deep_think_run_round_stream_v2(session_id: str, payload: dict):
        """V2 真流式执行：同一请求内完成“执行 + 事件推送 + 最终收口”。

        说明：
        - 与 v1 `/rounds` + `/stream` 回放不同，此接口在执行过程中实时输出事件。
        - 前端可直接消费 SSE 事件来更新轮次进度，减少“空等待”。
        """

        def event_gen():
            for event in svc.deep_think_run_round_stream_events(session_id, payload):
                event_name = str(event.get("event", "message"))
                data = event.get("data", {})
                yield f"event: {event_name}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"

        return StreamingResponse(
            event_gen(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    @app.get("/v1/deep-think/sessions/{session_id}")
    def deep_think_get_session(session_id: str):
        result = svc.deep_think_get_session(session_id)
        if "error" in result:
            raise HTTPException(status_code=404, detail=result)
        return result

    @app.get("/v1/deep-think/sessions/{session_id}/report-export")
    def deep_think_report_export(session_id: str, format: str = "markdown"):
        try:
            result = svc.deep_think_export_report(session_id, format=format)
        except ValueError as ex:
            raise HTTPException(status_code=400, detail=str(ex)) from ex
        if _error_code(result):
            raise HTTPException(status_code=404, detail=result)
        filename = str(result.get("filename", "deepthink-report.txt"))
        media_type = str(result.get("media_type", "text/plain; charset=utf-8"))
        content = result.get("content", "")
        return Response(
            content=content,
            media_type=media_type,
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )

    @app.get("/v1/deep-think/sessions/{session_id}/stream")
    def deep_think_stream(session_id: str):
        def event_gen():
            for event in svc.deep_think_stream_events(session_id):
                event_name = str(event.get("event", "message"))
                data = event.get("data", {})
                yield f"event: {event_name}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"

        return StreamingResponse(
            event_gen(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    @app.get("/v1/deep-think/sessions/{session_id}/events")
    def deep_think_events(
        session_id: str,
        round_id: str = "",
        limit: int = 200,
        event_name: str = "",
        cursor: int = 0,
        created_from: str = "",
        created_to: str = "",
    ):
        try:
            result = svc.deep_think_list_events(
                session_id,
                round_id=round_id.strip() or None,
                limit=limit,
                event_name=event_name.strip() or None,
                cursor=(cursor if cursor > 0 else None),
                created_from=created_from.strip() or None,
                created_to=created_to.strip() or None,
            )
        except ValueError as ex:
            raise HTTPException(status_code=400, detail=str(ex)) from ex
        if "error" in result:
            raise HTTPException(status_code=404, detail=result)
        return result

    @app.get("/v1/deep-think/sessions/{session_id}/events/export")
    def deep_think_events_export(
        session_id: str,
        format: str = "jsonl",
        round_id: str = "",
        limit: int = 200,
        event_name: str = "",
        cursor: int = 0,
        created_from: str = "",
        created_to: str = "",
    ):
        try:
            result = svc.deep_think_export_events(
                session_id,
                round_id=round_id.strip() or None,
                limit=limit,
                event_name=event_name.strip() or None,
                cursor=(cursor if cursor > 0 else None),
                created_from=created_from.strip() or None,
                created_to=created_to.strip() or None,
                format=format,
            )
        except ValueError as ex:
            raise HTTPException(status_code=400, detail=str(ex)) from ex
        if _error_code(result):
            raise HTTPException(status_code=404, detail=result)
        filename = str(result.get("filename", "deepthink-events.txt"))
        media_type = str(result.get("media_type", "text/plain; charset=utf-8"))
        content = str(result.get("content", ""))
        return Response(
            content=content,
            media_type=media_type,
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )

    @app.get("/v1/deep-think/sessions/{session_id}/business-export")
    def deep_think_business_export(
        session_id: str,
        format: str = "csv",
        round_id: str = "",
        limit: int = 400,
    ):
        try:
            result = svc.deep_think_export_business(
                session_id,
                round_id=round_id.strip() or None,
                format=format,
                limit=limit,
            )
        except ValueError as ex:
            raise HTTPException(status_code=400, detail=str(ex)) from ex
        if _error_code(result):
            raise HTTPException(status_code=404, detail=result)
        filename = str(result.get("filename", "deepthink-business.txt"))
        media_type = str(result.get("media_type", "text/plain; charset=utf-8"))
        content = str(result.get("content", ""))
        return Response(
            content=content,
            media_type=media_type,
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )

    @app.get("/v1/deep-think/intel/self-test")
    def deep_think_intel_self_test(stock_code: str = "SH600000", question: str = ""):
        """实时情报链路自检：输出开关/provider/tool/fallback 诊断信息。"""
        return svc.deep_think_intel_self_test(stock_code=stock_code, question=question)

    @app.get("/v1/deep-think/intel/traces/{trace_id}")
    def deep_think_intel_trace_events(trace_id: str, limit: int = 120):
        """按 trace_id 查询情报链路日志，便于前后端联调排障。"""
        return svc.deep_think_trace_events(trace_id, limit=limit)

    @app.post("/v1/deep-think/sessions/{session_id}/events/export-tasks")
    def deep_think_events_export_task_create(session_id: str, payload: dict):
        try:
            result = svc.deep_think_create_export_task(
                session_id,
                format=str(payload.get("format", "jsonl")),
                round_id=str(payload.get("round_id", "")).strip() or None,
                limit=int(payload.get("limit", 200)),
                event_name=str(payload.get("event_name", "")).strip() or None,
                cursor=int(payload.get("cursor", 0)) if str(payload.get("cursor", "")).strip() else None,
                created_from=str(payload.get("created_from", "")).strip() or None,
                created_to=str(payload.get("created_to", "")).strip() or None,
            )
        except ValueError as ex:
            raise HTTPException(status_code=400, detail=str(ex)) from ex
        if _error_code(result):
            raise HTTPException(status_code=404, detail=result)
        return result

    @app.get("/v1/deep-think/sessions/{session_id}/events/export-tasks/{task_id}")
    def deep_think_events_export_task_get(session_id: str, task_id: str):
        result = svc.deep_think_get_export_task(session_id, task_id)
        if _error_code(result):
            raise HTTPException(status_code=404, detail=result)
        return result

    @app.get("/v1/deep-think/sessions/{session_id}/events/export-tasks/{task_id}/download")
    def deep_think_events_export_task_download(session_id: str, task_id: str):
        result = svc.deep_think_download_export_task(session_id, task_id)
        error_code = _error_code(result)
        if error_code:
            if error_code == "not_found":
                raise HTTPException(status_code=404, detail=result)
            if error_code == "not_ready":
                raise HTTPException(status_code=409, detail=result)
            if error_code == "failed":
                raise HTTPException(status_code=500, detail=result)
            raise HTTPException(status_code=400, detail=result)
        filename = str(result.get("filename", "deepthink-export.txt"))
        media_type = str(result.get("media_type", "text/plain; charset=utf-8"))
        content = str(result.get("content", ""))
        return Response(
            content=content,
            media_type=media_type,
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )

    @app.post("/v1/report/generate")
    def report_generate(payload: dict):
        return svc.report_generate(payload)

    @app.get("/v1/report/{report_id}")
    def report_get(report_id: str):
        result = svc.report_get(report_id)
        if "error" in result:
            raise HTTPException(status_code=404, detail=result)
        return result

    @app.post("/v1/ingest/market-daily")
    def ingest_market_daily(payload: dict):
        return svc.ingest_market_daily(payload.get("stock_codes", []))

    @app.post("/v1/ingest/announcements")
    def ingest_announcements(payload: dict):
        return svc.ingest_announcements(payload.get("stock_codes", []))

    @app.post("/v1/ingest/financials")
    def ingest_financials(payload: dict):
        return svc.ingest_financials(payload.get("stock_codes", []))

    @app.post("/v1/ingest/news")
    def ingest_news(payload: dict):
        # `limit` lets operators control cost/latency in batch backfill scenarios.
        return svc.ingest_news(payload.get("stock_codes", []), limit=int(payload.get("limit", 20)))

    @app.post("/v1/ingest/research")
    def ingest_research(payload: dict):
        return svc.ingest_research_reports(payload.get("stock_codes", []), limit=int(payload.get("limit", 20)))

    @app.post("/v1/ingest/macro")
    def ingest_macro(payload: dict):
        return svc.ingest_macro_indicators(limit=int(payload.get("limit", 20)))

    @app.post("/v1/ingest/fund")
    def ingest_fund(payload: dict):
        return svc.ingest_fund_snapshots(payload.get("stock_codes", []))

    @app.get("/v1/datasources/sources")
    def datasource_sources(authorization: str | None = Header(default=None)):
        token = _extract_optional_bearer_token(authorization)
        try:
            return svc.datasource_sources(token)
        except Exception as ex:  # noqa: BLE001
            _raise_auth_http_error(ex)

    @app.get("/v1/datasources/health")
    def datasource_health(limit: int = 200, authorization: str | None = Header(default=None)):
        token = _extract_optional_bearer_token(authorization)
        try:
            return svc.datasource_health(token, limit=limit)
        except Exception as ex:  # noqa: BLE001
            _raise_auth_http_error(ex)

    @app.post("/v1/datasources/fetch")
    def datasource_fetch(payload: dict, authorization: str | None = Header(default=None)):
        token = _extract_optional_bearer_token(authorization)
        try:
            return svc.datasource_fetch(token, payload)
        except Exception as ex:  # noqa: BLE001
            _raise_auth_http_error(ex)

    @app.get("/v1/datasources/logs")
    def datasource_logs(
        source_id: str = "",
        status: str = "",
        limit: int = 100,
        authorization: str | None = Header(default=None),
    ):
        token = _extract_optional_bearer_token(authorization)
        try:
            return svc.datasource_logs(token, source_id=source_id, status=status, limit=limit)
        except Exception as ex:  # noqa: BLE001
            _raise_auth_http_error(ex)

    @app.post("/v1/docs/upload")
    def docs_upload(payload: dict):
        return svc.docs_upload(
            payload["doc_id"],
            payload["filename"],
            payload["content"],
            payload.get("source", "user_upload"),
        )

    @app.post("/v1/docs/{doc_id}/index")
    def docs_index(doc_id: str):
        return svc.docs_index(doc_id)

    @app.post("/v1/evals/run")
    def evals_run(payload: dict):
        return svc.evals_run(payload.get("samples"))

    @app.get("/v1/evals/{eval_run_id}")
    def evals_get(eval_run_id: str):
        return svc.evals_get(eval_run_id)

    @app.post("/v1/scheduler/run")
    def scheduler_run(payload: dict):
        return svc.scheduler_run(payload["job_name"])

    @app.get("/v1/scheduler/status")
    def scheduler_status():
        return svc.scheduler_status()

    # ---------------- Prediction APIs ----------------
    @app.post("/v1/predict/run")
    def predict_run(payload: dict, authorization: str | None = Header(default=None)):
        token = _extract_optional_bearer_token(authorization)
        if token and not str(payload.get("token", "")).strip():
            payload = dict(payload)
            payload["token"] = token
        return svc.predict_run(payload)

    @app.get("/v1/predict/evals/latest")
    def predict_eval_latest():
        return svc.predict_eval_latest()

    @app.get("/v1/factors/{stock_code}")
    def factor_snapshot(stock_code: str):
        return svc.factor_snapshot(stock_code)

    @app.get("/v1/predict/{run_id}")
    def predict_get(run_id: str):
        result = svc.predict_get(run_id)
        if "error" in result:
            raise HTTPException(status_code=404, detail=result)
        return result

    @app.get("/v1/market/overview/{stock_code}")
    def market_overview(stock_code: str):
        return svc.market_overview(stock_code)

    @app.get("/v1/analysis/intel-card")
    def analysis_intel_card(stock_code: str, horizon: str = "30d", risk_profile: str = "neutral"):
        try:
            return svc.analysis_intel_card(stock_code, horizon=horizon, risk_profile=risk_profile)
        except ValueError as ex:
            raise HTTPException(status_code=400, detail=str(ex)) from ex

    @app.post("/v1/analysis/intel-card/feedback")
    def analysis_intel_feedback(payload: dict):
        try:
            return svc.analysis_intel_feedback(payload)
        except ValueError as ex:
            raise HTTPException(status_code=400, detail=str(ex)) from ex

    @app.get("/v1/analysis/intel-card/review")
    def analysis_intel_review(stock_code: str = "", limit: int = 120):
        try:
            return svc.analysis_intel_review(stock_code, limit=limit)
        except ValueError as ex:
            raise HTTPException(status_code=400, detail=str(ex)) from ex

    @app.post("/v1/backtest/run")
    def backtest_run(payload: dict):
        try:
            return svc.backtest_run(payload)
        except ValueError as ex:
            raise HTTPException(status_code=400, detail=str(ex)) from ex

    @app.get("/v1/backtest/{run_id}")
    def backtest_get(run_id: str):
        result = svc.backtest_get(run_id)
        if "error" in result:
            raise HTTPException(status_code=404, detail=result)
        return result

    # ---------------- WEB-001 Auth ----------------
    @app.post("/v1/auth/register")
    def auth_register(payload: dict):
        try:
            return svc.auth_register(payload)
        except Exception as ex:  # noqa: BLE001
            raise HTTPException(status_code=400, detail=str(ex)) from ex

    @app.post("/v1/auth/login")
    def auth_login(payload: dict):
        try:
            return svc.auth_login(payload)
        except Exception as ex:  # noqa: BLE001
            raise HTTPException(status_code=401, detail=str(ex)) from ex

    @app.get("/v1/auth/me")
    def auth_me(authorization: str | None = Header(default=None)):
        token = _extract_bearer_token(authorization)
        try:
            return svc.auth_me(token)
        except Exception as ex:  # noqa: BLE001
            raise HTTPException(status_code=401, detail=str(ex)) from ex

    @app.post("/v1/auth/refresh")
    def auth_refresh(authorization: str | None = Header(default=None)):
        token = _extract_bearer_token(authorization)
        me = svc.auth_me(token)
        # 用登录逻辑返回新 token，当前实现保持简化
        return {"access_token": token, "token_type": "bearer", "user_id": me["user_id"]}

    # ---------------- WEB-002 Watchlist / Dashboard ----------------
    @app.get("/v1/watchlist")
    def watchlist_list(authorization: str | None = Header(default=None)):
        token = _extract_bearer_token(authorization)
        try:
            return svc.watchlist_list(token)
        except Exception as ex:  # noqa: BLE001
            _raise_auth_http_error(ex)

    @app.post("/v1/watchlist")
    def watchlist_add(payload: dict, authorization: str | None = Header(default=None)):
        token = _extract_bearer_token(authorization)
        try:
            return svc.watchlist_add(token, payload["stock_code"])
        except Exception as ex:  # noqa: BLE001
            _raise_auth_http_error(ex)

    @app.delete("/v1/watchlist/{stock_code}")
    def watchlist_delete(stock_code: str, authorization: str | None = Header(default=None)):
        token = _extract_bearer_token(authorization)
        try:
            return svc.watchlist_delete(token, stock_code)
        except Exception as ex:  # noqa: BLE001
            _raise_auth_http_error(ex)

    @app.get("/v1/watchlist/pools")
    def watchlist_pool_list(authorization: str | None = Header(default=None)):
        token = _extract_optional_bearer_token(authorization)
        try:
            return svc.watchlist_pool_list(token)
        except Exception as ex:  # noqa: BLE001
            _raise_auth_http_error(ex)

    @app.post("/v1/watchlist/pools")
    def watchlist_pool_create(payload: dict, authorization: str | None = Header(default=None)):
        token = _extract_optional_bearer_token(authorization)
        try:
            return svc.watchlist_pool_create(token, payload)
        except Exception as ex:  # noqa: BLE001
            _raise_auth_http_error(ex)

    @app.get("/v1/watchlist/pools/{pool_id}/stocks")
    def watchlist_pool_stocks(pool_id: str, authorization: str | None = Header(default=None)):
        token = _extract_optional_bearer_token(authorization)
        try:
            return svc.watchlist_pool_stocks(token, pool_id)
        except Exception as ex:  # noqa: BLE001
            _raise_auth_http_error(ex)

    @app.post("/v1/watchlist/pools/{pool_id}/stocks")
    def watchlist_pool_add_stock(pool_id: str, payload: dict, authorization: str | None = Header(default=None)):
        token = _extract_optional_bearer_token(authorization)
        try:
            return svc.watchlist_pool_add_stock(token, pool_id, payload)
        except Exception as ex:  # noqa: BLE001
            _raise_auth_http_error(ex)

    @app.delete("/v1/watchlist/pools/{pool_id}/stocks/{stock_code}")
    def watchlist_pool_delete_stock(pool_id: str, stock_code: str, authorization: str | None = Header(default=None)):
        token = _extract_optional_bearer_token(authorization)
        try:
            return svc.watchlist_pool_delete_stock(token, pool_id, stock_code)
        except Exception as ex:  # noqa: BLE001
            _raise_auth_http_error(ex)

    @app.get("/v1/dashboard/overview")
    def dashboard_overview(authorization: str | None = Header(default=None)):
        token = _extract_bearer_token(authorization)
        try:
            return svc.dashboard_overview(token)
        except Exception as ex:  # noqa: BLE001
            _raise_auth_http_error(ex)

    # ---------------- WEB-010 Portfolio Manager ----------------
    @app.post("/v1/portfolio")
    def portfolio_create(payload: dict, authorization: str | None = Header(default=None)):
        token = _extract_optional_bearer_token(authorization)
        try:
            return svc.portfolio_create(token, payload)
        except Exception as ex:  # noqa: BLE001
            _raise_auth_http_error(ex)

    @app.get("/v1/portfolio")
    def portfolio_list(authorization: str | None = Header(default=None)):
        token = _extract_optional_bearer_token(authorization)
        try:
            return svc.portfolio_list(token)
        except Exception as ex:  # noqa: BLE001
            _raise_auth_http_error(ex)

    @app.post("/v1/portfolio/{portfolio_id}/transactions")
    def portfolio_add_transaction(portfolio_id: int, payload: dict, authorization: str | None = Header(default=None)):
        token = _extract_optional_bearer_token(authorization)
        try:
            return svc.portfolio_add_transaction(token, portfolio_id, payload)
        except Exception as ex:  # noqa: BLE001
            _raise_auth_http_error(ex)

    @app.get("/v1/portfolio/{portfolio_id}")
    def portfolio_summary(portfolio_id: int, authorization: str | None = Header(default=None)):
        token = _extract_optional_bearer_token(authorization)
        try:
            return svc.portfolio_summary(token, portfolio_id)
        except Exception as ex:  # noqa: BLE001
            _raise_auth_http_error(ex)

    @app.get("/v1/portfolio/{portfolio_id}/transactions")
    def portfolio_transactions(portfolio_id: int, limit: int = 200, authorization: str | None = Header(default=None)):
        token = _extract_optional_bearer_token(authorization)
        try:
            return svc.portfolio_transactions(token, portfolio_id, limit=limit)
        except Exception as ex:  # noqa: BLE001
            _raise_auth_http_error(ex)

    # ---------------- WEB-011 Investment Journal ----------------
    @app.post("/v1/journal")
    def journal_create(payload: dict, authorization: str | None = Header(default=None)):
        token = _extract_optional_bearer_token(authorization)
        try:
            return svc.journal_create(token, payload)
        except Exception as ex:  # noqa: BLE001
            _raise_auth_http_error(ex)

    @app.get("/v1/journal")
    def journal_list(
        journal_type: str = "",
        stock_code: str = "",
        limit: int = 20,
        offset: int = 0,
        authorization: str | None = Header(default=None),
    ):
        token = _extract_optional_bearer_token(authorization)
        try:
            return svc.journal_list(
                token,
                journal_type=journal_type,
                stock_code=stock_code,
                limit=limit,
                offset=offset,
            )
        except Exception as ex:  # noqa: BLE001
            _raise_auth_http_error(ex)

    @app.get("/v1/journal/insights")
    def journal_insights(
        window_days: int = 90,
        limit: int = 400,
        timeline_days: int = 30,
        authorization: str | None = Header(default=None),
    ):
        token = _extract_optional_bearer_token(authorization)
        try:
            return svc.journal_insights(
                token,
                window_days=window_days,
                limit=limit,
                timeline_days=timeline_days,
            )
        except Exception as ex:  # noqa: BLE001
            _raise_auth_http_error(ex)

    @app.post("/v1/journal/{journal_id}/reflections")
    def journal_reflection_add(journal_id: int, payload: dict, authorization: str | None = Header(default=None)):
        token = _extract_optional_bearer_token(authorization)
        try:
            return svc.journal_reflection_add(token, journal_id, payload)
        except Exception as ex:  # noqa: BLE001
            _raise_auth_http_error(ex)

    @app.get("/v1/journal/{journal_id}/reflections")
    def journal_reflection_list(journal_id: int, limit: int = 50, authorization: str | None = Header(default=None)):
        token = _extract_optional_bearer_token(authorization)
        try:
            return svc.journal_reflection_list(token, journal_id, limit=limit)
        except Exception as ex:  # noqa: BLE001
            _raise_auth_http_error(ex)

    @app.post("/v1/journal/{journal_id}/ai-reflection/generate")
    def journal_ai_reflection_generate(journal_id: int, payload: dict, authorization: str | None = Header(default=None)):
        token = _extract_optional_bearer_token(authorization)
        try:
            return svc.journal_ai_reflection_generate(token, journal_id, payload)
        except Exception as ex:  # noqa: BLE001
            _raise_auth_http_error(ex)

    @app.get("/v1/journal/{journal_id}/ai-reflection")
    def journal_ai_reflection_get(journal_id: int, authorization: str | None = Header(default=None)):
        token = _extract_optional_bearer_token(authorization)
        try:
            return svc.journal_ai_reflection_get(token, journal_id)
        except Exception as ex:  # noqa: BLE001
            _raise_auth_http_error(ex)

    # ---------------- WEB-003 Report Center ----------------
    @app.get("/v1/reports")
    def reports_list(authorization: str | None = Header(default=None)):
        token = _extract_bearer_token(authorization)
        try:
            return svc.reports_list(token)
        except Exception as ex:  # noqa: BLE001
            _raise_auth_http_error(ex)

    @app.get("/v1/reports/{report_id}/versions")
    def report_versions(report_id: str, authorization: str | None = Header(default=None)):
        token = _extract_bearer_token(authorization)
        try:
            return svc.report_versions(token, report_id)
        except Exception as ex:  # noqa: BLE001
            _raise_auth_http_error(ex)

    @app.post("/v1/reports/{report_id}/export")
    def report_export(report_id: str, authorization: str | None = Header(default=None)):
        token = _extract_bearer_token(authorization)
        try:
            return svc.report_export(token, report_id)
        except Exception as ex:  # noqa: BLE001
            _raise_auth_http_error(ex)

    # ---------------- WEB-004 Docs Center ----------------
    @app.get("/v1/docs")
    def docs_list(authorization: str | None = Header(default=None)):
        token = _extract_bearer_token(authorization)
        try:
            return svc.docs_list(token)
        except Exception as ex:  # noqa: BLE001
            _raise_auth_http_error(ex)

    @app.get("/v1/docs/review-queue")
    def docs_review_queue(authorization: str | None = Header(default=None)):
        token = _extract_bearer_token(authorization)
        try:
            return svc.docs_review_queue(token)
        except Exception as ex:  # noqa: BLE001
            _raise_auth_http_error(ex)

    @app.get("/v1/docs/{doc_id}/versions")
    def docs_versions(doc_id: str, limit: int = 20, authorization: str | None = Header(default=None)):
        token = _extract_optional_bearer_token(authorization)
        try:
            return svc.docs_versions(token, doc_id, limit=limit)
        except ValueError as ex:
            raise HTTPException(status_code=400, detail=str(ex)) from ex
        except Exception as ex:  # noqa: BLE001
            _raise_auth_http_error(ex)

    @app.get("/v1/docs/{doc_id}/pipeline-runs")
    def docs_pipeline_runs(doc_id: str, limit: int = 30, authorization: str | None = Header(default=None)):
        token = _extract_optional_bearer_token(authorization)
        try:
            return svc.docs_pipeline_runs(token, doc_id, limit=limit)
        except ValueError as ex:
            raise HTTPException(status_code=400, detail=str(ex)) from ex
        except Exception as ex:  # noqa: BLE001
            _raise_auth_http_error(ex)

    @app.post("/v1/docs/recommend")
    def docs_recommend(payload: dict, authorization: str | None = Header(default=None)):
        token = _extract_optional_bearer_token(authorization)
        try:
            return svc.docs_recommend(token, payload)
        except ValueError as ex:
            raise HTTPException(status_code=400, detail=str(ex)) from ex
        except Exception as ex:  # noqa: BLE001
            _raise_auth_http_error(ex)

    @app.get("/v1/docs/{doc_id}/quality-report")
    def docs_quality_report(doc_id: str, authorization: str | None = Header(default=None)):
        token = _extract_optional_bearer_token(authorization)
        try:
            _ = svc.auth_me(token)
            result = svc.docs_quality_report(doc_id)
            if "error" in result:
                raise HTTPException(status_code=404, detail=result)
            return result
        except HTTPException:
            raise
        except Exception as ex:  # noqa: BLE001
            _raise_auth_http_error(ex)

    @app.get("/v1/knowledge/graph/{entity_id}")
    def knowledge_graph_view(entity_id: str, limit: int = 20):
        try:
            return svc.knowledge_graph_view(entity_id, limit=limit)
        except ValueError as ex:
            raise HTTPException(status_code=400, detail=str(ex)) from ex

    @app.post("/v1/docs/{doc_id}/review/approve")
    def docs_review_approve(doc_id: str, payload: dict, authorization: str | None = Header(default=None)):
        token = _extract_bearer_token(authorization)
        try:
            return svc.docs_review_action(token, doc_id, "approve", payload.get("comment", ""))
        except Exception as ex:  # noqa: BLE001
            _raise_auth_http_error(ex)

    @app.post("/v1/docs/{doc_id}/review/reject")
    def docs_review_reject(doc_id: str, payload: dict, authorization: str | None = Header(default=None)):
        token = _extract_bearer_token(authorization)
        try:
            return svc.docs_review_action(token, doc_id, "reject", payload.get("comment", ""))
        except Exception as ex:  # noqa: BLE001
            _raise_auth_http_error(ex)

    # ---------------- RAG Asset Management ----------------
    @app.get("/v1/rag/source-policy")
    def rag_source_policy_list(authorization: str | None = Header(default=None)):
        token = _extract_bearer_token(authorization)
        try:
            return svc.rag_source_policy_list(token)
        except Exception as ex:  # noqa: BLE001
            _raise_auth_http_error(ex)

    @app.post("/v1/rag/source-policy/{source}")
    def rag_source_policy_set(source: str, payload: dict, authorization: str | None = Header(default=None)):
        token = _extract_bearer_token(authorization)
        try:
            return svc.rag_source_policy_set(token, source, payload)
        except Exception as ex:  # noqa: BLE001
            _raise_auth_http_error(ex)

    @app.get("/v1/rag/docs/chunks")
    def rag_doc_chunks_list(
        authorization: str | None = Header(default=None),
        doc_id: str = "",
        status: str = "",
        source: str = "",
        stock_code: str = "",
        limit: int = 60,
        offset: int = 0,
    ):
        token = _extract_bearer_token(authorization)
        try:
            return svc.rag_doc_chunks_list(
                token,
                doc_id=doc_id,
                status=status,
                source=source,
                stock_code=stock_code,
                limit=limit,
                offset=offset,
            )
        except Exception as ex:  # noqa: BLE001
            _raise_auth_http_error(ex)

    @app.post("/v1/rag/docs/chunks/{chunk_id}/status")
    def rag_doc_chunk_status_set(chunk_id: str, payload: dict, authorization: str | None = Header(default=None)):
        token = _extract_bearer_token(authorization)
        try:
            return svc.rag_doc_chunk_status_set(token, chunk_id, payload)
        except Exception as ex:  # noqa: BLE001
            _raise_auth_http_error(ex)

    @app.get("/v1/rag/qa-memory")
    def rag_qa_memory_list(
        authorization: str | None = Header(default=None),
        stock_code: str = "",
        retrieval_enabled: int = -1,
        limit: int = 100,
        offset: int = 0,
    ):
        token = _extract_bearer_token(authorization)
        try:
            return svc.rag_qa_memory_list(
                token,
                stock_code=stock_code,
                retrieval_enabled=retrieval_enabled,
                limit=limit,
                offset=offset,
            )
        except Exception as ex:  # noqa: BLE001
            _raise_auth_http_error(ex)

    @app.post("/v1/rag/qa-memory/{memory_id}/toggle")
    def rag_qa_memory_toggle(memory_id: str, payload: dict, authorization: str | None = Header(default=None)):
        token = _extract_bearer_token(authorization)
        try:
            return svc.rag_qa_memory_toggle(token, memory_id, payload)
        except Exception as ex:  # noqa: BLE001
            _raise_auth_http_error(ex)

    @app.get("/v1/rag/dashboard")
    def rag_dashboard(authorization: str | None = Header(default=None)):
        token = _extract_bearer_token(authorization)
        try:
            return svc.rag_dashboard(token)
        except Exception as ex:  # noqa: BLE001
            _raise_auth_http_error(ex)

    @app.get("/v1/rag/uploads")
    def rag_uploads_list(
        authorization: str | None = Header(default=None),
        status: str = "",
        source: str = "",
        limit: int = 40,
        offset: int = 0,
    ):
        token = _extract_bearer_token(authorization)
        try:
            return svc.rag_uploads_list(
                token,
                status=status,
                source=source,
                limit=limit,
                offset=offset,
            )
        except Exception as ex:  # noqa: BLE001
            _raise_auth_http_error(ex)

    @app.post("/v1/rag/uploads")
    def rag_upload(payload: dict, authorization: str | None = Header(default=None)):
        token = _extract_bearer_token(authorization)
        try:
            return svc.rag_upload_from_payload(token, payload)
        except Exception as ex:  # noqa: BLE001
            _raise_auth_http_error(ex)

    @app.post("/v1/rag/workflow/upload-and-index")
    def rag_workflow_upload_and_index(payload: dict, authorization: str | None = Header(default=None)):
        token = _extract_bearer_token(authorization)
        try:
            return svc.rag_workflow_upload_and_index(token, payload)
        except Exception as ex:  # noqa: BLE001
            _raise_auth_http_error(ex)

    @app.get("/v1/rag/retrieval-preview")
    def rag_retrieval_preview(
        doc_id: str,
        max_queries: int = 3,
        top_k: int = 5,
        authorization: str | None = Header(default=None),
    ):
        token = _extract_bearer_token(authorization)
        try:
            return svc.rag_retrieval_preview_api(
                token,
                doc_id=doc_id,
                max_queries=max_queries,
                top_k=top_k,
            )
        except Exception as ex:  # noqa: BLE001
            _raise_auth_http_error(ex)

    # ---------------- WEB-005/006/007/008 Ops ----------------
    @app.get("/v1/ops/data-sources/health")
    def ops_source_health(authorization: str | None = Header(default=None)):
        token = _extract_bearer_token(authorization)
        try:
            return svc.ops_source_health(token)
        except Exception as ex:  # noqa: BLE001
            _raise_auth_http_error(ex)

    @app.get("/v1/ops/journal/health")
    def ops_journal_health(
        window_hours: int = 168,
        limit: int = 400,
        authorization: str | None = Header(default=None),
    ):
        token = _extract_bearer_token(authorization)
        try:
            return svc.ops_journal_health(token, window_hours=window_hours, limit=limit)
        except Exception as ex:  # noqa: BLE001
            _raise_auth_http_error(ex)

    @app.get("/v1/ops/deep-think/archive-metrics")
    def ops_deep_think_archive_metrics(window_hours: int = 24, authorization: str | None = Header(default=None)):
        token = _extract_bearer_token(authorization)
        try:
            return svc.ops_deep_think_archive_metrics(token, window_hours=window_hours)
        except Exception as ex:  # noqa: BLE001
            _raise_auth_http_error(ex)

    @app.get("/v1/ops/evals/history")
    def ops_evals_history(authorization: str | None = Header(default=None)):
        token = _extract_bearer_token(authorization)
        try:
            return svc.ops_evals_history(token)
        except Exception as ex:  # noqa: BLE001
            _raise_auth_http_error(ex)

    @app.get("/v1/ops/prompts/releases")
    def ops_prompt_releases(authorization: str | None = Header(default=None)):
        token = _extract_bearer_token(authorization)
        try:
            return svc.ops_prompt_releases(token)
        except Exception as ex:  # noqa: BLE001
            _raise_auth_http_error(ex)

    @app.get("/v1/ops/capabilities")
    def ops_capabilities():
        # 技术实现核查接口：用于确认关键技术点当前落地状态。
        return svc.ops_capabilities()

    @app.get("/v1/ops/agent/debate")
    def ops_agent_debate(stock_code: str, question: str = ""):
        return svc.ops_agent_debate(stock_code=stock_code, question=question)

    @app.get("/v1/ops/rag/quality")
    def ops_rag_quality():
        return svc.ops_rag_quality()

    @app.get("/v1/ops/rag/retrieval-trace")
    def ops_rag_retrieval_trace(
        authorization: str | None = Header(default=None),
        trace_id: str = "",
        limit: int = 120,
    ):
        token = _extract_bearer_token(authorization)
        try:
            return svc.ops_rag_retrieval_trace(token, trace_id=trace_id, limit=limit)
        except Exception as ex:  # noqa: BLE001
            _raise_auth_http_error(ex)

    @app.post("/v1/ops/rag/reindex")
    def ops_rag_reindex(payload: dict, authorization: str | None = Header(default=None)):
        token = _extract_bearer_token(authorization)
        try:
            return svc.ops_rag_reindex(token, limit=int(payload.get("limit", 2000)))
        except Exception as ex:  # noqa: BLE001
            _raise_auth_http_error(ex)

    @app.post("/v1/ops/prompts/compare")
    def ops_prompt_compare(payload: dict):
        return svc.ops_prompt_compare(
            prompt_id=payload.get("prompt_id", "fact_qa"),
            base_version=payload.get("base_version", "1.0.0"),
            candidate_version=payload.get("candidate_version", "1.1.0"),
            variables=payload.get(
                "variables",
                {"question": "请分析SH600000", "stock_codes": ["SH600000"], "evidence": "source:cninfo"},
            ),
        )

    @app.get("/v1/ops/prompts/{prompt_id}/versions")
    def ops_prompt_versions(prompt_id: str):
        return svc.ops_prompt_versions(prompt_id)

    @app.get("/v1/a2a/agent-cards")
    def a2a_agent_cards():
        return svc.a2a_agent_cards()

    @app.post("/v1/a2a/tasks")
    def a2a_create_task(payload: dict):
        return svc.a2a_create_task(payload)

    @app.get("/v1/a2a/tasks/{task_id}")
    def a2a_get_task(task_id: str):
        result = svc.a2a_get_task(task_id)
        if "error" in result:
            raise HTTPException(status_code=404, detail=result)
        return result

    @app.post("/v1/scheduler/pause")
    def scheduler_pause(payload: dict, authorization: str | None = Header(default=None)):
        token = _extract_bearer_token(authorization)
        try:
            svc.web.require_role(token, {"admin", "ops"})
            return svc.scheduler_pause(payload["job_name"])
        except Exception as ex:  # noqa: BLE001
            _raise_auth_http_error(ex)

    @app.post("/v1/scheduler/resume")
    def scheduler_resume(payload: dict, authorization: str | None = Header(default=None)):
        token = _extract_bearer_token(authorization)
        try:
            svc.web.require_role(token, {"admin", "ops"})
            return svc.scheduler_resume(payload["job_name"])
        except Exception as ex:  # noqa: BLE001
            _raise_auth_http_error(ex)

    @app.get("/v1/alerts")
    def alerts_list(authorization: str | None = Header(default=None)):
        token = _extract_bearer_token(authorization)
        try:
            return svc.alerts_list(token)
        except Exception as ex:  # noqa: BLE001
            _raise_auth_http_error(ex)

    @app.post("/v1/alerts/{alert_id}/ack")
    def alerts_ack(alert_id: int, authorization: str | None = Header(default=None)):
        token = _extract_bearer_token(authorization)
        try:
            return svc.alerts_ack(token, alert_id)
        except Exception as ex:  # noqa: BLE001
            _raise_auth_http_error(ex)

    @app.post("/v1/alerts/rules")
    def alert_rule_create(payload: dict, authorization: str | None = Header(default=None)):
        token = _extract_optional_bearer_token(authorization)
        try:
            return svc.alert_rule_create(token, payload)
        except Exception as ex:  # noqa: BLE001
            _raise_auth_http_error(ex)

    @app.get("/v1/alerts/rules")
    def alert_rule_list(authorization: str | None = Header(default=None)):
        token = _extract_optional_bearer_token(authorization)
        try:
            return svc.alert_rule_list(token)
        except Exception as ex:  # noqa: BLE001
            _raise_auth_http_error(ex)

    @app.delete("/v1/alerts/rules/{rule_id}")
    def alert_rule_delete(rule_id: int, authorization: str | None = Header(default=None)):
        token = _extract_optional_bearer_token(authorization)
        try:
            return svc.alert_rule_delete(token, rule_id)
        except Exception as ex:  # noqa: BLE001
            _raise_auth_http_error(ex)

    @app.post("/v1/alerts/check")
    def alert_rule_check(authorization: str | None = Header(default=None)):
        token = _extract_optional_bearer_token(authorization)
        try:
            return svc.alert_rule_check(token)
        except Exception as ex:  # noqa: BLE001
            _raise_auth_http_error(ex)

    @app.get("/v1/alerts/logs")
    def alert_trigger_logs(limit: int = 100, authorization: str | None = Header(default=None)):
        token = _extract_optional_bearer_token(authorization)
        try:
            return svc.alert_trigger_logs(token, limit=limit)
        except Exception as ex:  # noqa: BLE001
            _raise_auth_http_error(ex)

    # ---------------- WEB-009 Stock Universe ----------------
    @app.post("/v1/stocks/sync")
    def stocks_sync(authorization: str | None = Header(default=None)):
        token = _extract_bearer_token(authorization)
        try:
            return svc.stock_universe_sync(token)
        except Exception as ex:  # noqa: BLE001
            _raise_auth_http_error(ex)

    @app.get("/v1/stocks/search")
    def stocks_search(
        authorization: str | None = Header(default=None),
        keyword: str = "",
        exchange: str = "",
        market_tier: str = "",
        listing_board: str = "",
        industry_l1: str = "",
        industry_l2: str = "",
        industry_l3: str = "",
        limit: int = 30,
    ):
        token = _extract_optional_bearer_token(authorization)
        try:
            return svc.stock_universe_search(
                token,
                keyword=keyword,
                exchange=exchange,
                market_tier=market_tier,
                listing_board=listing_board,
                industry_l1=industry_l1,
                industry_l2=industry_l2,
                industry_l3=industry_l3,
                limit=limit,
            )
        except Exception as ex:  # noqa: BLE001
            _raise_auth_http_error(ex)

    @app.get("/v1/stocks/filters")
    def stocks_filters(authorization: str | None = Header(default=None)):
        token = _extract_optional_bearer_token(authorization)
        try:
            return svc.stock_universe_filters(token)
        except Exception as ex:  # noqa: BLE001
            _raise_auth_http_error(ex)

    return app
