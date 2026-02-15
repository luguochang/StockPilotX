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
        return svc.query(payload)

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
    def predict_run(payload: dict):
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

    @app.get("/v1/dashboard/overview")
    def dashboard_overview(authorization: str | None = Header(default=None)):
        token = _extract_bearer_token(authorization)
        try:
            return svc.dashboard_overview(token)
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

    # ---------------- WEB-005/006/007/008 Ops ----------------
    @app.get("/v1/ops/data-sources/health")
    def ops_source_health(authorization: str | None = Header(default=None)):
        token = _extract_bearer_token(authorization)
        try:
            return svc.ops_source_health(token)
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
