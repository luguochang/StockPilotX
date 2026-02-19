from __future__ import annotations

from backend.app.config import Settings
from backend.app.data.sources import AnnouncementService, HistoryService
from backend.app.datasources.financial.service import FinancialService
from backend.app.datasources.fund.service import FundService
from backend.app.datasources.macro.service import MacroService
from backend.app.datasources.news.service import NewsService
from backend.app.datasources.quote.service import QuoteService
from backend.app.datasources.research.service import ResearchService


def build_default_quote_service(settings: Settings | None = None) -> QuoteService:
    """Build quote service with datasource-aware runtime settings.

    Round-AC keeps the legacy adapters to minimize migration risk. New adapter
    implementations will gradually replace this fallback in later rounds.
    """

    cfg = settings or Settings.from_env()
    return QuoteService.build_default(
        xueqiu_cookie=(cfg.datasource_xueqiu_cookie or "").strip(),
        timeout_seconds=float(cfg.datasource_request_timeout_seconds),
        retry_count=int(cfg.datasource_retry_count),
        retry_backoff_seconds=float(cfg.datasource_retry_backoff_seconds),
        proxy_url=str(cfg.datasource_proxy_url or ""),
    )


def build_default_announcement_service(settings: Settings | None = None) -> AnnouncementService:
    """Build announcement service.

    The `settings` parameter is intentionally kept for a stable interface. It
    will be consumed by source-level toggles in upcoming rounds.
    """

    _ = settings
    return AnnouncementService()


def build_default_history_service(settings: Settings | None = None) -> HistoryService:
    """Build history service.

    Similar to announcements, this keeps a stable constructor contract for
    future datasource options (timeout, retries, provider selection).
    """

    _ = settings
    return HistoryService()


def build_default_financial_service(settings: Settings | None = None) -> FinancialService:
    """Build financial service with multi-source fallback."""

    cfg = settings or Settings.from_env()
    return FinancialService.build_default(
        tushare_token=str(cfg.datasource_tushare_token or ""),
        timeout_seconds=float(cfg.datasource_request_timeout_seconds),
        retry_count=int(cfg.datasource_retry_count),
        retry_backoff_seconds=float(cfg.datasource_retry_backoff_seconds),
        proxy_url=str(cfg.datasource_proxy_url or ""),
    )


def build_default_news_service(settings: Settings | None = None) -> NewsService:
    cfg = settings or Settings.from_env()
    return NewsService.build_default(
        xueqiu_cookie=str(cfg.datasource_xueqiu_cookie or ""),
        timeout_seconds=float(cfg.datasource_request_timeout_seconds),
        retry_count=int(cfg.datasource_retry_count),
        retry_backoff_seconds=float(cfg.datasource_retry_backoff_seconds),
        proxy_url=str(cfg.datasource_proxy_url or ""),
        tradingview_proxy_url=str(cfg.datasource_tradingview_proxy_url or ""),
    )


def build_default_research_service(settings: Settings | None = None) -> ResearchService:
    cfg = settings or Settings.from_env()
    return ResearchService.build_default(
        timeout_seconds=float(cfg.datasource_request_timeout_seconds),
        retry_count=int(cfg.datasource_retry_count),
        retry_backoff_seconds=float(cfg.datasource_retry_backoff_seconds),
        proxy_url=str(cfg.datasource_proxy_url or ""),
    )


def build_default_macro_service(settings: Settings | None = None) -> MacroService:
    cfg = settings or Settings.from_env()
    return MacroService.build_default(
        timeout_seconds=float(cfg.datasource_request_timeout_seconds),
        retry_count=int(cfg.datasource_retry_count),
        retry_backoff_seconds=float(cfg.datasource_retry_backoff_seconds),
        proxy_url=str(cfg.datasource_proxy_url or ""),
    )


def build_default_fund_service(settings: Settings | None = None) -> FundService:
    cfg = settings or Settings.from_env()
    return FundService.build_default(
        timeout_seconds=float(cfg.datasource_request_timeout_seconds),
        retry_count=int(cfg.datasource_retry_count),
        retry_backoff_seconds=float(cfg.datasource_retry_backoff_seconds),
        proxy_url=str(cfg.datasource_proxy_url or ""),
    )
