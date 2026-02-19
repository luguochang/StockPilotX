from __future__ import annotations

from backend.app.config import Settings
from backend.app.data.sources import AnnouncementService, HistoryService, QuoteService


def build_default_quote_service(settings: Settings | None = None) -> QuoteService:
    """Build quote service with datasource-aware runtime settings.

    Round-AC keeps the legacy adapters to minimize migration risk. New adapter
    implementations will gradually replace this fallback in later rounds.
    """

    cfg = settings or Settings.from_env()
    cookie = (cfg.datasource_xueqiu_cookie or "").strip() or None
    return QuoteService.build_default(xueqiu_cookie=cookie)


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

