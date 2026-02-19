from __future__ import annotations

from backend.app.datasources.news.cls import CLSNewsAdapter
from backend.app.datasources.news.service import MockNewsAdapter, NewsService
from backend.app.datasources.news.tradingview import TradingViewNewsAdapter
from backend.app.datasources.news.xueqiu_news import XueqiuNewsAdapter

__all__ = [
    "NewsService",
    "MockNewsAdapter",
    "CLSNewsAdapter",
    "TradingViewNewsAdapter",
    "XueqiuNewsAdapter",
]
