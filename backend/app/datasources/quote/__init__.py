from __future__ import annotations

from backend.app.datasources.quote.models import Quote
from backend.app.datasources.quote.netease import NeteaseQuoteAdapter
from backend.app.datasources.quote.service import MockQuoteAdapter, QuoteService
from backend.app.datasources.quote.sina import SinaQuoteAdapter
from backend.app.datasources.quote.tencent import TencentQuoteAdapter
from backend.app.datasources.quote.xueqiu import XueqiuQuoteAdapter

__all__ = [
    "Quote",
    "QuoteService",
    "MockQuoteAdapter",
    "TencentQuoteAdapter",
    "NeteaseQuoteAdapter",
    "SinaQuoteAdapter",
    "XueqiuQuoteAdapter",
]
