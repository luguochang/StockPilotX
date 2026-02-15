from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import re
from typing import Any, Protocol
from urllib.parse import quote
from urllib.request import ProxyHandler, Request, build_opener


def now_utc() -> datetime:
    """Return current UTC time."""
    return datetime.now(timezone.utc)


@dataclass(slots=True)
class Quote:
    """Market quote data object."""

    stock_code: str
    price: float
    pct_change: float
    volume: float
    turnover: float
    ts: datetime
    source_id: str
    source_url: str
    reliability_score: float


class QuoteAdapter(Protocol):
    source_id: str

    def fetch_quote(self, stock_code: str) -> Quote:
        ...


class BaseHttpQuoteAdapter:
    """HTTP quote adapter base with short timeout and no system proxy."""

    source_id = "http"
    reliability_score = 0.8
    timeout_sec = 0.5

    def _fetch_text(self, url: str, headers: dict[str, str] | None = None) -> str:
        req = Request(url=url, headers=headers or {"User-Agent": "StockPilotX/1.0"})
        opener = build_opener(ProxyHandler({}))
        with opener.open(req, timeout=self.timeout_sec) as resp:  # noqa: S310
            return resp.read().decode("utf-8", errors="ignore")

    @staticmethod
    def _to_api_code(stock_code: str) -> str:
        code = stock_code.upper().replace(".", "")
        if code.startswith("SH"):
            return "sh" + code[2:]
        if code.startswith("SZ"):
            return "sz" + code[2:]
        if code.startswith("6"):
            return "sh" + code
        return "sz" + code


class TencentLiveAdapter(BaseHttpQuoteAdapter):
    source_id = "tencent"
    reliability_score = 0.85

    def fetch_quote(self, stock_code: str) -> Quote:
        api_code = self._to_api_code(stock_code)
        url = f"https://qt.gtimg.cn/q={api_code}"
        text = self._fetch_text(url)
        m = re.search(r'"([^"]+)"', text)
        if not m:
            raise RuntimeError("tencent parse failed: empty payload")
        fields = m.group(1).split("~")
        if len(fields) < 4:
            raise RuntimeError("tencent parse failed: field too short")
        return _build_quote(
            stock_code=stock_code,
            price=_to_float(_safe_get(fields, 3)),
            pct_change=_to_float(_safe_get(fields, 32)),
            volume=_to_float(_safe_get(fields, 36)),
            turnover=_to_float(_safe_get(fields, 37)),
            source_id=self.source_id,
            source_url=url,
            reliability_score=self.reliability_score,
        )


class NeteaseLiveAdapter(BaseHttpQuoteAdapter):
    source_id = "netease"
    reliability_score = 0.82

    def fetch_quote(self, stock_code: str) -> Quote:
        api_code = _to_netease_code(stock_code)
        url = f"https://api.money.126.net/data/feed/{quote(api_code)},money.api?callback=_ntes_quote_callback"
        text = self._fetch_text(url)
        start = text.find("(")
        end = text.rfind(")")
        if start < 0 or end <= start:
            raise RuntimeError("netease parse failed: invalid callback json")
        payload = json.loads(text[start + 1 : end])
        item = payload.get(api_code)
        if not item:
            raise RuntimeError("netease parse failed: symbol not found")
        return _build_quote(
            stock_code=stock_code,
            price=_to_float(item.get("price")),
            pct_change=_to_float(item.get("percent")),
            volume=_to_float(item.get("volume")),
            turnover=_to_float(item.get("turnover")),
            source_id=self.source_id,
            source_url=url,
            reliability_score=self.reliability_score,
        )


class SinaLiveAdapter(BaseHttpQuoteAdapter):
    source_id = "sina"
    reliability_score = 0.8

    def fetch_quote(self, stock_code: str) -> Quote:
        api_code = self._to_api_code(stock_code)
        url = f"https://hq.sinajs.cn/list={api_code}"
        text = self._fetch_text(
            url,
            headers={"Referer": "https://finance.sina.com.cn", "User-Agent": "StockPilotX/1.0"},
        )
        m = re.search(r'"([^"]+)"', text)
        if not m:
            raise RuntimeError("sina parse failed: empty payload")
        fields = m.group(1).split(",")
        if len(fields) < 10:
            raise RuntimeError("sina parse failed: field too short")
        prev_close = _to_float(_safe_get(fields, 2))
        price = _to_float(_safe_get(fields, 3))
        pct_change = round(((price - prev_close) / prev_close) * 100, 4) if prev_close else 0.0
        return _build_quote(
            stock_code=stock_code,
            price=price,
            pct_change=pct_change,
            volume=_to_float(_safe_get(fields, 8)),
            turnover=_to_float(_safe_get(fields, 9)),
            source_id=self.source_id,
            source_url=url,
            reliability_score=self.reliability_score,
        )


class XueqiuLiveAdapter(BaseHttpQuoteAdapter):
    source_id = "xueqiu"
    reliability_score = 0.78

    def __init__(self, cookie: str | None = None) -> None:
        self.cookie = cookie

    def fetch_quote(self, stock_code: str) -> Quote:
        if not self.cookie:
            raise RuntimeError("xueqiu adapter disabled: missing cookie")
        symbol = _to_xueqiu_symbol(stock_code)
        url = f"https://stock.xueqiu.com/v5/stock/realtime/quotec.json?symbol={quote(symbol)}"
        text = self._fetch_text(
            url,
            headers={"Cookie": self.cookie, "User-Agent": "StockPilotX/1.0", "Referer": "https://xueqiu.com/"},
        )
        payload = json.loads(text)
        items = payload.get("data", [])
        if not items:
            raise RuntimeError("xueqiu parse failed: empty data")
        item = items[0]
        return _build_quote(
            stock_code=stock_code,
            price=_to_float(item.get("current")),
            pct_change=_to_float(item.get("percent")),
            volume=_to_float(item.get("volume")),
            turnover=_to_float(item.get("amount")),
            source_id=self.source_id,
            source_url=url,
            reliability_score=self.reliability_score,
        )


class BaseMockAdapter:
    source_id = "base"
    reliability_score = 0.7

    def __init__(self, fail_codes: set[str] | None = None) -> None:
        self.fail_codes = fail_codes or set()

    def fetch_quote(self, stock_code: str) -> Quote:
        if stock_code in self.fail_codes:
            raise RuntimeError(f"{self.source_id} temporary failure")
        seed = sum(ord(c) for c in stock_code)
        price = round((seed % 5000) / 100 + 5, 2)
        pct = round(((seed % 1000) - 500) / 100, 2)
        return Quote(
            stock_code=stock_code,
            price=price,
            pct_change=pct,
            volume=float(100000 + seed),
            turnover=float(1000000 + seed * 10),
            ts=now_utc(),
            source_id=self.source_id,
            source_url=f"https://{self.source_id}.example.com/{stock_code}",
            reliability_score=self.reliability_score,
        )


class TencentAdapter(BaseMockAdapter):
    source_id = "tencent"
    reliability_score = 0.85


class NeteaseAdapter(BaseMockAdapter):
    source_id = "netease"
    reliability_score = 0.82


class SinaAdapter(BaseMockAdapter):
    source_id = "sina"
    reliability_score = 0.8


class QuoteService:
    """Quote service with fallback chain."""

    def __init__(self, adapters: list[QuoteAdapter]) -> None:
        self.adapters = adapters

    def get_quote(self, stock_code: str) -> Quote:
        errors: list[str] = []
        for adapter in self.adapters:
            try:
                return adapter.fetch_quote(stock_code)
            except Exception as ex:  # noqa: BLE001
                errors.append(str(ex))
        raise RuntimeError("all quote sources failed: " + "; ".join(errors))

    @classmethod
    def build_default(cls, xueqiu_cookie: str | None = None) -> "QuoteService":
        adapters: list[QuoteAdapter] = [
            TencentLiveAdapter(),
            NeteaseLiveAdapter(),
            SinaLiveAdapter(),
            XueqiuLiveAdapter(cookie=xueqiu_cookie),
            TencentAdapter(),
            NeteaseAdapter(),
            SinaAdapter(),
        ]
        return cls(adapters)


class HistoryService:
    source_id = "eastmoney_history"
    reliability_score = 0.9
    timeout_sec = 0.8

    def fetch_daily_bars(self, stock_code: str, beg: str = "20240101", end: str = "20500101", limit: int = 240) -> list[dict]:
        secid = _to_eastmoney_secid(stock_code)
        url = (
            "https://push2his.eastmoney.com/api/qt/stock/kline/get"
            "?fields1=f1,f2,f3,f4,f5,f6"
            "&fields2=f51,f52,f53,f54,f55,f56,f57,f58"
            f"&beg={beg}&end={end}"
            "&ut=fa5fd1943c7b386f172d6893dbfba10b"
            "&rtntype=6"
            f"&secid={secid}"
            "&klt=101&fqt=1"
        )
        req = Request(url=url, headers={"User-Agent": "StockPilotX/1.0"})
        opener = build_opener(ProxyHandler({}))
        with opener.open(req, timeout=self.timeout_sec) as resp:  # noqa: S310
            payload = json.loads(resp.read().decode("utf-8", errors="ignore"))
        klines = (((payload or {}).get("data") or {}).get("klines")) or []
        bars: list[dict] = []
        for row in klines[-limit:]:
            parts = row.split(",")
            if len(parts) < 7:
                continue
            bars.append(
                {
                    "stock_code": _normalize_stock_code(stock_code),
                    "trade_date": parts[0],
                    "open": _to_float(parts[1]),
                    "close": _to_float(parts[2]),
                    "high": _to_float(parts[3]),
                    "low": _to_float(parts[4]),
                    "volume": _to_float(parts[5]),
                    "amount": _to_float(parts[6]),
                    "amplitude": _to_float(parts[7]) if len(parts) > 7 else 0.0,
                    "source_id": self.source_id,
                    "source_url": url,
                    "reliability_score": self.reliability_score,
                }
            )
        if not bars:
            raise RuntimeError("history parse failed: empty bars")
        return bars


class AnnouncementAdapter(Protocol):
    source_id: str

    def fetch_announcements(self, stock_code: str) -> list[dict]:
        ...


class BaseLiveAnnouncementAdapter:
    source_id = "announcement"
    source_url = ""
    reliability_score = 0.95
    timeout_sec = 0.4

    def _fetch_text(self, url: str) -> str:
        req = Request(url=url, headers={"User-Agent": "StockPilotX/1.0"})
        opener = build_opener(ProxyHandler({}))
        with opener.open(req, timeout=self.timeout_sec) as resp:  # noqa: S310
            return resp.read().decode("utf-8", errors="ignore")

    def _extract_title(self, html: str) -> str:
        m = re.search(r"<title>(.*?)</title>", html, re.IGNORECASE | re.DOTALL)
        if not m:
            return "公告源页面抓取成功"
        return re.sub(r"\s+", " ", m.group(1)).strip()

    def _to_event(self, stock_code: str, title: str) -> dict:
        return {
            "stock_code": stock_code,
            "event_type": "announcement_snapshot",
            "title": title[:120],
            "content": f"{self.source_id} 页面抓取成功，可后续扩展结构化公告解析。",
            "event_time": now_utc().isoformat(),
            "source_id": self.source_id,
            "source_url": self.source_url,
            "reliability_score": self.reliability_score,
        }


class CninfoLiveAnnouncementAdapter(BaseLiveAnnouncementAdapter):
    source_id = "cninfo"
    source_url = "https://www.cninfo.com.cn/"
    reliability_score = 0.98

    def fetch_announcements(self, stock_code: str) -> list[dict]:
        html = self._fetch_text(self.source_url)
        return [self._to_event(stock_code, self._extract_title(html))]


class SSELiveAnnouncementAdapter(BaseLiveAnnouncementAdapter):
    source_id = "sse"
    source_url = "https://www.sse.com.cn/disclosure/listedinfo/announcement/"
    reliability_score = 0.97

    def fetch_announcements(self, stock_code: str) -> list[dict]:
        html = self._fetch_text(self.source_url)
        return [self._to_event(stock_code, self._extract_title(html))]


class SZSELiveAnnouncementAdapter(BaseLiveAnnouncementAdapter):
    source_id = "szse"
    source_url = "https://www.szse.cn/disclosure/listed/index.html"
    reliability_score = 0.97

    def fetch_announcements(self, stock_code: str) -> list[dict]:
        html = self._fetch_text(self.source_url)
        return [self._to_event(stock_code, self._extract_title(html))]


class AnnouncementService:
    """Announcement service with live-first fallback."""

    def __init__(self, adapters: list[AnnouncementAdapter] | None = None) -> None:
        self.adapters = adapters or [
            CninfoLiveAnnouncementAdapter(),
            SSELiveAnnouncementAdapter(),
            SZSELiveAnnouncementAdapter(),
        ]

    def fetch_announcements(self, stock_code: str) -> list[dict]:
        events: list[dict] = []
        for adapter in self.adapters:
            try:
                events.extend(adapter.fetch_announcements(stock_code))
            except Exception:
                continue
        if events:
            return events
        return _mock_announcements(stock_code)


def _safe_get(values: list[str], idx: int) -> str:
    return values[idx] if 0 <= idx < len(values) else "0"


def _to_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _build_quote(
    *,
    stock_code: str,
    price: float,
    pct_change: float,
    volume: float,
    turnover: float,
    source_id: str,
    source_url: str,
    reliability_score: float,
) -> Quote:
    return Quote(
        stock_code=stock_code,
        price=round(price, 4),
        pct_change=round(pct_change, 4),
        volume=round(volume, 4),
        turnover=round(turnover, 4),
        ts=now_utc(),
        source_id=source_id,
        source_url=source_url,
        reliability_score=reliability_score,
    )


def _to_netease_code(stock_code: str) -> str:
    code = stock_code.upper().replace(".", "")
    if code.startswith("SH"):
        return "0" + code[2:]
    if code.startswith("SZ"):
        return "1" + code[2:]
    if code.startswith("6"):
        return "0" + code
    return "1" + code


def _to_xueqiu_symbol(stock_code: str) -> str:
    code = stock_code.upper().replace(".", "")
    if code.startswith(("SH", "SZ")):
        return code
    if code.startswith("6"):
        return "SH" + code
    return "SZ" + code


def _to_eastmoney_secid(stock_code: str) -> str:
    code = _normalize_stock_code(stock_code)
    if code.startswith("SH"):
        return "1." + code[2:]
    return "0." + code[2:]


def _normalize_stock_code(stock_code: str) -> str:
    code = stock_code.upper().replace(".", "")
    if code.startswith(("SH", "SZ")):
        return code
    if code.startswith("6"):
        return "SH" + code
    return "SZ" + code


def _mock_announcements(stock_code: str) -> list[dict]:
    year = datetime.now().year
    return [
        {
            "stock_code": stock_code,
            "event_type": "annual_report",
            "title": f"{year - 1}年年报披露",
            "content": "公司披露年度财务报告，营收同比增长。",
            "event_time": datetime(year, 3, 28, tzinfo=timezone.utc).isoformat(),
            "source_id": "cninfo_mock",
            "source_url": "https://www.cninfo.com.cn/",
            "reliability_score": 0.9,
        },
        {
            "stock_code": stock_code,
            "event_type": "board_resolution",
            "title": "董事会决议公告",
            "content": "审议通过年度利润分配预案。",
            "event_time": datetime(year, 4, 1, tzinfo=timezone.utc).isoformat(),
            "source_id": "sse_szse_mock",
            "source_url": "https://www.sse.com.cn/",
            "reliability_score": 0.88,
        },
    ]
