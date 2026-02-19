from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any
from urllib.request import ProxyHandler, Request, build_opener


@dataclass(slots=True)
class HttpClient:
    """Small HTTP helper with retry/backoff and optional proxy support.

    This implementation intentionally uses stdlib urllib to avoid introducing
    extra runtime dependencies during scaffold stage.
    """

    timeout_seconds: float = 2.0
    retry_count: int = 2
    retry_backoff_seconds: float = 0.3
    proxy_url: str = ""
    user_agent: str = "StockPilotX/1.0"

    def get_bytes(self, url: str, headers: dict[str, str] | None = None) -> bytes:
        return self._request_bytes(method="GET", url=url, headers=headers, data=None)

    def post_json_bytes(
        self,
        url: str,
        payload: dict[str, Any],
        headers: dict[str, str] | None = None,
    ) -> bytes:
        req_headers = {"Content-Type": "application/json"}
        if headers:
            req_headers.update(headers)
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        return self._request_bytes(method="POST", url=url, headers=req_headers, data=data)

    def _request_bytes(
        self,
        *,
        method: str,
        url: str,
        headers: dict[str, str] | None,
        data: bytes | None,
    ) -> bytes:
        last_error: Exception | None = None
        attempts = max(1, int(self.retry_count) + 1)
        for attempt in range(1, attempts + 1):
            try:
                req_headers = {"User-Agent": self.user_agent}
                if headers:
                    req_headers.update(headers)
                request = Request(url=url, headers=req_headers, data=data, method=method)
                opener = self._build_opener()
                with opener.open(request, timeout=self.timeout_seconds) as response:  # noqa: S310
                    return response.read()
            except Exception as ex:  # noqa: BLE001
                last_error = ex
                if attempt >= attempts:
                    break
                time.sleep(self.retry_backoff_seconds * attempt)
        raise RuntimeError(f"http request failed: method={method}, url={url}; error={last_error}") from last_error

    def get_text(
        self,
        url: str,
        *,
        headers: dict[str, str] | None = None,
        encoding: str = "utf-8",
        errors: str = "ignore",
    ) -> str:
        payload = self.get_bytes(url=url, headers=headers)
        return payload.decode(encoding, errors=errors)

    def _build_opener(self) -> Any:
        if self.proxy_url.strip():
            return build_opener(ProxyHandler({"http": self.proxy_url, "https": self.proxy_url}))
        # Explicitly disable system proxy to keep behavior deterministic.
        return build_opener(ProxyHandler({}))
