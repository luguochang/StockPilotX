from __future__ import annotations

import hashlib
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from dataclasses import dataclass
from typing import Any, Callable


@dataclass
class CacheEntry:
    """In-memory cache entry for query responses."""

    value: dict[str, Any]
    created_at: float


class QueryOptimizer:
    """Query optimizer for Phase 1 Query Hub.

    Responsibilities:
    - Build stable cache keys for semantically identical requests.
    - Keep a bounded in-memory cache with TTL.
    - Execute query work with timeout guard to prevent long blocking requests.
    """

    def __init__(self, cache_size: int = 300, ttl_seconds: int = 180, timeout_seconds: int = 30) -> None:
        self.cache_size = max(1, int(cache_size))
        self.ttl_seconds = max(1, int(ttl_seconds))
        self.timeout_seconds = max(1, int(timeout_seconds))
        self._cache: dict[str, CacheEntry] = {}
        self._lock = threading.RLock()

    def _build_cache_key(self, question: str, stock_codes: list[str], context: dict[str, Any] | None = None) -> str:
        """Generate deterministic cache key from normalized request fields."""
        normalized = {
            "q": str(question or "").strip(),
            "stocks": sorted({str(code or "").strip().upper() for code in stock_codes if str(code or "").strip()}),
            "ctx": context or {},
        }
        payload = json.dumps(normalized, ensure_ascii=False, sort_keys=True)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def get_cached(self, question: str, stock_codes: list[str], context: dict[str, Any] | None = None) -> dict[str, Any] | None:
        """Return cached result if key exists and is still fresh."""
        key = self._build_cache_key(question, stock_codes, context)
        now = time.time()
        with self._lock:
            entry = self._cache.get(key)
            if not entry:
                return None
            if (now - entry.created_at) > self.ttl_seconds:
                self._cache.pop(key, None)
                return None
            return json.loads(json.dumps(entry.value, ensure_ascii=False))

    def set_cached(
        self,
        question: str,
        stock_codes: list[str],
        value: dict[str, Any],
        context: dict[str, Any] | None = None,
    ) -> None:
        """Store a response into in-memory cache with bounded eviction."""
        key = self._build_cache_key(question, stock_codes, context)
        with self._lock:
            if len(self._cache) >= self.cache_size:
                oldest_key = min(self._cache.items(), key=lambda item: item[1].created_at)[0]
                self._cache.pop(oldest_key, None)
            self._cache[key] = CacheEntry(value=json.loads(json.dumps(value, ensure_ascii=False)), created_at=time.time())

    def run_with_timeout(self, fn: Callable[[], dict[str, Any]]) -> dict[str, Any]:
        """Execute query function with timeout control."""
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(fn)
            try:
                return future.result(timeout=self.timeout_seconds)
            except FutureTimeoutError as ex:
                raise TimeoutError(f"query timeout after {self.timeout_seconds}s") from ex
