from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import re
from typing import Any
import urllib.request


@dataclass(slots=True)
class EmbeddingRuntimeConfig:
    provider: str = "local_hash"
    model: str = ""
    base_url: str = ""
    api_key: str = ""
    dim: int = 256
    timeout_seconds: float = 12.0
    batch_size: int = 32
    fallback_to_local: bool = True


class EmbeddingProvider:
    """Embedding 适配层：支持本地哈希向量 + OpenAI 兼容 embeddings 接口。"""

    def __init__(self, config: EmbeddingRuntimeConfig, trace_emit: callable | None = None) -> None:
        self.config = config
        self.trace_emit = trace_emit

    def embed_query(self, query: str) -> list[float]:
        rows = self.embed_texts([query])
        return rows[0] if rows else self._local_hash_embedding("")

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        clean = [str(x or "") for x in texts]
        if not clean:
            return []
        provider = str(self.config.provider or "local_hash").strip().lower()
        if provider == "local_hash":
            return [self._local_hash_embedding(text) for text in clean]
        try:
            rows = self._embed_remote(clean)
            return [self._normalize(vec) for vec in rows]
        except Exception as ex:  # noqa: BLE001
            if not self.config.fallback_to_local:
                raise
            if self.trace_emit:
                self.trace_emit(
                    "embedding-runtime",
                    "embedding_fallback_local_hash",
                    {"provider": provider, "error": str(ex)},
                )
            return [self._local_hash_embedding(text) for text in clean]

    def _embed_remote(self, texts: list[str]) -> list[list[float]]:
        all_rows: list[list[float]] = []
        batch = max(1, int(self.config.batch_size))
        for start in range(0, len(texts), batch):
            part = texts[start : start + batch]
            all_rows.extend(self._embed_remote_batch(part))
        return all_rows

    def _embed_remote_batch(self, texts: list[str]) -> list[list[float]]:
        base = str(self.config.base_url or "").strip()
        if not base:
            raise RuntimeError("embedding base_url is empty")
        url = base if base.endswith("/embeddings") else f"{base.rstrip('/')}/embeddings"
        payload = {
            "model": str(self.config.model or ""),
            "input": texts,
        }
        headers = {"Content-Type": "application/json"}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        req = urllib.request.Request(
            url=url,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            method="POST",
            headers=headers,
        )
        with urllib.request.urlopen(req, timeout=float(self.config.timeout_seconds)) as resp:  # noqa: S310
            body = json.loads(resp.read().decode("utf-8"))
        data = body.get("data", [])
        if not isinstance(data, list) or len(data) != len(texts):
            raise RuntimeError("invalid embedding response shape")
        rows: list[list[float]] = []
        for item in data:
            emb = item.get("embedding", [])
            if not isinstance(emb, list) or not emb:
                raise RuntimeError("embedding vector missing")
            rows.append([float(x) for x in emb])
        return rows

    def _local_hash_embedding(self, text: str) -> list[float]:
        """本地哈希向量：用于离线自测和远端 embedding 不可用时兜底。"""
        dim = max(64, int(self.config.dim))
        vec = [0.0 for _ in range(dim)]
        tokens = [t for t in re.split(r"[^\w\u4e00-\u9fff]+", str(text).lower()) if t]
        if not tokens:
            tokens = [str(text).strip() or "_empty_"]
        for token in tokens:
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            idx = int.from_bytes(digest[:4], "big") % dim
            sign = 1.0 if (digest[4] % 2 == 0) else -1.0
            weight = 1.0 + (digest[5] / 255.0) * 0.5
            vec[idx] += sign * weight
        return self._normalize(vec)

    @staticmethod
    def _normalize(vec: list[float]) -> list[float]:
        norm = math.sqrt(sum(float(x) * float(x) for x in vec))
        if norm <= 1e-12:
            out = [0.0 for _ in vec]
            if out:
                out[0] = 1.0
            return out
        return [float(x) / norm for x in vec]
