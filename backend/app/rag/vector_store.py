from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

try:  # pragma: no cover - optional runtime dependency
    import numpy as np  # type: ignore
except Exception:  # pragma: no cover
    np = None

try:  # pragma: no cover - optional runtime dependency
    import faiss  # type: ignore
except Exception:  # pragma: no cover
    faiss = None

from backend.app.rag.embedding_provider import EmbeddingProvider


@dataclass(slots=True)
class VectorSummaryRecord:
    record_id: str
    kind: str
    summary_text: str
    parent_text: str
    source_id: str
    source_url: str
    event_time: str
    reliability_score: float
    stock_code: str
    updated_at: str
    metadata: dict[str, Any]


class LocalSummaryVectorStore:
    """本地摘要向量库：优先 FAISS，缺失依赖时回退到 JSON 向量检索。"""

    def __init__(
        self,
        *,
        index_dir: str,
        embedding_provider: EmbeddingProvider,
        dim: int,
        enable_faiss: bool = True,
    ) -> None:
        self.embedding_provider = embedding_provider
        self.dim = max(64, int(dim))
        self.enable_faiss = bool(enable_faiss)
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.meta_path = self.index_dir / "summary_meta.json"
        self.faiss_path = self.index_dir / "summary_index.faiss"
        self.vectors_path = self.index_dir / "summary_vectors.json"
        self._records: list[VectorSummaryRecord] = []
        self._vectors: list[list[float]] = []
        self._index: Any = None
        self._backend: str = "none"
        self._load()

    @property
    def backend(self) -> str:
        return self._backend

    @property
    def record_count(self) -> int:
        return len(self._records)

    def rebuild(self, records: list[VectorSummaryRecord]) -> dict[str, Any]:
        rows = list(records)
        if not rows:
            self._records = []
            self._vectors = []
            self._index = None
            self._backend = "none"
            self._persist_meta()
            if self.faiss_path.exists():
                self.faiss_path.unlink()
            if self.vectors_path.exists():
                self.vectors_path.unlink()
            return {"indexed_count": 0, "backend": self._backend}
        summaries = [r.summary_text for r in rows]
        vectors = self.embedding_provider.embed_texts(summaries)
        normalized = [self._fit_dim(v) for v in vectors]
        self._records = rows
        self._vectors = normalized
        self._build_index_and_persist()
        return {"indexed_count": len(self._records), "backend": self._backend}

    def search(self, query: str, top_k: int = 8) -> list[dict[str, Any]]:
        if not self._records:
            return []
        k = max(1, min(int(top_k), len(self._records)))
        qvec = self._fit_dim(self.embedding_provider.embed_query(query))
        if self._backend == "faiss" and self._index is not None and np is not None and faiss is not None:
            qarr = np.array([qvec], dtype="float32")
            dist, idx = self._index.search(qarr, k)
            out: list[dict[str, Any]] = []
            for rank, row_idx in enumerate(idx[0].tolist()):
                if row_idx < 0 or row_idx >= len(self._records):
                    continue
                out.append(
                    {
                        "rank": rank + 1,
                        "score": float(dist[0][rank]),
                        "record": asdict(self._records[row_idx]),
                    }
                )
            return out
        # JSON fallback：直接内积检索。
        scores: list[tuple[int, float]] = []
        for i, vec in enumerate(self._vectors):
            dot = sum(float(a) * float(b) for a, b in zip(qvec, vec))
            scores.append((i, dot))
        scores.sort(key=lambda x: x[1], reverse=True)
        out: list[dict[str, Any]] = []
        for rank, (i, score) in enumerate(scores[:k], start=1):
            out.append({"rank": rank, "score": float(score), "record": asdict(self._records[i])})
        return out

    def _build_index_and_persist(self) -> None:
        self._persist_meta()
        if self.enable_faiss and faiss is not None and np is not None:
            arr = np.array(self._vectors, dtype="float32")
            index = faiss.IndexFlatIP(self.dim)
            index.add(arr)
            faiss.write_index(index, str(self.faiss_path))
            if self.vectors_path.exists():
                self.vectors_path.unlink()
            self._index = index
            self._backend = "faiss"
            return
        # 缺失 faiss 时，落盘 JSON 向量并走纯 Python 检索。
        self.vectors_path.write_text(json.dumps(self._vectors, ensure_ascii=False), encoding="utf-8")
        if self.faiss_path.exists():
            self.faiss_path.unlink()
        self._index = None
        self._backend = "json_fallback"

    def _load(self) -> None:
        if not self.meta_path.exists():
            self._records = []
            self._vectors = []
            self._index = None
            self._backend = "none"
            return
        meta_payload = json.loads(self.meta_path.read_text(encoding="utf-8"))
        self._records = [
            VectorSummaryRecord(
                record_id=str(x.get("record_id", "")),
                kind=str(x.get("kind", "")),
                summary_text=str(x.get("summary_text", "")),
                parent_text=str(x.get("parent_text", "")),
                source_id=str(x.get("source_id", "")),
                source_url=str(x.get("source_url", "")),
                event_time=str(x.get("event_time", "")),
                reliability_score=float(x.get("reliability_score", 0.6)),
                stock_code=str(x.get("stock_code", "")),
                updated_at=str(x.get("updated_at", "")),
                metadata=dict(x.get("metadata", {})) if isinstance(x.get("metadata", {}), dict) else {},
            )
            for x in meta_payload
        ]
        if self.enable_faiss and faiss is not None and self.faiss_path.exists():
            self._index = faiss.read_index(str(self.faiss_path))
            self._vectors = []
            self._backend = "faiss"
            return
        if self.vectors_path.exists():
            payload = json.loads(self.vectors_path.read_text(encoding="utf-8"))
            self._vectors = [[float(y) for y in row] for row in payload]
            self._index = None
            self._backend = "json_fallback"
            return
        # 元数据存在但索引缺失，按无索引状态处理，等待下次重建。
        self._vectors = []
        self._index = None
        self._backend = "none"

    def _persist_meta(self) -> None:
        self.meta_path.write_text(
            json.dumps([asdict(x) for x in self._records], ensure_ascii=False),
            encoding="utf-8",
        )

    def _fit_dim(self, vec: list[float]) -> list[float]:
        if len(vec) == self.dim:
            return [float(x) for x in vec]
        if len(vec) > self.dim:
            return [float(x) for x in vec[: self.dim]]
        out = [float(x) for x in vec]
        out.extend([0.0 for _ in range(self.dim - len(out))])
        return out
