from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Any


@dataclass(slots=True)
class GraphRelation:
    src: str
    dst: str
    rel_type: str
    source_id: str
    source_url: str


class InMemoryGraphStore:
    """本地图存储兜底：无 Neo4j 时仍可运行 GraphRAG。"""

    def __init__(self) -> None:
        self.relations = [
            GraphRelation("SH600000", "银行业", "belong_to", "graph_seed", "neo4j://local"),
            GraphRelation("SH600000", "利率波动", "exposed_to", "graph_seed", "neo4j://local"),
            GraphRelation("SH600000", "信贷需求恢复", "benefit_from", "graph_seed", "neo4j://local"),
        ]

    def find_relations(self, stock_codes: list[str], limit: int = 20) -> list[GraphRelation]:
        matched = [r for r in self.relations if not stock_codes or r.src in stock_codes]
        return matched[:limit]


class Neo4jGraphStore:
    """Neo4j 适配层（可选）。

    若环境无 neo4j 驱动或连接失败，调用方应回退到 InMemoryGraphStore。
    """

    def __init__(self, uri: str, username: str, password: str) -> None:
        self.uri = uri
        self.username = username
        self.password = password
        try:
            from neo4j import GraphDatabase  # type: ignore
        except Exception as ex:  # pragma: no cover
            raise RuntimeError("neo4j driver not installed") from ex
        self._driver = GraphDatabase.driver(uri, auth=(username, password))

    def find_relations(self, stock_codes: list[str], limit: int = 20) -> list[GraphRelation]:
        cypher = """
        MATCH (c:Company)-[r]->(x)
        WHERE size($codes)=0 OR c.code IN $codes
        RETURN c.code AS src, x.name AS dst, type(r) AS rel_type
        LIMIT $limit
        """
        rows: list[GraphRelation] = []
        with self._driver.session() as sess:
            recs = sess.run(cypher, {"codes": stock_codes, "limit": limit})
            for rec in recs:
                rows.append(
                    GraphRelation(
                        src=rec["src"],
                        dst=rec["dst"],
                        rel_type=rec["rel_type"],
                        source_id="neo4j",
                        source_url=self.uri,
                    )
                )
        return rows


class GraphRAGService:
    """GraphRAG 服务：Neo4j 优先，失败回退 InMemory 图。"""

    def __init__(self, store: Any | None = None) -> None:
        self.store = store or self._build_default_store()

    def _build_default_store(self) -> Any:
        uri = os.getenv("NEO4J_URI", "").strip()
        user = os.getenv("NEO4J_USER", "").strip()
        pwd = os.getenv("NEO4J_PASSWORD", "").strip()
        if uri and user and pwd:
            try:
                return Neo4jGraphStore(uri=uri, username=user, password=pwd)
            except Exception:
                pass
        return InMemoryGraphStore()

    def query_subgraph(self, question: str, stock_codes: list[str]) -> dict:
        relations = self.store.find_relations(stock_codes, limit=20)
        if not relations:
            return {
                "mode": "graph_rag",
                "summary": "未检索到关系图谱结果，建议扩大时间窗或补充实体。",
                "relations": [],
                "citations": [],
            }

        relation_payload = [
            {"from": r.src, "to": r.dst, "type": r.rel_type, "source_id": r.source_id, "source_url": r.source_url}
            for r in relations
        ]
        summary = (
            f"图谱检索命中 {len(relations)} 条关系。"
            f"问题：{question}。"
            f"核心关系：{relations[0].src}->{relations[0].dst}({relations[0].rel_type})。"
        )
        citations = [
            {
                "source_id": r.source_id,
                "source_url": r.source_url,
                "event_time": None,
                "reliability_score": 0.9 if r.source_id == "neo4j" else 0.8,
                "excerpt": f"{r.src}->{r.dst} ({r.rel_type})",
            }
            for r in relations[:5]
        ]
        return {
            "mode": "graph_rag",
            "summary": summary,
            "relations": relation_payload,
            "citations": citations,
        }

