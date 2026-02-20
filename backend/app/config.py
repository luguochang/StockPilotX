from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path


_BACKEND_DIR = Path(__file__).resolve().parents[1]
_DATA_DIR = _BACKEND_DIR / "data"
_CONFIG_DIR = _BACKEND_DIR / "config"


def _to_bool(value: str | None, default: bool) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(slots=True)
class Settings:
    """系统配置。"""

    # 应用基础配置
    app_name: str = "a-share-agent-system"
    env: str = "dev"

    # SQLite 路径：用于本地 MVP 的记忆与 Prompt 资产化存储
    memory_db_path: str = str(_DATA_DIR / "memory.db")
    prompt_db_path: str = str(_DATA_DIR / "prompt.db")
    web_db_path: str = str(_DATA_DIR / "web.db")
    jwt_secret: str = "stockpilotx-dev-secret"
    jwt_expire_seconds: int = 60 * 60 * 8

    # 预算控制：限制模型调用、工具调用和上下文大小
    max_tool_calls: int = 12
    max_model_calls: int = 8
    max_context_chars: int = 12000

    # 触发 GraphRAG 的关键词（后续可演进为规则引擎）
    graph_trigger_keywords: tuple[str, ...] = ("关系", "演化", "关联", "产业链", "股权")

    # 外部大模型网关配置（支持多供应商回退）
    llm_external_enabled: bool = False
    llm_config_path: str = str(_CONFIG_DIR / "llm_providers.local.json")
    llm_request_timeout_seconds: float = 20.0
    llm_retry_count: int = 1
    llm_retry_backoff_seconds: float = 0.8
    llm_fallback_to_local: bool = True
    use_langgraph_runtime: bool = True
    deep_archive_max_events_default: int = 1200
    deep_archive_max_events_dev: int = 1200
    deep_archive_max_events_staging: int = 2400
    deep_archive_max_events_prod: int = 3600
    deep_archive_max_events_hard_cap: int = 5000
    deep_archive_tenant_policy_json: str = "{}"
    deep_archive_export_task_max_attempts: int = 2
    deep_archive_export_retry_backoff_seconds: float = 0.35
    # DeepThink runtime guard: cap one round wall-clock latency and emit warning before timeout.
    deep_round_timeout_seconds: float = 45.0
    deep_round_stage_soft_timeout_seconds: float = 28.0
    # Deep retrieval fanout timeout per subtask to keep deep mode responsive.
    deep_subtask_timeout_seconds: float = 2.5
    # Corrective RAG: retry retrieval with rewritten query when first-pass relevance is weak.
    corrective_rag_enabled: bool = True
    corrective_rag_rewrite_threshold: float = 0.42
    react_deep_enabled: bool = False
    react_max_iterations: int = 2
    rag_vector_enabled: bool = True
    rag_vector_index_dir: str = str(_DATA_DIR / "vector")
    rag_vector_top_k: int = 8
    embedding_provider: str = "local_hash"
    embedding_model: str = ""
    embedding_base_url: str = ""
    embedding_api_key: str = ""
    embedding_dim: int = 256
    embedding_timeout_seconds: float = 12.0
    embedding_batch_size: int = 32
    embedding_fallback_to_local: bool = True
    # Datasource runtime controls used by backend.app.datasources factory.
    datasource_request_timeout_seconds: float = 2.0
    datasource_retry_count: int = 2
    datasource_retry_backoff_seconds: float = 0.3
    datasource_proxy_url: str = ""
    datasource_xueqiu_cookie: str = ""
    datasource_tushare_token: str = ""
    datasource_tradingview_proxy_url: str = ""
    # A-share regime controls: model "bull short, bear long" market behavior using short/mid horizon signals.
    a_share_regime_enabled: bool = True
    a_share_regime_vol_threshold: float = 0.025
    a_share_regime_conf_discount_bear: float = 0.82
    a_share_regime_conf_discount_range: float = 0.88
    a_share_regime_conf_discount_bull_high_vol: float = 0.90

    @classmethod
    def from_env(cls) -> "Settings":
        """从环境变量构建配置对象。"""
        default_memory = str(_DATA_DIR / "memory.db")
        default_prompt = str(_DATA_DIR / "prompt.db")
        default_llm_config = str(_CONFIG_DIR / "llm_providers.local.json")
        graph_keywords_env = os.getenv("GRAPH_TRIGGER_KEYWORDS")
        graph_keywords = (
            tuple(x.strip() for x in graph_keywords_env.split(",") if x.strip())
            if graph_keywords_env
            else ("关系", "演化", "关联", "产业链", "股权")
        )
        return cls(
            env=os.getenv("APP_ENV", "dev"),
            memory_db_path=os.getenv("MEMORY_DB_PATH", default_memory),
            prompt_db_path=os.getenv("PROMPT_DB_PATH", default_prompt),
            web_db_path=os.getenv("WEB_DB_PATH", str(_DATA_DIR / "web.db")),
            jwt_secret=os.getenv("JWT_SECRET", "stockpilotx-dev-secret"),
            jwt_expire_seconds=int(os.getenv("JWT_EXPIRE_SECONDS", str(60 * 60 * 8))),
            graph_trigger_keywords=graph_keywords,
            llm_external_enabled=_to_bool(os.getenv("LLM_EXTERNAL_ENABLED"), False),
            llm_config_path=os.getenv("LLM_CONFIG_PATH", default_llm_config),
            llm_request_timeout_seconds=float(os.getenv("LLM_REQUEST_TIMEOUT_SECONDS", "20")),
            llm_retry_count=max(1, int(os.getenv("LLM_RETRY_COUNT", "1"))),
            llm_retry_backoff_seconds=max(0.0, float(os.getenv("LLM_RETRY_BACKOFF_SECONDS", "0.8"))),
            llm_fallback_to_local=_to_bool(os.getenv("LLM_FALLBACK_TO_LOCAL"), True),
            use_langgraph_runtime=_to_bool(os.getenv("USE_LANGGRAPH_RUNTIME"), True),
            deep_archive_max_events_default=max(1, int(os.getenv("DEEP_ARCHIVE_MAX_EVENTS_DEFAULT", "1200"))),
            deep_archive_max_events_dev=max(1, int(os.getenv("DEEP_ARCHIVE_MAX_EVENTS_DEV", "1200"))),
            deep_archive_max_events_staging=max(1, int(os.getenv("DEEP_ARCHIVE_MAX_EVENTS_STAGING", "2400"))),
            deep_archive_max_events_prod=max(1, int(os.getenv("DEEP_ARCHIVE_MAX_EVENTS_PROD", "3600"))),
            deep_archive_max_events_hard_cap=max(100, int(os.getenv("DEEP_ARCHIVE_MAX_EVENTS_HARD_CAP", "5000"))),
            deep_archive_tenant_policy_json=os.getenv("DEEP_ARCHIVE_TENANT_POLICY_JSON", "{}"),
            deep_archive_export_task_max_attempts=max(
                1, int(os.getenv("DEEP_ARCHIVE_EXPORT_TASK_MAX_ATTEMPTS", "2"))
            ),
            deep_archive_export_retry_backoff_seconds=max(
                0.0, float(os.getenv("DEEP_ARCHIVE_EXPORT_RETRY_BACKOFF_SECONDS", "0.35"))
            ),
            deep_round_timeout_seconds=max(0.1, float(os.getenv("DEEP_ROUND_TIMEOUT_SECONDS", "45"))),
            deep_round_stage_soft_timeout_seconds=max(
                0.05, float(os.getenv("DEEP_ROUND_STAGE_SOFT_TIMEOUT_SECONDS", "28"))
            ),
            deep_subtask_timeout_seconds=max(0.1, float(os.getenv("DEEP_SUBTASK_TIMEOUT_SECONDS", "2.5"))),
            corrective_rag_enabled=_to_bool(os.getenv("CORRECTIVE_RAG_ENABLED"), True),
            corrective_rag_rewrite_threshold=max(
                0.05, min(1.0, float(os.getenv("CORRECTIVE_RAG_REWRITE_THRESHOLD", "0.42")))
            ),
            react_deep_enabled=_to_bool(os.getenv("REACT_DEEP_ENABLED"), False),
            react_max_iterations=max(1, min(4, int(os.getenv("REACT_MAX_ITERATIONS", "2")))),
            rag_vector_enabled=_to_bool(os.getenv("RAG_VECTOR_ENABLED"), True),
            rag_vector_index_dir=os.getenv("RAG_VECTOR_INDEX_DIR", str(_DATA_DIR / "vector")),
            rag_vector_top_k=max(1, int(os.getenv("RAG_VECTOR_TOP_K", "8"))),
            embedding_provider=os.getenv("EMBEDDING_PROVIDER", "local_hash").strip() or "local_hash",
            embedding_model=os.getenv("EMBEDDING_MODEL", "").strip(),
            embedding_base_url=os.getenv("EMBEDDING_BASE_URL", "").strip(),
            embedding_api_key=os.getenv("EMBEDDING_API_KEY", "").strip(),
            embedding_dim=max(64, int(os.getenv("EMBEDDING_DIM", "256"))),
            embedding_timeout_seconds=max(1.0, float(os.getenv("EMBEDDING_TIMEOUT_SECONDS", "12"))),
            embedding_batch_size=max(1, int(os.getenv("EMBEDDING_BATCH_SIZE", "32"))),
            embedding_fallback_to_local=_to_bool(os.getenv("EMBEDDING_FALLBACK_TO_LOCAL"), True),
            datasource_request_timeout_seconds=max(
                0.1, float(os.getenv("DATASOURCE_REQUEST_TIMEOUT_SECONDS", "2.0"))
            ),
            datasource_retry_count=max(0, int(os.getenv("DATASOURCE_RETRY_COUNT", "2"))),
            datasource_retry_backoff_seconds=max(
                0.0, float(os.getenv("DATASOURCE_RETRY_BACKOFF_SECONDS", "0.3"))
            ),
            datasource_proxy_url=os.getenv("DATASOURCE_PROXY_URL", "").strip(),
            datasource_xueqiu_cookie=os.getenv("DATASOURCE_XUEQIU_COOKIE", "").strip(),
            datasource_tushare_token=os.getenv("DATASOURCE_TUSHARE_TOKEN", "").strip(),
            datasource_tradingview_proxy_url=os.getenv("DATASOURCE_TRADINGVIEW_PROXY_URL", "").strip(),
            a_share_regime_enabled=_to_bool(os.getenv("A_SHARE_REGIME_ENABLED"), True),
            a_share_regime_vol_threshold=max(0.005, float(os.getenv("A_SHARE_REGIME_VOL_THRESHOLD", "0.025"))),
            a_share_regime_conf_discount_bear=max(
                0.5, min(1.0, float(os.getenv("A_SHARE_REGIME_CONF_DISCOUNT_BEAR", "0.82")))
            ),
            a_share_regime_conf_discount_range=max(
                0.5, min(1.0, float(os.getenv("A_SHARE_REGIME_CONF_DISCOUNT_RANGE", "0.88")))
            ),
            a_share_regime_conf_discount_bull_high_vol=max(
                0.5,
                min(1.0, float(os.getenv("A_SHARE_REGIME_CONF_DISCOUNT_BULL_HIGH_VOL", "0.90"))),
            ),
        )

    def load_llm_provider_configs(self) -> list[dict]:
        """读取多提供商LLM配置（JSON数组）。"""
        path = Path(self.llm_config_path)
        if not path.exists():
            return []
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise ValueError("llm provider config must be a JSON array")
        return [x for x in payload if isinstance(x, dict)]
