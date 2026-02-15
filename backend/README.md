# A-Share Agent System Backend (MVP)

This backend implements a runnable MVP based on `docs/specs/a-share-agent-system-executable-spec.md`.

## Implemented
- Multi-agent workflow: Router -> Data -> RAG -> Analysis -> Report -> Critic
- Middleware engineering:
  - `before_agent`, `before_model`, `after_model`, `after_agent`
  - `wrap_model_call`, `wrap_tool_call`
- Long-term memory store (SQLite)
- Prompt registry/versioning store (SQLite)
- Free data source fallback strategy (Tencent -> Netease -> Sina)
- Agentic RAG default + GraphRAG trigger
- Eval runner with basic metrics
- API contract mapped to service methods

## Run quick demo
```powershell
python -m backend.main
```

## Run tests
```powershell
python -m unittest discover -s tests -p "test_*.py" -v
```

## Note
Network-restricted environment prevented installing FastAPI/LangChain packages.
So this MVP keeps architecture-compatible modules and pure-Python execution.
When dependencies are available, `backend/app/http_api.py` can be switched to actual FastAPI app wiring.

