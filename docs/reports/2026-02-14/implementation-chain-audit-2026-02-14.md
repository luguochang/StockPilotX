# 当前实现链路审计（2026-02-14）

## 1. 你提到的核心疑问，先给结论

1. 上涨概率当前是**规则公式**算出来的，不是大模型推理结果。  
2. 当前后端有 Agent/RAG 工作流，但“模型调用”是**本地模板函数**，没有外部 LLM API 调用。  
3. 数据源是“真实源优先 + mock 回退”，当真实源失败时会用 mock，返回里会带 `source_id` 可区分。  

## 2. “上涨概率”到底怎么算

实现位置：`backend/app/predict/service.py`

1. 先构造因子（技术指标）：
- `momentum_5`、`momentum_20`
- `ma5_bias`、`ma20_bias`
- `volatility_20`
- `drawdown_20`
- `risk_score`

代码定位：
- 因子计算：`backend/app/predict/service.py:129`
- 风险分：`backend/app/predict/service.py:141`

2. 再计算 `alpha_score`（线性组合）：
- `0.35*momentum_5 + 0.35*momentum_20 + 0.15*ma5_bias + 0.15*ma20_bias - 0.2*volatility_20`

代码定位：
- `backend/app/predict/service.py:157`

3. 最后用 Sigmoid 转概率：
- `up_probability = sigmoid(alpha_score * 15)`

代码定位：
- `backend/app/predict/service.py:164`

说明：这是一版可解释的启发式评分，不是训练好的预测模型，也不是 LLM 判断。

## 3. 当前 `/v1/query` 的真实执行链路

实现位置：`backend/app/service.py` + `backend/app/agents/workflow.py`

1. `POST /v1/query` 进入服务层：`backend/app/http_api.py:42`
2. 先尝试刷新数据（行情/公告/历史K线）：`backend/app/service.py:107`
3. 用最新摄取数据动态构造检索语料：`backend/app/service.py:124` + `backend/app/service.py:235`
4. 进入 AgentWorkflow：
- 意图路由：`backend/app/agents/workflow.py:49`
- GraphRAG/AgenticRAG 选择：`backend/app/agents/workflow.py:67`
- 检索：`backend/app/agents/workflow.py:94`
- 分析统计：`backend/app/agents/workflow.py:159`
- Prompt组装：`backend/app/agents/workflow.py:170`
- 模型调用（当前为本地合成函数）：`backend/app/agents/workflow.py:117` + `backend/app/agents/workflow.py:193`

## 4. 现在到底有没有接入“大模型”

结论：**没有外部大模型调用**（OpenAI/其他云模型都没有实际调用代码）。

证据：
1. 代码里明确写了“不依赖外部LLM”：`backend/app/agents/workflow.py:193`
2. 模型输出由 `_synthesize_model_output` 本地函数拼接：`backend/app/agents/workflow.py:193`
3. 全仓检索 `backend/frontend`（排除业务代码后）未发现有效 OpenAI/LangChain/LangGraph 调用链（当前只看到注释/文档层面的提法）。

## 5. Agent / RAG 当前“用了什么”

已用：
1. 多阶段 Agent 工作流（路由、检索、分析、输出、引用）
2. Agentic RAG（HybridRetriever）
3. GraphRAG 分支（可走 Neo4j，不配置时走 InMemory 图）
4. Middleware（Guardrail + Budget）

代码定位：
- Workflow 主体：`backend/app/agents/workflow.py:25`
- Hybrid Retriever：`backend/app/rag/retriever.py:34`
- GraphRAG：`backend/app/rag/graphrag.py:69`
- Middleware：`backend/app/middleware/hooks.py:122`

限制：
1. Retriever 是本地 BM25 + ngram 相似度，不是向量数据库 + embedding 模型。
2. GraphRAG 默认 InMemory 种子图；只有配置 Neo4j 环境变量才切真实图库。  
3. 最终答案生成仍是规则模板函数，不是 LLM 推理。  

## 6. 数据源是实时还是 Mock

实现位置：`backend/app/data/sources.py`

行情 QuoteService 回退顺序：
1. 腾讯实时：`TencentLiveAdapter`
2. 网易实时：`NeteaseLiveAdapter`
3. 新浪实时：`SinaLiveAdapter`
4. 雪球（可选，需 cookie）：`XueqiuLiveAdapter`
5. mock 兜底：`TencentAdapter/NeteaseAdapter/SinaAdapter`

代码定位：
- 默认适配器链：`backend/app/data/sources.py:240`

历史K线：
- 东方财富历史K线接口：`HistoryService.fetch_daily_bars`
- 代码定位：`backend/app/data/sources.py:261`

公告：
- cninfo/sse/szse 页面抓取，全部失败时回退 `_mock_announcements`
- 代码定位：`backend/app/data/sources.py:315`、`backend/app/data/sources.py:467`

## 7. 本地实际运行核验（2026-02-14）

我在当前环境直接跑过服务对象，得到以下结果：

1. `predict_run` 返回 `source_id=tencent`（说明本次命中真实源）  
2. 返回示例 `up_prob_5d=0.464961`（由公式计算）  
3. `query` 返回 `mode=agentic_rag`，并带 `citation_count=7`  

说明：当前系统并非“全是同一条固定结果”，但“解释文本生成”确实还是规则模板，不是 LLM 深度推理。

## 8. 为什么你会感觉“分析说服力不够”

根因是架构层面的：
1. 概率来自启发式公式，不是训练模型或因果框架。
2. 文本结论不是 LLM 推理链，而是模板化合成。
3. 财报结构化、事件抽取、时间序列建模还没进主链。

所以它现在更像“可运行 MVP 骨架 + 数据证据拼装”，还不是“生产级投研智能体”。

