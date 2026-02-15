# 中国A股智能分析系统：需求背景与技术方案（V1）

## 1. 项目背景与目标
- 背景主题：构建一个以中国 A 股为核心的智能分析系统。
- 核心目标：
  - 对指定股票输出“事实依据充分、可追溯”的分析报告。
  - 支持公司历史信息、财报、公告、行业资料、外部研究文档的统一检索与推理。
  - 通过定时任务持续积累数据资产，形成长期可复用知识底座。
- 技术导向：优先采用可落地的前沿 Agent + RAG 架构，而不是只做单点问答。

## 2. 业务能力边界（先定规则，再做能力）
- 能做：
  - 事实检索与证据引用（公告、财报、历史事件、行业资料）。
  - 结构化分析（财务指标变化、风险点、事件时间线、同业对比）。
  - 观点归纳（来自上传文档和公开信息），并明确“观点来源”。
- 不直接做（默认）：
  - 明确个股买卖建议、保证收益、确定性预测。
  - 面向个人的个性化投资建议（若要做，需要额外合规评估与牌照路径）。
- 输出准则：
  - 每个关键结论必须可追溯到证据（source id + 时间 + 链接/文档片段）。
  - 明确区分“事实”“推断”“观点”三类内容。

## 3. 用户场景（MVP）
- 场景 1：输入股票代码，生成“公司基本面 + 最近公告事件 + 风险摘要”。
- 场景 2：上传财报 PDF/行业报告，提问“营收质量是否改善、原因是什么”。
- 场景 3：关注股票后自动跟踪，按日/周生成变化简报。
- 场景 4：多轮追问（例如“再和同行比较一下毛利率与现金流质量”）。

## 4. 技术架构总览（你指定技术点全部纳入）
## 4.1 应用层
- Web/API 网关
- 鉴权与租户隔离
- 报告导出（Markdown/PDF）

## 4.2 Agent 编排层（LangChain 1.x + LangGraph）
- `langchain`（1.x）：
  - 快速构建 tool-calling agent
  - middleware 注入（风控、提示词策略、上下文裁剪、输出规范）
- `langgraph`（1.x）：
  - 有状态图编排（stateful workflow）
  - durable execution（长任务可恢复）
  - human-in-the-loop（高风险请求审批/复核）

## 4.3 多 Agent 协作层
- Router Agent：意图识别（事实查询 / 深度分析 / 文档问答 / 监控任务）
- Data Agent：拉取行情、公告、财务和元数据
- RAG Agent：检索与证据组装
- Analysis Agent：财务分析、事件归因、风险项抽取
- Report Agent：结构化写作 + 引用对齐 + 免责声明注入
- Critic Agent：一致性与证据充分性复核

## 4.4 数据与记忆层
- OLTP：PostgreSQL（任务、用户、配置、结构化结果）
- 时序/分析：ClickHouse 或 TimescaleDB（行情与指标）
- 对象存储：MinIO/S3（原始文档、解析结果）
- 向量库：Qdrant/Milvus（语义检索）
- 图数据库：Neo4j（实体关系 + 事件关系，支撑 GraphRAG）
- 长期记忆（Long-term Memory）：
  - 用户偏好记忆（风险偏好、关注板块）
  - 任务记忆（过去分析结论、反馈修正）
  - 组织级知识记忆（行业规则、内部研究模板）

## 4.5 可观测与评测层（LangSmith）
- Tracing：全链路跟踪（prompt/tool/retrieval/model cost）
- Offline Eval：基于数据集的离线评测
- Online Eval：线上抽样质量监控与告警
- A/B 实验：比较不同检索策略、不同 agent 流程

## 5. RAG 方案设计（Baseline -> Hybrid -> GraphRAG -> Agentic RAG）
## 5.1 Baseline RAG（第一阶段）
- 文档切分（语义切分 + 表格单元保留）
- 向量召回 top-k
- 生成时强制引用证据片段

## 5.2 Hybrid RAG（第二阶段）
- BM25 + 向量混合召回
- reranker 重排（降低“看起来相关但不关键”的噪声）
- 元数据过滤（股票代码、时间窗、文档类型）

## 5.3 GraphRAG（第三阶段）
- 从公告/财报中抽取实体与关系（公司、子公司、供应商、事件、指标）
- 建立实体图谱 + 社区摘要
- 面向“全局问题”增强（例如“过去三年主要风险主题如何演变”）

## 5.4 Agentic RAG（第四阶段）
- 先规划再检索：agent 先分解子问题，再多轮检索补证
- 检索失败自恢复：自动换检索策略或扩展时间窗口
- 证据冲突处理：多来源交叉验证并输出“冲突说明”

## 6. 文档处理工程（PDF/DOC/网页）
- 解析流水线：
  - 文档接入 -> OCR/结构化解析 -> 表格抽取 -> 去噪清洗 -> chunk 化 -> 嵌入入库
- 版本管理：
  - 同一文档多版本追踪（更新时间、来源、哈希）
  - 可回滚与可审计
- 质量门禁：
  - 低质量解析（表格错位、OCR 置信度低）进入人工复核队列

## 7. 状态管理与 Middleware 工程化
## 7.1 状态管理（LangGraph state）
- Thread 级短期状态：对话、当前任务上下文、临时检索结果
- 跨线程长期状态：用户偏好、关注列表、历史报告摘要
- 状态裁剪策略：控制 token 成本，保留关键事实链

## 7.2 Middleware（LangChain 1.x）
- 输入层中间件：意图识别、敏感请求识别、参数校验
- 模型前中间件：上下文压缩、提示词拼装、风险规则注入
- 模型后中间件：格式标准化、引用检查、术语解释补全
- 工具层中间件：超时、重试、熔断、缓存、审计日志

## 8. 定时任务与数据抓取设计
- 调度器：Airflow/Temporal/Celery（建议优先 Temporal 或 Airflow）
- 任务类别：
  - 日级：行情、公告、新闻/观点增量
  - 周级：行业专题、指标回算、知识图谱增量更新
  - 触发式：用户关注股票时提升抓取频率
- 抓取原则：
  - 优先官方/权威披露源（交易所、法定披露平台）
  - 非官方源用于“补充线索”，不得单源直接作为结论

## 9. 数据源分层建议（A股）
## 9.1 一级可信源（事实主源）
- 巨潮资讯（法定信息披露平台）
- 上交所/深交所公告与规则页面

## 9.2 二级源（行情与研究增强）
- 合规数据服务商（商业授权）
- 开源接口（如 AKShare / Tushare）用于研究验证或内部原型

## 9.3 数据治理
- source reliability score（来源可信度评分）
- 时间戳与生效期管理（避免过期结论）
- 去重与冲突检测（同事件多来源归并）

## 10. 评测体系（必须先建）
## 10.1 离线评测
- 检索：Recall@k、MRR、nDCG
- 生成：事实一致性、引用命中率、幻觉率
- 业务：报告可读性、行动建议可执行性、错误严重度分级

## 10.2 在线评测
- 关键指标：首 token 延迟、P95 响应、单请求成本、任务成功率
- 用户反馈：有用/无用、证据充分/不充分、是否需人工复核
- 回归机制：每次改动必须跑固定测试集并留档

## 11. 多 Agent 架构取舍（同类方案对比）
- 单 Agent：
  - 优点：开发快、链路短、成本低
  - 缺点：复杂任务容易失控、可观测性较弱
- 多 Agent：
  - 优点：职责隔离清晰、便于扩展和调优、质量上限更高
  - 缺点：编排复杂、延迟和成本上升
- 结论：
  - MVP 用“1 主 2 辅”（Router + RAG + Report）起步
  - 在“复杂分析”流量占比上升后，再扩展到 5 角色协作

## 12. 合规与风险控制（A股场景必须前置）
- 显式免责声明：系统输出仅供研究参考，不构成投资建议。
- 高风险内容策略：
  - 避免确定性收益表达
  - 对“买点/卖点”输出改为“情景假设 + 风险揭示”
- 审计要求：
  - 完整保存输入、证据、推理链、输出版本
  - 支持监管或内部审计追溯

## 13. 实施路线图（8周可落地）
- 第 1-2 周：数据接入 + 文档处理 + Baseline RAG
- 第 3-4 周：LangGraph 工作流 + Middleware + LangSmith tracing
- 第 5-6 周：多 Agent（Router/RAG/Report）+ 离线评测集
- 第 7 周：GraphRAG 或 Agentic RAG 试点（选其一先做深）
- 第 8 周：线上试运行 + 质量看板 + 回归流程固化

## 14. 需要你确认的关键决策（下一步）
1. MVP 首批覆盖多少只股票（例如 50 / 300 / 1000）？
2. 是否允许使用商业数据源（影响数据质量与合法性）？
3. 报告输出默认偏“事实报告”还是“策略研究报告”？
4. 你希望先做 GraphRAG 还是 Agentic RAG（建议先 Agentic RAG）？
5. 是否从 Day 1 做多租户权限隔离？

## 15. 当前推荐技术选型（初版）
- `langchain` 1.x + `langgraph` 1.x
- `langsmith`（追踪 + 评测）
- `postgresql` + `clickhouse` + `qdrant` + `neo4j` + `minio`
- `temporal`（或 `airflow`）用于定时/异步任务编排
- `fastapi` 作为统一服务层（API + webhook + job control）

## 16. 外部依据（用于后续技术决策追溯）
- LangChain / LangGraph 1.0 里程碑（2025-10-22）：
  - https://blog.langchain.com/langchain-langgraph-1dot0/
- LangChain v1 迁移指南（`create_agent`、middleware 等）：
  - https://docs.langchain.com/oss/python/migrate/langchain-v1
- LangGraph 概览（durable execution、memory、HITL）：
  - https://docs.langchain.com/oss/python/langgraph/overview
- LangSmith 概览与评测（offline/online eval）：
  - https://docs.langchain.com/langsmith/home
  - https://docs.langchain.com/langsmith/evaluation
- GraphRAG（Microsoft Research + 开源仓库）：
  - https://www.microsoft.com/en-us/research/project/graphrag/
  - https://github.com/microsoft/graphrag
- Agentic RAG（综述）：
  - https://arxiv.org/abs/2501.09136
  - https://arxiv.org/abs/2507.09477
- A2A 协议（多 Agent 互操作）：
  - https://github.com/a2aproject/A2A
- MCP 规范（工具与上下文标准协议）：
  - https://modelcontextprotocol.io/docs/getting-started/intro
  - https://modelcontextprotocol.io/specification/2025-06-18
- A股披露与公告主源（事实层）：
  - https://www.cninfo.com.cn/
  - https://www.sse.com.cn/disclosure/listedinfo/announcement/
  - https://www.szse.cn/disclosure/listed/index.html
- 开源数据接入参考（研究/原型阶段）：
  - https://akshare.akfamily.xyz/
  - https://tushare.pro/

## 17. 外部参考仓库接入方案（不作为本仓库内容）
- 外部参考仓库：
  - `go-stock`（`https://github.com/ArvinLovegood/go-stock`）
  - `stock-api`（`https://github.com/zhangxiangliang/stock-api`）
- 说明：
  - `StockPilotX` 仓库不纳入上述第三方仓库代码。
  - 仅按需拷贝所需能力实现到本项目模块，并保留来源备注。

### 17.1 go-stock 可复用能力映射
- 可复用的数据抓取能力：
  - 浏览器抓取与页面等待策略：`go-stock/backend/data/crawler_api.go`
  - Tushare 数据拉取与代码转换：`go-stock/backend/data/tushare_data_api.go`
  - 市场新闻、公告、资金、研报等数据接口：`go-stock/backend/data/*.go`
- 可复用的 Agent 工具化思路：
  - 工具注册与调用入口：`go-stock/backend/agent/agent.go`
  - 财报工具示例：`go-stock/backend/agent/tools/financial_reports_tool.go`
  - 其它工具：`go-stock/backend/agent/tools/*`
- 接入建议：
  - 不直接复用其 UI/桌面端结构（Wails），只复用“数据接口模式 + 工具抽象”。
  - 将工具协议统一到 LangChain 1.x tool schema（参数校验、超时、重试、审计）。

### 17.2 stock-api 可复用能力映射
- 可复用点：
  - 多数据源统一抽象：`stock-api/src/stocks/base/index.ts`
  - 数据源适配器实现（腾讯/网易/新浪/雪球）：`stock-api/src/stocks/*/index.ts`
  - 股票代码转换与字段标准化 transform：`stock-api/src/stocks/*/transforms/*.ts`
- 接入建议：
  - 将其作为“实时行情快速补充层”，放在二级数据源。
  - 对每个 source 打“可靠性等级 + 可用性SLA”，避免单点源故障影响结论。

### 17.3 你这个项目的落地集成方式
- 统一 `ingestion-service`：
  - Adapter-1：官方披露源（主源，优先级最高）
  - Adapter-2：商业/开源行情源（补充源）
  - Adapter-3：网页爬取源（线索源，需二次验证）
- 统一输出到 canonical schema（必须做）
  - `instrument`, `event`, `financial_statement`, `news`, `research_note`, `source_meta`
- 写入策略：
  - 原始数据入对象存储
  - 结构化数据入 OLTP/OLAP
  - 文本与向量入 RAG 索引
  - 实体关系入图数据库（GraphRAG）

## 18. 财经分析能力补充（基于检索）
### 18.1 建议纳入的分析主轴
- 公司经营与现金流主轴：
  - 不是只看利润，要看现金流质量与可持续性。
- 财务结构主轴：
  - 偿债能力、资本结构、流动性风险、再融资压力。
- 估值与市场定价主轴：
  - 估值区间、同业对比、历史分位、情景分析。
- 风险与不确定性主轴：
  - 业务/政策/供应链/治理/会计估计风险。
- 外部环境主轴：
  - 宏观、监管、行业周期、利率与流动性。

### 18.2 报告结构建议（与国际披露框架对齐）
- 建议报告固定 6 区块：
  - 商业模式
  - 发展策略
  - 资源与关系
  - 风险
  - 外部环境
  - 财务业绩与财务状况
- 这是为了让输出结构稳定、可评测、可横向对比，而不是每次自由发挥。

### 18.3 A股场景下必须增加的工程约束
- 证据优先级机制：
  - 法定披露（巨潮/交易所） > 公司官网 > 主流财经媒体 > 社区观点。
- 结论可信度分级：
  - A（多源一致且含法定披露）
  - B（双源一致但缺法定披露）
  - C（单源线索，需人工复核）
- 合规输出规则：
  - 默认输出“研究参考”而非“投资建议”。
  - 买卖点相关问题改写为“情景假设 + 风险揭示 + 证据出处”。

### 18.4 评测指标补充（财经专用）
- 事实正确性：数值、日期、主体、同比环比口径正确率。
- 证据充分性：关键结论的引用覆盖率。
- 冲突处理质量：多源冲突时是否给出冲突说明与置信度。
- 时效性：新公告出现后系统纳入时延。
- 可执行性：报告是否给出可复核步骤（而非空泛观点）。

### 18.5 检索来源（本轮补充）
- LangChain / LangGraph 1.0 发布：
  - https://blog.langchain.com/langchain-langgraph-1dot0/
  - https://changelog.langchain.com/announcements/langchain-1-0-now-generally-available
  - https://changelog.langchain.com/announcements/langgraph-1-0-is-now-generally-available
- LangChain v1 迁移（create_agent / middleware）：
  - https://docs.langchain.com/oss/python/migrate/langchain-v1
- IFRS 管理层讨论框架（用于报告结构）：
  - https://www.ifrs.org/issued-standards/list-of-standards/management-commentary-practice-statement/
  - https://www.ifrs.org/content/dam/ifrs/project/management-commentary/practice-statement/projectsummary-mc-ps1.pdf
- SEC 对 MD&A 重点（用于“分析必须解释趋势与不确定性”）：
  - https://www.sec.gov/news/press/2003-179.htm
- A股信息披露与监管指引（用于证据优先级与合规边界）：
  - https://www.cninfo.com.cn/
  - https://www.sse.com.cn/lawandrules/sselawsrules/stocks/staripo/c/c_20250314_10775831.shtml
  - https://www.sse.com.cn/lawandrules/sselawsrules2025/stocks/mainipo/c/c_20250515_10779032.shtml
  - https://www.szse.cn/disclosure/listed/index.html
- 投资者适当性管理（用于产品提示与能力边界）：
  - https://www.ine.com.cn/regulation/productrules/202503/t20250313_824789.html

## 19. V2 实施清单（可直接开工）
## 19.1 里程碑与交付件
1. M1（第1周）：数据底座打通
- 交付：`ingestion-service`、`canonical schema`、首批 50 只股票基础数据入库。
2. M2（第2-3周）：RAG Baseline 可用
- 交付：文档上传/解析/索引/问答全链路，支持证据引用。
3. M3（第4周）：LangGraph 工作流 + 中间件
- 交付：Router/RAG/Report 三节点图，超时重试、审计日志、输出规范中间件。
4. M4（第5-6周）：评测体系上线
- 交付：离线评测脚本 + 回归集 + LangSmith tracing 与质量看板。
5. M5（第7-8周）：多 Agent + 高级RAG试点
- 交付：Agentic RAG 首版（可选并行做 GraphRAG PoC）。

## 19.2 工程任务清单（按模块）
1. 数据接入模块
- 接入主源：巨潮/上交所/深交所公告接口与页面抓取。
- 接入补充源：Tushare/AKShare/stock-api 适配层。
- 实现统一规范化：代码、日期、币种、口径、时间戳、来源ID。
- 增加去重和冲突合并：同事件多源归并、冲突打标。
2. 文档处理模块
- 支持 PDF/DOCX/HTML 上传与解析。
- 表格识别与结构化抽取（财报表格优先）。
- 分块策略：按章节+语义双通道切分。
- 建立文档版本与溯源（hash、来源、抓取时间、版本号）。
3. RAG 模块
- Baseline：向量检索 + 引用强制输出。
- Hybrid：BM25 + 向量 + reranker。
- Agentic RAG：规划-检索-验证-汇总流程。
- GraphRAG（可选PoC）：实体关系抽取与子图检索。
4. Agent 编排模块
- LangGraph state 定义：`query_state`、`retrieval_state`、`analysis_state`。
- 节点实现：Router、Data、RAG、Analysis、Report、Critic。
- 中间件：输入校验、风险策略、输出标准化、失败重试。
- HITL：高风险输出进入人工复核队列。
5. 评测与观测模块
- LangSmith 接入 tracing、dataset、experiment。
- 离线评测：检索、事实、引用、时效、成本五类指标。
- 在线评测：用户反馈、异常告警、回归门禁。
- A/B 实验：检索参数与模型路由对比。

## 19.3 数据表与存储清单（首版）
1. PostgreSQL
- `instrument`（股票基础信息）
- `watchlist`（用户关注列表）
- `event_fact`（公告/新闻/事件事实）
- `report_run`（报告任务记录）
- `agent_trace_ref`（内部任务与LangSmith trace映射）
2. 对象存储（MinIO/S3）
- `raw_docs/`（原始文档）
- `parsed_docs/`（解析结果JSON）
- `report_exports/`（导出报告）
3. 向量库（Qdrant/Milvus）
- `kb_chunks`（通用知识）
- `stock_chunks`（个股相关）
- `report_chunks`（历史分析报告）
4. 图数据库（Neo4j，可后置）
- `Company`、`Person`、`Event`、`Metric` 节点
- `RELATED_TO`、`IMPACTS`、`DISCLOSED_IN` 关系

## 19.4 定时任务 DAG（首版）
1. `daily_market_ingest`（交易日 16:30）
- 拉取日行情、成交、资金流向、公告增量。
2. `nightly_doc_pipeline`（每日 22:00）
- 文档解析、分块、向量入库、关系抽取。
3. `watchlist_digest`（每日 07:30）
- 对关注股票生成晨报（变化点+风险点+证据）。
4. `weekly_reindex`（每周日）
- 重建部分索引、质量抽检、低质量文档重处理。
5. `eval_regression_gate`（每次发布前）
- 回归集全量跑分，不达阈值禁止发布。

## 19.5 报告输出模板（首版）
1. 基本信息：公司、行业、时间窗、数据截止时间。
2. 核心结论：3-5条，标注事实/推断/观点类别。
3. 财务与经营：增长、盈利、现金流、偿债、效率。
4. 关键事件：公告、监管、产业链、舆情变化。
5. 风险清单：触发条件、影响路径、监控指标。
6. 情景分析：乐观/基准/保守三情景。
7. 引用证据：逐条列 source id + 时间 + 链接/片段。
8. 免责声明：研究参考，不构成投资建议。

## 19.6 验收标准（Definition of Done）
1. 数据层
- 主源抓取成功率 >= 99%（按任务批次统计）。
- 结构化入库延迟 <= 30 分钟（交易日增量）。
2. RAG层
- 离线事实正确率 >= 85%（首版门槛，可持续提升）。
- 关键结论引用覆盖率 >= 95%。
3. Agent层
- 多步骤任务成功率 >= 90%（含重试）。
- 高风险请求 100% 命中风控策略。
4. 体验与成本
- 报告生成 P95 < 25s（MVP目标）。
- 单次报告成本可观测且可追溯到节点。

## 19.7 第一批开工任务（本周）
1. 建仓与服务骨架
- 新建 `services/ingestion`、`services/agent-orchestrator`、`services/reporting`。
2. 数据模型落库
- 建立 `instrument/event_fact/report_run/watchlist` 四张核心表。
3. 接入一个主源 + 一个补充源
- 主源先接 `cninfo` 公告。
- 补充源先接 `tushare` 日线与基础财务字段。
4. 打通最小闭环
- 输入股票代码 -> 检索 -> 生成带引用报告 -> 保存历史。
5. 接入 LangSmith
- 每次任务自动打 trace，并记录到 `agent_trace_ref`。

## 20. V3 技术架构定稿（前后端 + Agent + RAG）
## 20.1 前后端技术栈（明确落地）
1. 前端（Web）
- 语言与框架：TypeScript + Next.js 15（App Router）+ React 19
- UI：Ant Design + ECharts（金融图表）
- 状态管理：
  - 服务端状态：TanStack Query（行情、报告、任务状态）
  - 客户端状态：Zustand（筛选条件、工作台上下文、UI状态）
- 实时通信：SSE（报告生成流）+ WebSocket（任务进度/告警）

2. 后端（核心服务）
- Agent 编排服务：Python 3.12 + FastAPI + LangChain 1.x + LangGraph
- 数据抓取服务：Python（Playwright + httpx + AKShare/Tushare adapter）
- 任务调度：Temporal（优先）或 Airflow（团队已熟悉时可选）
- 文档处理服务：Python（unstructured + marker/pymupdf + OCR）
- 报告服务：Python（Jinja2 模板 + Markdown/PDF 导出）

3. 数据与基础设施
- PostgreSQL（事务与配置）
- Redis（队列、缓存、幂等键）
- Qdrant（向量检索）
- Neo4j（GraphRAG）
- MinIO（原始文档与解析产物）
- LangSmith（Tracing + Evals + Regression）

4. 为什么后端选 Python 而不是 Node
- 你的核心诉求是 LangChain/LangGraph + RAG 工程化 + 金融数据处理，Python 生态在文档解析、数据科学与RAG评测侧更成熟。
- 前端仍保持 TypeScript，保证交互效率与工程可维护性。

## 20.2 你要求的技术项 -> 具体落位
1. `状态管理`
- 前端：TanStack Query + Zustand
- 后端：LangGraph state（`query_state/retrieval_state/analysis_state`）+ PostgreSQL 持久化会话
2. `LangSmith`
- 所有 agent run 写 trace；离线/在线评测共用 datasets；发布前回归门禁
3. `多 Agent 协作`
- Router / Data / RAG / Analysis / Report / Critic 六角色图
4. `Long-term Memory 架构`
- 三层记忆：用户偏好记忆、任务记忆、领域知识记忆（向量+图谱）
5. `LangGraph 集成`
- 用 StateGraph 编排节点、分支、重试、HITL 审批
6. `Middleware 工程化`
- 在 LangChain 1.x middleware 中实现输入校验、风控、提示词拼装、输出标准化、审计
7. `Deep Agents`
- 用在复杂“深度分析任务”（子任务并行、长上下文文件系统记忆、工具上限控制）
8. `文档处理工程`
- 统一解析流水线（PDF/DOCX/HTML）+ 表格抽取 + 质量门禁 + 版本化
9. `RAG 集成 GraphRAG/AgenticRAG + 测评`
- 先 Agentic RAG（快出效果），后 GraphRAG（提升全局关系推理）
10. `LangChain 1.0`
- `create_agent` + middleware hooks + structured output + tool strategy 全量采用 v1 范式

## 21. Agent 架构与执行流程（具体）
## 21.1 主流程（单请求）
1. `Router Agent`
- 识别意图：事实查询 / 深度分析 / 文档问答 / 对比研究
2. `Data Agent`
- 拉取结构化数据（行情、公告、财务）并标准化
3. `RAG Agent`
- 规划检索 -> 多路召回 -> 重排 -> 证据包构建
4. `Analysis Agent`
- 指标分析、事件归因、风险路径抽取
5. `Report Agent`
- 按模板生成报告并绑定引用
6. `Critic Agent`
- 检查事实冲突、引用完整性、合规表达

## 21.2 Deep Agent 触发条件
- 用户请求包含“多维对比 + 历史回溯 + 情景分析”
- 需要并行子任务（例如 3 家同业对比 + 近 8 季财报）
- 触发后进入 `SubAgent Lane`，并发执行后回收结果

## 21.3 多 Agent 的工具权限控制
- 默认最小权限（allowlist）
- 每个 agent 独立工具白名单（Data 可抓取，Report 禁止外网写操作）
- 高风险工具（shell/browser）必须走审批中间件

## 22. RAG 设计（从可用到前沿）
## 22.1 索引分层
1. Fact Index（结构化事实）
- 财务指标、公告事件、时间序列
2. Doc Index（非结构化文档）
- 财报、研报、行业文档、上传资料
3. Graph Index（关系图）
- 公司-事件-指标-行业链路

## 22.2 查询执行链路
1. Query Planner：拆问题与检索计划
2. Retriever Orchestrator：
- 向量召回 + BM25 + 元数据过滤
- 必要时触发图检索（GraphRAG）
3. Evidence Packager：构造证据包（带 source_id）
4. Answer Synthesizer：生成“事实/推断/观点”分层输出
5. Citation Verifier：逐条核对引用有效性

## 22.3 Agentic RAG 与 GraphRAG 的分工
- Agentic RAG：解决“复杂问题拆解 + 多步补证 + 冲突处理”
- GraphRAG：解决“跨文档关系推理 + 全局主题总结”
- 组合策略：默认 Agentic RAG；当问题是“关系网络/演化路径”时切 GraphRAG

## 22.4 RAG 评测设计（必须上线）
1. 检索指标：Recall@k、nDCG、MRR
2. 生成指标：事实正确率、引用覆盖率、幻觉率
3. 业务指标：报告可执行性、冲突解释质量、时效性
4. 成本指标：token、P95、单报告成本
5. 发布门禁：LangSmith 回归分数低于阈值禁止上线

## 23. OpenClaw 技术分析与可借鉴思想（针对 `openclaw/openclaw`）
## 23.1 该仓库已体现的前沿 Agent 技术（结论）
1. `Gateway 控制平面`
- 单一长连接网关，统一接入多消息渠道，WS 协议有 schema 校验与设备配对。
2. `多 Agent 路由`
- 通过 `bindings` 按 channel/account/peer 做确定性路由；每个 agent 独立 workspace/session/auth。
3. `Sub-Agents`
- 支持后台子代理并行任务，不阻塞主会话。
4. `技能系统（Skills）`
- `SKILL.md` 指令化扩展；支持 bundled/managed/workspace 多层优先级与动态刷新。
5. `安全沙箱`
- 支持 `non-main/all` sandbox 模式；多层工具策略叠加（deny 优先）。
6. `记忆机制`
- 文件化记忆（Markdown）+ 可选向量检索 + 会话压缩（compaction）。
7. `模型路由与 failover`
- 支持主模型+后备模型，provider/model 规范化。

## 23.2 我们应吸收的“思想”，不是照抄实现
1. 吸收点 A：`控制平面`思想
- 我们也做统一 Agent Gateway（但专注 A股分析，不做泛聊天渠道全接入）。
2. 吸收点 B：`确定性路由`思想
- 按“用户/租户/关注池/任务类型”绑定到专用 agent 配置。
3. 吸收点 C：`分层技能+权限`思想
- 金融工具（估值、财报解析、事件抽取）都走 skill/tool registry + allowlist。
4. 吸收点 D：`会话压缩 + 长期记忆`思想
- 避免上下文膨胀，把长历史沉淀为结构化记忆与压缩摘要。
5. 吸收点 E：`子代理并行`思想
- 同业对比、时间序列回溯、新闻事件聚类可并行子任务执行。

## 23.3 不建议照搬的部分
1. 不建议先做超多聊天渠道
- 你当前目标是“股票分析系统”，先做 Web + API + 定时任务即可。
2. 不建议直接开放第三方技能市场
- 金融场景风险高，必须企业内控审核后再发布技能。
3. 不建议默认主机高权限执行
- 默认沙箱 + 最小权限，否则合规和安全风险过大。

## 24. 本项目最终技术实现建议（定版）
1. 架构形态
- `Frontend(BFF)` + `Agent Gateway` + `Data Ingestion` + `Doc Pipeline` + `Eval Service`
2. 主语言
- 前端 TypeScript，后端 Python
3. Agent框架
- LangChain 1.x + LangGraph（核心），Deep Agents 用于复杂子任务
4. 评测
- LangSmith 为主，补充自定义财经评测脚本
5. RAG路线
- 先 Agentic RAG，后 GraphRAG
6. 安全
- 默认 sandbox、工具分级授权、审计全量落盘

## 25. 关键参考链接（本节新增）
1. OpenClaw 仓库与文档
- https://github.com/openclaw/openclaw
- https://raw.githubusercontent.com/openclaw/openclaw/main/README.md
- https://docs.openclaw.ai/configuration/
- https://docs.openclaw.ai/skills/
- https://docs.openclaw.ai/architecture/overview/
- https://docs.openclaw.ai/architecture/memory/
- https://docs.openclaw.ai/architecture/routing/
2. LangChain / LangGraph / Deep Agents
- https://blog.langchain.com/langchain-langgraph-1dot0/
- https://docs.langchain.com/oss/python/migrate/langchain-v1
- https://docs.langchain.com/oss/python/langchain/deepagents
3. LangSmith
- https://docs.langchain.com/langsmith/home
- https://docs.langchain.com/langsmith/evaluation
4. GraphRAG / Agentic RAG
- https://www.microsoft.com/en-us/research/project/graphrag/
- https://github.com/microsoft/graphrag
- https://arxiv.org/abs/2501.09136
- https://arxiv.org/abs/2507.09477

## 26. V4 可开工蓝图（代码结构 + API + Graph）
## 26.1 仓库目录结构（Monorepo）
```txt
project-root/
  apps/
    web/                          # Next.js 前端
  services/
    agent-gateway/                # FastAPI: 对话/报告入口，LangGraph 编排
    ingestion-service/            # FastAPI: 数据抓取与标准化
    doc-pipeline/                 # FastAPI/worker: 文档解析与索引
    eval-service/                 # FastAPI/worker: 离线评测与回归门禁
    report-service/               # FastAPI: 报告模板渲染与导出
  packages/
    py-common/                    # Python 公共库（schema、logging、errors）
    ts-sdk/                       # 前端调用SDK（OpenAPI 生成）
  infra/
    docker/                       # 本地依赖（pg/redis/qdrant/neo4j/minio）
    migrations/                   # SQL 迁移脚本
    temporal/                     # workflow 定义
  docs/
    a-share-agent-system-tech-solution.md
```

## 26.2 首批 API 合同（MVP）
1. Agent Gateway（`services/agent-gateway`）
- `POST /v1/query`
  - 入参：`user_id, tenant_id, prompt, stock_codes[], mode(fact|deep), context_refs[]`
  - 出参：`run_id, answer_md, citations[], risk_flags[], trace_id`
- `POST /v1/report/generate`
  - 入参：`stock_code, horizon(1q|1y|3y), report_type(fact|research), options`
  - 出参：`report_id, status, eta_seconds`
- `GET /v1/report/{report_id}`
  - 出参：`status, sections[], citations[], export_urls`

2. Ingestion Service（`services/ingestion-service`）
- `POST /v1/ingest/announcements`
  - 入参：`stock_code, start_date, end_date, source_priority[]`
  - 出参：`job_id, pulled_count, dedup_count`
- `POST /v1/ingest/market-daily`
  - 入参：`trade_date, symbols[]`
  - 出参：`job_id, rows_written`
- `POST /v1/watchlist/refresh`
  - 入参：`user_id, stock_codes[]`
  - 出参：`job_id, scheduled_tasks[]`

3. Doc Pipeline（`services/doc-pipeline`）
- `POST /v1/docs/upload`
  - 入参：`file, doc_type(financial|research|other), stock_code?`
  - 出参：`doc_id, parse_status`
- `POST /v1/docs/{doc_id}/index`
  - 出参：`chunk_count, vector_count, graph_entities`
- `GET /v1/docs/{doc_id}`
  - 出参：`metadata, versions[], quality_flags[]`

4. Eval Service（`services/eval-service`）
- `POST /v1/evals/run`
  - 入参：`suite_id, candidate_version, baseline_version`
  - 出参：`eval_run_id, status`
- `GET /v1/evals/{eval_run_id}`
  - 出参：`retrieval_metrics, generation_metrics, business_metrics, pass_gate`

## 26.3 LangGraph 设计（节点与状态）
1. 状态对象（`AgentState`）
```json
{
  "run_id": "string",
  "user_id": "string",
  "tenant_id": "string",
  "intent": "fact|deep|doc_qa|compare",
  "stock_codes": ["string"],
  "query": "string",
  "retrieval_plan": {},
  "evidence_pack": [],
  "analysis_result": {},
  "report_draft": {},
  "risk_flags": [],
  "citations": [],
  "trace_id": "string"
}
```
2. 节点定义
- `intent_router`
- `data_fetch_node`
- `retrieval_planner_node`
- `hybrid_retriever_node`
- `graphrag_node`（按需分支）
- `analysis_node`
- `report_writer_node`
- `citation_verify_node`
- `compliance_guard_node`
- `critic_node`
- `finalize_node`
3. 关键分支
- `intent == deep` -> 启动 `retrieval_planner_node` + `analysis_node` 并行子任务
- `query_type == relation/global` -> 启用 `graphrag_node`
- `risk_flags contains HIGH` -> `human_review_queue`

## 26.4 Middleware 实现点（LangChain 1.x）
1. `pre_model_middleware`
- 输入清洗、意图标签注入、金融术语词典扩展、最大上下文预算控制
2. `tool_middleware`
- 每个工具统一超时（默认10s）、重试（2次）、熔断、审计日志
3. `post_model_middleware`
- 输出 schema 校验、引用缺失补救、禁止词/违规表达替换
4. `risk_middleware`
- 识别“保证收益/明确买卖建议”并改写为情景分析模板

## 26.5 Long-term Memory 结构（可直接建表）
1. `memory_user_profile`
- `user_id, risk_preference, sectors, updated_at`
2. `memory_investment_focus`
- `user_id, stock_code, watch_reason, confidence, updated_at`
3. `memory_task_summary`
- `run_id, summary, key_facts, unresolved_questions`
4. `memory_feedback`
- `run_id, user_feedback, correction_label, corrected_fact`
5. memory 写入策略
- 只写“高价值低噪声信息”（用户确认、评测通过结论、人工修正）

## 26.6 RAG 检索参数初始值（MVP 默认）
1. Chunk
- `chunk_size=900`, `chunk_overlap=120`
2. Retrieval
- `top_k_vector=12`, `top_k_bm25=20`, `rerank_top_n=10`
3. GraphRAG 触发阈值
- 问题长度 > 60 字且包含“关系/影响/演化/关联方”等关键词
4. 引用规则
- 每个核心结论至少 1 条引用；报告总引用不低于 6 条

## 26.7 数据库 DDL 草案（首批4表）
```sql
create table instrument (
  instrument_id bigserial primary key,
  stock_code varchar(16) unique not null,
  stock_name varchar(128) not null,
  market varchar(16) not null,
  industry varchar(128),
  listed_at date,
  updated_at timestamptz default now()
);

create table event_fact (
  event_id bigserial primary key,
  stock_code varchar(16) not null,
  event_type varchar(64) not null,
  event_time timestamptz not null,
  title text not null,
  content text,
  source_id varchar(64) not null,
  source_url text,
  reliability_score numeric(3,2) default 0.50,
  created_at timestamptz default now()
);

create table report_run (
  report_id bigserial primary key,
  run_id varchar(64) unique not null,
  user_id varchar(64) not null,
  stock_code varchar(16) not null,
  report_type varchar(16) not null,
  status varchar(16) not null,
  trace_id varchar(128),
  created_at timestamptz default now(),
  finished_at timestamptz
);

create table watchlist (
  id bigserial primary key,
  user_id varchar(64) not null,
  stock_code varchar(16) not null,
  priority smallint default 5,
  created_at timestamptz default now(),
  unique(user_id, stock_code)
);
```

## 26.8 本周开发排期（工程任务到人天）
1. Day 1-2
- 建目录与基础依赖；起 `agent-gateway` 与 `ingestion-service`
2. Day 3
- 完成4张核心表迁移；打通 `POST /v1/ingest/announcements`
3. Day 4
- 打通 `POST /v1/query` 最小链路（Router->Retriever->Writer）
4. Day 5
- 接入 LangSmith tracing + 一组离线评测样本（>=30条）
5. Day 6-7
- 完成报告导出与 watchlist 定时任务；跑第一轮回归

## 27. Prompt 工程化设计（企业级必做）
## 27.1 目标
- 让 Prompt 从“文本技巧”变成“可版本化、可评测、可回滚”的工程资产。
- 在金融场景下保证输出稳定性、合规性、可追溯性。

## 27.2 Prompt 资产模型（建议）
1. 三层结构
- `system layer`：角色与全局规则（合规、引用、禁用表达）。
- `policy layer`：场景策略（事实问答/深度分析/研报总结）。
- `task layer`：当前任务模板（变量、输出格式、工具调用约束）。
2. 配置化对象
- `prompt_id, version, owner, scenario, model_family, template, variables_schema, guardrails, created_at`
3. 版本与发布
- 支持 `draft -> canary -> stable -> deprecated` 生命周期。

## 27.3 Prompt 运行时编排
1. 上下文构建顺序
- 用户输入 -> 意图标签 -> 检索证据 -> 记忆摘要 -> policy 注入 -> 最终 prompt。
2. 变量注入规则
- 所有变量必须 schema 校验，禁止直接拼接未清洗文本。
3. Token 预算
- 按模块配额：规则区、证据区、历史区分别限额，避免上下文爆炸。
4. 动态裁剪
- 低相关历史对话与低可信证据优先裁剪。

## 27.4 Prompt 质量门禁（必须自动化）
1. 单测（Prompt Unit Tests）
- 给定固定输入，断言输出结构、字段完整性、引用格式。
2. 回归测试（Prompt Regression）
- 新旧版本在同一评测集对比：事实正确率、引用覆盖率、幻觉率、成本。
3. 红队测试（Prompt Red Team）
- 越权请求、提示词注入、投资建议诱导、伪造数据场景。
4. 上线门槛
- 未达到阈值（如引用覆盖率<95%）禁止合并/发布。

## 27.5 金融场景专属 Guardrails
1. 表达约束
- 禁止“保证收益/确定性买卖点”表述。
- 对买卖问题强制改写为“情景分析 + 风险揭示 + 数据依据”。
2. 证据约束
- 关键结论无引用时禁止输出最终结论。
3. 冲突约束
- 多源冲突必须输出冲突说明与置信度，不允许静默忽略。
4. 时效约束
- 超过时效窗口的数据必须打“可能过期”标签。

## 27.6 Prompt 可观测与审计
- 每次运行记录：`prompt_version, model, input_hash, context_hash, tool_calls, citations, output_hash, trace_id`
- 关联 LangSmith trace，支持按版本回放和问题复现。

## 27.7 推荐落地清单
1. 建 `prompt_registry`（版本库）
2. 建 `prompt_experiment`（A/B 对比）
3. 建 `prompt_eval_suite`（固定回归集）
4. 建 `prompt_release_gate`（自动门禁）
