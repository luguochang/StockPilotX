# 免费数据源实施细则（A股智能分析系统）

## 1. 目标与边界
- 目标：在“不购买商业数据源”的前提下，构建可运行、可扩展、可审计的数据底座。
- 边界：
  - 只使用公开可访问的数据接口/网页。
  - 不使用绕过登录、破解、逆向加密等高风险抓取方式。
  - 输出仅供研究参考，不构成投资建议。
- 治理绑定（新增）：
  - 执行清单：`docs/implementation-checklist.md`
  - 追踪矩阵：`docs/spec-traceability-matrix.md`
  - 全局门禁：`docs/global-constraints.md`

## 2. 数据源分层（免费方案）
## 2.1 一级事实源（结论主依据）
- 巨潮资讯：`https://www.cninfo.com.cn/`
- 上交所公告：`https://www.sse.com.cn/disclosure/listedinfo/announcement/`
- 深交所披露：`https://www.szse.cn/disclosure/listed/index.html`

用途：
- 公告、财报、监管问询、公司治理信息。
- 系统“事实结论”默认优先引用一级源。

## 2.2 二级行情源（免费补充）
- `stock-api` 仓库适配源（免费）：
  - 腾讯：`qt.gtimg.cn`, `smartbox.gtimg.cn`
  - 网易：`api.money.126.net`, `quotes.money.163.com`
  - 新浪：`hq.sinajs.cn`, `suggest3.sinajs.cn`
  - 雪球：`stock.xueqiu.com`, `xueqiu.com`

代码参考：
- `stock-api/src/stocks/tencent/index.ts`
- `stock-api/src/stocks/netease/index.ts`
- `stock-api/src/stocks/sina/index.ts`
- `stock-api/src/stocks/xueqiu/index.ts`
- 说明：以上为第三方仓库路径，仅作能力参考；`StockPilotX` 不直接纳入该仓库代码。

用途：
- 实时行情、股票搜索、批量行情拉取。

## 2.3 三级资讯源（线索补充）
- `go-stock` 已实现的一批免费源（财经资讯/事件）：
  - 东方财富、财联社、新浪、雪球、腾讯等

代码参考：
- `go-stock/backend/data/market_news_api.go`
- `go-stock/backend/data/openai_api.go`
- `go-stock/backend/data/crawler_api.go`
- 说明：以上为第三方仓库路径，仅作能力参考；需按需拷贝实现到本项目。

用途：
- 事件线索、市场情绪、题材热度。
- 默认不作为单一事实结论依据，需与一级/二级交叉验证。

## 2.4 可选源（免费但有限制）
- Tushare：免费可注册，部分数据有积分/频次限制。
- 代码参考：`go-stock/backend/data/tushare_data_api.go`
- 说明：该参考实现不作为本仓库依赖。

用途：
- 日线、基础财务字段补充；不依赖其作为唯一主源。

## 3. 抓取实施架构
## 3.1 组件划分
- `ingestion-service`：统一调度抓取任务。
- `source-adapters`：每个数据源一个适配器（source plugin）。
- `normalizer`：字段标准化（代码、时间、单位、口径）。
- `validator`：数据质量校验（缺失/异常/重复/冲突）。
- `writer`：写入 PostgreSQL + MinIO + Qdrant + Neo4j。

## 3.2 适配器接口（统一）
```python
class SourceAdapter(Protocol):
    source_id: str
    source_tier: str
    async def fetch(self, query: dict) -> list[dict]: ...
    def normalize(self, raw: dict) -> dict: ...
    def reliability_score(self, item: dict) -> float: ...
```

## 4. 字段标准化规范（Canonical Schema）
## 4.1 基础实体
- `instrument`：股票基础信息
  - `stock_code`, `stock_name`, `market`, `industry`
- `quote_tick`：行情快照
  - `stock_code`, `price`, `pct_change`, `volume`, `turnover`, `ts`
- `event_fact`：事件事实
  - `stock_code`, `event_type`, `title`, `content`, `event_time`
- `source_meta`：来源信息
  - `source_id`, `source_url`, `crawl_time`, `reliability_score`

## 4.2 编码与时间统一
- 股票代码统一：`SH600000` / `SZ000001` 格式。
- 时间统一：UTC 存储 + Asia/Shanghai 展示。
- 数值统一：金额统一为 `CNY`、比例统一为小数或百分比（字段中明确口径）。

## 5. 更新频率与任务编排
## 5.1 任务清单
1. `daily_announcement_ingest`（交易日 18:00）
- 增量抓取公告/财报/问询。
2. `intraday_quote_ingest`（交易时段每 1-5 分钟）
- 拉取重点池行情快照（watchlist + 热门池）。
3. `nightly_news_ingest`（每日 22:00）
- 抓资讯与事件线索，构建次日素材。
4. `weekly_rebuild`（周日）
- 去重、修复、索引重建、质量回归。

## 5.2 调度策略
- 首选 Temporal（可重试、可观测、可恢复）。
- 每个任务设置幂等键：`source + symbol + date + page`。

## 6. 回退与容错策略（免费源必备）
## 6.1 多源回退顺序（行情示例）
1. 腾讯
2. 网易
3. 新浪
4. 雪球

规则：
- 主源失败或字段异常时自动切换下一源。
- 同时返回多源时做一致性检查（价格偏差阈值）。

## 6.2 重试与熔断
- 重试：指数退避（1s/2s/4s），最多 3 次。
- 熔断：同源 5 分钟内连续失败超过阈值则临时熔断 10 分钟。
- 降级：熔断期间仅返回历史缓存 + 明确“数据可能延迟”标签。

## 6.3 缓存策略
- 实时行情缓存 TTL：15-60 秒（按接口限频调整）。
- 公告/财报缓存 TTL：24 小时。
- 缓存命中时仍保留来源时间戳用于新鲜度判断。

## 7. 数据质量与可信度
## 7.1 质量校验
- 完整性：关键字段不能为空（代码、时间、值、来源）。
- 合法性：数值范围检查（涨跌幅、成交量等）。
- 唯一性：按 `source_id + source_event_id/hash` 去重。
- 一致性：多源数据冲突时打 `conflict_flag`。

## 7.2 来源可信度评分（建议）
- 一级源（交易所/法定披露）：`0.95-1.00`
- 二级源（主流行情接口）：`0.75-0.90`
- 三级源（资讯线索）：`0.50-0.75`

输出要求：
- 报告中每条关键结论必须带 `source_id + source_url + event_time + score`。

## 8. 与 RAG 的衔接
## 8.1 入库分流
- 结构化事实 -> PostgreSQL/ClickHouse
- 非结构化文本 -> MinIO + Qdrant
- 关系抽取 -> Neo4j

## 8.2 检索优先级
- 先查结构化事实（快且稳定）
- 再查文档块（解释与上下文）
- 关系问题再查图谱（GraphRAG）

## 8.3 引用规则
- 核心结论至少 1 条一级或二级来源。
- 若只来自三级源，必须标记“线索级结论，待确认”。

## 9. 合规与风险控制
- 遵守目标站点公开使用规则与 robots/policy（可访问范围内抓取）。
- 不做登录态盗用、不抓个人敏感信息。
- 不输出“保证收益”“明确买卖建议”。
- 全链路审计：记录请求、来源、解析、输出版本。

## 10. 最小可行实施（两周）
## 周1
1. 接入一级源公告抓取（巨潮/上交所/深交所）
2. 接入二级行情源（腾讯+网易）并完成字段标准化
3. 建立 `event_fact` 与 `quote_tick` 入库

## 周2
1. 接入三级资讯源（东财/财联社/新浪）作为线索层
2. 建立回退、缓存、重试、熔断
3. 打通“问答/报告”引用链路（source_meta 可追溯）

## 11. 你给的两个仓库在本项目中的定位
- `stock-api`：行情聚合适配层（快速、轻量、免费）
- `go-stock`：资讯与工具能力参考（抓取与多源思路）
- 本项目不直接照搬其完整架构，且不将其代码纳入本仓库；仅复用“免费数据接入模式 + 源适配思路”
