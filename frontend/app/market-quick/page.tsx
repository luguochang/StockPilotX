"use client";

import { useMemo, useState } from "react";
import { motion } from "framer-motion";
import { Alert, Button, Card, Col, Input, Progress, Row, Space, Statistic, Tag, Typography } from "antd";
import StockSelectorModal from "../components/StockSelectorModal";
import StructuredAnswerCard from "../components/analysis/StructuredAnswerCard";
import CitationsPanel, { type Citation } from "../components/analysis/CitationsPanel";
import { readSSEAndConsume } from "../components/analysis/sse";
import styles from "./market-quick.module.css";

const API_BASE = process.env.NEXT_PUBLIC_API_BASE ?? "http://127.0.0.1:8000";
const { TextArea } = Input;
const { Title, Paragraph, Text } = Typography;

type QueryResponse = { trace_id: string; answer: string; citations: Citation[] };
type AnalysisBrief = {
  confidence_level: string;
  confidence_reason: string;
  citation_count: number;
  citation_avg_reliability: number;
};
type MarketOverview = {
  realtime: { price?: number; pct_change?: number; source_id?: string; ts?: string };
};

type QueryStageKey = "idle" | "data_refresh" | "retriever" | "model" | "done";

const QUERY_STAGE_ORDER: QueryStageKey[] = ["idle", "data_refresh", "retriever", "model", "done"];
const QUERY_STAGE_LABEL: Record<QueryStageKey, string> = {
  idle: "待执行",
  data_refresh: "数据刷新",
  retriever: "检索准备",
  model: "模型输出",
  done: "执行完成",
};

const QUICK_QUESTIONS = [
  { label: "30天结论", value: "请给出未来30天的核心观点、风险触发条件和执行建议。" },
  { label: "证据链模式", value: "请基于近3个月走势与事件，给出可验证结论与证据链。" },
  { label: "情景对比", value: "请给出多空两套情景、触发条件、仓位节奏和止损规则。" },
  { label: "摘要模式", value: "请仅输出结论摘要 + 关键证据 + 不确定性说明。" },
];

function resolveQueryStage(eventName: string, payload: Record<string, unknown>): QueryStageKey {
  if (eventName === "answer_delta") return "model";
  if (eventName === "done") return "done";
  if (eventName !== "progress") return "idle";

  const phase = String(payload?.phase ?? "").trim();
  if (phase === "data_refresh") return "data_refresh";
  if (phase === "retriever") return "retriever";
  if (phase === "model" || phase === "model_wait") return "model";
  return "idle";
}

export default function MarketQuickPage() {
  const [stockCode, setStockCode] = useState("SH600000");
  const [question, setQuestion] = useState("请给出可验证的交易结论，并拆解风险与执行条件");
  const [error, setError] = useState("");

  const [result, setResult] = useState<QueryResponse>({ trace_id: "", answer: "", citations: [] });
  const [analysisBrief, setAnalysisBrief] = useState<AnalysisBrief | null>(null);
  const [knowledgePersistedTraceId, setKnowledgePersistedTraceId] = useState("");

  const [queryStreaming, setQueryStreaming] = useState(false);
  const [queryProgressText, setQueryProgressText] = useState("");
  const [queryStage, setQueryStage] = useState<QueryStageKey>("idle");
  const [streamSource, setStreamSource] = useState("unknown");
  const [runtimeEngine, setRuntimeEngine] = useState("unknown");
  const [activeProvider, setActiveProvider] = useState("");
  const [activeModel, setActiveModel] = useState("");
  const [activeApiStyle, setActiveApiStyle] = useState("");
  const [overview, setOverview] = useState<MarketOverview | null>(null);

  const queryStagePercent = useMemo(() => {
    const idx = QUERY_STAGE_ORDER.indexOf(queryStage);
    if (idx <= 0) return 0;
    return Number(((idx / (QUERY_STAGE_ORDER.length - 1)) * 100).toFixed(1));
  }, [queryStage]);

  async function ensureStockInUniverse(codes: string[]) {
    for (const code of codes) {
      const normalized = code.trim().toUpperCase();
      const resp = await fetch(`${API_BASE}/v1/stocks/search?keyword=${encodeURIComponent(normalized)}&limit=30`);
      if (!resp.ok) throw new Error("股票库检索失败，请稍后重试");
      const rows = (await resp.json()) as Array<Record<string, unknown>>;
      const hit = Array.isArray(rows) && rows.some((x) => String(x?.stock_code ?? "").toUpperCase() === normalized);
      if (!hit) throw new Error(`股票 ${normalized} 不在已同步股票库中，请重新选择`);
    }
  }

  // 市场快析只负责单次高级分析流，不承担 DeepThink 多轮控制台职责。
  async function runMarketQuickAnalysis() {
    setError("");
    setQueryStreaming(true);
    setQueryProgressText("");
    setQueryStage("idle");
    setKnowledgePersistedTraceId("");
    setAnalysisBrief(null);
    setStreamSource("unknown");
    setRuntimeEngine("unknown");
    setActiveProvider("");
    setActiveModel("");
    setActiveApiStyle("");
    setResult({ trace_id: "", answer: "", citations: [] });

    try {
      await ensureStockInUniverse([stockCode]);

      const overviewPromise = fetch(`${API_BASE}/v1/market/overview/${stockCode}`);
      const streamResp = await fetch(`${API_BASE}/v1/query/stream`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ user_id: "frontend-market-quick", stock_codes: [stockCode], question }),
      });

      if (!streamResp.ok) {
        const txt = await streamResp.text();
        throw new Error(txt || `query/stream HTTP ${streamResp.status}`);
      }

      let gotEvent = false;
      await readSSEAndConsume(streamResp, (eventName, payload) => {
        gotEvent = true;
        const stage = resolveQueryStage(eventName, payload as Record<string, unknown>);
        if (stage !== "idle") setQueryStage(stage);

        if (eventName === "meta") {
          setResult((prev) => ({ ...prev, trace_id: String(payload?.trace_id ?? prev.trace_id ?? "") }));
          return;
        }
        if (eventName === "answer_delta") {
          const delta = String(payload?.delta ?? "");
          if (!delta) return;
          setResult((prev) => ({ ...prev, answer: `${prev.answer}${delta}` }));
          return;
        }
        if (eventName === "stream_source") {
          setActiveProvider(String(payload?.provider ?? ""));
          setActiveModel(String(payload?.model ?? ""));
          setActiveApiStyle(String(payload?.api_style ?? ""));
          setStreamSource(String(payload?.source ?? "unknown"));
          return;
        }
        if (eventName === "stream_runtime") {
          setRuntimeEngine(String(payload?.runtime ?? "unknown"));
          return;
        }
        if (eventName === "progress") {
          const phase = String(payload?.phase ?? "").trim();
          const message = String(payload?.message ?? "").trim();
          const waitMs = Number(payload?.wait_ms ?? 0);
          if (phase === "model_wait" && Number.isFinite(waitMs) && waitMs > 0) {
            setQueryProgressText(`${message || "模型推理中"}（${(waitMs / 1000).toFixed(1)}s）`);
          } else {
            setQueryProgressText(message || phase || "处理中");
          }
          return;
        }
        if (eventName === "citations") {
          const citations = Array.isArray(payload?.citations) ? (payload.citations as Citation[]) : [];
          setResult((prev) => ({ ...prev, citations }));
          return;
        }
        if (eventName === "analysis_brief") {
          setAnalysisBrief(payload as AnalysisBrief);
          return;
        }
        if (eventName === "knowledge_persisted") {
          const trace = String(payload?.trace_id ?? "").trim();
          if (trace) setKnowledgePersistedTraceId(trace);
          setQueryProgressText("回答已沉淀到共享语料，可用于后续检索复用");
          return;
        }
        if (eventName === "done") {
          setQueryStage("done");
          setQueryProgressText("分析已完成");
        }
      });

      if (!gotEvent) throw new Error("query/stream 连接成功但未收到事件");

      const overviewResp = await overviewPromise;
      const overviewBody = (await overviewResp.json()) as MarketOverview;
      if (!overviewResp.ok) throw new Error(`overview HTTP ${overviewResp.status}`);
      setOverview(overviewBody);
    } catch (e) {
      setError(e instanceof Error ? e.message : "执行失败");
      setQueryStage("idle");
    } finally {
      setQueryStreaming(false);
    }
  }

  return (
    <main className="container shell-fade-in">
      <section className={styles.page}>
        <motion.div initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} className={styles.hero}>
          <Space direction="vertical" size={10} style={{ width: "100%" }}>
            <Tag color="processing" style={{ width: "fit-content" }}>市场快析</Tag>
            <Title level={2} className={styles.heroTitle}>高级分析（流式）</Title>
            <Paragraph className={styles.heroDesc}>
              面向单次问题的快速深度回答，重点输出结论、证据和不确定性。若需要多轮协商、冲突复核与预算治理，请进入 DeepThink。
            </Paragraph>
            <div className={styles.inputCard}>
              <Space direction="vertical" style={{ width: "100%" }}>
                <StockSelectorModal
                  value={stockCode}
                  onChange={(next) => setStockCode(Array.isArray(next) ? (next[0] ?? "") : next)}
                  title="选择分析标的"
                  placeholder="请先选择要分析的股票"
                />
                <TextArea rows={4} value={question} onChange={(e) => setQuestion(e.target.value)} />
                <Space wrap>
                  {QUICK_QUESTIONS.map((item) => (
                    <Button key={item.label} size="small" onClick={() => setQuestion(item.value)}>
                      {item.label}
                    </Button>
                  ))}
                </Space>
                <Space wrap>
                  <Button type="primary" size="large" onClick={runMarketQuickAnalysis} loading={queryStreaming}>开始市场快析</Button>
                  <Button href="/deep-think">DeepThink 业务研判</Button>
                  <Button href="/deep-think/console">DeepThink 工程控制台</Button>
                  <Tag color={queryStreaming ? "processing" : "default"}>{queryStreaming ? "执行中" : "待执行"}</Tag>
                  {knowledgePersistedTraceId ? <Tag color="green">已沉淀共享语料</Tag> : null}
                </Space>
              </Space>
            </div>
          </Space>
        </motion.div>

        {error ? <Alert type="error" showIcon message={error} /> : null}

        <Row gutter={[12, 12]}>
          <Col xs={24} md={8}>
            <Card className="premium-card" title="行情快照">
              <Space direction="vertical" style={{ width: "100%" }}>
                <Statistic title="最新价" value={overview?.realtime?.price ?? 0} precision={3} />
                <Statistic title="涨跌幅" value={overview?.realtime?.pct_change ?? 0} precision={2} suffix="%" />
                <Text style={{ color: "#64748b" }}>{overview?.realtime?.source_id ?? "unknown"} | {overview?.realtime?.ts ?? "--"}</Text>
              </Space>
            </Card>
          </Col>
          <Col xs={24} md={16}>
            <Card className="premium-card" title="执行状态">
              <Space direction="vertical" style={{ width: "100%" }}>
                <Text style={{ color: "#334155" }}>当前阶段：{QUERY_STAGE_LABEL[queryStage]}</Text>
                <Progress percent={queryStagePercent} status={error ? "exception" : queryStreaming ? "active" : "normal"} />
                <Text style={{ color: "#64748b" }}>{queryProgressText || "等待/完成"}</Text>
              </Space>
            </Card>
          </Col>
        </Row>

        {analysisBrief ? (
          <div className={styles.briefRow}>
            <Card className="premium-card" title="置信等级"><Text>{analysisBrief.confidence_level}</Text></Card>
            <Card className="premium-card" title="引用指标"><Text>{analysisBrief.citation_count} / {analysisBrief.citation_avg_reliability.toFixed(2)}</Text></Card>
            <Card className="premium-card" title="原因说明"><Text>{analysisBrief.confidence_reason}</Text></Card>
          </div>
        ) : null}

        <div className={styles.grid}>
          <div>
            <StructuredAnswerCard
              answer={result.answer}
              traceId={result.trace_id}
              streaming={queryStreaming}
              progressText={queryProgressText}
              streamSource={streamSource}
              model={activeModel}
              provider={activeProvider}
              apiStyle={activeApiStyle}
              runtime={runtimeEngine}
            />
            <CitationsPanel citations={result.citations} />
          </div>
          <div>
            <Card className="premium-card" title="模块边界说明">
              <Space direction="vertical" style={{ width: "100%" }} size={8}>
                <Text style={{ color: "#334155" }}>市场快析：单次问题的快速流式分析，不执行 DeepThink 多轮任务图。</Text>
                <Text style={{ color: "#334155" }}>DeepThink：多 Agent 按轮推进，适用于冲突复核、预算治理和事件归档。</Text>
                <Text style={{ color: "#64748b" }}>
                  如果你需要“执行下一轮 / 冲突源 / 任务图 / 导出审计”，请直接进入 DeepThink 页面。
                </Text>
              </Space>
            </Card>
          </div>
        </div>
      </section>
    </main>
  );
}
