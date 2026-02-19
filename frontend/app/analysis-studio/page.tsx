"use client";

import { useMemo, useState } from "react";
import { motion } from "framer-motion";
import { Alert, Button, Card, Col, Input, Progress, Row, Space, Statistic, Tag, Typography } from "antd";
import StockSelectorModal from "../components/StockSelectorModal";
import StructuredAnswerCard from "../components/analysis/StructuredAnswerCard";
import CitationsPanel, { type Citation } from "../components/analysis/CitationsPanel";
import DeepRoundSummaryPanel from "../components/analysis/DeepRoundSummaryPanel";
import StreamEventPanel, { type StreamEvent } from "../components/analysis/StreamEventPanel";
import { readSSEAndConsume } from "../components/analysis/sse";
import styles from "./analysis-studio.module.css";

const API_BASE = process.env.NEXT_PUBLIC_API_BASE ?? "http://127.0.0.1:8000";
const ENABLE_DEEPTHINK_V2_STREAM = process.env.NEXT_PUBLIC_DEEPTHINK_V2_STREAM !== "0";
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
type DeepThinkOpinion = {
  agent_id: string;
  signal: "buy" | "hold" | "reduce";
  confidence: number;
  reason: string;
};
type DeepThinkRound = {
  round_id: string;
  round_no: number;
  consensus_signal: string;
  disagreement_score: number;
  conflict_sources: string[];
  stop_reason: string;
  opinions: DeepThinkOpinion[];
};
type DeepThinkSession = {
  session_id: string;
  status: string;
  current_round: number;
  max_rounds: number;
  rounds: DeepThinkRound[];
};

type DeepRoundStageKey = "idle" | "planning" | "data_refresh" | "intel_search" | "debate" | "arbitration" | "persist" | "done";
const DEEP_ROUND_STAGE_ORDER: DeepRoundStageKey[] = ["idle", "planning", "data_refresh", "intel_search", "debate", "arbitration", "persist", "done"];
const DEEP_ROUND_STAGE_LABEL: Record<DeepRoundStageKey, string> = {
  idle: "待执行",
  planning: "任务规划",
  data_refresh: "数据刷新",
  intel_search: "情报检索",
  debate: "多Agent协商",
  arbitration: "仲裁收敛",
  persist: "结果落库",
  done: "执行完成",
};

function resolveDeepStageFromEvent(event: string, data: Record<string, any>): DeepRoundStageKey {
  const normalized = String(event ?? "").trim();
  const stage = String(data?.stage ?? data?.phase ?? "").trim();
  if (normalized === "round_started") return "planning";
  if (stage === "planning") return "planning";
  if (stage === "data_refresh") return "data_refresh";
  if (stage === "intel_search") return "intel_search";
  if (stage === "debate") return "debate";
  if (normalized === "arbitration_final") return "arbitration";
  if (normalized === "round_persisted") return "persist";
  if (normalized === "done") return "done";
  return "idle";
}

export default function AnalysisStudioPage() {
  const [stockCode, setStockCode] = useState("SH600000");
  const [question, setQuestion] = useState("请给出可验证的交易结论，并拆解风险与执行条件");
  const [error, setError] = useState("");

  const [result, setResult] = useState<QueryResponse>({ trace_id: "", answer: "", citations: [] });
  const [analysisBrief, setAnalysisBrief] = useState<AnalysisBrief | null>(null);
  const [knowledgePersistedTraceId, setKnowledgePersistedTraceId] = useState("");
  const [queryStreaming, setQueryStreaming] = useState(false);
  const [queryProgressText, setQueryProgressText] = useState("");
  const [streamSource, setStreamSource] = useState("unknown");
  const [runtimeEngine, setRuntimeEngine] = useState("unknown");
  const [activeProvider, setActiveProvider] = useState("");
  const [activeModel, setActiveModel] = useState("");
  const [activeApiStyle, setActiveApiStyle] = useState("");
  const [overview, setOverview] = useState<MarketOverview | null>(null);

  const [deepSession, setDeepSession] = useState<DeepThinkSession | null>(null);
  const [deepEvents, setDeepEvents] = useState<StreamEvent[]>([]);
  const [deepProgressText, setDeepProgressText] = useState("");
  const [deepRunning, setDeepRunning] = useState(false);
  const [deepError, setDeepError] = useState("");

  const running = queryStreaming || deepRunning;

  const deepCurrentStage = useMemo(() => {
    let current: DeepRoundStageKey = "idle";
    for (const item of deepEvents) {
      const stage = resolveDeepStageFromEvent(item.event, item.data);
      if (stage !== "idle") current = stage;
      if (stage === "done") return "done";
    }
    if (running && current === "idle") return "planning";
    return current;
  }, [deepEvents, running]);

  const deepStagePercent = useMemo(() => {
    const idx = DEEP_ROUND_STAGE_ORDER.indexOf(deepCurrentStage);
    if (idx <= 0) return 0;
    return Number(((idx / (DEEP_ROUND_STAGE_ORDER.length - 1)) * 100).toFixed(1));
  }, [deepCurrentStage]);

  function appendDeepEvent(event: string, data: Record<string, any>) {
    setDeepEvents((prev) => [...prev, { event, data, emitted_at: new Date().toISOString() }].slice(-120));
    const stage = resolveDeepStageFromEvent(event, data);
    if (stage !== "idle") {
      const message = String(data?.message ?? data?.reason ?? "").trim();
      setDeepProgressText(message || DEEP_ROUND_STAGE_LABEL[stage]);
    }
    if (String(event) === "done") setDeepProgressText("轮次执行完成，结果已同步。");
    if (String(event) === "error") setDeepProgressText("轮次执行失败，请查看错误信息。");
  }

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

  async function runQueryStream() {
    setQueryStreaming(true);
    setQueryProgressText("");
    setKnowledgePersistedTraceId("");
    setAnalysisBrief(null);
    setStreamSource("unknown");
    setRuntimeEngine("unknown");
    setActiveProvider("");
    setActiveModel("");
    setActiveApiStyle("");
    setResult({ trace_id: "", answer: "", citations: [] });
    try {
      const overviewPromise = fetch(`${API_BASE}/v1/market/overview/${stockCode}`);
      const streamResp = await fetch(`${API_BASE}/v1/query/stream`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ user_id: "frontend-user", stock_codes: [stockCode], question }),
      });
      if (!streamResp.ok) {
        const txt = await streamResp.text();
        throw new Error(txt || `query/stream HTTP ${streamResp.status}`);
      }
      let gotEvent = false;
      await readSSEAndConsume(streamResp, (eventName, payload) => {
        gotEvent = true;
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
        }
      });
      if (!gotEvent) throw new Error("query/stream 连接成功但未收到事件");

      const overviewResp = await overviewPromise;
      const overviewBody = (await overviewResp.json()) as MarketOverview;
      if (!overviewResp.ok) throw new Error(`overview HTTP ${overviewResp.status}`);
      setOverview(overviewBody);
      setQueryProgressText("");
    } finally {
      setQueryStreaming(false);
    }
  }

  async function createDeepThinkSession(): Promise<DeepThinkSession> {
    const resp = await fetch(`${API_BASE}/v1/deep-think/sessions`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        user_id: "frontend-deepthink-user",
        question,
        stock_codes: [stockCode],
        max_rounds: 6,
        mode: "analysis",
      }),
    });
    const body = (await resp.json()) as DeepThinkSession;
    if (!resp.ok) throw new Error(`create deep session failed: HTTP ${resp.status}`);
    return body;
  }

  async function replayDeepThinkStream(sessionId: string) {
    const resp = await fetch(`${API_BASE}/v1/deep-think/sessions/${sessionId}/stream`);
    if (!resp.ok) throw new Error(`replay stream failed: HTTP ${resp.status}`);
    await readSSEAndConsume(resp, (eventName, payload) => appendDeepEvent(eventName, payload));
  }

  async function runDeepThinkRound() {
    setDeepRunning(true);
    setDeepError("");
    setDeepProgressText("准备执行下一轮 DeepThink...");
    setDeepEvents([]);
    try {
      const session = await createDeepThinkSession();
      setDeepSession(session);
      const roundPayload = {
        question,
        stock_codes: [stockCode],
        archive_max_events: 220,
      };
      if (ENABLE_DEEPTHINK_V2_STREAM) {
        try {
          const resp = await fetch(`${API_BASE}/v2/deep-think/sessions/${session.session_id}/rounds/stream`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(roundPayload),
          });
          if (!resp.ok) {
            const detail = await resp.text();
            throw new Error(detail || `HTTP ${resp.status}`);
          }
          let streamFailure = "";
          await readSSEAndConsume(resp, (eventName, payload) => {
            appendDeepEvent(eventName, payload);
            if (eventName === "done" && payload && payload.ok === false) {
              streamFailure = String(payload.error ?? payload.message ?? "DeepThink round failed");
            }
          });
          if (streamFailure) throw new Error(streamFailure);
        } catch (streamErr) {
          appendDeepEvent("progress", {
            stage: "fallback",
            message: `v2流式失败，回退v1：${streamErr instanceof Error ? streamErr.message : "unknown"}`,
          });
          const fallbackResp = await fetch(`${API_BASE}/v1/deep-think/sessions/${session.session_id}/rounds`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(roundPayload),
          });
          const fallbackBody = (await fallbackResp.json()) as DeepThinkSession;
          if (!fallbackResp.ok) throw new Error(`fallback round failed: HTTP ${fallbackResp.status}`);
          setDeepSession(fallbackBody);
          await replayDeepThinkStream(session.session_id);
        }
      } else {
        const fallbackResp = await fetch(`${API_BASE}/v1/deep-think/sessions/${session.session_id}/rounds`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(roundPayload),
        });
        const fallbackBody = (await fallbackResp.json()) as DeepThinkSession;
        if (!fallbackResp.ok) throw new Error(`round failed: HTTP ${fallbackResp.status}`);
        setDeepSession(fallbackBody);
        await replayDeepThinkStream(session.session_id);
      }

      const refreshResp = await fetch(`${API_BASE}/v1/deep-think/sessions/${session.session_id}`);
      const refreshBody = (await refreshResp.json()) as DeepThinkSession;
      if (!refreshResp.ok) throw new Error(`refresh session failed: HTTP ${refreshResp.status}`);
      setDeepSession(refreshBody);
    } catch (e) {
      setDeepError(e instanceof Error ? e.message : "执行 DeepThink 失败");
    } finally {
      setDeepRunning(false);
    }
  }

  async function runAll() {
    setError("");
    try {
      await ensureStockInUniverse([stockCode]);
      const [queryResult, deepResult] = await Promise.allSettled([runQueryStream(), runDeepThinkRound()]);
      const errs: string[] = [];
      if (queryResult.status === "rejected") errs.push(queryResult.reason instanceof Error ? queryResult.reason.message : "query/stream failed");
      if (deepResult.status === "rejected") errs.push(deepResult.reason instanceof Error ? deepResult.reason.message : "deep-think failed");
      if (errs.length) setError(errs.join(" | "));
    } catch (e) {
      setError(e instanceof Error ? e.message : "执行失败");
    }
  }

  return (
    <main className="container shell-fade-in">
      <section className={styles.page}>
        <motion.div initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} className={styles.hero}>
          <Space direction="vertical" size={10} style={{ width: "100%" }}>
            <Tag color="processing" style={{ width: "fit-content" }}>Analysis Studio</Tag>
            <Title level={2} className={styles.heroTitle}>联动分析工作台</Title>
            <Paragraph className={styles.heroDesc}>
              一键联动 `query/stream` 与 `DeepThink round`，输出结构化结论、证据与过程摘要。
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
                  <Button type="primary" size="large" onClick={runAll} loading={running}>一键联动执行</Button>
                  <Button href="/deep-think">业务分析页</Button>
                  <Button href="/deep-think/console">工程控制台</Button>
                  <Tag color={running ? "processing" : "default"}>{running ? "执行中" : "待执行"}</Tag>
                  {knowledgePersistedTraceId ? <Tag color="green">已沉淀共享语料</Tag> : null}
                </Space>
              </Space>
            </div>
          </Space>
        </motion.div>

        {error ? <Alert type="error" showIcon message={error} /> : null}
        {deepError ? <Alert type="warning" showIcon message={deepError} /> : null}

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
                <Text style={{ color: "#334155" }}>DeepThink 阶段：{DEEP_ROUND_STAGE_LABEL[deepCurrentStage]}</Text>
                <Progress percent={deepStagePercent} status={deepError ? "exception" : running ? "active" : "normal"} />
                <Text style={{ color: "#64748b" }}>问答阶段：{queryProgressText || "等待/完成"}</Text>
                <Text style={{ color: "#64748b" }}>轮次阶段：{deepProgressText || "等待/完成"}</Text>
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
            <DeepRoundSummaryPanel
              session={deepSession}
              stageLabel={DEEP_ROUND_STAGE_LABEL[deepCurrentStage]}
              stagePercent={deepStagePercent}
              running={deepRunning}
              deepError={deepError}
            />
            <StreamEventPanel events={deepEvents} />
          </div>
        </div>
      </section>
    </main>
  );
}
