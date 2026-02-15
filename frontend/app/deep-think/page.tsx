"use client";

import { useEffect, useMemo, useState } from "react";
import { motion } from "framer-motion";
import ReactECharts from "echarts-for-react";
import { Alert, Button, Card, Col, Input, List, Progress, Row, Space, Statistic, Table, Tag, Timeline, Typography } from "antd";
import MediaCarousel from "../components/MediaCarousel";
import StockSelectorModal from "../components/StockSelectorModal";

const API_BASE = process.env.NEXT_PUBLIC_API_BASE ?? "http://127.0.0.1:8000";
const HERO_VIDEO_URL =
  process.env.NEXT_PUBLIC_HERO_VIDEO_URL ??
  "/assets/media/hero-stock-analysis.mp4";
const { Title, Text, Paragraph } = Typography;
const { TextArea } = Input;

type Citation = { source_id: string; source_url: string; excerpt: string };
type QueryResponse = { trace_id: string; answer: string; citations: Citation[] };
type AnalysisBrief = {
  confidence_level: string;
  confidence_reason: string;
  citation_count: number;
  citation_avg_reliability: number;
  stocks: Array<{
    stock_code: string;
    history_sample_size: number;
    realtime: {
      price?: number;
      pct_change?: number;
      source_id?: string;
      ts?: string;
      freshness_seconds?: number | null;
    };
    trend: {
      ma20?: number;
      ma60?: number;
      ma20_slope?: number;
      ma60_slope?: number;
      momentum_20?: number;
      volatility_20?: number;
      max_drawdown_60?: number;
    };
  }>;
};
type MarketOverview = {
  stock_code: string;
  realtime: { price?: number; pct_change?: number; source_id?: string; source_url?: string; ts?: string };
  history: Array<{ trade_date: string; open: number; close: number; high: number; low: number }>;
  events: Array<{ event_time: string; title: string; source_id: string }>;
  trend: { ma20?: number; ma60?: number; momentum_20?: number; volatility_20?: number; max_drawdown_60?: number };
};
type FactorSnapshot = Record<string, number>;
type DeepThinkTask = { task_id: string; agent: string; title: string; priority: string };
type DeepThinkOpinion = {
  agent_id: string;
  signal: "buy" | "hold" | "reduce";
  confidence: number;
  reason: string;
  evidence_ids: string[];
  risk_tags: string[];
  created_at: string;
};
type DeepThinkBudgetUsage = {
  limit: { token_budget: number; time_budget_ms: number; tool_call_budget: number };
  used: { token_used: number; time_used_ms: number; tool_calls_used: number };
  remaining: { token_budget: number; time_budget_ms: number; tool_call_budget: number };
  warn: boolean;
  exceeded: boolean;
};
type DeepThinkRound = {
  round_id: string;
  session_id: string;
  round_no: number;
  status: string;
  consensus_signal: string;
  disagreement_score: number;
  conflict_sources: string[];
  counter_view: string;
  task_graph: DeepThinkTask[];
  replan_triggered: boolean;
  stop_reason: string;
  budget_usage: DeepThinkBudgetUsage;
  created_at: string;
  opinions: DeepThinkOpinion[];
};
type DeepThinkSession = {
  session_id: string;
  user_id: string;
  question: string;
  stock_codes: string[];
  agent_profile: string[];
  max_rounds: number;
  current_round: number;
  mode: string;
  status: string;
  trace_id: string;
  created_at: string;
  updated_at: string;
  budget: { token_budget: number; time_budget_ms: number; tool_call_budget: number };
  rounds: DeepThinkRound[];
};
type DeepThinkStreamEvent = { event: string; data: Record<string, any>; emitted_at: string };

function CountUp({ value, suffix = "" }: { value: number; suffix?: string }) {
  const [display, setDisplay] = useState(0);

  useEffect(() => {
    const start = performance.now();
    const from = display;
    const to = value;
    const duration = 700;
    let raf = 0;

    const tick = (now: number) => {
      const p = Math.min((now - start) / duration, 1);
      const eased = 1 - (1 - p) * (1 - p);
      const current = from + (to - from) * eased;
      setDisplay(current);
      if (p < 1) raf = requestAnimationFrame(tick);
    };

    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [value]);

  return <>{display.toFixed(2)}{suffix}</>;
}

export default function DeepThinkPage() {
  const [stockCode, setStockCode] = useState("SH600000");
  const [compareCode, setCompareCode] = useState("SZ000001");
  const [question, setQuestion] = useState("请结合实时数据与历史趋势，给出可验证的交易观察结论");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [result, setResult] = useState<QueryResponse | null>(null);
  const [overview, setOverview] = useState<MarketOverview | null>(null);
  const [factorData, setFactorData] = useState<FactorSnapshot | null>(null);
  const [factorLoading, setFactorLoading] = useState(false);
  const [showFactors, setShowFactors] = useState(false);
  const [compareOverview, setCompareOverview] = useState<MarketOverview | null>(null);
  const [compareLoading, setCompareLoading] = useState(false);
  const [streaming, setStreaming] = useState(false);
  const [streamSource, setStreamSource] = useState<"external_llm_stream" | "local_fallback_stream" | "unknown">("unknown");
  const [runtimeEngine, setRuntimeEngine] = useState("unknown");
  const [activeProvider, setActiveProvider] = useState("");
  const [activeModel, setActiveModel] = useState("");
  const [activeApiStyle, setActiveApiStyle] = useState("");
  const [analysisBrief, setAnalysisBrief] = useState<AnalysisBrief | null>(null);
  const [deepSession, setDeepSession] = useState<DeepThinkSession | null>(null);
  const [deepLoading, setDeepLoading] = useState(false);
  const [deepStreaming, setDeepStreaming] = useState(false);
  const [deepError, setDeepError] = useState("");
  const [deepStreamEvents, setDeepStreamEvents] = useState<DeepThinkStreamEvent[]>([]);
  const [deepLastA2ATask, setDeepLastA2ATask] = useState<{ task_id: string; status: string; agent_id: string } | null>(null);

  function formatDeepPercent(used: number, limit: number): number {
    const safeLimit = Number(limit) <= 0 ? 1 : Number(limit);
    const value = (Number(used) / safeLimit) * 100;
    return Math.max(0, Math.min(100, Number(value.toFixed(1))));
  }

  function appendDeepEvent(event: string, data: Record<string, any>) {
    setDeepStreamEvents((prev) => {
      const next = [...prev, { event, data, emitted_at: new Date().toISOString() }];
      return next.slice(-80);
    });
  }

  async function readSSEAndConsume(resp: Response, onEvent: (event: string, payload: Record<string, any>) => void) {
    if (!resp.body) throw new Error("浏览器不支持流式响应读取");
    const reader = resp.body.getReader();
    const decoder = new TextDecoder("utf-8");
    let buffer = "";
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      while (true) {
        const splitAt = buffer.indexOf("\n\n");
        if (splitAt < 0) break;
        const rawEvent = buffer.slice(0, splitAt);
        buffer = buffer.slice(splitAt + 2);
        const lines = rawEvent.split("\n");
        let eventName = "message";
        const dataLines: string[] = [];
        for (const line of lines) {
          if (line.startsWith("event:")) eventName = line.slice(6).trim();
          if (line.startsWith("data:")) dataLines.push(line.slice(5).trim());
        }
        if (!dataLines.length) continue;
        try {
          const payload = JSON.parse(dataLines.join("\n"));
          onEvent(eventName, payload as Record<string, any>);
        } catch {
          // ignore malformed event payload
        }
      }
    }
  }

  async function ensureStockInUniverse(codes: string[]) {
    for (const code of codes) {
      const normalized = code.trim().toUpperCase();
      const resp = await fetch(`${API_BASE}/v1/stocks/search?keyword=${encodeURIComponent(normalized)}&limit=30`);
      if (!resp.ok) throw new Error("股票库检索失败，请稍后重试");
      const rows = await resp.json();
      const hit = Array.isArray(rows) && rows.some((x: any) => String(x?.stock_code ?? "").toUpperCase() === normalized);
      if (!hit) throw new Error(`股票 ${normalized} 不在已同步股票库中，请重新选择`);
    }
  }

  async function runAnalysis() {
    setLoading(true);
    setStreaming(true);
    setStreamSource("unknown");
    setRuntimeEngine("unknown");
    setActiveProvider("");
    setActiveModel("");
    setActiveApiStyle("");
    setAnalysisBrief(null);
    setError("");
    try {
      await ensureStockInUniverse([stockCode]);
      // 同步请求结构化行情总览，和流式问答并行执行。
      const overviewPromise = fetch(`${API_BASE}/v1/market/overview/${stockCode}`);
      const streamResp = await fetch(`${API_BASE}/v1/query/stream`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ user_id: "frontend-user", stock_codes: [stockCode], question })
      });
      if (!streamResp.ok) {
        const txt = await streamResp.text();
        throw new Error(txt || `query/stream HTTP ${streamResp.status}`);
      }
      if (!streamResp.body) {
        throw new Error("浏览器不支持流式响应读取");
      }

      // 初始化结果容器，让前端边收边展示。
      setResult({ trace_id: "", answer: "", citations: [] });
      const reader = streamResp.body.getReader();
      const decoder = new TextDecoder("utf-8");
      let buffer = "";

      const consumeEvent = (raw: string) => {
        const lines = raw.split("\n");
        let eventName = "message";
        const dataLines: string[] = [];
        for (const line of lines) {
          if (line.startsWith("event:")) eventName = line.slice(6).trim();
          if (line.startsWith("data:")) dataLines.push(line.slice(5).trim());
        }
        if (!dataLines.length) return;
        let payload: any = {};
        try {
          payload = JSON.parse(dataLines.join("\n"));
        } catch {
          return;
        }
        if (eventName === "meta") {
          setResult((prev) => ({
            trace_id: String(payload?.trace_id ?? prev?.trace_id ?? ""),
            answer: String(prev?.answer ?? ""),
            citations: prev?.citations ?? []
          }));
          return;
        }
        if (eventName === "answer_delta") {
          const delta = String(payload?.delta ?? "");
          if (!delta) return;
          setResult((prev) => ({
            trace_id: prev?.trace_id ?? "",
            answer: `${prev?.answer ?? ""}${delta}`,
            citations: prev?.citations ?? []
          }));
          return;
        }
        if (eventName === "stream_source") {
          const src = String(payload?.source ?? "unknown");
          setActiveProvider(String(payload?.provider ?? ""));
          setActiveModel(String(payload?.model ?? ""));
          setActiveApiStyle(String(payload?.api_style ?? ""));
          if (src === "external_llm_stream" || src === "local_fallback_stream") {
            setStreamSource(src);
          } else {
            setStreamSource("unknown");
          }
          return;
        }
        if (eventName === "stream_runtime") {
          setRuntimeEngine(String(payload?.runtime ?? "unknown"));
          return;
        }
        if (eventName === "citations") {
          const citations = Array.isArray(payload?.citations) ? payload.citations : [];
          setResult((prev) => ({
            trace_id: prev?.trace_id ?? "",
            answer: prev?.answer ?? "",
            citations
          }));
          return;
        }
        if (eventName === "analysis_brief") {
          setAnalysisBrief(payload as AnalysisBrief);
        }
      };

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        while (true) {
          const splitAt = buffer.indexOf("\n\n");
          if (splitAt < 0) break;
          const rawEvent = buffer.slice(0, splitAt);
          buffer = buffer.slice(splitAt + 2);
          consumeEvent(rawEvent);
        }
      }

      const overviewResp = await overviewPromise;
      const overviewBody = await overviewResp.json();
      if (!overviewResp.ok) throw new Error(overviewBody?.detail ?? `overview HTTP ${overviewResp.status}`);
      setOverview(overviewBody as MarketOverview);
    } catch (e) {
      setError(e instanceof Error ? e.message : "请求失败");
      setResult(null);
      setOverview(null);
    } finally {
      setStreaming(false);
      setLoading(false);
    }
  }

  async function loadFactors() {
    setFactorLoading(true);
    try {
      await ensureStockInUniverse([stockCode]);
      const resp = await fetch(`${API_BASE}/v1/factors/${stockCode}`);
      const body = await resp.json();
      if (!resp.ok) throw new Error(body?.detail ?? `factors HTTP ${resp.status}`);
      setFactorData(body as FactorSnapshot);
      setShowFactors(true);
    } catch (e) {
      setError(e instanceof Error ? e.message : "因子请求失败");
      setFactorData(null);
      setShowFactors(false);
    } finally {
      setFactorLoading(false);
    }
  }

  async function runCompare() {
    setCompareLoading(true);
    try {
      await ensureStockInUniverse([stockCode, compareCode]);
      const resp = await fetch(`${API_BASE}/v1/market/overview/${compareCode}`);
      const body = await resp.json();
      if (!resp.ok) throw new Error(body?.detail ?? `compare HTTP ${resp.status}`);
      setCompareOverview(body as MarketOverview);
    } catch (e) {
      setError(e instanceof Error ? e.message : "对比请求失败");
      setCompareOverview(null);
    } finally {
      setCompareLoading(false);
    }
  }

  async function createDeepThinkSessionRequest(): Promise<DeepThinkSession> {
    const resp = await fetch(`${API_BASE}/v1/deep-think/sessions`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        user_id: "frontend-deepthink-user",
        question,
        stock_codes: [stockCode],
        max_rounds: 3
      })
    });
    const body = await resp.json();
    if (!resp.ok) {
      const detail = typeof body?.detail === "string" ? body.detail : body?.error ?? `HTTP ${resp.status}`;
      throw new Error(String(detail));
    }
    return body as DeepThinkSession;
  }

  async function startDeepThinkSession() {
    setDeepLoading(true);
    setDeepError("");
    try {
      await ensureStockInUniverse([stockCode]);
      const created = await createDeepThinkSessionRequest();
      setDeepSession(created);
      setDeepLastA2ATask(null);
      setDeepStreamEvents([]);
    } catch (e) {
      setDeepError(e instanceof Error ? e.message : "创建 DeepThink 会话失败");
    } finally {
      setDeepLoading(false);
    }
  }

  async function refreshDeepThinkSession() {
    if (!deepSession?.session_id) return;
    setDeepLoading(true);
    setDeepError("");
    try {
      const resp = await fetch(`${API_BASE}/v1/deep-think/sessions/${deepSession.session_id}`);
      const body = await resp.json();
      if (!resp.ok) throw new Error(body?.detail ?? `HTTP ${resp.status}`);
      setDeepSession(body as DeepThinkSession);
    } catch (e) {
      setDeepError(e instanceof Error ? e.message : "刷新 DeepThink 会话失败");
    } finally {
      setDeepLoading(false);
    }
  }

  async function replayDeepThinkStream(sessionId: string) {
    setDeepStreaming(true);
    setDeepError("");
    try {
      const resp = await fetch(`${API_BASE}/v1/deep-think/sessions/${sessionId}/stream`);
      if (!resp.ok) {
        const detail = await resp.text();
        throw new Error(detail || `HTTP ${resp.status}`);
      }
      await readSSEAndConsume(resp, (eventName, payload) => appendDeepEvent(eventName, payload));
    } catch (e) {
      setDeepError(e instanceof Error ? e.message : "回放 DeepThink 事件流失败");
    } finally {
      setDeepStreaming(false);
    }
  }

  async function runDeepThinkRound() {
    setDeepLoading(true);
    setDeepError("");
    try {
      await ensureStockInUniverse([stockCode]);
      const session = deepSession ?? (await createDeepThinkSessionRequest());
      const resp = await fetch(`${API_BASE}/v1/deep-think/sessions/${session.session_id}/rounds`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question, stock_codes: [stockCode] })
      });
      const body = await resp.json();
      if (!resp.ok) throw new Error(body?.detail ?? `HTTP ${resp.status}`);
      setDeepSession(body as DeepThinkSession);
      await replayDeepThinkStream(session.session_id);
    } catch (e) {
      setDeepError(e instanceof Error ? e.message : "执行 DeepThink 下一轮失败");
    } finally {
      setDeepLoading(false);
    }
  }

  async function runDeepThinkRoundViaA2A() {
    setDeepLoading(true);
    setDeepError("");
    try {
      await ensureStockInUniverse([stockCode]);
      const session = deepSession ?? (await createDeepThinkSessionRequest());
      const resp = await fetch(`${API_BASE}/v1/a2a/tasks`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          agent_id: "supervisor_agent",
          session_id: session.session_id,
          task_type: "deep_round",
          question
        })
      });
      const body = await resp.json();
      if (!resp.ok) throw new Error(body?.detail ?? body?.error ?? `HTTP ${resp.status}`);
      const snapshot = body?.result?.payload_result?.deep_think_snapshot;
      if (snapshot?.session_id) setDeepSession(snapshot as DeepThinkSession);
      setDeepLastA2ATask({
        task_id: String(body?.task_id ?? ""),
        status: String(body?.status ?? "unknown"),
        agent_id: String(body?.agent_id ?? "supervisor_agent")
      });
      await replayDeepThinkStream(session.session_id);
    } catch (e) {
      setDeepError(e instanceof Error ? e.message : "通过 A2A 执行 DeepThink 轮次失败");
    } finally {
      setDeepLoading(false);
    }
  }

  const trendOption = useMemo(() => {
    const bars = overview?.history ?? [];
    return {
      grid: { top: 16, left: 44, right: 16, bottom: 30 },
      tooltip: { trigger: "axis" },
      xAxis: {
        type: "category",
        data: bars.map((x) => x.trade_date),
        axisLabel: { color: "#475569", show: false },
        axisLine: { lineStyle: { color: "rgba(100,116,139,0.28)" } }
      },
      yAxis: {
        type: "value",
        axisLabel: { color: "#475569" },
        splitLine: { lineStyle: { color: "rgba(100,116,139,0.16)" } }
      },
      series: [
        {
          type: "line",
          data: bars.map((x) => x.close),
          smooth: true,
          showSymbol: false,
          lineStyle: { width: 2, color: "#2563eb" },
          areaStyle: {
            color: {
              type: "linear",
              x: 0,
              y: 0,
              x2: 0,
              y2: 1,
              colorStops: [
                { offset: 0, color: "rgba(37,99,235,0.24)" },
                { offset: 1, color: "rgba(37,99,235,0.02)" }
              ]
            }
          }
        }
      ]
    };
  }, [overview]);

  const pct = Number(overview?.realtime?.pct_change ?? 0);
  const barsCount = Number(overview?.history?.length ?? 0);
  const eventCount = Number(overview?.events?.length ?? 0);
  const bannerItems = [
    "实时行情聚合",
    "历史K线趋势",
    "公告事件时间线",
    "多Agent研判",
    "证据可追溯引用",
    "策略预测评测"
  ];
  const storySlides = [
    { src: "/assets/images/nyse-floor-2014.jpg", alt: "NYSE trading floor 2014", caption: "现代交易大厅" },
    { src: "/assets/images/nyse-floor-1963.jpg", alt: "NYSE trading floor 1963", caption: "历史市场情绪" },
    { src: "/assets/images/nyse-floor-1930.png", alt: "NYSE trading floor 1930", caption: "长期周期视角" }
  ];
  const sourceStatus = [
    { name: "Tencent 实时行情", state: overview?.realtime?.source_id ? "在线" : "待拉取" },
    { name: "Eastmoney 历史K线", state: barsCount > 0 ? "在线" : "待拉取" },
    { name: "公告事件聚合", state: eventCount > 0 ? "在线" : "待拉取" }
  ];
  const recentBars = (overview?.history ?? []).slice(-5).reverse();
  const quickFacts = [
    `MA20 ${Number(overview?.trend?.ma20 ?? 0).toFixed(2)}`,
    `MA60 ${Number(overview?.trend?.ma60 ?? 0).toFixed(2)}`,
    `动量 ${Number(overview?.trend?.momentum_20 ?? 0).toFixed(4)}`,
    `波动 ${Number(overview?.trend?.volatility_20 ?? 0).toFixed(4)}`,
    `回撤 ${Number(overview?.trend?.max_drawdown_60 ?? 0).toFixed(4)}`
  ];
  const factorRows = Object.entries(factorData ?? {})
    .slice(0, 16)
    .map(([k, v]) => ({ key: k, factor: k, value: Number(v).toFixed(6) }));
  const deepRounds = deepSession?.rounds ?? [];
  const latestDeepRound = deepRounds.length ? deepRounds[deepRounds.length - 1] : null;
  const latestBudget = latestDeepRound?.budget_usage;
  const deepTimelineItems = deepRounds.map((round) => ({
    color: round.replan_triggered ? "orange" : round.stop_reason ? "red" : "blue",
    children: (
      <Space direction="vertical" size={2}>
        <Text style={{ color: "#0f172a" }}>
          第 {round.round_no} 轮 | 共识={round.consensus_signal} | 分歧={Number(round.disagreement_score).toFixed(3)}
        </Text>
        <Space size={6} wrap>
          {round.replan_triggered ? <Tag color="orange">replan_triggered</Tag> : null}
          {round.stop_reason ? <Tag color="red">{round.stop_reason}</Tag> : null}
          {(round.conflict_sources ?? []).map((src) => (
            <Tag key={`${round.round_id}-${src}`} color="gold">
              {src}
            </Tag>
          ))}
        </Space>
      </Space>
    )
  }));
  const deepConflictOption = useMemo(() => {
    const rounds = deepSession?.rounds ?? [];
    return {
      grid: { top: 24, left: 44, right: 16, bottom: 30 },
      tooltip: { trigger: "axis" },
      legend: { data: ["分歧得分", "冲突源数量"], top: 0, textStyle: { color: "#475569", fontSize: 12 } },
      xAxis: {
        type: "category",
        data: rounds.map((x) => `R${x.round_no}`),
        axisLabel: { color: "#475569" },
        axisLine: { lineStyle: { color: "rgba(100,116,139,0.28)" } }
      },
      yAxis: [
        {
          type: "value",
          min: 0,
          max: 1,
          axisLabel: { color: "#475569" },
          splitLine: { lineStyle: { color: "rgba(100,116,139,0.16)" } }
        },
        {
          type: "value",
          min: 0,
          axisLabel: { color: "#64748b" },
          splitLine: { show: false }
        }
      ],
      series: [
        {
          name: "分歧得分",
          type: "line",
          smooth: true,
          yAxisIndex: 0,
          data: rounds.map((x) => Number(x.disagreement_score ?? 0)),
          lineStyle: { width: 2, color: "#2563eb" },
          showSymbol: true,
          symbolSize: 8
        },
        {
          name: "冲突源数量",
          type: "bar",
          yAxisIndex: 1,
          data: rounds.map((x) => Number((x.conflict_sources ?? []).length)),
          itemStyle: { color: "rgba(249, 115, 22, 0.7)" },
          barMaxWidth: 28
        }
      ]
    };
  }, [deepSession]);
  const deepTaskRows = (latestDeepRound?.task_graph ?? []).map((task) => ({
    key: task.task_id,
    task_id: task.task_id,
    agent: task.agent,
    title: task.title,
    priority: task.priority
  }));
  const deepOpinionRows = (latestDeepRound?.opinions ?? []).map((opinion) => ({
    key: `${opinion.agent_id}-${opinion.created_at}-${opinion.signal}`,
    agent_id: opinion.agent_id,
    signal: opinion.signal,
    confidence: Number(opinion.confidence ?? 0),
    risk_tags: (opinion.risk_tags ?? []).join(", "),
    reason: opinion.reason
  }));
  const deepTokenPercent = latestBudget ? formatDeepPercent(latestBudget.used.token_used, latestBudget.limit.token_budget) : 0;
  const deepTimePercent = latestBudget ? formatDeepPercent(latestBudget.used.time_used_ms, latestBudget.limit.time_budget_ms) : 0;
  const deepToolPercent = latestBudget ? formatDeepPercent(latestBudget.used.tool_calls_used, latestBudget.limit.tool_call_budget) : 0;

  return (
    <main className="container">
      <motion.section
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
        className="hero-video-section"
      >
        <video className="hero-video-media" autoPlay muted loop playsInline preload="metadata">
          <source src={HERO_VIDEO_URL} type="video/mp4" />
        </video>
        <div className="hero-video-mask" />
        <div className="hero-video-content">
          <Tag color="processing">Live Research Experience</Tag>
          <Title level={1} style={{ margin: 0, color: "#ffffff" }}>
            DeepThink 多Agent深度研判台
          </Title>
          <Paragraph style={{ margin: 0, color: "rgba(255,255,255,0.86)", maxWidth: 640 }}>
            这里聚焦深度分析执行链路：多角色协商、证据追溯、流式过程可见与结果可回放。
          </Paragraph>
          <Space>
            <Button type="primary" size="large" onClick={runAnalysis}>
              启动深度分析
            </Button>
            <Button size="large" ghost>
              查看能力链路
            </Button>
          </Space>
        </div>
      </motion.section>

      <motion.section
        initial={{ opacity: 0, y: 14 }}
        whileInView={{ opacity: 1, y: 0 }}
        viewport={{ once: true, amount: 0.3 }}
        transition={{ duration: 0.45 }}
        className="banner-marquee"
      >
        <div className="banner-track">
          {[...bannerItems, ...bannerItems].map((item, idx) => (
            <span key={`${item}-${idx}`} className="banner-chip">
              {item}
            </span>
          ))}
        </div>
      </motion.section>

      <motion.section
        initial={{ opacity: 0, y: 18 }}
        whileInView={{ opacity: 1, y: 0 }}
        viewport={{ once: true, amount: 0.25 }}
        transition={{ duration: 0.45 }}
      >
        <MediaCarousel items={storySlides} />
      </motion.section>

      <motion.section
        initial={{ opacity: 0, y: 14 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.4 }}
        style={{ marginBottom: 8 }}
      >
        <Card className="premium-card" style={{ background: "linear-gradient(132deg, rgba(255,255,255,0.98), rgba(246,249,252,0.95))" }}>
          <Space direction="vertical" size={8} style={{ width: "100%" }}>
            <Tag color="processing" style={{ width: "fit-content" }}>Agent-Native Research Workspace</Tag>
            <Title level={2} style={{ margin: 0, color: "#0f172a" }}>A股智能研判工作台</Title>
            <Paragraph style={{ margin: 0, color: "#475569", maxWidth: 760 }}>
              实时行情、历史趋势、事件时间线和可追溯证据在同一视图联动，避免“只给结论没有依据”的伪分析。
            </Paragraph>
          </Space>
        </Card>
      </motion.section>

      <motion.section
        initial={{ opacity: 0, y: 20 }}
        whileInView={{ opacity: 1, y: 0 }}
        viewport={{ once: true, amount: 0.3 }}
        transition={{ duration: 0.45 }}
        style={{ marginBottom: 8 }}
      >
        <Row gutter={[14, 14]}>
          <Col xs={24} md={8}>
            <Card className="premium-card">
              <Statistic title="最新涨跌幅" valueRender={() => <CountUp value={pct} suffix="%" />} valueStyle={{ color: pct >= 0 ? "#059669" : "#dc2626" }} />
            </Card>
          </Col>
          <Col xs={24} md={8}>
            <Card className="premium-card">
              <Statistic title="历史K线样本" valueRender={() => <CountUp value={barsCount} />} valueStyle={{ color: "#0f172a" }} />
            </Card>
          </Col>
          <Col xs={24} md={8}>
            <Card className="premium-card">
              <Statistic title="事件样本" valueRender={() => <CountUp value={eventCount} />} valueStyle={{ color: "#0f172a" }} />
            </Card>
          </Col>
        </Row>
      </motion.section>

      <motion.section
        initial={{ opacity: 0, y: 20 }}
        whileInView={{ opacity: 1, y: 0 }}
        viewport={{ once: true, amount: 0.2 }}
        transition={{ duration: 0.5 }}
      >
        <Row gutter={[14, 14]}>
          <Col xs={24} xl={14}>
            <Card className="premium-card" style={{ borderColor: "rgba(15,23,42,0.12)" }}>
              <Space direction="vertical" style={{ width: "100%" }}>
                <Text style={{ color: "#475569" }}>输入标的和问题，触发 DeepThink 联合推理</Text>
                <StockSelectorModal
                  value={stockCode}
                  onChange={(next: string | string[]) => setStockCode(Array.isArray(next) ? (next[0] ?? "") : next)}
                  title="选择分析标的"
                  placeholder="请先选择要分析的股票"
                />
                <TextArea rows={5} value={question} onChange={(e) => setQuestion(e.target.value)} />
                <Button type="primary" size="large" loading={loading} onClick={runAnalysis}>开始高级分析</Button>
                {streaming ? <Text style={{ color: "#2563eb" }}>流式输出中...</Text> : null}
              </Space>
            </Card>

            {error ? <Alert style={{ marginTop: 12 }} message={error} type="error" showIcon /> : null}

            {result ? (
              <Card className="premium-card" title={<span style={{ color: "#0f172a" }}>分析主报告（可验证）</span>} style={{ marginTop: 12 }}>
                <Space style={{ marginBottom: 8 }}>
                  <Tag color={streamSource === "external_llm_stream" ? "green" : streamSource === "local_fallback_stream" ? "gold" : "default"}>
                    {streamSource === "external_llm_stream"
                      ? "来源：外部LLM流式"
                      : streamSource === "local_fallback_stream"
                        ? "来源：本地回退流式"
                        : "来源：待识别"}
                  </Tag>
                  <Tag color={activeModel === "gpt-5.2" ? "blue" : "default"}>
                    模型：{activeModel || "unknown"}
                  </Tag>
                  <Tag color={runtimeEngine === "langgraph" ? "purple" : "default"}>Engine：{runtimeEngine}</Tag>
                  <Tag color="cyan">Provider：{activeProvider || "unknown"}</Tag>
                  <Tag>API：{activeApiStyle || "unknown"}</Tag>
                </Space>
                <pre style={{ whiteSpace: "pre-wrap", color: "#0f172a", margin: 0 }}>{result.answer}</pre>
                <Text style={{ color: "#64748b" }}>trace_id: {result.trace_id}</Text>
              </Card>
            ) : null}
          </Col>

          <Col xs={24} xl={10}>
            <Card className="premium-card" title={<span style={{ color: "#0f172a" }}>实时快照</span>}>
              <Row gutter={10}>
                <Col span={12}>
                  <Statistic title="最新价" value={overview?.realtime?.price ?? 0} precision={3} valueStyle={{ color: "#0f172a" }} />
                </Col>
                <Col span={12}>
                  <Statistic title="涨跌幅" value={pct} suffix="%" precision={2} valueStyle={{ color: pct >= 0 ? "#059669" : "#dc2626" }} />
                </Col>
              </Row>
              <Space style={{ marginTop: 8 }}>
                <Tag color="blue">{overview?.realtime?.source_id ?? "unknown"}</Tag>
                <Text style={{ color: "#64748b" }}>{overview?.realtime?.ts ?? "--"}</Text>
              </Space>
            </Card>

            <Card className="premium-card" title={<span style={{ color: "#0f172a" }}>历史趋势</span>} style={{ marginTop: 12 }}>
              <ReactECharts option={trendOption} style={{ height: 220 }} />
              <Space wrap>
                <Tag color="cyan">MA20 {(overview?.trend?.ma20 ?? 0).toFixed(2)}</Tag>
                <Tag color="geekblue">MA60 {(overview?.trend?.ma60 ?? 0).toFixed(2)}</Tag>
                <Tag color="gold">动量 {(overview?.trend?.momentum_20 ?? 0).toFixed(4)}</Tag>
                <Tag color="volcano">回撤 {(overview?.trend?.max_drawdown_60 ?? 0).toFixed(4)}</Tag>
              </Space>
            </Card>

            <Card className="premium-card" title={<span style={{ color: "#0f172a" }}>事件时间线</span>} style={{ marginTop: 12 }}>
              <Timeline
                items={(overview?.events ?? []).slice(-5).reverse().map((x) => ({
                  color: "blue",
                  children: `${x.event_time} | ${x.title} (${x.source_id})`
                }))}
              />
            </Card>

            {result?.citations?.length ? (
              <Card className="premium-card" title={<span style={{ color: "#0f172a" }}>证据引用</span>} style={{ marginTop: 12 }}>
                <List
                  size="small"
                  dataSource={result.citations}
                  renderItem={(x) => (
                    <List.Item>
                      <Space direction="vertical" size={1}>
                        <Text style={{ color: "#0f172a" }}>{x.source_id}</Text>
                        <a href={x.source_url} target="_blank" rel="noreferrer" style={{ color: "#2563eb" }}>{x.source_url}</a>
                        <Text style={{ color: "#64748b" }}>{x.excerpt}</Text>
                      </Space>
                    </List.Item>
                  )}
                />
              </Card>
            ) : null}

            {analysisBrief ? (
              <Card className="premium-card" title={<span style={{ color: "#0f172a" }}>分析置信度</span>} style={{ marginTop: 12 }}>
                <Space direction="vertical" style={{ width: "100%" }}>
                  <Space>
                    <Tag color={analysisBrief.confidence_level === "high" ? "green" : analysisBrief.confidence_level === "medium" ? "gold" : "red"}>
                      {analysisBrief.confidence_level}
                    </Tag>
                    <Tag>引用数: {analysisBrief.citation_count}</Tag>
                    <Tag>平均可信度: {analysisBrief.citation_avg_reliability.toFixed(2)}</Tag>
                  </Space>
                  <Text style={{ color: "#64748b" }}>{analysisBrief.confidence_reason}</Text>
                  {analysisBrief.stocks.slice(0, 2).map((s) => (
                    <Space key={s.stock_code} direction="vertical" style={{ width: "100%" }}>
                      <Text style={{ color: "#0f172a" }}>
                        {s.stock_code} | 样本={s.history_sample_size} | 数据新鲜度={s.realtime?.freshness_seconds ?? "-"}s
                      </Text>
                      <Text style={{ color: "#64748b" }}>
                        MA20={Number(s.trend?.ma20 ?? 0).toFixed(2)}, MA60={Number(s.trend?.ma60 ?? 0).toFixed(2)}, 动量20={Number(s.trend?.momentum_20 ?? 0).toFixed(4)}
                      </Text>
                    </Space>
                  ))}
                </Space>
              </Card>
            ) : null}
          </Col>
        </Row>
      </motion.section>

      <motion.section
        initial={{ opacity: 0, y: 18 }}
        whileInView={{ opacity: 1, y: 0 }}
        viewport={{ once: true, amount: 0.2 }}
        transition={{ duration: 0.45 }}
        style={{ marginTop: 8 }}
      >
        <Row gutter={[14, 14]}>
          <Col xs={24} xl={14}>
            <Card className="premium-card" title={<span style={{ color: "#0f172a" }}>DeepThink 轮次控制台</span>}>
              <Space direction="vertical" style={{ width: "100%" }}>
                <Text style={{ color: "#475569" }}>
                  基于后端 `/v1/deep-think/*` 接口按轮执行，展示 task graph、冲突源、预算消耗和重规划信号。
                </Text>
                <Space wrap>
                  <Button onClick={startDeepThinkSession} loading={deepLoading}>
                    新建会话
                  </Button>
                  <Button type="primary" onClick={runDeepThinkRound} loading={deepLoading}>
                    执行下一轮
                  </Button>
                  <Button onClick={runDeepThinkRoundViaA2A} loading={deepLoading}>
                    A2A派发下一轮
                  </Button>
                  <Button onClick={refreshDeepThinkSession} disabled={!deepSession?.session_id || deepLoading}>
                    刷新会话
                  </Button>
                  <Button onClick={() => replayDeepThinkStream(deepSession?.session_id ?? "")} disabled={!deepSession?.session_id} loading={deepStreaming}>
                    回放最新轮次流
                  </Button>
                </Space>
                <Space wrap>
                  <Tag color={deepSession ? "blue" : "default"}>session: {deepSession?.session_id ?? "未创建"}</Tag>
                  <Tag color={deepSession?.status === "completed" ? "green" : "processing"}>status: {deepSession?.status ?? "idle"}</Tag>
                  <Tag>round: {deepSession?.current_round ?? 0}/{deepSession?.max_rounds ?? 0}</Tag>
                  {latestDeepRound?.replan_triggered ? <Tag color="orange">replan_triggered</Tag> : null}
                  {latestDeepRound?.stop_reason ? <Tag color="red">{latestDeepRound.stop_reason}</Tag> : null}
                </Space>
                {deepLastA2ATask ? (
                  <Space wrap>
                    <Tag color="cyan">A2A Task: {deepLastA2ATask.task_id}</Tag>
                    <Tag color={deepLastA2ATask.status === "completed" ? "green" : "gold"}>A2A Status: {deepLastA2ATask.status}</Tag>
                    <Tag>A2A Agent: {deepLastA2ATask.agent_id}</Tag>
                  </Space>
                ) : null}
                {deepError ? <Alert message={deepError} type="error" showIcon /> : null}
              </Space>
            </Card>

            <Card className="premium-card" title={<span style={{ color: "#0f172a" }}>Round Timeline</span>} style={{ marginTop: 12 }}>
              {deepTimelineItems.length ? (
                <Timeline items={deepTimelineItems} />
              ) : (
                <Text style={{ color: "#64748b" }}>尚未执行 DeepThink 轮次。可先点击「新建会话」再执行。</Text>
              )}
            </Card>

            <Card className="premium-card" title={<span style={{ color: "#0f172a" }}>最新轮次 Task Graph</span>} style={{ marginTop: 12 }}>
              <Table
                size="small"
                pagination={false}
                locale={{ emptyText: "无任务图数据" }}
                dataSource={deepTaskRows}
                columns={[
                  { title: "Task ID", dataIndex: "task_id", key: "task_id", width: 110 },
                  { title: "Agent", dataIndex: "agent", key: "agent", width: 130 },
                  { title: "Title", dataIndex: "title", key: "title" },
                  {
                    title: "Priority",
                    dataIndex: "priority",
                    key: "priority",
                    width: 100,
                    render: (v: string) => <Tag color={v === "high" ? "red" : v === "medium" ? "gold" : "blue"}>{v}</Tag>
                  }
                ]}
              />
            </Card>
          </Col>

          <Col xs={24} xl={10}>
            <Card className="premium-card" title={<span style={{ color: "#0f172a" }}>预算使用与剩余</span>}>
              {latestBudget ? (
                <Space direction="vertical" style={{ width: "100%" }} size={10}>
                  <div>
                    <Text style={{ color: "#334155" }}>Token: {latestBudget.used.token_used} / {latestBudget.limit.token_budget}</Text>
                    <Progress percent={deepTokenPercent} status={latestBudget.exceeded ? "exception" : latestBudget.warn ? "active" : "normal"} />
                  </div>
                  <div>
                    <Text style={{ color: "#334155" }}>Time(ms): {latestBudget.used.time_used_ms} / {latestBudget.limit.time_budget_ms}</Text>
                    <Progress percent={deepTimePercent} status={latestBudget.exceeded ? "exception" : latestBudget.warn ? "active" : "normal"} />
                  </div>
                  <div>
                    <Text style={{ color: "#334155" }}>Tool Calls: {latestBudget.used.tool_calls_used} / {latestBudget.limit.tool_call_budget}</Text>
                    <Progress percent={deepToolPercent} status={latestBudget.exceeded ? "exception" : latestBudget.warn ? "active" : "normal"} />
                  </div>
                  <Space wrap>
                    <Tag color={latestBudget.warn ? "gold" : "green"}>warn: {String(latestBudget.warn)}</Tag>
                    <Tag color={latestBudget.exceeded ? "red" : "green"}>exceeded: {String(latestBudget.exceeded)}</Tag>
                  </Space>
                </Space>
              ) : (
                <Text style={{ color: "#64748b" }}>执行轮次后展示预算治理指标。</Text>
              )}
            </Card>

            <Card className="premium-card" title={<span style={{ color: "#0f172a" }}>冲突源可视化</span>} style={{ marginTop: 12 }}>
              <ReactECharts option={deepConflictOption} style={{ height: 240 }} />
              <Space wrap>
                {(latestDeepRound?.conflict_sources ?? []).map((src) => (
                  <Tag key={src} color="orange">{src}</Tag>
                ))}
                {!(latestDeepRound?.conflict_sources ?? []).length ? <Tag>暂无冲突源</Tag> : null}
              </Space>
            </Card>

            <Card className="premium-card" title={<span style={{ color: "#0f172a" }}>最新轮次 Agent 观点</span>} style={{ marginTop: 12 }}>
              <Table
                size="small"
                pagination={false}
                locale={{ emptyText: "暂无观点数据" }}
                dataSource={deepOpinionRows}
                columns={[
                  { title: "Agent", dataIndex: "agent_id", key: "agent_id", width: 126 },
                  {
                    title: "Signal",
                    dataIndex: "signal",
                    key: "signal",
                    width: 92,
                    render: (v: string) => <Tag color={v === "buy" ? "green" : v === "reduce" ? "red" : "blue"}>{v}</Tag>
                  },
                  {
                    title: "Confidence",
                    dataIndex: "confidence",
                    key: "confidence",
                    width: 104,
                    render: (v: number) => v.toFixed(3)
                  }
                ]}
              />
            </Card>

            <Card className="premium-card" title={<span style={{ color: "#0f172a" }}>SSE 回放事件</span>} style={{ marginTop: 12 }}>
              <List
                size="small"
                locale={{ emptyText: "暂无流事件记录" }}
                dataSource={[...deepStreamEvents].reverse().slice(0, 16)}
                renderItem={(item) => (
                  <List.Item>
                    <Space direction="vertical" size={1}>
                      <Space>
                        <Tag color="processing">{item.event}</Tag>
                        <Text style={{ color: "#64748b" }}>{item.emitted_at}</Text>
                      </Space>
                      <Text style={{ color: "#475569" }}>
                        {JSON.stringify(item.data).slice(0, 180)}
                      </Text>
                    </Space>
                  </List.Item>
                )}
              />
            </Card>
          </Col>
        </Row>
      </motion.section>

      <motion.section
        initial={{ opacity: 0, y: 18 }}
        whileInView={{ opacity: 1, y: 0 }}
        viewport={{ once: true, amount: 0.2 }}
        transition={{ duration: 0.45 }}
        style={{ marginTop: 8 }}
      >
        <Row gutter={[14, 14]}>
          <Col xs={24} xl={14}>
            <Card className="premium-card" title={<span style={{ color: "#0f172a" }}>最近5个交易日明细</span>}>
              <Table
                size="small"
                pagination={false}
                dataSource={recentBars.map((b) => ({ key: b.trade_date, ...b }))}
                columns={[
                  { title: "日期", dataIndex: "trade_date", key: "trade_date" },
                  { title: "开", dataIndex: "open", key: "open" },
                  { title: "收", dataIndex: "close", key: "close" },
                  { title: "高", dataIndex: "high", key: "high" },
                  { title: "低", dataIndex: "low", key: "low" }
                ]}
              />
            </Card>
          </Col>
          <Col xs={24} xl={10}>
            <Card className="premium-card" title={<span style={{ color: "#0f172a" }}>关键指标速览</span>}>
              <Space wrap>
                {quickFacts.map((f) => (
                  <Tag key={f} color="blue">
                    {f}
                  </Tag>
                ))}
              </Space>
            </Card>
          </Col>
        </Row>
      </motion.section>

      <motion.section
        initial={{ opacity: 0, y: 16 }}
        whileInView={{ opacity: 1, y: 0 }}
        viewport={{ once: true, amount: 0.2 }}
        transition={{ duration: 0.4 }}
        style={{ marginTop: 8 }}
      >
        <Row gutter={[14, 14]}>
          <Col xs={24} xl={12}>
            <Card className="premium-card" title={<span style={{ color: "#0f172a" }}>数据源状态</span>}>
              <Space direction="vertical" style={{ width: "100%" }}>
                {sourceStatus.map((s) => (
                  <div key={s.name} style={{ display: "flex", justifyContent: "space-between" }}>
                    <Text style={{ color: "#334155" }}>{s.name}</Text>
                    <Tag color={s.state === "在线" ? "green" : "gold"}>{s.state}</Tag>
                  </div>
                ))}
              </Space>
            </Card>
          </Col>
          <Col xs={24} xl={12}>
            <Card className="premium-card" title={<span style={{ color: "#0f172a" }}>多 Agent 执行链</span>}>
              <Space wrap>
                <Tag color="processing">Intent Agent</Tag>
                <Tag color="processing">Data Agent</Tag>
                <Tag color="processing">RAG Agent</Tag>
                <Tag color="processing">Risk Agent</Tag>
                <Tag color="processing">Report Agent</Tag>
              </Space>
            </Card>
          </Col>
        </Row>
      </motion.section>

      <motion.section
        initial={{ opacity: 0, y: 16 }}
        whileInView={{ opacity: 1, y: 0 }}
        viewport={{ once: true, amount: 0.2 }}
        transition={{ duration: 0.4 }}
        style={{ marginTop: 8 }}
      >
        <Row gutter={[14, 14]}>
          <Col xs={24} xl={12}>
            <Card className="premium-card" title={<span style={{ color: "#0f172a" }}>财务/因子面板</span>}>
              <Space style={{ width: "100%", justifyContent: "space-between" }}>
                <Text style={{ color: "#475569" }}>按当前标的加载因子快照（用于策略解释）</Text>
                <Button type="primary" loading={factorLoading} onClick={loadFactors}>
                  加载因子
                </Button>
              </Space>
              {showFactors && factorRows.length > 0 ? (
                <Table
                  size="small"
                  pagination={false}
                  style={{ marginTop: 8 }}
                  dataSource={factorRows}
                  columns={[
                    { title: "因子", dataIndex: "factor", key: "factor" },
                    { title: "值", dataIndex: "value", key: "value" }
                  ]}
                />
              ) : null}
            </Card>
          </Col>
          <Col xs={24} xl={12}>
            <Card className="premium-card" title={<span style={{ color: "#0f172a" }}>双股票快速对比</span>}>
              <Space direction="vertical" style={{ width: "100%" }}>
                <StockSelectorModal
                  value={compareCode}
                  onChange={(next: string | string[]) => setCompareCode(Array.isArray(next) ? (next[0] ?? "") : next)}
                  title="选择对比标的"
                  placeholder="请先选择要对比的股票"
                />
                <Button type="primary" loading={compareLoading} onClick={runCompare}>
                  对比
                </Button>
              </Space>
              <Row gutter={10} style={{ marginTop: 10 }}>
                <Col span={12}>
                  <Card className="premium-card" title={<span style={{ color: "#334155" }}>{stockCode}</span>}>
                    <Text>价: {Number(overview?.realtime?.price ?? 0).toFixed(3)}</Text>
                    <br />
                    <Text>涨跌: {Number(overview?.realtime?.pct_change ?? 0).toFixed(2)}%</Text>
                    <br />
                    <Text>MA20: {Number(overview?.trend?.ma20 ?? 0).toFixed(2)}</Text>
                  </Card>
                </Col>
                <Col span={12}>
                  <Card className="premium-card" title={<span style={{ color: "#334155" }}>{compareCode}</span>}>
                    <Text>价: {Number(compareOverview?.realtime?.price ?? 0).toFixed(3)}</Text>
                    <br />
                    <Text>涨跌: {Number(compareOverview?.realtime?.pct_change ?? 0).toFixed(2)}%</Text>
                    <br />
                    <Text>MA20: {Number(compareOverview?.trend?.ma20 ?? 0).toFixed(2)}</Text>
                  </Card>
                </Col>
              </Row>
            </Card>
          </Col>
        </Row>
      </motion.section>
    </main>
  );
}

