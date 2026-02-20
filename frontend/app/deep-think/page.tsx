"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { motion } from "framer-motion";
import ReactECharts from "echarts-for-react";
import { Alert, Button, Card, Col, Input, InputNumber, List, Popover, Progress, Row, Segmented, Select, Space, Statistic, Table, Tag, Timeline, Typography } from "antd";
import MediaCarousel from "../components/MediaCarousel";
import StockSelectorModal from "../components/StockSelectorModal";
import { validatePromptQuality } from "../lib/analysis/guardrails";
import { composeStructuredQuestion } from "../lib/analysis/template-compose";
import { ANALYSIS_TEMPLATES, type AnalysisTemplateId, type HorizonOption, type PositionStateOption, type RiskProfileOption } from "../lib/analysis/template-config";

const API_BASE = process.env.NEXT_PUBLIC_API_BASE ?? "http://127.0.0.1:8000";
// 默认开启 V2 真流式；设置 NEXT_PUBLIC_DEEPTHINK_V2_STREAM=0 可临时回退 V1。
const ENABLE_DEEPTHINK_V2_STREAM = process.env.NEXT_PUBLIC_DEEPTHINK_V2_STREAM !== "0";
const HERO_VIDEO_URL =
  process.env.NEXT_PUBLIC_HERO_VIDEO_URL ??
  "/assets/media/hero-stock-analysis.mp4";
const { Title, Text, Paragraph } = Typography;
const { TextArea } = Input;

type Citation = {
  source_id: string;
  source_url: string;
  excerpt: string;
  retrieval_track?: string;
  rerank_score?: number;
  reliability_score?: number;
  event_time?: string;
};
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
type DeepThinkEventArchiveSnapshot = {
  session_id: string;
  round_id: string;
  event_name: string;
  cursor: number;
  limit: number;
  created_from: string;
  created_to: string;
  has_more: boolean;
  next_cursor: number | null;
  count: number;
  events: Array<{
    event_id?: number;
    session_id: string;
    round_id: string;
    round_no: number;
    event_seq: number;
    event: string;
    data: Record<string, any>;
    created_at: string;
  }>;
};
type DeepThinkArchiveLoadOptions = {
  roundId?: string;
  eventName?: string;
  limit?: number;
  cursor?: number;
  createdFrom?: string;
  createdTo?: string;
  historyMode?: "keep" | "push" | "reset";
};
type DeepThinkExportTaskSnapshot = {
  task_id: string;
  session_id: string;
  status: "queued" | "running" | "completed" | "failed";
  format: "jsonl" | "csv";
  filename: string;
  media_type: string;
  row_count: number;
  attempt_count: number;
  max_attempts: number;
  error?: string;
  failure_reason?: string;
  created_at: string;
  updated_at: string;
  completed_at?: string;
  download_ready: boolean;
};

type IntelCardEvent = {
  date: string;
  title: string;
  event_type: string;
  source_id: string;
};

type IntelCardEvidence = {
  kind: string;
  title: string;
  summary: string;
  source_id: string;
  source_url: string;
  event_time: string;
  reliability_score: number;
  retrieval_track: string;
  rerank_score?: number | null;
};

type IntelCardScenario = {
  scenario: string;
  expected_return_pct: number;
  probability: number;
};

type IntelCardSnapshot = {
  stock_code: string;
  time_horizon: "7d" | "30d" | "90d";
  horizon_days: number;
  risk_profile: "conservative" | "neutral" | "aggressive";
  overall_signal: "buy" | "hold" | "reduce";
  confidence: number;
  risk_level: "low" | "medium" | "high";
  position_hint: string;
  market_snapshot: {
    price: number;
    pct_change: number;
    ma20: number;
    ma60: number;
    momentum_20: number;
    volatility_20: number;
    max_drawdown_60: number;
    main_inflow: number;
  };
  key_catalysts: Array<Record<string, any>>;
  risk_watch: Array<Record<string, any>>;
  event_calendar: IntelCardEvent[];
  scenario_matrix: IntelCardScenario[];
  evidence: IntelCardEvidence[];
  trigger_conditions: string[];
  invalidation_conditions: string[];
  execution_plan: {
    entry_mode: string;
    cadence_hint: string;
    max_single_step_pct: number;
    max_position_cap: string;
    stop_loss_hint_pct: number;
    recheck_interval_hours: number;
  };
  risk_thresholds: {
    volatility_20_max: number;
    max_drawdown_60_max: number;
    min_evidence_count: number;
    max_data_staleness_minutes: number;
  };
  degrade_status: {
    level: "normal" | "watch" | "degraded";
    reasons: string[];
  };
  next_review_time: string;
  data_freshness: Record<string, number | null>;
};

type IntelReviewSnapshot = {
  stock_code: string;
  count: number;
  stats: Record<string, { count: number; avg_return: number | null; hit_rate: number | null }>;
  items: Array<Record<string, any>>;
};

type DeepConsoleMode = "analysis" | "engineering";
type DeepRoundStageKey = "idle" | "planning" | "data_refresh" | "intel_search" | "debate" | "arbitration" | "persist" | "done";
type DeepThinkWorkspace = "business" | "console";
type DeepFullFlowStage = "idle" | "query" | "round" | "done" | "error";

const DEEP_ROUND_STAGE_ORDER: DeepRoundStageKey[] = ["idle", "planning", "data_refresh", "intel_search", "debate", "arbitration", "persist", "done"];
const DEEP_ROUND_STAGE_LABEL: Record<DeepRoundStageKey, string> = {
  idle: "待执行",
  planning: "任务规划",
  data_refresh: "数据刷新",
  intel_search: "实时情报检索",
  debate: "多Agent协商",
  arbitration: "仲裁收敛",
  persist: "结果落库",
  done: "执行完成",
};
const DEEP_FULL_FLOW_ORDER: DeepFullFlowStage[] = ["idle", "query", "round", "done"];
const DEEP_FULL_FLOW_STAGE_LABEL: Record<DeepFullFlowStage, string> = {
  idle: "待执行",
  query: "流式分析",
  round: "轮次研判",
  done: "执行完成",
  error: "执行失败",
};

type AgentMeta = { name: string; role: string };
// 统一的 Agent 中文展示映射：让业务用户能看懂每个角色在做什么。
const AGENT_META_MAP: Record<string, AgentMeta> = {
  supervisor_agent: { name: "监督仲裁Agent", role: "统筹轮次推进并输出最终仲裁结论" },
  pm_agent: { name: "主题叙事Agent", role: "评估题材逻辑与叙事一致性" },
  quant_agent: { name: "量化评估Agent", role: "评估估值、收益风险比与概率信号" },
  risk_agent: { name: "风险控制Agent", role: "评估回撤、波动与下行风险" },
  critic_agent: { name: "质检复核Agent", role: "检查证据完整性与逻辑一致性" },
  macro_agent: { name: "宏观研判Agent", role: "评估政策与宏观冲击影响" },
  execution_agent: { name: "执行策略Agent", role: "评估仓位节奏与执行约束" },
  compliance_agent: { name: "合规审查Agent", role: "评估合规边界与表达风险" },
};

function getAgentMeta(agentId: string): AgentMeta {
  return AGENT_META_MAP[agentId] ?? { name: agentId, role: "未配置角色说明" };
}

function getSignalLabel(signal: string): string {
  if (signal === "buy") return "增配";
  if (signal === "reduce") return "减配";
  if (signal === "hold") return "持有";
  return signal || "未知";
}

function getPriorityLabel(priority: string): string {
  if (priority === "high") return "高";
  if (priority === "medium") return "中";
  if (priority === "low") return "低";
  return priority || "未知";
}

function getConflictSourceLabel(source: string): string {
  const mapping: Record<string, string> = {
    signal_conflict: "信号分歧",
    signal_divergence: "信号分歧",
    confidence_gap: "置信度分化",
    confidence_divergence: "置信度分化",
    risk_veto: "风险否决",
    compliance_veto: "合规否决",
    evidence_gap: "证据缺口",
    evidence_conflict: "证据冲突",
    budget_limit: "预算约束",
  };
  return mapping[source] ?? source;
}

function normalizeArchiveTimestamp(raw: string): string {
  const value = String(raw ?? "").trim();
  if (!value) return "";
  const normalized = value.replace("T", " ");
  const match = normalized.match(/^(\d{4}-\d{2}-\d{2}) (\d{2}):(\d{2})(?::(\d{2}))?$/);
  if (!match) return normalized;
  const sec = match[4] ?? "00";
  return `${match[1]} ${match[2]}:${match[3]}:${sec}`;
}

function toDatetimeLocalValue(raw: string): string {
  const normalized = normalizeArchiveTimestamp(raw);
  const match = normalized.match(/^(\d{4}-\d{2}-\d{2}) (\d{2}):(\d{2}):(\d{2})$/);
  if (!match) return "";
  return `${match[1]}T${match[2]}:${match[3]}:${match[4]}`;
}

function formatArchiveNow(date: Date): string {
  const y = date.getFullYear();
  const mo = String(date.getMonth() + 1).padStart(2, "0");
  const d = String(date.getDate()).padStart(2, "0");
  const h = String(date.getHours()).padStart(2, "0");
  const mi = String(date.getMinutes()).padStart(2, "0");
  const s = String(date.getSeconds()).padStart(2, "0");
  return `${y}-${mo}-${d} ${h}:${mi}:${s}`;
}

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
  const [workspace, setWorkspace] = useState<DeepThinkWorkspace>("business");
  useEffect(() => {
    // Client-side URL parsing keeps app router page typed as standard Page component.
    const params = new URLSearchParams(window.location.search);
    setWorkspace(params.get("workspace") === "console" ? "console" : "business");
  }, []);
  const isConsoleWorkspace = workspace === "console";
  const [stockCode, setStockCode] = useState("SH600000");
  const [compareCode, setCompareCode] = useState("SZ000001");
  const [question, setQuestion] = useState(
    composeStructuredQuestion({
      templateId: "mid_term_trend_risk",
      stockCode: "SH600000",
      horizon: "30d",
      riskProfile: "neutral",
      positionState: "flat",
    }),
  );
  // 模板优先：默认引导用户通过固定模板构建问题，减少自由输入导致的低质量请求。
  const [questionInputMode, setQuestionInputMode] = useState<"template" | "free">("template");
  const [selectedTemplateId, setSelectedTemplateId] = useState<AnalysisTemplateId>("mid_term_trend_risk");
  const [selectedHorizon, setSelectedHorizon] = useState<HorizonOption>("30d");
  const [selectedRiskProfile, setSelectedRiskProfile] = useState<RiskProfileOption>("neutral");
  const [selectedPositionState, setSelectedPositionState] = useState<PositionStateOption>("flat");
  const [promptQualityScore, setPromptQualityScore] = useState(100);
  const [promptGuardrailWarnings, setPromptGuardrailWarnings] = useState<string[]>([]);
  const [loading, setLoading] = useState(false);
  // 统一编排状态：将“流式分析 + DeepThink轮次”串成一次业务动作。
  const [fullFlowRunning, setFullFlowRunning] = useState(false);
  const [fullFlowStage, setFullFlowStage] = useState<DeepFullFlowStage>("idle");
  const [fullFlowMessage, setFullFlowMessage] = useState("");
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
  const [queryProgressText, setQueryProgressText] = useState("");
  const [analysisBrief, setAnalysisBrief] = useState<AnalysisBrief | null>(null);
  const [intelCard, setIntelCard] = useState<IntelCardSnapshot | null>(null);
  const [intelCardLoading, setIntelCardLoading] = useState(false);
  const [intelCardError, setIntelCardError] = useState("");
  const [intelCardHorizon, setIntelCardHorizon] = useState<"7d" | "30d" | "90d">("30d");
  const [intelCardRiskProfile, setIntelCardRiskProfile] = useState<"conservative" | "neutral" | "aggressive">("neutral");
  const [intelFeedbackLoading, setIntelFeedbackLoading] = useState(false);
  const [intelFeedbackMessage, setIntelFeedbackMessage] = useState("");
  const [intelReviewLoading, setIntelReviewLoading] = useState(false);
  const [intelReview, setIntelReview] = useState<IntelReviewSnapshot | null>(null);
  // query/stream 在后端落库后会发 knowledge_persisted，前端据此显示“已沉淀共享语料”反馈。
  const [knowledgePersistedTraceId, setKnowledgePersistedTraceId] = useState("");
  const [deepSession, setDeepSession] = useState<DeepThinkSession | null>(null);
  const [deepLoading, setDeepLoading] = useState(false);
  const [deepStreaming, setDeepStreaming] = useState(false);
  const [deepError, setDeepError] = useState("");
  const [deepStreamEvents, setDeepStreamEvents] = useState<DeepThinkStreamEvent[]>([]);
  // 三层反馈中的“阶段反馈”文本，会在流事件消费时持续更新。
  const [deepProgressText, setDeepProgressText] = useState("");
  const [deepLastA2ATask, setDeepLastA2ATask] = useState<{ task_id: string; status: string; agent_id: string } | null>(null);
  const [deepArchiveLoading, setDeepArchiveLoading] = useState(false);
  const [deepArchiveCount, setDeepArchiveCount] = useState(0);
  const [deepArchiveRoundId, setDeepArchiveRoundId] = useState("");
  const [deepArchiveEventName, setDeepArchiveEventName] = useState("");
  const [deepArchiveLimit, setDeepArchiveLimit] = useState(220);
  const [deepArchiveCursor, setDeepArchiveCursor] = useState(0);
  const [deepArchiveHasMore, setDeepArchiveHasMore] = useState(false);
  const [deepArchiveNextCursor, setDeepArchiveNextCursor] = useState<number | null>(null);
  const [deepArchiveCreatedFrom, setDeepArchiveCreatedFrom] = useState("");
  const [deepArchiveCreatedTo, setDeepArchiveCreatedTo] = useState("");
  const [deepArchiveCursorHistory, setDeepArchiveCursorHistory] = useState<number[]>([]);
  const [deepArchiveExporting, setDeepArchiveExporting] = useState(false);
  const [deepArchiveExportTask, setDeepArchiveExportTask] = useState<DeepThinkExportTaskSnapshot | null>(null);
  // 实时情报链路自检结果：用于快速确认 external/websearch/fallback 状态。
  const [deepIntelProbe, setDeepIntelProbe] = useState<Record<string, any> | null>(null);
  const [deepIntelProbeLoading, setDeepIntelProbeLoading] = useState(false);
  // 股票切换后提示：明确告知 DeepThink 数据已按标的隔离重置。
  const [deepStockSwitchNotice, setDeepStockSwitchNotice] = useState("");
  const deepTrackedStockRef = useRef(stockCode);
  const analysisPanelRef = useRef<HTMLDivElement | null>(null);
  const effectiveConsoleMode: DeepConsoleMode = isConsoleWorkspace ? "engineering" : "analysis";

  useEffect(() => {
    if (questionInputMode !== "template") return;
    // 模板槽位变化时自动重组问题，保证默认输入具备可执行语义。
    const next = composeStructuredQuestion({
      templateId: selectedTemplateId,
      stockCode,
      horizon: selectedHorizon,
      riskProfile: selectedRiskProfile,
      positionState: selectedPositionState,
    });
    setQuestion(next);
  }, [questionInputMode, selectedTemplateId, selectedHorizon, selectedRiskProfile, selectedPositionState, stockCode]);

  useEffect(() => {
    const guard = validatePromptQuality({
      stockCode,
      question,
      horizon: selectedHorizon,
    });
    setPromptQualityScore(guard.score);
    setPromptGuardrailWarnings([...guard.errors, ...guard.warnings, ...guard.suggestions].slice(0, 4));
  }, [stockCode, question, selectedHorizon]);

  function formatDeepPercent(used: number, limit: number): number {
    const safeLimit = Number(limit) <= 0 ? 1 : Number(limit);
    const value = (Number(used) / safeLimit) * 100;
    return Math.max(0, Math.min(100, Number(value.toFixed(1))));
  }

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

  function appendDeepEvent(event: string, data: Record<string, any>) {
    setDeepStreamEvents((prev) => {
      const next = [...prev, { event, data, emitted_at: new Date().toISOString() }];
      return next.slice(-80);
    });
    // 将底层流事件翻译成用户可感知的阶段反馈，避免“在跑但无反馈”的体验。
    const stage = resolveDeepStageFromEvent(event, data);
    if (stage !== "idle") {
      const message = String(data?.message ?? data?.reason ?? "").trim();
      setDeepProgressText(message || DEEP_ROUND_STAGE_LABEL[stage]);
    }
    if (String(event) === "done") {
      setDeepProgressText("轮次执行完成，结果已同步。");
    }
    if (String(event) === "error") {
      setDeepProgressText("轮次执行失败，请查看错误信息。");
    }
  }

  function setDeepArchiveQuickWindow(hours: number) {
    const now = new Date();
    const from = new Date(now.getTime() - Math.max(1, hours) * 60 * 60 * 1000);
    setDeepArchiveCreatedFrom(formatArchiveNow(from));
    setDeepArchiveCreatedTo(formatArchiveNow(now));
  }

  async function readSSEAndConsume(resp: Response, onEvent: (event: string, payload: Record<string, any>) => void) {
    if (!resp.body) throw new Error("浏览器不支持流式响应读取");
    const reader = resp.body.getReader();
    const decoder = new TextDecoder("utf-8");
    let buffer = "";
    // 兼容不同网关/代理下的分隔符：\n\n 或 \r\n\r\n。
    const nextEventSplitAt = (text: string): { at: number; len: number } => {
      const lf = text.indexOf("\n\n");
      const crlf = text.indexOf("\r\n\r\n");
      if (lf < 0 && crlf < 0) return { at: -1, len: 0 };
      if (lf >= 0 && (crlf < 0 || lf <= crlf)) return { at: lf, len: 2 };
      return { at: crlf, len: 4 };
    };
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      while (true) {
        const split = nextEventSplitAt(buffer);
        if (split.at < 0) break;
        const rawEvent = buffer.slice(0, split.at);
        buffer = buffer.slice(split.at + split.len);
        const lines = rawEvent.split("\n");
        let eventName = "message";
        const dataLines: string[] = [];
        for (const line of lines) {
          const normalized = line.replace(/\r$/, "");
          if (normalized.startsWith("event:")) eventName = normalized.slice(6).trim();
          if (normalized.startsWith("data:")) dataLines.push(normalized.slice(5).trim());
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
    // 连接结束时尝试消费尾包（某些实现不会在末尾补全空行分隔）。
    const tail = buffer.trim();
    if (tail) {
      const lines = tail.split("\n");
      let eventName = "message";
      const dataLines: string[] = [];
      for (const line of lines) {
        const normalized = line.replace(/\r$/, "");
        if (normalized.startsWith("event:")) eventName = normalized.slice(6).trim();
        if (normalized.startsWith("data:")) dataLines.push(normalized.slice(5).trim());
      }
      if (dataLines.length) {
        try {
          const payload = JSON.parse(dataLines.join("\n"));
          onEvent(eventName, payload as Record<string, any>);
        } catch {
          // ignore malformed tail payload
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

  async function loadIntelCard(options?: { code?: string; showLoading?: boolean; skipUniverseCheck?: boolean }) {
    const code = String(options?.code ?? stockCode).trim().toUpperCase();
    const showLoading = options?.showLoading !== false;
    const skipUniverseCheck = Boolean(options?.skipUniverseCheck);
    if (showLoading) setIntelCardLoading(true);
    setIntelCardError("");
    try {
      if (!skipUniverseCheck) {
        await ensureStockInUniverse([code]);
      }
      const params = new URLSearchParams({
        stock_code: code,
        horizon: intelCardHorizon,
        risk_profile: intelCardRiskProfile,
      });
      const resp = await fetch(`${API_BASE}/v1/analysis/intel-card?${params.toString()}`);
      const body = await resp.json();
      if (!resp.ok) throw new Error(body?.detail ?? `intel-card HTTP ${resp.status}`);
      setIntelCard(body as IntelCardSnapshot);
      void loadIntelReview(code, false);
    } catch (e) {
      setIntelCardError(e instanceof Error ? e.message : "业务情报卡片加载失败");
    } finally {
      if (showLoading) setIntelCardLoading(false);
    }
  }

  async function loadIntelReview(code?: string, showLoading = true) {
    const target = String(code ?? stockCode).trim().toUpperCase();
    if (showLoading) setIntelReviewLoading(true);
    try {
      const resp = await fetch(`${API_BASE}/v1/analysis/intel-card/review?stock_code=${encodeURIComponent(target)}&limit=50`);
      const body = await resp.json();
      if (!resp.ok) throw new Error(body?.detail ?? `intel-review HTTP ${resp.status}`);
      setIntelReview(body as IntelReviewSnapshot);
    } catch (e) {
      setIntelFeedbackMessage(e instanceof Error ? e.message : "复盘统计加载失败");
    } finally {
      if (showLoading) setIntelReviewLoading(false);
    }
  }

  async function submitIntelFeedback(signal: "adopt" | "watch" | "reject") {
    if (!intelCard) return;
    setIntelFeedbackLoading(true);
    setIntelFeedbackMessage("");
    try {
      const payload = {
        stock_code: stockCode,
        trace_id: result?.trace_id ?? "",
        signal: intelCard.overall_signal,
        confidence: intelCard.confidence,
        position_hint: intelCard.position_hint,
        feedback: signal,
      };
      const resp = await fetch(`${API_BASE}/v1/analysis/intel-card/feedback`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      const body = await resp.json();
      if (!resp.ok) throw new Error(body?.detail ?? `intel-feedback HTTP ${resp.status}`);
      setIntelFeedbackMessage(`反馈已记录：${signal}`);
      await loadIntelReview(stockCode, false);
    } catch (e) {
      setIntelFeedbackMessage(e instanceof Error ? e.message : "反馈提交失败");
    } finally {
      setIntelFeedbackLoading(false);
    }
  }

  async function runAnalysis(questionOverride?: string): Promise<boolean> {
    const submitQuestion = String(questionOverride ?? resolveComposedQuestion()).trim();
    const guard = validatePromptQuality({
      stockCode,
      question: submitQuestion,
      horizon: selectedHorizon,
    });
    if (guard.errors.length) {
      setError(guard.errors.join("；"));
      return false;
    }
    setLoading(true);
    setStreaming(true);
    setStreamSource("unknown");
    setRuntimeEngine("unknown");
    setActiveProvider("");
    setActiveModel("");
    setActiveApiStyle("");
    setQueryProgressText("");
    setAnalysisBrief(null);
    setIntelCardError("");
    setKnowledgePersistedTraceId("");
    setError("");
    try {
      await ensureStockInUniverse([stockCode]);
      // 提交前统一收敛为本次真实问题，确保 query/deep-think 后续链路一致。
      setQuestion(submitQuestion);
      // 与 query stream 并行刷新业务卡片，缩短“等待后才看到业务结论”的感知延迟。
      const intelCardPromise = loadIntelCard({ code: stockCode, showLoading: false, skipUniverseCheck: true });
      // 同步请求结构化行情总览，和流式问答并行执行。
      const overviewPromise = fetch(`${API_BASE}/v1/market/overview/${stockCode}`);
      const streamResp = await fetch(`${API_BASE}/v1/query/stream`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ user_id: "frontend-user", stock_codes: [stockCode], question: submitQuestion })
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
      let gotEvent = false;
      const consumeEvent = (eventName: string, payload: any) => {
        gotEvent = true;
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
          return;
        }
        if (eventName === "knowledge_persisted") {
          const trace = String(payload?.trace_id ?? "").trim();
          if (trace) setKnowledgePersistedTraceId(trace);
          setQueryProgressText("回答已沉淀到共享语料，可用于后续检索复用");
        }
      };
      // 复用统一 SSE 读取器，避免 query/deep-think 两套解析逻辑分叉。
      await readSSEAndConsume(streamResp, consumeEvent);
      if (!gotEvent) {
        throw new Error("query/stream 连接成功但未收到事件");
      }

      const overviewResp = await overviewPromise;
      const overviewBody = await overviewResp.json();
      if (!overviewResp.ok) throw new Error(overviewBody?.detail ?? `overview HTTP ${overviewResp.status}`);
      setOverview(overviewBody as MarketOverview);
      await intelCardPromise;
      setQueryProgressText("");
      return true;
    } catch (e) {
      setError(e instanceof Error ? e.message : "请求失败");
      setResult(null);
      setOverview(null);
      setQueryProgressText("");
      setKnowledgePersistedTraceId("");
      return false;
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

  useEffect(() => {
    // 切换标的后先清空旧卡片，避免把前一只股票的结论误读为当前标的。
    setIntelCard(null);
    setIntelCardError("");
    setIntelReview(null);
    setIntelFeedbackMessage("");
  }, [stockCode]);

  function resolveComposedQuestion(): string {
    if (questionInputMode !== "template") return String(question).trim();
    return composeStructuredQuestion({
      templateId: selectedTemplateId,
      stockCode,
      horizon: selectedHorizon,
      riskProfile: selectedRiskProfile,
      positionState: selectedPositionState,
    }).trim();
  }

  function scrollToAnalysisPanel() {
    analysisPanelRef.current?.scrollIntoView({ behavior: "smooth", block: "start" });
  }

  async function createDeepThinkSessionRequest(): Promise<DeepThinkSession> {
    const composedQuestion = resolveComposedQuestion();
    const resp = await fetch(`${API_BASE}/v1/deep-think/sessions`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        user_id: "frontend-deepthink-user",
        question: composedQuestion,
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

  async function loadDeepThinkEventArchive(
    sessionId: string,
    options?: DeepThinkArchiveLoadOptions
  ) {
    if (!sessionId) return;
    setDeepArchiveLoading(true);
    try {
      const roundId = String(options?.roundId ?? deepArchiveRoundId).trim();
      const eventName = String(options?.eventName ?? deepArchiveEventName).trim();
      const limitRaw = Number(options?.limit ?? deepArchiveLimit);
      const limit = Number.isFinite(limitRaw) ? Math.max(20, Math.min(2000, Math.floor(limitRaw))) : 220;
      const cursorRaw = Number(options?.cursor ?? 0);
      const cursor = Number.isFinite(cursorRaw) ? Math.max(0, Math.floor(cursorRaw)) : 0;
      const createdFrom = String(options?.createdFrom ?? deepArchiveCreatedFrom).trim();
      const createdTo = String(options?.createdTo ?? deepArchiveCreatedTo).trim();
      const historyMode = options?.historyMode ?? "keep";
      if (historyMode === "reset") setDeepArchiveCursorHistory([]);
      if (historyMode === "push") setDeepArchiveCursorHistory((prev) => [...prev, deepArchiveCursor]);
      const params = new URLSearchParams();
      if (roundId) params.set("round_id", roundId);
      if (eventName) params.set("event_name", eventName);
      if (cursor > 0) params.set("cursor", String(cursor));
      if (createdFrom) params.set("created_from", createdFrom);
      if (createdTo) params.set("created_to", createdTo);
      params.set("limit", String(limit));
      const qs = params.toString();
      const resp = await fetch(`${API_BASE}/v1/deep-think/sessions/${sessionId}/events${qs ? `?${qs}` : ""}`);
      const body = await resp.json();
      if (!resp.ok) throw new Error(body?.detail ?? `HTTP ${resp.status}`);
      const payload = body as DeepThinkEventArchiveSnapshot;
      const rows = Array.isArray(payload.events) ? payload.events : [];
      const mappedRows = rows.map((row) => ({
        event: String(row.event ?? "message"),
        data: (row.data ?? {}) as Record<string, any>,
        emitted_at: String(row.created_at ?? "")
      }));
      const payloadCount = Number(payload.count ?? rows.length);
      setDeepArchiveCount(payloadCount);
      setDeepArchiveRoundId(roundId);
      setDeepArchiveEventName(eventName);
      setDeepArchiveLimit(limit);
      setDeepArchiveCreatedFrom(createdFrom);
      setDeepArchiveCreatedTo(createdTo);
      setDeepArchiveCursor(Number(payload.cursor ?? cursor));
      setDeepArchiveHasMore(Boolean(payload.has_more));
      setDeepArchiveNextCursor(payload.next_cursor == null ? null : Number(payload.next_cursor));
      setDeepStreamEvents(mappedRows);
    } catch (e) {
      setDeepError(e instanceof Error ? e.message : "加载 DeepThink 事件存档失败");
    } finally {
      setDeepArchiveLoading(false);
    }
  }

  async function loadNextDeepThinkArchivePage() {
    if (!deepSession?.session_id) return;
    if (!deepArchiveHasMore || !deepArchiveNextCursor) return;
    await loadDeepThinkEventArchive(deepSession.session_id, {
      roundId: deepArchiveRoundId,
      eventName: deepArchiveEventName,
      limit: deepArchiveLimit,
      cursor: deepArchiveNextCursor,
      createdFrom: deepArchiveCreatedFrom,
      createdTo: deepArchiveCreatedTo,
      historyMode: "push"
    });
  }

  async function loadPrevDeepThinkArchivePage() {
    if (!deepSession?.session_id) return;
    if (!deepArchiveCursorHistory.length) return;
    const prevCursor = deepArchiveCursorHistory[deepArchiveCursorHistory.length - 1];
    setDeepArchiveCursorHistory((prev) => prev.slice(0, -1));
    await loadDeepThinkEventArchive(deepSession.session_id, {
      roundId: deepArchiveRoundId,
      eventName: deepArchiveEventName,
      limit: deepArchiveLimit,
      cursor: prevCursor,
      createdFrom: deepArchiveCreatedFrom,
      createdTo: deepArchiveCreatedTo,
      historyMode: "keep"
    });
  }

  async function loadFirstDeepThinkArchivePage() {
    if (!deepSession?.session_id) return;
    await loadDeepThinkEventArchive(deepSession.session_id, {
      roundId: deepArchiveRoundId,
      eventName: deepArchiveEventName,
      limit: deepArchiveLimit,
      cursor: 0,
      createdFrom: deepArchiveCreatedFrom,
      createdTo: deepArchiveCreatedTo,
      historyMode: "reset"
    });
  }

  async function pollDeepThinkExportTask(sessionId: string, taskId: string): Promise<DeepThinkExportTaskSnapshot> {
    for (let idx = 0; idx < 45; idx += 1) {
      const resp = await fetch(`${API_BASE}/v1/deep-think/sessions/${sessionId}/events/export-tasks/${taskId}`);
      const body = await resp.json();
      if (!resp.ok) {
        const detail = typeof body?.detail === "string" ? body.detail : JSON.stringify(body?.detail ?? body);
        throw new Error(detail || `HTTP ${resp.status}`);
      }
      const snapshot = body as DeepThinkExportTaskSnapshot;
      setDeepArchiveExportTask(snapshot);
      if (snapshot.status === "completed") return snapshot;
      if (snapshot.status === "failed") throw new Error(snapshot.failure_reason || snapshot.error || "导出任务执行失败");
      await new Promise((resolve) => setTimeout(resolve, 800));
    }
    throw new Error("导出任务超时，请稍后在任务状态中重试下载");
  }

  async function downloadDeepThinkExportTask(sessionId: string, taskId: string, format: "jsonl" | "csv") {
    const resp = await fetch(`${API_BASE}/v1/deep-think/sessions/${sessionId}/events/export-tasks/${taskId}/download`);
    if (!resp.ok) {
      const detail = await resp.text();
      throw new Error(detail || `HTTP ${resp.status}`);
    }
    const blob = await resp.blob();
    const disposition = resp.headers.get("Content-Disposition") ?? "";
    const match = disposition.match(/filename=\"?([^\";]+)\"?/i);
    const filename = (match?.[1] || `deepthink-events-${sessionId}.${format}`).trim();
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = filename;
    document.body.appendChild(anchor);
    anchor.click();
    anchor.remove();
    URL.revokeObjectURL(url);
  }

  async function exportDeepThinkBusiness(format: "csv" | "json") {
    if (!deepSession?.session_id) return;
    setDeepArchiveExporting(true);
    try {
      const params = new URLSearchParams();
      params.set("format", format);
      if (deepArchiveRoundId) params.set("round_id", deepArchiveRoundId);
      params.set("limit", String(Math.max(20, Math.min(2000, Math.floor(deepArchiveLimit)))));
      const resp = await fetch(`${API_BASE}/v1/deep-think/sessions/${deepSession.session_id}/business-export?${params.toString()}`);
      if (!resp.ok) {
        const detail = await resp.text();
        throw new Error(detail || `HTTP ${resp.status}`);
      }
      const blob = await resp.blob();
      const disposition = resp.headers.get("Content-Disposition") ?? "";
      const match = disposition.match(/filename=\"?([^\";]+)\"?/i);
      const filename = (match?.[1] || `deepthink-business-${deepSession.session_id}.${format}`).trim();
      const url = URL.createObjectURL(blob);
      const anchor = document.createElement("a");
      anchor.href = url;
      anchor.download = filename;
      document.body.appendChild(anchor);
      anchor.click();
      anchor.remove();
      URL.revokeObjectURL(url);
    } catch (e) {
      setDeepError(e instanceof Error ? e.message : "导出 DeepThink 业务摘要失败");
    } finally {
      setDeepArchiveExporting(false);
    }
  }

  async function exportDeepThinkEventArchive(format: "jsonl" | "csv") {
    if (!deepSession?.session_id) return;
    setDeepArchiveExporting(true);
    try {
      const payload = {
        format,
        round_id: deepArchiveRoundId || "",
        event_name: deepArchiveEventName || "",
        limit: Math.max(20, Math.min(2000, Math.floor(deepArchiveLimit))),
        cursor: deepArchiveCursor > 0 ? deepArchiveCursor : 0,
        created_from: deepArchiveCreatedFrom || "",
        created_to: deepArchiveCreatedTo || ""
      };
      const createResp = await fetch(`${API_BASE}/v1/deep-think/sessions/${deepSession.session_id}/events/export-tasks`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
      });
      const createBody = await createResp.json();
      if (!createResp.ok) {
        const detail = typeof createBody?.detail === "string" ? createBody.detail : JSON.stringify(createBody?.detail ?? createBody);
        throw new Error(detail || `HTTP ${createResp.status}`);
      }
      const created = createBody as DeepThinkExportTaskSnapshot;
      setDeepArchiveExportTask(created);
      const done = created.status === "completed" ? created : await pollDeepThinkExportTask(deepSession.session_id, created.task_id);
      await downloadDeepThinkExportTask(deepSession.session_id, done.task_id, format);
    } catch (e) {
      setDeepError(e instanceof Error ? e.message : "导出 DeepThink 事件存档失败");
    } finally {
      setDeepArchiveExporting(false);
    }
  }

  async function startDeepThinkSession() {
    setDeepLoading(true);
    setDeepError("");
    setDeepStockSwitchNotice("");
    setDeepProgressText("正在创建会话...");
    try {
      await ensureStockInUniverse([stockCode]);
      const created = await createDeepThinkSessionRequest();
      setDeepSession(created);
      setDeepLastA2ATask(null);
      setDeepStreamEvents([]);
      setDeepArchiveCount(0);
      setDeepArchiveCursor(0);
      setDeepArchiveHasMore(false);
      setDeepArchiveNextCursor(null);
      setDeepArchiveCursorHistory([]);
      setDeepArchiveExportTask(null);
      await loadDeepThinkEventArchive(created.session_id, { cursor: 0, historyMode: "reset" });
    } catch (e) {
      setDeepError(e instanceof Error ? e.message : "创建 DeepThink 会话失败");
      setDeepProgressText("会话创建失败。");
    } finally {
      setDeepLoading(false);
    }
  }

  async function refreshDeepThinkSession() {
    if (!deepSession?.session_id) return;
    setDeepLoading(true);
    setDeepError("");
    setDeepProgressText("正在刷新会话快照...");
    try {
      const resp = await fetch(`${API_BASE}/v1/deep-think/sessions/${deepSession.session_id}`);
      const body = await resp.json();
      if (!resp.ok) throw new Error(body?.detail ?? `HTTP ${resp.status}`);
      const loaded = body as DeepThinkSession;
      setDeepSession(loaded);
      const latest = loaded.rounds?.length ? loaded.rounds[loaded.rounds.length - 1] : null;
      setDeepArchiveExportTask(null);
      await loadDeepThinkEventArchive(loaded.session_id, { roundId: String(latest?.round_id ?? ""), cursor: 0, historyMode: "reset" });
    } catch (e) {
      setDeepError(e instanceof Error ? e.message : "刷新 DeepThink 会话失败");
      setDeepProgressText("刷新失败。");
    } finally {
      setDeepLoading(false);
    }
  }

  async function replayDeepThinkStream(sessionId: string) {
    setDeepStreaming(true);
    setDeepError("");
    setDeepProgressText("正在回放最新轮次事件...");
    try {
      const resp = await fetch(`${API_BASE}/v1/deep-think/sessions/${sessionId}/stream`);
      if (!resp.ok) {
        const detail = await resp.text();
        throw new Error(detail || `HTTP ${resp.status}`);
      }
      await readSSEAndConsume(resp, (eventName, payload) => appendDeepEvent(eventName, payload));
      await loadDeepThinkEventArchive(sessionId, { cursor: 0, historyMode: "reset" });
    } catch (e) {
      setDeepError(e instanceof Error ? e.message : "回放 DeepThink 事件流失败");
      setDeepProgressText("回放失败，请检查会话与网络状态。");
    } finally {
      setDeepStreaming(false);
    }
  }

  async function runDeepThinkRoundStreamV2(sessionId: string, requestPayload: Record<string, any>) {
    // V2 路径：单个请求内完成“执行 + 流式推送”，不再先等 round 完成再回放。
    setDeepStreaming(true);
    setDeepError("");
    setDeepProgressText("已连接流式执行通道，等待首个事件...");
    let streamFailure = "";
    let gotEvent = false;
    try {
      const resp = await fetch(`${API_BASE}/v2/deep-think/sessions/${sessionId}/rounds/stream`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(requestPayload)
      });
      if (!resp.ok) {
        const detail = await resp.text();
        throw new Error(detail || `HTTP ${resp.status}`);
      }
      await readSSEAndConsume(resp, (eventName, payload) => {
        // 所有流事件都写入回放面板，便于在线排障与复盘。
        gotEvent = true;
        appendDeepEvent(eventName, payload);
        // done(ok=false) 统一视作本轮失败，外层再抛错中断流程。
        if (eventName === "done" && payload && payload.ok === false) {
          streamFailure = String(payload.error ?? payload.message ?? "DeepThink round failed");
        }
      });
      // 保护性兜底：HTTP成功但没有任何事件时，按异常处理并触发回退。
      if (!gotEvent) {
        throw new Error("v2 stream connected but no events received");
      }
      if (streamFailure) throw new Error(streamFailure);
    } finally {
      setDeepStreaming(false);
    }
  }

  async function runDeepThinkRoundV1Fallback(sessionId: string, roundPayload: Record<string, any>) {
    // 回退路径：保持旧行为，先执行 round 再通过 stream 回放事件。
    const resp = await fetch(`${API_BASE}/v1/deep-think/sessions/${sessionId}/rounds`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(roundPayload)
    });
    const body = await resp.json();
    if (!resp.ok) throw new Error(body?.detail ?? `HTTP ${resp.status}`);
    setDeepSession(body as DeepThinkSession);
    await replayDeepThinkStream(sessionId);
  }

  async function runDeepThinkRound(): Promise<boolean> {
    setDeepLoading(true);
    setDeepError("");
    setDeepStockSwitchNotice("");
    setDeepProgressText("准备执行下一轮 DeepThink...");
    try {
      await ensureStockInUniverse([stockCode]);
      const autoCreateSession = !deepSession;
      const session = deepSession ?? (await createDeepThinkSessionRequest());
      if (autoCreateSession) {
        // 执行下一轮可自动建会话，业务页无需暴露“新建会话”按钮。
        setDeepProgressText("未检测到会话，已自动创建并开始执行...");
      }
      setDeepSession(session);
      setDeepStreamEvents([]);
      // 每次执行前清空事件过滤，避免“过滤条件隐藏实时事件”的误判。
      setDeepArchiveEventName("");
      const composedQuestion = resolveComposedQuestion();
      setQuestion(composedQuestion);

      const roundPayload = {
        question: composedQuestion,
        stock_codes: [stockCode],
        archive_max_events: deepArchiveLimit
      };
      if (ENABLE_DEEPTHINK_V2_STREAM) {
        // 默认走 V2 真流式：用户可在执行过程中持续看到事件。
        try {
          await runDeepThinkRoundStreamV2(session.session_id, roundPayload);
        } catch (streamErr) {
          // 若 v2 失败，自动回退 v1，保证本轮任务仍能执行完成。
          const reason = streamErr instanceof Error ? streamErr.message : "unknown stream error";
          appendDeepEvent("progress", { stage: "fallback", message: `v2流式失败，回退v1：${reason}` });
          await runDeepThinkRoundV1Fallback(session.session_id, roundPayload);
        }
      } else {
        await runDeepThinkRoundV1Fallback(session.session_id, roundPayload);
      }

      // 无论 V1/V2，最后都刷新一次会话快照并重载归档，确保前端状态与后端一致。
      const refreshResp = await fetch(`${API_BASE}/v1/deep-think/sessions/${session.session_id}`);
      const refreshBody = await refreshResp.json();
      if (!refreshResp.ok) throw new Error(refreshBody?.detail ?? `HTTP ${refreshResp.status}`);
      const loaded = refreshBody as DeepThinkSession;
      setDeepSession(loaded);
      const latest = loaded.rounds?.length ? loaded.rounds[loaded.rounds.length - 1] : null;
      await loadDeepThinkEventArchive(loaded.session_id, { roundId: String(latest?.round_id ?? ""), cursor: 0, historyMode: "reset" });
      return true;
    } catch (e) {
      setDeepError(e instanceof Error ? e.message : "执行 DeepThink 下一轮失败");
      setDeepProgressText("执行失败，请调整问题或稍后重试。");
      return false;
    } finally {
      setDeepLoading(false);
    }
  }

  // 一键全流程：先跑流式分析，再推进 DeepThink 轮次，避免用户在两个区域来回点击。
  async function runDeepThinkFullFlow(questionOverride?: string) {
    const submitQuestion = String(questionOverride ?? resolveComposedQuestion()).trim();
    if (!submitQuestion) {
      setError("请输入分析问题后再执行。");
      return;
    }
    if (fullFlowRunning) return;
    setFullFlowRunning(true);
    setFullFlowStage("query");
    setFullFlowMessage("步骤 1/2：正在执行流式分析...");
    setError("");
    setDeepError("");
    try {
      const queryOk = await runAnalysis(submitQuestion);
      if (!queryOk) {
        setFullFlowStage("error");
        setFullFlowMessage("流式分析失败，请检查输入或稍后重试。");
        return;
      }

      setFullFlowStage("round");
      setFullFlowMessage("步骤 2/2：正在执行 DeepThink 下一轮...");
      const roundOk = await runDeepThinkRound();
      if (!roundOk) {
        setFullFlowStage("error");
        setFullFlowMessage("DeepThink 轮次执行失败，请稍后重试。");
        return;
      }

      setFullFlowStage("done");
      setFullFlowMessage("全流程执行完成，已生成结论与证据。");
    } finally {
      setFullFlowRunning(false);
    }
  }

  async function runDeepIntelSelfTest() {
    setDeepIntelProbeLoading(true);
    setDeepError("");
    try {
      const params = new URLSearchParams({
        stock_code: stockCode,
        question: question || `请自检 ${stockCode} 的实时情报链路`,
      });
      const resp = await fetch(`${API_BASE}/v1/deep-think/intel/self-test?${params.toString()}`);
      const body = await resp.json();
      if (!resp.ok) throw new Error(body?.detail ?? `HTTP ${resp.status}`);
      setDeepIntelProbe(body as Record<string, any>);
      appendDeepEvent("intel_self_test", body as Record<string, any>);
      setDeepProgressText("情报链路自检已完成。");
    } catch (e) {
      setDeepError(e instanceof Error ? e.message : "情报链路自检失败");
    } finally {
      setDeepIntelProbeLoading(false);
    }
  }

  async function runDeepThinkRoundViaA2A() {
    setDeepLoading(true);
    setDeepError("");
    setDeepProgressText("正在通过 A2A 派发执行任务...");
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
      setDeepProgressText("A2A 任务执行失败。");
    } finally {
      setDeepLoading(false);
    }
  }

  useEffect(() => {
    if (!deepSession) {
      setDeepArchiveRoundId("");
      setDeepArchiveCursor(0);
      setDeepArchiveHasMore(false);
      setDeepArchiveNextCursor(null);
      setDeepArchiveCursorHistory([]);
      setDeepArchiveExportTask(null);
      setDeepProgressText("");
      return;
    }
    const roundIds = new Set((deepSession.rounds ?? []).map((x) => String(x.round_id)));
    setDeepArchiveRoundId((prev) => {
      if (prev && roundIds.has(prev)) return prev;
      const latestId = deepSession.rounds?.length ? String(deepSession.rounds[deepSession.rounds.length - 1].round_id) : "";
      return latestId;
    });
  }, [deepSession]);

  useEffect(() => {
    const prevStock = deepTrackedStockRef.current;
    if (prevStock === stockCode) return;
    deepTrackedStockRef.current = stockCode;
    const hasDeepThinkState = Boolean(
      deepSession || deepStreamEvents.length || deepArchiveCount > 0 || deepLastA2ATask || deepIntelProbe
    );
    if (!hasDeepThinkState) return;
    // 切换标的后自动清理 DeepThink 上下文，避免跨股票串用旧轮次数据。
    setDeepSession(null);
    setDeepStreamEvents([]);
    setDeepLastA2ATask(null);
    setDeepArchiveCount(0);
    setDeepArchiveRoundId("");
    setDeepArchiveEventName("");
    setDeepArchiveCursor(0);
    setDeepArchiveHasMore(false);
    setDeepArchiveNextCursor(null);
    setDeepArchiveCursorHistory([]);
    setDeepArchiveExportTask(null);
    setDeepIntelProbe(null);
    setDeepProgressText("已切换标的，DeepThink 数据已自动清空，请重新执行。");
    setDeepStockSwitchNotice(`已从 ${prevStock} 切换到 ${stockCode}，DeepThink 会话已清空。`);
  }, [stockCode, deepArchiveCount, deepIntelProbe, deepLastA2ATask, deepSession, deepStreamEvents.length]);

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
  // 固定展示最近三个月连续样本摘要，避免用户只能从长文本里找样本覆盖信息。
  const history3mSummary = useMemo(() => {
    const bars = [...(overview?.history ?? [])].sort((a, b) => String(a.trade_date).localeCompare(String(b.trade_date)));
    const window = bars.slice(-90);
    if (window.length < 2) return null;
    const start = window[0];
    const end = window[window.length - 1];
    const startClose = Number(start.close ?? 0);
    const endClose = Number(end.close ?? 0);
    const pctChange = startClose > 0 ? ((endClose / startClose) - 1) * 100 : 0;
    return {
      sampleCount: window.length,
      totalCount: bars.length,
      startDate: String(start.trade_date ?? ""),
      endDate: String(end.trade_date ?? ""),
      startClose,
      endClose,
      pctChange,
    };
  }, [overview]);
  const recentFiveTradeRows = useMemo(() => {
    const bars = [...(overview?.history ?? [])].sort((a, b) => String(a.trade_date).localeCompare(String(b.trade_date)));
    const recent = bars.slice(-5).reverse();
    return recent.map((row, idx) => {
      const open = Number(row.open ?? 0);
      const close = Number(row.close ?? 0);
      const pctChange = open > 0 ? ((close / open) - 1) * 100 : 0;
      return {
        key: `${row.trade_date}-${idx}`,
        trade_date: String(row.trade_date ?? ""),
        open,
        close,
        high: Number(row.high ?? 0),
        low: Number(row.low ?? 0),
        pct_change: pctChange,
      };
    });
  }, [overview]);
  // 业务语义：共享语料命中表示本次回答引用了“文档库/RAG问答记忆”，而不只依赖即时上下文。
  const sharedKnowledgeHits = useMemo(() => {
    const rows = result?.citations ?? [];
    const filtered = rows.filter((x) => {
      const sourceId = String(x?.source_id ?? "");
      return sourceId.startsWith("doc::") || sourceId === "qa_memory_summary";
    });
    const dedup = new Map<string, Citation>();
    for (const item of filtered) {
      const key = `${String(item.source_id ?? "")}|${String(item.source_url ?? "")}`;
      if (!dedup.has(key)) dedup.set(key, item);
    }
    return Array.from(dedup.values()).slice(0, 6);
  }, [result]);
  const intelEvidenceRows = useMemo(
    () =>
      (intelCard?.evidence ?? []).map((row, idx) => ({
        ...row,
        key: `${row.source_id}-${row.event_time}-${idx}`,
      })),
    [intelCard]
  );
  const intelScenarioRows = useMemo(
    () =>
      (intelCard?.scenario_matrix ?? []).map((row, idx) => ({
        ...row,
        key: `${row.scenario}-${idx}`,
      })),
    [intelCard]
  );
  const intelFreshnessRows = useMemo(
    () =>
      Object.entries(intelCard?.data_freshness ?? {}).map(([k, v]) => ({
        key: k,
        item: k,
        minutes: typeof v === "number" ? v : null,
      })),
    [intelCard]
  );
  const intelReviewStatsRows = useMemo(
    () =>
      Object.entries(intelReview?.stats ?? {}).map(([k, v]) => ({
        key: k,
        horizon: k,
        count: Number(v?.count ?? 0),
        avg_return: typeof v?.avg_return === "number" ? v.avg_return : null,
        hit_rate: typeof v?.hit_rate === "number" ? v.hit_rate : null,
      })),
    [intelReview]
  );
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
  const deepArchiveRoundOptions = useMemo(
    () => [
      { label: "全部轮次", value: "" },
      ...deepRounds.map((round) => ({
        label: `R${round.round_no} · ${String(round.round_id).slice(-6)}`,
        value: String(round.round_id),
      })),
    ],
    [deepRounds]
  );
  const deepArchiveEventOptions = useMemo(() => {
    const known = [
      "round_started",
      "budget_warning",
      "agent_opinion_delta",
      "agent_opinion_final",
      "critic_feedback",
      "arbitration_final",
      "replan_triggered",
      "intel_snapshot",
      "intel_status",
      "intel_self_test",
      "calendar_watchlist",
      "business_summary",
      "done",
    ];
    const dynamic = deepStreamEvents
      .map((x) => String(x.event ?? "").trim())
      .filter((x) => x.length > 0);
    const unique = Array.from(new Set([...known, ...dynamic]));
    return unique.map((eventName) => ({ label: eventName, value: eventName }));
  }, [deepStreamEvents]);
  const deepReplayRows = useMemo(() => {
    const filtered = deepArchiveEventName
      ? deepStreamEvents.filter((x) => String(x.event ?? "") === deepArchiveEventName)
      : deepStreamEvents;
    return [...filtered].reverse().slice(0, 16);
  }, [deepArchiveEventName, deepStreamEvents]);
  const deepCurrentStage = useMemo<DeepRoundStageKey>(() => {
    // 基于流事件推断当前阶段，供分析模式展示可视进度。
    let current: DeepRoundStageKey = "idle";
    for (const item of deepStreamEvents) {
      const candidate = resolveDeepStageFromEvent(item.event, item.data);
      if (DEEP_ROUND_STAGE_ORDER.indexOf(candidate) > DEEP_ROUND_STAGE_ORDER.indexOf(current)) current = candidate;
    }
    if (deepLoading || deepStreaming) {
      if (current === "idle") return "planning";
      if (current === "done") return "persist";
    }
    return current;
  }, [deepLoading, deepStreaming, deepStreamEvents]);
  const deepStagePercent = useMemo(() => {
    const idx = DEEP_ROUND_STAGE_ORDER.indexOf(deepCurrentStage);
    if (idx <= 0) return 0;
    return Number(((idx / (DEEP_ROUND_STAGE_ORDER.length - 1)) * 100).toFixed(1));
  }, [deepCurrentStage]);
  const deepActionStatusText = useMemo(() => {
    if (deepError) return "执行失败";
    if (deepLoading || deepStreaming) return "执行中";
    if (deepCurrentStage === "done") return "已完成";
    return "待执行";
  }, [deepCurrentStage, deepError, deepLoading, deepStreaming]);
  const fullFlowPercent = useMemo(() => {
    if (fullFlowStage === "error") return 100;
    const idx = DEEP_FULL_FLOW_ORDER.indexOf(fullFlowStage);
    if (idx <= 0) return 0;
    return Number(((idx / (DEEP_FULL_FLOW_ORDER.length - 1)) * 100).toFixed(1));
  }, [fullFlowStage]);
  const deepDecisionSummary = useMemo(() => {
    if (!latestDeepRound) return null;
    const signal = String(latestDeepRound.consensus_signal ?? "hold");
    const disagreement = Number(latestDeepRound.disagreement_score ?? 0);
    const confidence = Number((1 - disagreement).toFixed(3));
    const riskSources = (latestDeepRound.conflict_sources ?? []).slice(0, 4);
    return {
      signal,
      confidence,
      disagreement,
      riskSources,
      replan: Boolean(latestDeepRound.replan_triggered),
      stopReason: String(latestDeepRound.stop_reason ?? ""),
      nextAction:
        latestDeepRound.stop_reason
          ? "建议先处理风险后再发起下一轮。"
          : latestDeepRound.replan_triggered
            ? "建议继续执行下一轮，重点补齐冲突证据。"
            : "建议结合实时情报与仓位约束执行跟踪。"
    };
  }, [latestDeepRound]);
  const deepDecisionExplainModel = useMemo(() => {
    if (!latestDeepRound) return null;
    const finalSignal = String(latestDeepRound.consensus_signal ?? "hold");
    const allOpinions = Array.isArray(latestDeepRound.opinions) ? latestDeepRound.opinions : [];
    const coreRows = allOpinions.filter((x) => String(x.agent_id) !== "supervisor_agent");
    const supporting = coreRows
      .filter((x) => String(x.signal) === finalSignal)
      .sort((a, b) => Number(b.confidence ?? 0) - Number(a.confidence ?? 0))
      .slice(0, 3);
    const counter = coreRows
      .filter((x) => String(x.signal) !== finalSignal)
      .sort((a, b) => Number(b.confidence ?? 0) - Number(a.confidence ?? 0))
      .slice(0, 2);
    const disagreement = Number(latestDeepRound.disagreement_score ?? 0);
    const stability =
      disagreement <= 0.3 ? "稳定" : disagreement <= 0.55 ? "中等分歧" : "高分歧";
    return { finalSignal, supporting, counter, stability };
  }, [latestDeepRound]);
  const deepRiskActionModel = useMemo(() => {
    if (!latestDeepRound) return null;
    const disagreement = Number(latestDeepRound.disagreement_score ?? 0);
    const sources = Array.isArray(latestDeepRound.conflict_sources) ? latestDeepRound.conflict_sources : [];
    const hasVeto = sources.includes("risk_veto") || sources.includes("compliance_veto");
    let riskLevel = "低";
    let action = "按计划跟踪执行，保持事件复核。";
    if (disagreement > 0.55 || hasVeto) {
      riskLevel = "高";
      action = "建议降低仓位或暂停新增，等待关键风险事件确认。";
    } else if (disagreement > 0.35) {
      riskLevel = "中";
      action = "建议轻仓观察，等待下一轮补证后再决策。";
    }
    return { riskLevel, action, sources };
  }, [latestDeepRound]);
  const deepAgentRoleRows = useMemo(() => {
    const ids = new Set<string>();
    for (const id of deepSession?.agent_profile ?? []) ids.add(String(id));
    for (const task of latestDeepRound?.task_graph ?? []) ids.add(String(task.agent));
    for (const op of latestDeepRound?.opinions ?? []) ids.add(String(op.agent_id));
    if (!ids.size) {
      Object.keys(AGENT_META_MAP).forEach((id) => ids.add(id));
    }
    return Array.from(ids).map((id) => ({ id, ...getAgentMeta(id) }));
  }, [deepSession, latestDeepRound]);
  const deepLatestBusinessSummary = useMemo(() => {
    const reversed = [...deepStreamEvents].reverse();
    const hit = reversed.find((item) => String(item.event) === "business_summary");
    return (hit?.data ?? null) as Record<string, any> | null;
  }, [deepStreamEvents]);
  const deepLatestIntelSnapshot = useMemo(() => {
    const reversed = [...deepStreamEvents].reverse();
    const hit = reversed.find((item) => String(item.event) === "intel_snapshot");
    return (hit?.data ?? null) as Record<string, any> | null;
  }, [deepStreamEvents]);
  const deepCalendarWatch = useMemo(() => {
    const reversed = [...deepStreamEvents].reverse();
    const hit = reversed.find((item) => String(item.event) === "calendar_watchlist");
    const rows = hit?.data?.items;
    return Array.isArray(rows) ? rows.slice(0, 6) : [];
  }, [deepStreamEvents]);
  const latestBudget = latestDeepRound?.budget_usage;
  const deepTimelineItems = deepRounds.map((round) => ({
    color: round.replan_triggered ? "orange" : round.stop_reason ? "red" : "blue",
    // antd Timeline 已改为 content 字段，避免 children 废弃告警。
    content: (
      <Space direction="vertical" size={2}>
        <Text style={{ color: "#0f172a" }}>
          第 {round.round_no} 轮 | 共识={getSignalLabel(String(round.consensus_signal ?? "hold"))} | 分歧={Number(round.disagreement_score).toFixed(3)}
        </Text>
        <Space size={6} wrap>
          {round.replan_triggered ? <Tag color="orange">replan_triggered</Tag> : null}
          {round.stop_reason ? <Tag color="red">{round.stop_reason}</Tag> : null}
          {(round.conflict_sources ?? []).map((src) => (
            <Tag key={`${round.round_id}-${src}`} color="gold">
              {getConflictSourceLabel(String(src))}
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
    agent_meta: getAgentMeta(String(task.agent)),
    title: task.title,
    priority: task.priority,
    priority_label: getPriorityLabel(String(task.priority ?? ""))
  }));
  const deepOpinionRows = (latestDeepRound?.opinions ?? []).map((opinion) => ({
    key: `${opinion.agent_id}-${opinion.created_at}-${opinion.signal}`,
    agent_id: opinion.agent_id,
    agent_meta: getAgentMeta(String(opinion.agent_id)),
    signal: opinion.signal,
    signal_label: getSignalLabel(String(opinion.signal ?? "")),
    confidence: Number(opinion.confidence ?? 0),
    risk_tags: (opinion.risk_tags ?? []).join(", "),
    reason: opinion.reason
  }));
  const deepOpinionDiffRows = useMemo(() => {
    if (deepRounds.length < 2) return [];
    const prevRound = deepRounds[deepRounds.length - 2];
    const currRound = deepRounds[deepRounds.length - 1];
    const prevByAgent = new Map((prevRound.opinions ?? []).map((x) => [x.agent_id, x]));
    const currByAgent = new Map((currRound.opinions ?? []).map((x) => [x.agent_id, x]));
    const agents = Array.from(new Set([...Array.from(prevByAgent.keys()), ...Array.from(currByAgent.keys())]));
    return agents.map((agentId) => {
      const prev = prevByAgent.get(agentId);
      const curr = currByAgent.get(agentId);
      const prevSignal = String(prev?.signal ?? "-");
      const currSignal = String(curr?.signal ?? "-");
      const prevConfidence = Number(prev?.confidence ?? 0);
      const currConfidence = Number(curr?.confidence ?? 0);
      const deltaConfidence = Number((currConfidence - prevConfidence).toFixed(4));
      let changeType = "unchanged";
      if (prevSignal !== currSignal) changeType = "signal_changed";
      else if (Math.abs(deltaConfidence) >= 0.08) changeType = "confidence_shift";
      return {
        key: `${currRound.round_id}-${agentId}`,
        agent_id: agentId,
        agent_meta: getAgentMeta(String(agentId)),
        prev_signal: prevSignal,
        prev_signal_label: getSignalLabel(prevSignal),
        curr_signal: currSignal,
        curr_signal_label: getSignalLabel(currSignal),
        prev_confidence: prevConfidence,
        curr_confidence: currConfidence,
        delta_confidence: deltaConfidence,
        change_type: changeType
      };
    });
  }, [deepRounds]);
  const deepConflictDrillRows = useMemo(() => {
    if (!latestDeepRound) return [];
    const consensusSignal = String(latestDeepRound.consensus_signal ?? "hold");
    const sources = latestDeepRound.conflict_sources ?? [];
    const needRiskVeto = sources.includes("risk_veto");
    const needComplianceVeto = sources.includes("compliance_veto");
    return (latestDeepRound.opinions ?? [])
      .filter((opinion) => {
        if (String(opinion.signal) !== consensusSignal) return true;
        if (needRiskVeto && opinion.agent_id === "risk_agent") return true;
        if (needComplianceVeto && opinion.agent_id === "compliance_agent") return true;
        return false;
      })
      .map((opinion) => ({
        key: `${latestDeepRound.round_id}-conflict-${opinion.agent_id}`,
        agent_id: opinion.agent_id,
        agent_meta: getAgentMeta(String(opinion.agent_id)),
        signal: opinion.signal,
        signal_label: getSignalLabel(String(opinion.signal ?? "")),
        confidence: Number(opinion.confidence ?? 0),
        evidence_ids: (opinion.evidence_ids ?? []).join(", "),
        risk_tags: (opinion.risk_tags ?? []).join(", "),
        reason: opinion.reason
      }));
  }, [latestDeepRound]);
  const deepTokenPercent = latestBudget ? formatDeepPercent(latestBudget.used.token_used, latestBudget.limit.token_budget) : 0;
  const deepTimePercent = latestBudget ? formatDeepPercent(latestBudget.used.time_used_ms, latestBudget.limit.time_budget_ms) : 0;
  const deepToolPercent = latestBudget ? formatDeepPercent(latestBudget.used.tool_calls_used, latestBudget.limit.tool_call_budget) : 0;

  return (
    <main className="container">
      {isConsoleWorkspace ? (
        <motion.section initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.35 }}>
          <Card className="premium-card" style={{ marginBottom: 8 }}>
            <Space direction="vertical" size={6} style={{ width: "100%" }}>
              <Tag color="processing" style={{ width: "fit-content" }}>DeepThink Console</Tag>
              <Title level={3} style={{ margin: 0, color: "#0f172a" }}>工程控制台</Title>
              <Text style={{ color: "#475569" }}>
                本页仅展示轮次排障、事件存档、预算治理与导出能力。业务分析请使用主页面。
              </Text>
              <Space>
                <Button href="/deep-think">返回业务分析页</Button>
              </Space>
            </Space>
          </Card>
        </motion.section>
      ) : null}

      {!isConsoleWorkspace ? (
        <>
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
            <Button type="primary" size="large" onClick={scrollToAnalysisPanel}>
              进入分析面板
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
            <div ref={analysisPanelRef}>
              <Card className="premium-card" style={{ borderColor: "rgba(15,23,42,0.12)" }}>
              <Space direction="vertical" style={{ width: "100%" }}>
                <Text style={{ color: "#475569" }}>
                  标准分析默认使用模板构建问题，减少自由输入造成的语义偏差；需要时可切换到自由输入。
                </Text>
                <StockSelectorModal
                  value={stockCode}
                  onChange={(next: string | string[]) => setStockCode(Array.isArray(next) ? (next[0] ?? "") : next)}
                  title="选择分析标的"
                  placeholder="请先选择要分析的股票"
                />
                <Segmented
                  value={questionInputMode}
                  onChange={(value) => setQuestionInputMode(value as "template" | "free")}
                  options={[
                    { label: "标准分析（模板）", value: "template" },
                    { label: "自由输入", value: "free" },
                  ]}
                />
                {questionInputMode === "template" ? (
                  <Space direction="vertical" size={8} style={{ width: "100%" }}>
                    <Space wrap>
                      {ANALYSIS_TEMPLATES.map((item) => (
                        <Button
                          key={item.id}
                          size="small"
                          type={item.id === selectedTemplateId ? "primary" : "default"}
                          onClick={() => setSelectedTemplateId(item.id)}
                        >
                          {item.title}
                        </Button>
                      ))}
                    </Space>
                    <Text style={{ color: "#64748b" }}>
                      {ANALYSIS_TEMPLATES.find((item) => item.id === selectedTemplateId)?.description ?? ""}
                    </Text>
                    <Space wrap>
                      <Select
                        style={{ width: 120 }}
                        value={selectedHorizon}
                        onChange={(value) => setSelectedHorizon(value as HorizonOption)}
                        options={[
                          { label: "7天", value: "7d" },
                          { label: "30天", value: "30d" },
                          { label: "90天", value: "90d" },
                        ]}
                      />
                      <Select
                        style={{ width: 140 }}
                        value={selectedRiskProfile}
                        onChange={(value) => setSelectedRiskProfile(value as RiskProfileOption)}
                        options={[
                          { label: "保守", value: "conservative" },
                          { label: "中性", value: "neutral" },
                          { label: "积极", value: "aggressive" },
                        ]}
                      />
                      <Select
                        style={{ width: 140 }}
                        value={selectedPositionState}
                        onChange={(value) => setSelectedPositionState(value as PositionStateOption)}
                        options={[
                          { label: "空仓", value: "flat" },
                          { label: "已持仓", value: "holding" },
                        ]}
                      />
                    </Space>
                  </Space>
                ) : null}
                <TextArea rows={5} value={question} onChange={(e) => setQuestion(e.target.value)} />
                <Space wrap>
                  <Tag color={promptQualityScore >= 80 ? "green" : promptQualityScore >= 60 ? "gold" : "red"}>
                    输入质量分：{promptQualityScore}
                  </Tag>
                  {promptGuardrailWarnings.slice(0, 2).map((item, idx) => (
                    <Tag key={`guard-${idx}`} color="blue">{item}</Tag>
                  ))}
                </Space>
                <Button type="primary" size="large" loading={fullFlowRunning || loading || deepLoading} onClick={() => runDeepThinkFullFlow(resolveComposedQuestion())}>
                  {deepSession?.session_id ? "继续全流程分析（流式+下一轮）" : "开始全流程分析"}
                </Button>
                <Space wrap>
                  <Tag color={fullFlowStage === "done" ? "green" : fullFlowStage === "error" ? "red" : fullFlowRunning ? "processing" : "default"}>
                    全流程阶段：{DEEP_FULL_FLOW_STAGE_LABEL[fullFlowStage]}
                  </Tag>
                  {fullFlowMessage ? <Tag color="blue">{fullFlowMessage}</Tag> : null}
                </Space>
                <Progress percent={fullFlowPercent} status={fullFlowStage === "error" ? "exception" : fullFlowRunning ? "active" : "normal"} />
                {streaming ? <Text style={{ color: "#2563eb" }}>流式输出中...</Text> : null}
                {streaming && queryProgressText ? <Text style={{ color: "#475569" }}>阶段：{queryProgressText}</Text> : null}
              </Space>
              </Card>
            </div>

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
                <pre style={{ whiteSpace: "pre-wrap", color: "#0f172a", margin: 0, maxHeight: 420, overflowY: "auto" }}>{result.answer}</pre>
                <Text style={{ color: "#64748b" }}>trace_id: {result.trace_id}</Text>
              </Card>
            ) : null}
          </Col>

          <Col xs={24} xl={10} style={{ alignSelf: "flex-start", position: "sticky", top: 88 }}>
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

            <Card className="premium-card" title={<span style={{ color: "#0f172a" }}>最近5个交易日明细</span>} style={{ marginTop: 12 }}>
              <Table
                size="small"
                pagination={false}
                dataSource={recentFiveTradeRows}
                columns={[
                  { title: "日期", dataIndex: "trade_date", key: "trade_date", width: 96 },
                  { title: "开", dataIndex: "open", key: "open", width: 72, render: (v: number) => v.toFixed(2) },
                  { title: "收", dataIndex: "close", key: "close", width: 72, render: (v: number) => v.toFixed(2) },
                  {
                    title: "涨跌",
                    dataIndex: "pct_change",
                    key: "pct_change",
                    width: 88,
                    render: (v: number) => <Text style={{ color: v >= 0 ? "#059669" : "#dc2626" }}>{v.toFixed(2)}%</Text>,
                  },
                ]}
                locale={{ emptyText: "暂无最近交易日数据" }}
              />
            </Card>

            <Card className="premium-card" title={<span style={{ color: "#0f172a" }}>最近三个月连续样本</span>} style={{ marginTop: 12 }}>
              {history3mSummary ? (
                <Space direction="vertical" style={{ width: "100%" }} size={6}>
                  <Space wrap>
                    <Tag color="blue">样本: {history3mSummary.sampleCount} / 总样本: {history3mSummary.totalCount}</Tag>
                    <Tag color={history3mSummary.pctChange >= 0 ? "green" : "red"}>
                      区间涨跌: {history3mSummary.pctChange.toFixed(2)}%
                    </Tag>
                  </Space>
                  <Text style={{ color: "#334155" }}>
                    区间：{history3mSummary.startDate} {"->"} {history3mSummary.endDate}
                  </Text>
                  <Text style={{ color: "#334155" }}>
                    收盘：{history3mSummary.startClose.toFixed(3)} {"->"} {history3mSummary.endClose.toFixed(3)}
                  </Text>
                  <Text style={{ color: "#64748b" }}>
                    该卡片固定基于连续日线窗口计算，优先用于判断样本覆盖是否足够。
                  </Text>
                </Space>
              ) : (
                <Text style={{ color: "#64748b" }}>连续样本不足，建议先点击“开始全流程分析”刷新历史数据。</Text>
              )}
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

            {result ? (
              <Card className="premium-card" title={<span style={{ color: "#0f172a" }}>共享知识命中</span>} style={{ marginTop: 12 }}>
                <Space direction="vertical" style={{ width: "100%" }} size={8}>
                  <Text style={{ color: "#64748b" }}>
                    作用：显示本次回答是否命中共享 RAG 语料（文档库/历史问答记忆），帮助判断结论是否可复用。
                  </Text>
                  <Space wrap>
                    <Tag color={sharedKnowledgeHits.length ? "green" : "default"}>命中数：{sharedKnowledgeHits.length}</Tag>
                    {knowledgePersistedTraceId ? (
                      <Tag color="blue">已沉淀共享语料：{knowledgePersistedTraceId.slice(0, 12)}</Tag>
                    ) : null}
                  </Space>
                  {sharedKnowledgeHits.length ? (
                    <List
                      size="small"
                      dataSource={sharedKnowledgeHits}
                      renderItem={(item) => {
                        const sourceId = String(item.source_id ?? "");
                        const hitType = sourceId === "qa_memory_summary" ? "历史问答摘要" : "文档语料";
                        return (
                          <List.Item>
                            <Space direction="vertical" size={1}>
                              <Space wrap>
                                <Tag color={hitType === "历史问答摘要" ? "purple" : "cyan"}>{hitType}</Tag>
                                <Text style={{ color: "#0f172a" }}>{sourceId}</Text>
                              </Space>
                              <Text style={{ color: "#64748b" }}>{String(item.excerpt ?? "").slice(0, 120)}</Text>
                            </Space>
                          </List.Item>
                        );
                      }}
                    />
                  ) : (
                    <Text style={{ color: "#64748b" }}>
                      本轮回答暂未命中共享语料，仅基于实时拉取数据与当前上下文生成。可通过上传资料或历史问答沉淀提升复用率。
                    </Text>
                  )}
                </Space>
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

            <Card className="premium-card" title={<span style={{ color: "#0f172a" }}>个股情报卡片（业务版）</span>} style={{ marginTop: 12 }}>
              <Space direction="vertical" style={{ width: "100%" }} size={10}>
                <Text style={{ color: "#64748b" }}>
                  作用：把多源数据转换成可执行结论，优先回答“该做什么、为什么、何时失效”。
                </Text>
                <Space wrap>
                  <Select
                    value={intelCardHorizon}
                    style={{ width: 120 }}
                    options={[
                      { label: "7天", value: "7d" },
                      { label: "30天", value: "30d" },
                      { label: "90天", value: "90d" },
                    ]}
                    onChange={(value) => setIntelCardHorizon(value as "7d" | "30d" | "90d")}
                  />
                  <Select
                    value={intelCardRiskProfile}
                    style={{ width: 160 }}
                    options={[
                      { label: "保守风险偏好", value: "conservative" },
                      { label: "中性风险偏好", value: "neutral" },
                      { label: "积极风险偏好", value: "aggressive" },
                    ]}
                    onChange={(value) => setIntelCardRiskProfile(value as "conservative" | "neutral" | "aggressive")}
                  />
                  <Button loading={intelCardLoading} onClick={() => loadIntelCard({ showLoading: true })}>刷新业务卡片</Button>
                </Space>

                {intelCardError ? <Alert type="warning" showIcon message={intelCardError} /> : null}

                {intelCard ? (
                  <>
                    <Space wrap>
                      <Tag color={intelCard.overall_signal === "buy" ? "green" : intelCard.overall_signal === "reduce" ? "red" : "blue"}>
                        建议动作：{getSignalLabel(intelCard.overall_signal)}
                      </Tag>
                      <Tag color={intelCard.confidence >= 0.7 ? "green" : intelCard.confidence >= 0.5 ? "gold" : "red"}>
                        置信度：{Number(intelCard.confidence ?? 0).toFixed(3)}
                      </Tag>
                      <Tag color={intelCard.risk_level === "high" ? "red" : intelCard.risk_level === "medium" ? "gold" : "green"}>
                        风险等级：{intelCard.risk_level}
                      </Tag>
                      <Tag>仓位建议：{intelCard.position_hint}</Tag>
                      <Tag>复核时间：{String(intelCard.next_review_time ?? "-")}</Tag>
                      <Tag color={intelCard.degrade_status?.level === "degraded" ? "red" : intelCard.degrade_status?.level === "watch" ? "gold" : "green"}>
                        降级状态：{String(intelCard.degrade_status?.level ?? "normal")}
                      </Tag>
                      {(intelCard.degrade_status?.reasons ?? []).slice(0, 3).map((reason) => (
                        <Tag key={`degrade-${reason}`} color="orange">{String(reason)}</Tag>
                      ))}
                    </Space>
                    <Space wrap>
                      <Button size="small" loading={intelFeedbackLoading} onClick={() => submitIntelFeedback("adopt")}>
                        采纳本次建议
                      </Button>
                      <Button size="small" loading={intelFeedbackLoading} onClick={() => submitIntelFeedback("watch")}>
                        继续观察
                      </Button>
                      <Button size="small" danger loading={intelFeedbackLoading} onClick={() => submitIntelFeedback("reject")}>
                        拒绝本次建议
                      </Button>
                      <Button size="small" loading={intelReviewLoading} onClick={() => loadIntelReview(stockCode, true)}>
                        刷新复盘统计
                      </Button>
                    </Space>
                    {intelFeedbackMessage ? <Alert type="info" showIcon message={intelFeedbackMessage} /> : null}

                    <Row gutter={[12, 12]}>
                      <Col xs={24} md={12}>
                        <Card size="small" title={<span style={{ color: "#0f172a" }}>触发条件</span>}>
                          <List
                            size="small"
                            dataSource={intelCard.trigger_conditions ?? []}
                            renderItem={(item) => <List.Item><Text style={{ color: "#334155" }}>{String(item)}</Text></List.Item>}
                          />
                        </Card>
                      </Col>
                      <Col xs={24} md={12}>
                        <Card size="small" title={<span style={{ color: "#0f172a" }}>失效条件</span>}>
                          <List
                            size="small"
                            dataSource={intelCard.invalidation_conditions ?? []}
                            renderItem={(item) => <List.Item><Text style={{ color: "#334155" }}>{String(item)}</Text></List.Item>}
                          />
                        </Card>
                      </Col>
                    </Row>

                    <Row gutter={[12, 12]}>
                      <Col xs={24} md={12}>
                        <Card size="small" title={<span style={{ color: "#0f172a" }}>执行节奏建议</span>}>
                          <Space direction="vertical" size={4} style={{ width: "100%" }}>
                            <Text style={{ color: "#334155" }}>模式：{String(intelCard.execution_plan?.entry_mode ?? "-")}</Text>
                            <Text style={{ color: "#334155" }}>节奏：{String(intelCard.execution_plan?.cadence_hint ?? "-")}</Text>
                            <Text style={{ color: "#334155" }}>
                              单步上限：{Number(intelCard.execution_plan?.max_single_step_pct ?? 0).toFixed(2)}
                            </Text>
                            <Text style={{ color: "#334155" }}>仓位上限：{String(intelCard.execution_plan?.max_position_cap ?? "-")}</Text>
                            <Text style={{ color: "#334155" }}>
                              风险止损提示：{Number(intelCard.execution_plan?.stop_loss_hint_pct ?? 0).toFixed(2)}%
                            </Text>
                            <Text style={{ color: "#334155" }}>
                              复核间隔：{Number(intelCard.execution_plan?.recheck_interval_hours ?? 0)} 小时
                            </Text>
                          </Space>
                        </Card>
                      </Col>
                      <Col xs={24} md={12}>
                        <Card size="small" title={<span style={{ color: "#0f172a" }}>风险阈值</span>}>
                          <Space direction="vertical" size={4} style={{ width: "100%" }}>
                            <Text style={{ color: "#334155" }}>
                              波动阈值：{Number(intelCard.risk_thresholds?.volatility_20_max ?? 0).toFixed(4)}
                            </Text>
                            <Text style={{ color: "#334155" }}>
                              回撤阈值：{Number(intelCard.risk_thresholds?.max_drawdown_60_max ?? 0).toFixed(4)}
                            </Text>
                            <Text style={{ color: "#334155" }}>
                              最小证据数：{Number(intelCard.risk_thresholds?.min_evidence_count ?? 0)}
                            </Text>
                            <Text style={{ color: "#334155" }}>
                              数据时效阈值：{Number(intelCard.risk_thresholds?.max_data_staleness_minutes ?? 0)} 分钟
                            </Text>
                          </Space>
                        </Card>
                      </Col>
                    </Row>

                    <Row gutter={[12, 12]}>
                      <Col xs={24} md={12}>
                        <Card size="small" title={<span style={{ color: "#0f172a" }}>关键催化</span>}>
                          <List
                            size="small"
                            dataSource={(intelCard.key_catalysts ?? []).slice(0, 6)}
                            renderItem={(row: any) => (
                              <List.Item>
                                <Space direction="vertical" size={1} style={{ width: "100%" }}>
                                  <Text style={{ color: "#0f172a" }}>{String(row?.title ?? "-")}</Text>
                                  <Text style={{ color: "#64748b" }}>{String(row?.summary ?? "")}</Text>
                                </Space>
                              </List.Item>
                            )}
                          />
                        </Card>
                      </Col>
                      <Col xs={24} md={12}>
                        <Card size="small" title={<span style={{ color: "#0f172a" }}>风险观察</span>}>
                          <List
                            size="small"
                            dataSource={(intelCard.risk_watch ?? []).slice(0, 6)}
                            renderItem={(row: any) => (
                              <List.Item>
                                <Space direction="vertical" size={1} style={{ width: "100%" }}>
                                  <Text style={{ color: "#0f172a" }}>{String(row?.title ?? "-")}</Text>
                                  <Text style={{ color: "#64748b" }}>{String(row?.summary ?? "")}</Text>
                                </Space>
                              </List.Item>
                            )}
                          />
                        </Card>
                      </Col>
                    </Row>

                    <Card size="small" title={<span style={{ color: "#0f172a" }}>事件日历</span>}>
                      <List
                        size="small"
                        dataSource={(intelCard.event_calendar ?? []).slice(0, 8)}
                        renderItem={(item) => (
                          <List.Item>
                            <Space wrap>
                              <Tag color={String(item.event_type) === "macro" ? "gold" : "blue"}>{String(item.event_type)}</Tag>
                              <Text style={{ color: "#334155" }}>{String(item.date)}</Text>
                              <Text style={{ color: "#0f172a" }}>{String(item.title)}</Text>
                            </Space>
                          </List.Item>
                        )}
                      />
                    </Card>

                    <Card size="small" title={<span style={{ color: "#0f172a" }}>情景矩阵</span>}>
                      <Table
                        rowKey="key"
                        pagination={false}
                        dataSource={intelScenarioRows}
                        columns={[
                          { title: "情景", dataIndex: "scenario", key: "scenario" },
                          {
                            title: "预期收益(%)",
                            dataIndex: "expected_return_pct",
                            key: "expected_return_pct",
                            render: (v: number) => Number(v ?? 0).toFixed(2),
                          },
                          {
                            title: "概率",
                            dataIndex: "probability",
                            key: "probability",
                            render: (v: number) => Number(v ?? 0).toFixed(2),
                          },
                        ]}
                      />
                    </Card>

                    <Card size="small" title={<span style={{ color: "#0f172a" }}>证据链路（粗排+精排归因）</span>}>
                      <Table
                        rowKey="key"
                        pagination={false}
                        dataSource={intelEvidenceRows}
                        scroll={{ x: 960 }}
                        columns={[
                          { title: "来源", dataIndex: "source_id", key: "source_id" },
                          { title: "轨道", dataIndex: "retrieval_track", key: "retrieval_track" },
                          {
                            title: "可信度",
                            dataIndex: "reliability_score",
                            key: "reliability_score",
                            render: (v: number) => Number(v ?? 0).toFixed(2),
                          },
                          {
                            title: "摘要",
                            dataIndex: "summary",
                            key: "summary",
                            render: (v: string) => <Text style={{ color: "#334155" }}>{String(v ?? "").slice(0, 90)}</Text>,
                          },
                          {
                            title: "链接",
                            dataIndex: "source_url",
                            key: "source_url",
                            render: (v: string) => (
                              <a href={v} target="_blank" rel="noreferrer" style={{ color: "#2563eb" }}>
                                查看来源
                              </a>
                            ),
                          },
                        ]}
                      />
                    </Card>

                    <Card size="small" title={<span style={{ color: "#0f172a" }}>数据新鲜度</span>}>
                      <Table
                        rowKey="key"
                        pagination={false}
                        dataSource={intelFreshnessRows}
                        columns={[
                          { title: "数据项", dataIndex: "item", key: "item" },
                          {
                            title: "距今分钟",
                            dataIndex: "minutes",
                            key: "minutes",
                            render: (v: number | null) => (typeof v === "number" ? `${v} 分钟前` : "未知"),
                          },
                        ]}
                      />
                    </Card>

                    <Card size="small" title={<span style={{ color: "#0f172a" }}>采纳与偏差复盘（T+1/T+5/T+20）</span>}>
                      <Space direction="vertical" style={{ width: "100%" }} size={8}>
                        <Text style={{ color: "#64748b" }}>
                          说明：统计用户对建议的采纳结果，并追踪后续区间偏差，用于迭代阈值与策略节奏。
                        </Text>
                        <Table
                          rowKey="key"
                          pagination={false}
                          dataSource={intelReviewStatsRows}
                          columns={[
                            { title: "窗口", dataIndex: "horizon", key: "horizon" },
                            { title: "样本数", dataIndex: "count", key: "count" },
                            {
                              title: "平均收益",
                              dataIndex: "avg_return",
                              key: "avg_return",
                              render: (v: number | null) => (typeof v === "number" ? `${(v * 100).toFixed(2)}%` : "-"),
                            },
                            {
                              title: "命中率",
                              dataIndex: "hit_rate",
                              key: "hit_rate",
                              render: (v: number | null) => (typeof v === "number" ? `${(v * 100).toFixed(2)}%` : "-"),
                            },
                          ]}
                        />
                        <List
                          size="small"
                          dataSource={(intelReview?.items ?? []).slice(0, 6)}
                          renderItem={(row: any) => (
                            <List.Item>
                              <Space wrap>
                                <Tag color="blue">{String(row?.feedback ?? "-")}</Tag>
                                <Tag color="purple">{String(row?.signal ?? "-")}</Tag>
                                <Text style={{ color: "#334155" }}>{String(row?.created_at ?? "")}</Text>
                                <Text style={{ color: "#64748b" }}>
                                  T+1={row?.realized?.t1 == null ? "-" : `${(Number(row.realized.t1) * 100).toFixed(2)}%`}
                                </Text>
                                <Text style={{ color: "#64748b" }}>
                                  T+5={row?.realized?.t5 == null ? "-" : `${(Number(row.realized.t5) * 100).toFixed(2)}%`}
                                </Text>
                                <Text style={{ color: "#64748b" }}>
                                  T+20={row?.realized?.t20 == null ? "-" : `${(Number(row.realized.t20) * 100).toFixed(2)}%`}
                                </Text>
                              </Space>
                            </List.Item>
                          )}
                        />
                      </Space>
                    </Card>
                  </>
                ) : (
                  <Text style={{ color: "#64748b" }}>暂无业务卡片，请点击“刷新业务卡片”或先完成一次分析后查看。</Text>
                )}
              </Space>
            </Card>
          </Col>
        </Row>
      </motion.section>
        </>
      ) : null}

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
                  {isConsoleWorkspace
                    ? "当前为工程控制台：聚焦轮次治理、事件归档、导出和排障。"
                    : "当前为业务分析页：聚焦结论、风险与执行建议。"}
                </Text>
                <Space wrap>
                  <Tag color={effectiveConsoleMode === "analysis" ? "blue" : "purple"}>
                    当前视图：{effectiveConsoleMode === "analysis" ? "分析模式" : "工程模式"}
                  </Tag>
                  {isConsoleWorkspace ? (
                    <Button href="/deep-think">返回业务分析页</Button>
                  ) : (
                    <Button href="/deep-think/console">进入工程控制台</Button>
                  )}
                </Space>
                {effectiveConsoleMode === "analysis" ? (
                  <>
                    <Text style={{ color: "#64748b" }}>
                      业务模式已简化为“一键全流程”。请在上方输入区点击“开始全流程分析”，系统会自动完成流式分析与下一轮研判。
                    </Text>
                    {deepStockSwitchNotice ? (
                      <Alert
                        type="info"
                        showIcon
                        message={deepStockSwitchNotice}
                        closable
                        onClose={() => setDeepStockSwitchNotice("")}
                      />
                    ) : null}
                    <Space direction="vertical" size={4} style={{ width: "100%" }}>
                      <Text style={{ color: "#334155" }}>
                        全流程状态：{DEEP_FULL_FLOW_STAGE_LABEL[fullFlowStage]} | 轮次状态：{deepActionStatusText}
                      </Text>
                      <Progress percent={fullFlowPercent} status={fullFlowStage === "error" ? "exception" : fullFlowRunning ? "active" : "normal"} />
                      <Text style={{ color: "#64748b" }}>{fullFlowMessage || deepProgressText || "点击“开始全流程分析”启动业务流程。"}</Text>
                    </Space>
                    {deepDecisionSummary ? (
                      <Card size="small" title={<span style={{ color: "#0f172a" }}>本轮结论摘要</span>}>
                        <Space direction="vertical" size={6} style={{ width: "100%" }}>
                          <Space wrap>
                            <Tag color={deepDecisionSummary.signal === "buy" ? "green" : deepDecisionSummary.signal === "reduce" ? "red" : "blue"}>
                              建议动作：{getSignalLabel(deepDecisionSummary.signal)}
                            </Tag>
                            <Tag color={deepDecisionSummary.confidence >= 0.7 ? "green" : deepDecisionSummary.confidence >= 0.5 ? "gold" : "red"}>
                              置信度：{deepDecisionSummary.confidence.toFixed(3)}
                            </Tag>
                            <Tag>分歧度：{deepDecisionSummary.disagreement.toFixed(3)}</Tag>
                            {deepDecisionSummary.replan ? <Tag color="orange">触发补证重规划</Tag> : null}
                            {deepDecisionSummary.stopReason ? <Tag color="red">{deepDecisionSummary.stopReason}</Tag> : null}
                          </Space>
                          <Text style={{ color: "#334155" }}>{deepDecisionSummary.nextAction}</Text>
                          <Space wrap>
                            {(deepDecisionSummary.riskSources ?? []).map((src) => (
                              <Tag key={`summary-${src}`} color="orange">{getConflictSourceLabel(String(src))}</Tag>
                            ))}
                            {!deepDecisionSummary.riskSources.length ? <Tag color="green">当前无显著冲突源</Tag> : null}
                          </Space>
                        </Space>
                      </Card>
                    ) : (
                      <Text style={{ color: "#64748b" }}>暂无轮次结果，请先点击“开始全流程分析”。</Text>
                    )}
                    {deepDecisionExplainModel ? (
                      <Card size="small" title={<span style={{ color: "#0f172a" }}>为什么是这个结论</span>}>
                        <Space direction="vertical" size={8} style={{ width: "100%" }}>
                          <Space wrap>
                            <Tag color={deepDecisionExplainModel.finalSignal === "buy" ? "green" : deepDecisionExplainModel.finalSignal === "reduce" ? "red" : "blue"}>
                              最终信号：{getSignalLabel(deepDecisionExplainModel.finalSignal)}
                            </Tag>
                            <Tag color="cyan">稳定性：{deepDecisionExplainModel.stability}</Tag>
                          </Space>
                          <Text style={{ color: "#334155" }}>支持结论的关键依据</Text>
                          {(deepDecisionExplainModel.supporting ?? []).length ? (
                            (deepDecisionExplainModel.supporting ?? []).map((row) => {
                              const meta = getAgentMeta(String(row.agent_id));
                              return (
                                <Space key={`support-${row.agent_id}-${row.created_at}`} direction="vertical" size={1} style={{ width: "100%" }}>
                                  <Text style={{ color: "#0f172a" }}>
                                    {meta.name}（{meta.role}） | {String(row.signal)} | {Number(row.confidence ?? 0).toFixed(3)}
                                  </Text>
                                  <Text style={{ color: "#475569" }}>
                                    观点：{getSignalLabel(String(row.signal ?? "hold"))}
                                  </Text>
                                  <Text style={{ color: "#64748b" }}>{String(row.reason ?? "")}</Text>
                                </Space>
                              );
                            })
                          ) : (
                            <Text style={{ color: "#64748b" }}>暂无支持结论的可解释依据。</Text>
                          )}
                          <Text style={{ color: "#334155" }}>被压制的反对意见</Text>
                          {(deepDecisionExplainModel.counter ?? []).length ? (
                            (deepDecisionExplainModel.counter ?? []).map((row) => {
                              const meta = getAgentMeta(String(row.agent_id));
                              return (
                                <Space key={`counter-${row.agent_id}-${row.created_at}`} direction="vertical" size={1} style={{ width: "100%" }}>
                                  <Text style={{ color: "#0f172a" }}>
                                    {meta.name}（{meta.role}） | {String(row.signal)} | {Number(row.confidence ?? 0).toFixed(3)}
                                  </Text>
                                  <Text style={{ color: "#475569" }}>
                                    观点：{getSignalLabel(String(row.signal ?? "hold"))}
                                  </Text>
                                  <Text style={{ color: "#64748b" }}>{String(row.reason ?? "")}</Text>
                                </Space>
                              );
                            })
                          ) : (
                            <Text style={{ color: "#64748b" }}>当前无显著反对意见。</Text>
                          )}
                        </Space>
                      </Card>
                    ) : null}
                    {deepRiskActionModel ? (
                      <Card size="small" title={<span style={{ color: "#0f172a" }}>风险与行动建议</span>}>
                        <Space direction="vertical" size={8} style={{ width: "100%" }}>
                          <Space wrap>
                            <Tag color={deepRiskActionModel.riskLevel === "高" ? "red" : deepRiskActionModel.riskLevel === "中" ? "gold" : "green"}>
                              风险等级：{deepRiskActionModel.riskLevel}
                            </Tag>
                            {(deepRiskActionModel.sources ?? []).map((src) => (
                              <Tag key={`risk-src-${src}`} color="orange">{getConflictSourceLabel(String(src))}</Tag>
                            ))}
                            {!(deepRiskActionModel.sources ?? []).length ? <Tag color="green">暂无显著冲突源</Tag> : null}
                          </Space>
                          <Text style={{ color: "#334155" }}>{deepRiskActionModel.action}</Text>
                        </Space>
                      </Card>
                    ) : null}
                    {deepAgentRoleRows.length ? (
                      <div style={{ display: "flex", justifyContent: "flex-end" }}>
                        {/* 角色说明改为 Popover：默认不占版面，业务用户按需点击查看。 */}
                        <Popover
                          trigger="click"
                          placement="bottomRight"
                          overlayStyle={{ maxWidth: 420 }}
                          content={(
                            <Space direction="vertical" size={6} style={{ width: "100%" }}>
                              {deepAgentRoleRows.map((row) => (
                                <Space key={`agent-role-${row.id}`} direction="vertical" size={0} style={{ width: "100%" }}>
                                  <Text style={{ color: "#0f172a" }}>{row.name}</Text>
                                  <Text style={{ color: "#64748b" }}>{row.role}</Text>
                                </Space>
                              ))}
                            </Space>
                          )}
                        >
                          <Button size="small" type="text" style={{ color: "#475569", paddingInline: 6 }}>
                            查看角色说明（{deepAgentRoleRows.length}）
                          </Button>
                        </Popover>
                      </div>
                    ) : null}
                    {deepLatestBusinessSummary ? (
                      <Card size="small" title={<span style={{ color: "#0f172a" }}>业务融合结论</span>}>
                        <Space direction="vertical" size={6} style={{ width: "100%" }}>
                          <Space wrap>
                            <Tag color={String(deepLatestBusinessSummary.signal) === "buy" ? "green" : String(deepLatestBusinessSummary.signal) === "reduce" ? "red" : "blue"}>
                              建议动作：{getSignalLabel(String(deepLatestBusinessSummary.signal ?? "hold"))}
                            </Tag>
                            <Tag>置信度：{Number(deepLatestBusinessSummary.confidence ?? 0).toFixed(3)}</Tag>
                            <Tag>复核周期：{String(deepLatestBusinessSummary.review_time_hint ?? "-")}</Tag>
                          </Space>
                          <Space wrap>
                            <Tag color={String(deepLatestBusinessSummary.intel_status) === "external_ok" ? "green" : "gold"}>
                              情报状态：{String(deepLatestBusinessSummary.intel_status ?? "-")}
                            </Tag>
                            {String(deepLatestBusinessSummary.intel_fallback_reason ?? "").trim() ? (
                              <Tag color="orange">降级原因：{String(deepLatestBusinessSummary.intel_fallback_reason)}</Tag>
                            ) : null}
                            {String(deepLatestBusinessSummary.intel_trace_id ?? "").trim() ? (
                              <Tag color="cyan">追踪ID：{String(deepLatestBusinessSummary.intel_trace_id)}</Tag>
                            ) : null}
                          </Space>
                          <Text style={{ color: "#334155" }}>
                            触发条件：{String(deepLatestBusinessSummary.trigger_condition ?? "-")}
                          </Text>
                          <Text style={{ color: "#334155" }}>
                            失效条件：{String(deepLatestBusinessSummary.invalidation_condition ?? "-")}
                          </Text>
                        </Space>
                      </Card>
                    ) : null}
                    {deepLatestIntelSnapshot ? (
                      <Card size="small" title={<span style={{ color: "#0f172a" }}>实时情报摘要</span>}>
                        <Space direction="vertical" size={6} style={{ width: "100%" }}>
                          <Text style={{ color: "#64748b" }}>as_of: {String(deepLatestIntelSnapshot.as_of ?? "-")}</Text>
                          <Space wrap>
                            <Tag color={String(deepLatestIntelSnapshot.intel_status) === "external_ok" ? "green" : "gold"}>
                              情报状态：{String(deepLatestIntelSnapshot.intel_status ?? "-")}
                            </Tag>
                            <Tag>引用数：{Number(deepLatestIntelSnapshot.citations_count ?? 0)}</Tag>
                            <Tag color={Boolean(deepLatestIntelSnapshot.websearch_tool_applied) ? "green" : "orange"}>
                              检索工具启用：{String(Boolean(deepLatestIntelSnapshot.websearch_tool_applied))}
                            </Tag>
                          </Space>
                          {String(deepLatestIntelSnapshot.fallback_reason ?? "").trim() ? (
                            <Text style={{ color: "#b45309" }}>
                              降级原因：{String(deepLatestIntelSnapshot.fallback_reason)}
                            </Text>
                          ) : null}
                          {String(deepLatestIntelSnapshot.trace_id ?? "").trim() ? (
                            <Text style={{ color: "#64748b" }}>trace_id: {String(deepLatestIntelSnapshot.trace_id)}</Text>
                          ) : null}
                          {(Array.isArray(deepLatestIntelSnapshot.macro_signals) ? deepLatestIntelSnapshot.macro_signals.slice(0, 3) : []).map((item: any, idx: number) => (
                            <Space key={`intel-macro-${idx}`} direction="vertical" size={1} style={{ width: "100%" }}>
                              <Text style={{ color: "#0f172a" }}>{String(item?.title ?? "")}</Text>
                              <Text style={{ color: "#475569" }}>{String(item?.summary ?? "")}</Text>
                            </Space>
                          ))}
                          {!Array.isArray(deepLatestIntelSnapshot.macro_signals) || deepLatestIntelSnapshot.macro_signals.length === 0 ? (
                            <Text style={{ color: "#64748b" }}>暂无可用宏观情报。</Text>
                          ) : null}
                        </Space>
                      </Card>
                    ) : null}
                    {deepIntelProbe ? (
                      <Card size="small" title={<span style={{ color: "#0f172a" }}>情报链路自检结果</span>}>
                        <Space direction="vertical" size={6} style={{ width: "100%" }}>
                          <Space wrap>
                            <Tag color={Boolean(deepIntelProbe.ok) ? "green" : "gold"}>链路可用：{String(Boolean(deepIntelProbe.ok))}</Tag>
                            <Tag color={String(deepIntelProbe.intel_status) === "external_ok" ? "green" : "orange"}>
                              情报状态：{String(deepIntelProbe.intel_status ?? "-")}
                            </Tag>
                            <Tag>引用数：{Number(deepIntelProbe.citation_count ?? 0)}</Tag>
                          </Space>
                          <Space wrap>
                            <Tag>外部检索开关：{String(Boolean(deepIntelProbe.external_enabled))}</Tag>
                            <Tag>可用Provider数：{Number(deepIntelProbe.provider_count ?? 0)}</Tag>
                            <Tag color={Boolean(deepIntelProbe.websearch_tool_applied) ? "green" : "orange"}>
                              检索工具启用：{String(Boolean(deepIntelProbe.websearch_tool_applied))}
                            </Tag>
                          </Space>
                          {String(deepIntelProbe.fallback_reason ?? "").trim() ? (
                            <Text style={{ color: "#b45309" }}>
                              降级原因：{String(deepIntelProbe.fallback_reason)}
                            </Text>
                          ) : null}
                          {String(deepIntelProbe.trace_id ?? "").trim() ? (
                            <Text style={{ color: "#64748b" }}>trace_id: {String(deepIntelProbe.trace_id)}</Text>
                          ) : null}
                        </Space>
                      </Card>
                    ) : null}
                    {deepCalendarWatch.length ? (
                      <Card size="small" title={<span style={{ color: "#0f172a" }}>未来事件关注清单</span>}>
                        <List
                          size="small"
                          dataSource={deepCalendarWatch}
                          renderItem={(item: any) => (
                            <List.Item>
                              <Space direction="vertical" size={1}>
                                <Text style={{ color: "#0f172a" }}>{String(item?.title ?? "-")}</Text>
                                <Text style={{ color: "#64748b" }}>
                                  {String(item?.published_at ?? "-")} | {String(item?.impact_direction ?? "uncertain")} | {String(item?.impact_horizon ?? "1w")}
                                </Text>
                              </Space>
                            </List.Item>
                          )}
                        />
                      </Card>
                    ) : null}
                  </>
                ) : (
                  <>
                    <Space wrap>
                      <Select
                        style={{ minWidth: 140 }}
                        value={deepArchiveRoundId}
                        onChange={(v) => setDeepArchiveRoundId(String(v ?? ""))}
                        options={deepArchiveRoundOptions}
                      />
                      <Select
                        style={{ minWidth: 190 }}
                        value={deepArchiveEventName}
                        onChange={(v) => setDeepArchiveEventName(String(v ?? ""))}
                        options={[{ label: "全部事件", value: "" }, ...deepArchiveEventOptions]}
                      />
                      <InputNumber
                        min={20}
                        max={2000}
                        step={20}
                        value={deepArchiveLimit}
                        onChange={(v) => setDeepArchiveLimit(Number(v ?? 220))}
                      />
                      <Input
                        style={{ minWidth: 200 }}
                        type="datetime-local"
                        step={1}
                        value={toDatetimeLocalValue(deepArchiveCreatedFrom)}
                        onChange={(e) => setDeepArchiveCreatedFrom(normalizeArchiveTimestamp(e.target.value))}
                        placeholder="created_from"
                      />
                      <Input
                        style={{ minWidth: 200 }}
                        type="datetime-local"
                        step={1}
                        value={toDatetimeLocalValue(deepArchiveCreatedTo)}
                        onChange={(e) => setDeepArchiveCreatedTo(normalizeArchiveTimestamp(e.target.value))}
                        placeholder="created_to"
                      />
                      <Button onClick={() => setDeepArchiveQuickWindow(24)}>最近24小时</Button>
                      <Button onClick={() => { setDeepArchiveCreatedFrom(""); setDeepArchiveCreatedTo(""); }}>清空时间过滤</Button>
                    </Space>
                    <Space wrap>
                      <Button onClick={startDeepThinkSession} loading={deepLoading}>
                        新建会话
                      </Button>
                      <Button type="primary" onClick={runDeepThinkRound} loading={deepLoading}>
                        {deepSession?.session_id ? "执行下一轮" : "执行下一轮（自动建会话）"}
                      </Button>
                      <Button onClick={runDeepThinkRoundViaA2A} loading={deepLoading}>
                        A2A派发下一轮
                      </Button>
                      <Button onClick={refreshDeepThinkSession} disabled={!deepSession?.session_id || deepLoading}>
                        刷新会话
                      </Button>
                      <Button
                        onClick={() =>
                          loadDeepThinkEventArchive(deepSession?.session_id ?? "", {
                            roundId: deepArchiveRoundId,
                            eventName: deepArchiveEventName,
                            limit: deepArchiveLimit,
                            cursor: 0,
                            createdFrom: deepArchiveCreatedFrom,
                            createdTo: deepArchiveCreatedTo,
                            historyMode: "reset"
                          })
                        }
                        disabled={!deepSession?.session_id}
                        loading={deepArchiveLoading}
                      >
                        加载会话存档
                      </Button>
                      <Button
                        onClick={loadFirstDeepThinkArchivePage}
                        disabled={!deepSession?.session_id || (deepArchiveCursor === 0 && !deepArchiveCursorHistory.length)}
                        loading={deepArchiveLoading}
                      >
                        回到第一页
                      </Button>
                      <Button
                        onClick={loadPrevDeepThinkArchivePage}
                        disabled={!deepSession?.session_id || !deepArchiveCursorHistory.length || deepArchiveLoading}
                        loading={deepArchiveLoading}
                      >
                        上一页存档
                      </Button>
                      <Button
                        onClick={loadNextDeepThinkArchivePage}
                        disabled={!deepSession?.session_id || !deepArchiveHasMore || deepArchiveLoading}
                        loading={deepArchiveLoading}
                      >
                        下一页存档
                      </Button>
                      <Button
                        onClick={() => exportDeepThinkEventArchive("jsonl")}
                        disabled={!deepSession?.session_id}
                        loading={deepArchiveExporting}
                      >
                        导出审计JSONL
                      </Button>
                      <Button
                        onClick={() => exportDeepThinkEventArchive("csv")}
                        disabled={!deepSession?.session_id}
                        loading={deepArchiveExporting}
                      >
                        导出审计CSV
                      </Button>
                      <Button onClick={() => replayDeepThinkStream(deepSession?.session_id ?? "")} disabled={!deepSession?.session_id} loading={deepStreaming}>
                        回放最新轮次流
                      </Button>
                    </Space>
                    <Text style={{ color: "#64748b" }}>
                      提示：工程模式可直接“执行下一轮”，系统会自动建会话并推进轮次。
                    </Text>
                    {deepStockSwitchNotice ? (
                      <Alert
                        type="info"
                        showIcon
                        message={deepStockSwitchNotice}
                        closable
                        onClose={() => setDeepStockSwitchNotice("")}
                      />
                    ) : null}
                  </>
                )}
                {effectiveConsoleMode === "analysis" ? (
                  <Card size="small" title={<span style={{ color: "#0f172a" }}>如何使用这块面板</span>}>
                    <Space direction="vertical" size={4} style={{ width: "100%" }}>
                      <Text style={{ color: "#334155" }}>1. 在上方输入区点击“开始全流程分析”，系统会自动完成流式分析和轮次推进。</Text>
                      <Text style={{ color: "#334155" }}>2. 重点查看“本轮结论摘要/为什么是这个结论/风险与行动建议”。</Text>
                      <Text style={{ color: "#334155" }}>3. 若触发补证重规划或存在高风险冲突，可在工程控制台继续执行复核轮次。</Text>
                      <Space wrap>
                        <Tag color={deepSession ? "blue" : "default"}>会话：{deepSession?.session_id ?? "未创建"}</Tag>
                        <Tag color={deepSession?.status === "completed" ? "green" : "processing"}>状态：{deepSession?.status ?? "idle"}</Tag>
                        <Tag>轮次：{deepSession?.current_round ?? 0}/{deepSession?.max_rounds ?? 0}</Tag>
                        {latestDeepRound?.replan_triggered ? <Tag color="orange">已触发补证重规划</Tag> : null}
                        {latestDeepRound?.stop_reason ? <Tag color="red">停止原因：{latestDeepRound.stop_reason}</Tag> : null}
                      </Space>
                    </Space>
                  </Card>
                ) : (
                  <>
                    <Space wrap>
                      <Tag color={deepSession ? "blue" : "default"}>session: {deepSession?.session_id ?? "未创建"}</Tag>
                      <Tag color={deepSession?.status === "completed" ? "green" : "processing"}>status: {deepSession?.status ?? "idle"}</Tag>
                      <Tag>round: {deepSession?.current_round ?? 0}/{deepSession?.max_rounds ?? 0}</Tag>
                      <Tag color={deepArchiveCount > 0 ? "cyan" : "default"}>archive_events: {deepArchiveCount}</Tag>
                      <Tag color={deepArchiveHasMore ? "gold" : "default"}>has_more: {String(deepArchiveHasMore)}</Tag>
                      <Tag>next_cursor: {deepArchiveNextCursor ?? "-"}</Tag>
                      <Tag>cursor_stack: {deepArchiveCursorHistory.length}</Tag>
                      <Tag color={deepArchiveExportTask?.status === "failed" ? "red" : deepArchiveExportTask?.status === "completed" ? "green" : "blue"}>
                        export_task: {deepArchiveExportTask?.status ?? "idle"}
                      </Tag>
                      {deepArchiveExportTask?.task_id ? <Tag>task_id: {deepArchiveExportTask.task_id}</Tag> : null}
                      {deepArchiveExportTask?.task_id ? <Tag>attempt: {deepArchiveExportTask.attempt_count}/{deepArchiveExportTask.max_attempts}</Tag> : null}
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
                  </>
                )}
                {deepError ? <Alert message={deepError} type="error" showIcon /> : null}
              </Space>
            </Card>

            {/* 分析模式聚焦业务决策解释；工程模式展示全量排障与治理数据。 */}
            {effectiveConsoleMode === "analysis" ? (
              <Card className="premium-card" title={<span style={{ color: "#0f172a" }}>执行过程摘要（业务视角）</span>} style={{ marginTop: 12 }}>
                <Space direction="vertical" size={8} style={{ width: "100%" }}>
                  <Text style={{ color: "#334155" }}>
                    已执行轮次：{deepRounds.length} | 当前最新轮次任务数：{deepTaskRows.length}
                  </Text>
                  <Space wrap>
                    {(latestDeepRound?.conflict_sources ?? []).map((src) => (
                      <Tag key={`analysis-conflict-${src}`} color="orange">
                        {getConflictSourceLabel(String(src))}
                      </Tag>
                    ))}
                    {!(latestDeepRound?.conflict_sources ?? []).length ? <Tag color="green">当前无显著冲突源</Tag> : null}
                  </Space>
                  {deepTimelineItems.length ? (
                    <Timeline items={deepTimelineItems.slice(-3)} />
                  ) : (
                    <Text style={{ color: "#64748b" }}>尚未执行 DeepThink 轮次。请先点击“开始全流程分析”。</Text>
                  )}
                </Space>
              </Card>
            ) : (
              <>
                <Card className="premium-card" title={<span style={{ color: "#0f172a" }}>Round Timeline</span>} style={{ marginTop: 12 }}>
                  <Text style={{ color: "#64748b" }}>
                    作用：按时间查看每轮是否触发重规划、在哪一轮停止，以及分歧是否在收敛。
                  </Text>
                  {deepTimelineItems.length ? (
                    <Timeline items={deepTimelineItems} />
                  ) : (
                    <Text style={{ color: "#64748b" }}>尚未执行 DeepThink 轮次。可先在业务页发起全流程，或在工程模式手动建会话后执行。</Text>
                  )}
                </Card>

                <Card className="premium-card" title={<span style={{ color: "#0f172a" }}>最新轮次 Task Graph</span>} style={{ marginTop: 12 }}>
                  <Text style={{ color: "#64748b" }}>
                    作用：展示本轮各 Agent 的任务分工与优先级，判断当前轮次在解决什么问题。
                  </Text>
                  <Table
                    size="small"
                    pagination={false}
                    locale={{ emptyText: "无任务图数据" }}
                    dataSource={deepTaskRows}
                    columns={[
                      { title: "任务ID", dataIndex: "task_id", key: "task_id", width: 110 },
                      {
                        title: "Agent",
                        dataIndex: "agent_meta",
                        key: "agent_meta",
                        width: 170,
                        render: (_: unknown, row: any) => (
                          <Space direction="vertical" size={0}>
                            <Text style={{ color: "#0f172a" }}>{row.agent_meta?.name ?? row.agent}</Text>
                            <Text style={{ color: "#64748b" }}>{row.agent_meta?.role ?? row.agent}</Text>
                          </Space>
                        )
                      },
                      { title: "任务说明", dataIndex: "title", key: "title" },
                      {
                        title: "优先级",
                        dataIndex: "priority",
                        key: "priority",
                        width: 100,
                        render: (v: string) => <Tag color={v === "high" ? "red" : v === "medium" ? "gold" : "blue"}>{getPriorityLabel(v)}</Tag>
                      }
                    ]}
                  />
                </Card>
              </>
            )}
          </Col>

          <Col xs={24} xl={10}>
            {/* 右侧面板在分析模式给出可行动信息，避免业务用户被技术噪音淹没。 */}
            {effectiveConsoleMode === "analysis" ? (
              <>
                <Card className="premium-card" title={<span style={{ color: "#0f172a" }}>关键 Agent 观点（业务版）</span>}>
                  <Table
                    size="small"
                    pagination={false}
                    locale={{ emptyText: "暂无观点数据" }}
                    dataSource={deepOpinionRows}
                    columns={[
                      {
                        title: "角色",
                        dataIndex: "agent_meta",
                        key: "agent_meta",
                        width: 186,
                        render: (_: unknown, row: any) => (
                          <Space direction="vertical" size={0}>
                            <Text style={{ color: "#0f172a" }}>{row.agent_meta?.name ?? row.agent_id}</Text>
                            <Text style={{ color: "#64748b" }}>{row.agent_meta?.role ?? row.agent_id}</Text>
                          </Space>
                        )
                      },
                      {
                        title: "观点",
                        dataIndex: "signal",
                        key: "signal",
                        width: 84,
                        render: (v: string) => <Tag color={v === "buy" ? "green" : v === "reduce" ? "red" : "blue"}>{getSignalLabel(v)}</Tag>
                      },
                      {
                        title: "置信度",
                        dataIndex: "confidence",
                        key: "confidence",
                        width: 88,
                        render: (v: number) => v.toFixed(3)
                      },
                      { title: "关键理由", dataIndex: "reason", key: "reason" }
                    ]}
                  />
                </Card>
                <Card className="premium-card" title={<span style={{ color: "#0f172a" }}>冲突与行动提示</span>} style={{ marginTop: 12 }}>
                  <Space direction="vertical" size={8} style={{ width: "100%" }}>
                    <Space wrap>
                      {(latestDeepRound?.conflict_sources ?? []).map((src) => (
                        <Tag key={`analysis-right-${src}`} color="orange">{getConflictSourceLabel(String(src))}</Tag>
                      ))}
                      {!(latestDeepRound?.conflict_sources ?? []).length ? <Tag color="green">暂无冲突源</Tag> : null}
                    </Space>
                    <Text style={{ color: "#334155" }}>
                      当前轮次的分歧度越高，建议动作越保守；出现风险/合规否决时优先执行风险控制。
                    </Text>
                  </Space>
                </Card>
              </>
            ) : (
              <>
                <Card className="premium-card" title={<span style={{ color: "#0f172a" }}>预算使用与剩余</span>}>
                  <Text style={{ color: "#64748b" }}>
                    作用：监控 token/时间/工具调用预算，防止单轮过度消耗导致降级或中断。
                  </Text>
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
                  <Text style={{ color: "#64748b" }}>
                    作用：识别“信号分歧、证据冲突、风险否决”等冲突来源，辅助决定是否继续补证。
                  </Text>
                  <ReactECharts option={deepConflictOption} style={{ height: 240 }} />
                  <Space wrap>
                    {(latestDeepRound?.conflict_sources ?? []).map((src) => (
                      <Tag key={src} color="orange">{getConflictSourceLabel(String(src))}</Tag>
                    ))}
                    {!(latestDeepRound?.conflict_sources ?? []).length ? <Tag>暂无冲突源</Tag> : null}
                  </Space>
                </Card>

                <Card className="premium-card" title={<span style={{ color: "#0f172a" }}>最新轮次 Agent 观点</span>} style={{ marginTop: 12 }}>
                  <Text style={{ color: "#64748b" }}>
                    作用：查看各角色对同一标的的动作建议与置信度，快速定位谁在拉高/拉低共识。
                  </Text>
                  <Table
                    size="small"
                    pagination={false}
                    locale={{ emptyText: "暂无观点数据" }}
                    dataSource={deepOpinionRows}
                    columns={[
                      {
                        title: "Agent",
                        dataIndex: "agent_meta",
                        key: "agent_meta",
                        width: 168,
                        render: (_: unknown, row: any) => (
                          <Space direction="vertical" size={0}>
                            <Text style={{ color: "#0f172a" }}>{row.agent_meta?.name ?? row.agent_id}</Text>
                            <Text style={{ color: "#64748b" }}>{row.agent_id}</Text>
                          </Space>
                        )
                      },
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

                <Card className="premium-card" title={<span style={{ color: "#0f172a" }}>跨轮次观点差分</span>} style={{ marginTop: 12 }}>
                  <Table
                    size="small"
                    pagination={false}
                    locale={{ emptyText: "至少执行两轮后可查看差分" }}
                    dataSource={deepOpinionDiffRows}
                    columns={[
                      {
                        title: "Agent",
                        dataIndex: "agent_meta",
                        key: "agent_meta",
                        width: 128,
                        render: (_: unknown, row: any) => row.agent_meta?.name ?? row.agent_id
                      },
                      {
                        title: "Signal",
                        key: "signal_diff",
                        width: 160,
                        render: (_: unknown, row: any) => (
                          <Text style={{ color: "#334155" }}>{row.prev_signal_label} → {row.curr_signal_label}</Text>
                        )
                      },
                      {
                        title: "ΔConf",
                        dataIndex: "delta_confidence",
                        key: "delta_confidence",
                        width: 92,
                        render: (v: number) => (
                          <Text style={{ color: v > 0 ? "#059669" : v < 0 ? "#dc2626" : "#475569" }}>{v.toFixed(4)}</Text>
                        )
                      },
                      {
                        title: "Type",
                        dataIndex: "change_type",
                        key: "change_type",
                        width: 120,
                        render: (v: string) => (
                          <Tag color={v === "signal_changed" ? "red" : v === "confidence_shift" ? "gold" : "blue"}>{v}</Tag>
                        )
                      }
                    ]}
                  />
                </Card>

                <Card className="premium-card" title={<span style={{ color: "#0f172a" }}>冲突源下钻（证据视角）</span>} style={{ marginTop: 12 }}>
                  <Table
                    size="small"
                    pagination={false}
                    locale={{ emptyText: "暂无冲突观点可下钻" }}
                    dataSource={deepConflictDrillRows}
                    columns={[
                      {
                        title: "Agent",
                        dataIndex: "agent_meta",
                        key: "agent_meta",
                        width: 146,
                        render: (_: unknown, row: any) => row.agent_meta?.name ?? row.agent_id
                      },
                      {
                        title: "Signal",
                        dataIndex: "signal",
                        key: "signal",
                        width: 84,
                        render: (v: string) => <Tag color={v === "buy" ? "green" : v === "reduce" ? "red" : "blue"}>{v}</Tag>
                      },
                      {
                        title: "Confidence",
                        dataIndex: "confidence",
                        key: "confidence",
                        width: 100,
                        render: (v: number) => v.toFixed(3)
                      },
                      { title: "Evidence IDs", dataIndex: "evidence_ids", key: "evidence_ids" }
                    ]}
                  />
                </Card>

                <Card className="premium-card" title={<span style={{ color: "#0f172a" }}>SSE 回放事件</span>} style={{ marginTop: 12 }}>
                  <Text style={{ color: "#64748b" }}>
                    当前筛选：round={deepArchiveRoundId ? deepArchiveRoundId : "all"} | event={deepArchiveEventName || "all"} | limit={deepArchiveLimit} | cursor={deepArchiveCursor} | from={deepArchiveCreatedFrom || "-"} | to={deepArchiveCreatedTo || "-"}
                  </Text>
                  <List
                    size="small"
                    locale={{ emptyText: "暂无流事件记录" }}
                    dataSource={deepReplayRows}
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
              </>
            )}
          </Col>
        </Row>
      </motion.section>

      {!isConsoleWorkspace ? (
        <>
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
        </>
      ) : null}
    </main>
  );
}

