"use client";

import { useEffect, useMemo, useState } from "react";
import {
  Alert,
  Button,
  Card,
  Col,
  Collapse,
  Drawer,
  Input,
  InputNumber,
  Progress,
  Row,
  Segmented,
  Select,
  Space,
  Switch,
  Table,
  Tag,
  Typography,
} from "antd";

const API_BASE = process.env.NEXT_PUBLIC_API_BASE ?? "http://127.0.0.1:8000";
const { Title, Text, Paragraph } = Typography;

type RagMode = "business" | "ops";
type RagOpsTab = "source" | "chunk" | "memory" | "trace";
type ChunkStatus = "active" | "review" | "rejected" | "archived";
type RagUploadPreset = "financial_report" | "announcement" | "research" | "meeting_note" | "custom";

const CHUNK_STATUS_OPTIONS: Array<{ label: string; value: ChunkStatus }> = [
  { label: "active", value: "active" },
  { label: "review", value: "review" },
  { label: "rejected", value: "rejected" },
  { label: "archived", value: "archived" },
];

const SOURCE_OPTIONS = [
  { label: "user_upload", value: "user_upload" },
  { label: "cninfo", value: "cninfo" },
  { label: "eastmoney", value: "eastmoney" },
  { label: "research", value: "research" },
];

// 业务模式预设：用于降低输入复杂度，默认给出来源与标签。
const PRESET_CONFIG: Record<RagUploadPreset, { label: string; source: string; tags: string[]; hint: string }> = {
  financial_report: { label: "财报", source: "eastmoney", tags: ["财报"], hint: "适合财报、业绩快报和财务附注。" },
  announcement: { label: "公告", source: "cninfo", tags: ["公告"], hint: "适合公告、问询函、回复函和重大事项披露。" },
  research: { label: "研报", source: "research", tags: ["研报"], hint: "适合券商研报、行业点评和策略报告。" },
  meeting_note: { label: "会议纪要", source: "user_upload", tags: ["会议纪要"], hint: "适合路演纪要、电话会纪要和调研摘要。" },
  custom: { label: "自定义", source: "user_upload", tags: [], hint: "保留完整自定义能力，可手动配置来源和标签。" },
};

function parseCommaValues(raw: string): string[] {
  return raw
    .split(",")
    .map((x) => x.trim())
    .filter((x) => x.length > 0);
}

export default function RagCenterPage() {
  const [mode, setMode] = useState<RagMode>("business");
  const [tab, setTab] = useState<RagOpsTab>("source");
  const [loading, setLoading] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState("");
  const [message, setMessage] = useState("");

  const [dashboard, setDashboard] = useState<Record<string, any> | null>(null);
  const [uploadRows, setUploadRows] = useState<Record<string, any>[]>([]);
  const [sourceRows, setSourceRows] = useState<Record<string, any>[]>([]);
  const [chunkRows, setChunkRows] = useState<Record<string, any>[]>([]);
  const [memoryRows, setMemoryRows] = useState<Record<string, any>[]>([]);
  const [traceRows, setTraceRows] = useState<Record<string, any>[]>([]);
  const [traceCount, setTraceCount] = useState(0);

  const [uploadFile, setUploadFile] = useState<File | null>(null);
  const [uploadPreset, setUploadPreset] = useState<RagUploadPreset>("financial_report");
  const [uploadSource, setUploadSource] = useState("user_upload");
  const [uploadStockCodes, setUploadStockCodes] = useState("");
  const [uploadTags, setUploadTags] = useState("");
  const [uploadAutoIndex, setUploadAutoIndex] = useState(true);
  const [showUploadAdvanced, setShowUploadAdvanced] = useState(false);
  const [retrievalPreview, setRetrievalPreview] = useState<Record<string, any> | null>(null);
  const [detailOpen, setDetailOpen] = useState(false);
  const [detailLoading, setDetailLoading] = useState(false);
  const [detailChunk, setDetailChunk] = useState<Record<string, any> | null>(null);
  const [detailContext, setDetailContext] = useState<Record<string, any> | null>(null);
  const [detailQuery, setDetailQuery] = useState("");

  const [editSource, setEditSource] = useState("user_upload");
  const [editAutoApprove, setEditAutoApprove] = useState(false);
  const [editTrustScore, setEditTrustScore] = useState(0.7);
  const [editEnabled, setEditEnabled] = useState(true);

  const [chunkDocId, setChunkDocId] = useState("");
  const [chunkSource, setChunkSource] = useState("");
  const [chunkStatusFilter, setChunkStatusFilter] = useState("");
  const [chunkStockCode, setChunkStockCode] = useState("");
  const [chunkLimit, setChunkLimit] = useState(40);
  const [chunkActionStatus, setChunkActionStatus] = useState<ChunkStatus>("review");

  const [memoryStockCode, setMemoryStockCode] = useState("");
  const [memoryLimit, setMemoryLimit] = useState(40);
  const [memoryRetrievalEnabled, setMemoryRetrievalEnabled] = useState<number>(-1);

  const [traceId, setTraceId] = useState("");
  const [traceLimit, setTraceLimit] = useState(50);

  useEffect(() => {
    refreshBusiness().catch(() => undefined);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // 当用户切换到非 custom 预设时，同步默认来源与标签，避免重复填写。
  useEffect(() => {
    if (uploadPreset === "custom") return;
    const preset = PRESET_CONFIG[uploadPreset];
    setUploadSource(preset.source);
    setUploadTags(preset.tags.join(","));
  }, [uploadPreset]);

  async function parseOrThrow(resp: Response) {
    const body = await resp.json();
    if (!resp.ok) throw new Error(body?.detail ?? body?.error ?? `HTTP ${resp.status}`);
    return body;
  }

  function resetFeedback() {
    setError("");
    setMessage("");
  }

  async function fileToBase64(file: File): Promise<string> {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = () => {
        const value = reader.result;
        if (!(value instanceof ArrayBuffer)) {
          reject(new Error("文件读取失败"));
          return;
        }
        const bytes = new Uint8Array(value);
        let binary = "";
        for (let i = 0; i < bytes.length; i += 1) binary += String.fromCharCode(bytes[i]);
        resolve(btoa(binary));
      };
      reader.onerror = () => reject(new Error("文件读取失败"));
      reader.readAsArrayBuffer(file);
    });
  }

  async function refreshBusiness() {
    setLoading(true);
    resetFeedback();
    try {
      // 业务模式只依赖看板和上传记录，保证刷新速度与稳定性。
      const [dashboardResp, uploadsResp] = await Promise.all([
        fetch(`${API_BASE}/v1/rag/dashboard`),
        fetch(`${API_BASE}/v1/rag/uploads?limit=30`),
      ]);
      setDashboard((await parseOrThrow(dashboardResp)) as Record<string, any>);
      setUploadRows(((await parseOrThrow(uploadsResp)) as Record<string, any>[]) ?? []);
      setMessage("RAG 业务看板已刷新");
    } catch (e) {
      setError(e instanceof Error ? e.message : "刷新业务看板失败");
    } finally {
      setLoading(false);
    }
  }

  async function loadRetrievalPreview(docId: string): Promise<Record<string, any> | null> {
    const normalized = docId.trim();
    if (!normalized) return null;
    try {
      const params = new URLSearchParams({
        doc_id: normalized,
        max_queries: "2",
        top_k: "4",
      });
      const resp = await fetch(`${API_BASE}/v1/rag/retrieval-preview?${params.toString()}`);
      return (await parseOrThrow(resp)) as Record<string, any>;
    } catch {
      // Preview failure should not block upload flow.
      return null;
    }
  }

  async function openChunkDetail(hit: Record<string, any>, queryText: string) {
    const chunkId = String(hit?.chunk_id ?? "").trim();
    if (!chunkId) {
      setError("当前命中项没有 chunk_id，无法定位到文档片段");
      return;
    }
    setDetailOpen(true);
    setDetailLoading(true);
    setDetailChunk(null);
    setDetailContext(null);
    setDetailQuery(queryText);
    try {
      const params = new URLSearchParams({ context_window: "1" });
      const resp = await fetch(`${API_BASE}/v1/rag/docs/chunks/${encodeURIComponent(chunkId)}?${params.toString()}`);
      const body = (await parseOrThrow(resp)) as Record<string, any>;
      setDetailChunk((body?.chunk as Record<string, any> | undefined) ?? null);
      setDetailContext((body?.context as Record<string, any> | undefined) ?? null);
    } catch (e) {
      setError(e instanceof Error ? e.message : "加载 chunk 定位详情失败");
    } finally {
      setDetailLoading(false);
    }
  }

  async function submitBusinessUpload() {
    if (!uploadFile) {
      setError("请先选择附件");
      return;
    }
    setUploading(true);
    resetFeedback();
    setRetrievalPreview(null);
    try {
      const contentBase64 = await fileToBase64(uploadFile);
      const preset = PRESET_CONFIG[uploadPreset];
      const normalizedTags = parseCommaValues(uploadTags);
      const payload = {
        filename: uploadFile.name,
        content_type: uploadFile.type,
        content_base64: contentBase64,
        source: uploadSource || preset.source,
        stock_codes: parseCommaValues(uploadStockCodes).map((x) => x.toUpperCase()),
        // 若用户未填标签则自动退回预设标签，保证检索可用元数据。
        tags: normalizedTags.length ? normalizedTags : preset.tags,
        auto_index: uploadAutoIndex,
        user_id: "frontend-rag",
      };
      const resp = await fetch(`${API_BASE}/v1/rag/workflow/upload-and-index`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      const body = (await parseOrThrow(resp)) as Record<string, any>;
      const docId = String(body?.result?.doc_id ?? "").trim();
      // Prefer workflow embedded preview, and fallback to dedicated endpoint for compatibility.
      const preview =
        (body?.retrieval_preview as Record<string, any> | undefined) ??
        (docId ? await loadRetrievalPreview(docId) : null);
      if (preview) setRetrievalPreview(preview);
      if (Boolean(body?.result?.dedupe_hit)) {
        setMessage(`检测到重复文件，已复用既有资产：${String(body?.result?.doc_id ?? "-")}`);
      } else {
        setMessage(`上传并入库完成：${String(body?.result?.doc_id ?? "-")}`);
      }
      await refreshBusiness();
    } catch (e) {
      setError(e instanceof Error ? e.message : "上传失败");
    } finally {
      setUploading(false);
    }
  }
  async function loadSourcePolicy() {
    setLoading(true);
    resetFeedback();
    try {
      const resp = await fetch(`${API_BASE}/v1/rag/source-policy`);
      const rows = (await parseOrThrow(resp)) as Record<string, any>[];
      setSourceRows(rows);
      setMessage(`来源策略已加载，共 ${rows.length} 条`);
    } catch (e) {
      setError(e instanceof Error ? e.message : "加载来源策略失败");
    } finally {
      setLoading(false);
    }
  }

  async function saveSourcePolicy() {
    setLoading(true);
    resetFeedback();
    try {
      const source = editSource.trim().toLowerCase();
      if (!source) throw new Error("source 不能为空");
      const resp = await fetch(`${API_BASE}/v1/rag/source-policy/${encodeURIComponent(source)}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ auto_approve: editAutoApprove, trust_score: Number(editTrustScore), enabled: editEnabled }),
      });
      await parseOrThrow(resp);
      await loadSourcePolicy();
      setMessage(`来源策略已更新：${source}`);
    } catch (e) {
      setError(e instanceof Error ? e.message : "保存来源策略失败");
    } finally {
      setLoading(false);
    }
  }

  async function loadChunks() {
    setLoading(true);
    resetFeedback();
    try {
      const params = new URLSearchParams();
      if (chunkDocId.trim()) params.set("doc_id", chunkDocId.trim());
      if (chunkSource.trim()) params.set("source", chunkSource.trim().toLowerCase());
      if (chunkStatusFilter.trim()) params.set("status", chunkStatusFilter.trim().toLowerCase());
      if (chunkStockCode.trim()) params.set("stock_code", chunkStockCode.trim().toUpperCase());
      params.set("limit", String(Math.max(1, Math.min(200, chunkLimit))));
      const resp = await fetch(`${API_BASE}/v1/rag/docs/chunks?${params.toString()}`);
      const rows = (await parseOrThrow(resp)) as Record<string, any>[];
      setChunkRows(rows);
      setMessage(`文档资产已加载，共 ${rows.length} 条`);
    } catch (e) {
      setError(e instanceof Error ? e.message : "加载文档资产失败");
    } finally {
      setLoading(false);
    }
  }

  async function setChunkStatus(chunkId: string, status: ChunkStatus) {
    setLoading(true);
    resetFeedback();
    try {
      const resp = await fetch(`${API_BASE}/v1/rag/docs/chunks/${encodeURIComponent(chunkId)}/status`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ status }),
      });
      await parseOrThrow(resp);
      await loadChunks();
      setMessage(`Chunk 状态已更新：${chunkId} -> ${status}`);
    } catch (e) {
      setError(e instanceof Error ? e.message : "更新 Chunk 状态失败");
    } finally {
      setLoading(false);
    }
  }

  async function loadQAMemory() {
    setLoading(true);
    resetFeedback();
    try {
      const params = new URLSearchParams();
      if (memoryStockCode.trim()) params.set("stock_code", memoryStockCode.trim().toUpperCase());
      if (memoryRetrievalEnabled >= 0) params.set("retrieval_enabled", String(memoryRetrievalEnabled));
      params.set("limit", String(Math.max(1, Math.min(200, memoryLimit))));
      const resp = await fetch(`${API_BASE}/v1/rag/qa-memory?${params.toString()}`);
      const rows = (await parseOrThrow(resp)) as Record<string, any>[];
      setMemoryRows(rows);
      setMessage(`共享问答语料已加载，共 ${rows.length} 条`);
    } catch (e) {
      setError(e instanceof Error ? e.message : "加载共享问答语料失败");
    } finally {
      setLoading(false);
    }
  }

  async function toggleMemory(memoryId: string, retrievalEnabled: boolean) {
    setLoading(true);
    resetFeedback();
    try {
      const resp = await fetch(`${API_BASE}/v1/rag/qa-memory/${encodeURIComponent(memoryId)}/toggle`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ retrieval_enabled: retrievalEnabled }),
      });
      await parseOrThrow(resp);
      await loadQAMemory();
      setMessage(`问答语料状态已更新：${memoryId}`);
    } catch (e) {
      setError(e instanceof Error ? e.message : "更新问答语料状态失败");
    } finally {
      setLoading(false);
    }
  }

  async function loadTrace() {
    setLoading(true);
    resetFeedback();
    try {
      const params = new URLSearchParams();
      if (traceId.trim()) params.set("trace_id", traceId.trim());
      params.set("limit", String(Math.max(1, Math.min(300, traceLimit))));
      const resp = await fetch(`${API_BASE}/v1/ops/rag/retrieval-trace?${params.toString()}`);
      const body = (await parseOrThrow(resp)) as Record<string, any>;
      const rows = (body.items ?? []) as Record<string, any>[];
      setTraceRows(rows);
      setTraceCount(Number(body.count ?? rows.length));
      setMessage(`检索追踪已加载，共 ${Number(body.count ?? rows.length)} 条`);
    } catch (e) {
      setError(e instanceof Error ? e.message : "加载检索追踪失败");
    } finally {
      setLoading(false);
    }
  }

  async function runReindex() {
    setLoading(true);
    resetFeedback();
    try {
      const resp = await fetch(`${API_BASE}/v1/ops/rag/reindex`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ limit: 2000 }),
      });
      const body = await parseOrThrow(resp);
      setMessage(`索引重建完成：backend=${String(body.index_backend ?? "-")} count=${String(body.indexed_count ?? "-")}`);
      await refreshBusiness();
    } catch (e) {
      setError(e instanceof Error ? e.message : "重建索引失败");
    } finally {
      setLoading(false);
    }
  }

  const sourceColumns: any[] = [
    { title: "Source", dataIndex: "source", key: "source", width: 220 },
    { title: "Auto Approve", dataIndex: "auto_approve", key: "auto_approve", width: 130, render: (v: boolean) => <Tag color={v ? "green" : "orange"}>{String(Boolean(v))}</Tag> },
    { title: "Trust", dataIndex: "trust_score", key: "trust_score", width: 100, render: (v: number) => Number(v ?? 0).toFixed(2) },
    { title: "Enabled", dataIndex: "enabled", key: "enabled", width: 110, render: (v: boolean) => <Tag color={v ? "green" : "red"}>{String(Boolean(v))}</Tag> },
    { title: "Updated", dataIndex: "updated_at", key: "updated_at" },
  ];

  const chunkColumns: any[] = [
    { title: "Chunk ID", dataIndex: "chunk_id", key: "chunk_id", width: 220 },
    { title: "Doc ID", dataIndex: "doc_id", key: "doc_id", width: 150 },
    { title: "No", dataIndex: "chunk_no", key: "chunk_no", width: 70 },
    { title: "Source", dataIndex: "source", key: "source", width: 120 },
    { title: "Status", dataIndex: "effective_status", key: "effective_status", width: 120, render: (v: string) => <Tag color={v === "active" ? "green" : v === "review" ? "gold" : v === "rejected" ? "red" : "default"}>{v}</Tag> },
    { title: "Quality", dataIndex: "quality_score", key: "quality_score", width: 90, render: (v: number) => Number(v ?? 0).toFixed(2) },
    { title: "Stocks", dataIndex: "stock_codes", key: "stock_codes", render: (v: string[]) => <Text style={{ color: "#475569" }}>{Array.isArray(v) ? v.join(", ") : ""}</Text> },
  ];

  const memoryColumns: any[] = [
    { title: "Memory ID", dataIndex: "memory_id", key: "memory_id", width: 220 },
    { title: "Stock", dataIndex: "stock_code", key: "stock_code", width: 100 },
    { title: "Intent", dataIndex: "intent", key: "intent", width: 96 },
    { title: "Quality", dataIndex: "quality_score", key: "quality_score", width: 96, render: (v: number) => Number(v ?? 0).toFixed(2) },
    { title: "Retrieval", dataIndex: "retrieval_enabled", key: "retrieval_enabled", width: 110, render: (v: boolean) => <Tag color={v ? "green" : "orange"}>{String(Boolean(v))}</Tag> },
    { title: "Summary", dataIndex: "summary_text", key: "summary_text", render: (v: string) => <Text style={{ color: "#334155" }}>{String(v ?? "").slice(0, 120)}</Text> },
  ];

  const traceColumns: any[] = [
    { title: "ID", dataIndex: "id", key: "id", width: 70 },
    { title: "Trace ID", dataIndex: "trace_id", key: "trace_id", width: 220 },
    { title: "Type", dataIndex: "query_type", key: "query_type", width: 120 },
    { title: "Latency(ms)", dataIndex: "latency_ms", key: "latency_ms", width: 120 },
    { title: "Retrieved -> Selected", key: "selected", render: (_: unknown, row: any) => <Text style={{ color: "#334155" }}>{(row.retrieved_ids ?? []).slice(0, 3).join(", ")} {"->"} {(row.selected_ids ?? []).slice(0, 3).join(", ")}</Text> },
  ];

  const uploadColumns: any[] = [
    { title: "文件", dataIndex: "filename", key: "filename", width: 220 },
    { title: "来源", dataIndex: "source", key: "source", width: 110 },
    { title: "状态", dataIndex: "status", key: "status", width: 120, render: (v: string) => <Tag color={v === "active" ? "green" : v === "review" ? "gold" : v === "rejected" ? "red" : "blue"}>{v}</Tag> },
    { title: "大小", dataIndex: "file_size", key: "file_size", width: 110, render: (v: number) => `${(Number(v ?? 0) / 1024).toFixed(1)} KB` },
    { title: "Doc ID", dataIndex: "doc_id", key: "doc_id", width: 170 },
    { title: "更新时间", dataIndex: "updated_at", key: "updated_at" },
  ];

  const opsTabHint = useMemo(() => {
    if (tab === "source") return "按来源控制文档自动生效与信任分。";
    if (tab === "chunk") return "查看/下线已入库 chunk，控制在线检索质量。";
    if (tab === "memory") return "管理共享问答语料，快速禁用污染样本。";
    return "观察检索召回轨迹并触发向量索引重建。";
  }, [tab]);

  return (
    <main className="container shell-fade-in">
      <Card className="premium-card">
        <Space direction="vertical" style={{ width: "100%" }} size={8}>
          <Title level={2} style={{ margin: 0 }}>RAG 语料中心</Title>
          <Paragraph style={{ margin: 0, color: "#475569", maxWidth: 920 }}>
            默认提供业务模式：只保留上传入库主流程。运维模式保留完整治理能力，按需切换。
          </Paragraph>
          <Segmented
            value={mode}
            onChange={(value) => setMode(value as RagMode)}
            options={[
              { label: "业务模式", value: "business" },
              { label: "运维模式", value: "ops" },
            ]}
          />
        </Space>
      </Card>

      {message ? <Alert style={{ marginTop: 10 }} type="success" showIcon message={message} /> : null}
      {error ? <Alert style={{ marginTop: 10 }} type="error" showIcon message={error} /> : null}

      {mode === "business" ? (
        <>
          <Card className="premium-card" style={{ marginTop: 10 }} title="快速上传（业务入口）">
            <Space direction="vertical" style={{ width: "100%" }} size={12}>
              <Text style={{ color: "#64748b" }}>
                步骤：选择资料类型 → 选择文件 → 点击“上传并生效”。默认无需填写复杂参数。
              </Text>

              <Space wrap>
                <Text style={{ color: "#334155" }}>资料类型</Text>
                <Segmented
                  value={uploadPreset}
                  onChange={(v) => setUploadPreset(v as RagUploadPreset)}
                  options={[
                    { label: "财报", value: "financial_report" },
                    { label: "公告", value: "announcement" },
                    { label: "研报", value: "research" },
                    { label: "会议纪要", value: "meeting_note" },
                    { label: "自定义", value: "custom" },
                  ]}
                />
              </Space>

              <Alert
                type="info"
                showIcon
                message={`当前类型：${PRESET_CONFIG[uploadPreset].label}`}
                description={PRESET_CONFIG[uploadPreset].hint}
              />

              <input
                type="file"
                onChange={(e) => {
                  const file = e.target.files?.[0] ?? null;
                  setUploadFile(file);
                  setRetrievalPreview(null);
                }}
              />

              <Space wrap>
                <Tag color={uploadFile ? "green" : "default"}>{uploadFile ? "文件已就绪" : "请先选择文件"}</Tag>
                <Tag>来源：{uploadSource || "-"}</Tag>
                <Tag>自动索引：{String(uploadAutoIndex)}</Tag>
              </Space>

              <Space wrap>
                <Button type="primary" onClick={submitBusinessUpload} loading={uploading} disabled={!uploadFile}>
                  上传并生效
                </Button>
                <Button onClick={refreshBusiness} loading={loading}>刷新看板</Button>
                <Button onClick={runReindex} loading={loading}>重建向量索引</Button>
              </Space>

              <Text style={{ color: "#475569" }}>
                当前文件：{uploadFile ? `${uploadFile.name} (${(uploadFile.size / 1024).toFixed(1)} KB)` : "未选择"}
              </Text>

              {retrievalPreview ? (
                <Card size="small" title="可检索样本预览">
                  <Space direction="vertical" style={{ width: "100%" }} size={8}>
                    <Alert
                      type={Boolean(retrievalPreview.passed) ? "success" : "warning"}
                      showIcon
                      message={
                        Boolean(retrievalPreview.passed)
                          ? "已验证：当前文档可被在线检索命中"
                          : "提示：当前样本未命中本次文档，建议重试或检查入库状态"
                      }
                      description={`doc_id=${String(retrievalPreview.doc_id ?? "-")}，命中率=${(
                        Number(retrievalPreview.target_hit_rate ?? 0) * 100
                      ).toFixed(1)}%，命中查询=${String(retrievalPreview.matched_query_count ?? 0)}/${String(
                        retrievalPreview.query_count ?? 0,
                      )}`}
                    />
                    {(retrievalPreview.items ?? []).map((item: any, index: number) => {
                      const hits = Array.isArray(item?.top_hits) ? item.top_hits.slice(0, 2) : [];
                      return (
                        <Card key={`${String(item?.query ?? "q")}-${index}`} size="small" style={{ background: "#f8fafc" }}>
                          <Space direction="vertical" size={4} style={{ width: "100%" }}>
                            <Text strong>Query {index + 1}: {String(item?.query ?? "")}</Text>
                            <Text style={{ color: item?.target_hit ? "#15803d" : "#b45309" }}>
                              {item?.target_hit ? `命中当前文档，rank=${String(item?.target_hit_rank ?? "-")}` : "未命中当前文档"}
                              {`，latency=${String(item?.latency_ms ?? 0)}ms`}
                            </Text>
                            {hits.length > 0 ? (
                              <Space direction="vertical" size={4} style={{ width: "100%" }}>
                                {hits.map((hit: any, hitIndex: number) => (
                                  <Card key={`${String(item?.query ?? "q")}-${hitIndex}`} size="small">
                                    <Space direction="vertical" size={4} style={{ width: "100%" }}>
                                      <Space wrap>
                                        <Tag color={Boolean(hit?.is_target_doc) ? "green" : "blue"}>
                                          #{String(hit?.rank ?? hitIndex + 1)} {Boolean(hit?.is_target_doc) ? "当前文档" : "其他文档"}
                                        </Tag>
                                        <Tag>{String(hit?.retrieval_track ?? "unknown_track")}</Tag>
                                        {String(hit?.chunk_id ?? "").trim() ? (
                                          <Button size="small" onClick={() => openChunkDetail(hit, String(item?.query ?? ""))}>
                                            定位查看
                                          </Button>
                                        ) : (
                                          <Tag>无 chunk_id</Tag>
                                        )}
                                        {String(hit?.chunk_id ?? "").trim() && String(hit?.doc_id ?? "").trim() ? (
                                          <a
                                            href={`/docs-center?doc_id=${encodeURIComponent(String(hit.doc_id))}&chunk_id=${encodeURIComponent(
                                              String(hit.chunk_id),
                                            )}&q=${encodeURIComponent(String(item?.query ?? ""))}`}
                                            target="_blank"
                                            rel="noreferrer"
                                          >
                                            在文档页打开
                                          </a>
                                        ) : null}
                                      </Space>
                                      <Text style={{ color: "#475569" }}>{String(hit?.excerpt ?? "") || "暂无摘录"}</Text>
                                    </Space>
                                  </Card>
                                ))}
                              </Space>
                            ) : (
                              <Text style={{ color: "#475569" }}>Top 命中：暂无</Text>
                            )}
                          </Space>
                        </Card>
                      );
                    })}
                  </Space>
                </Card>
              ) : null}

              <Collapse
                size="small"
                activeKey={showUploadAdvanced ? ["upload-advanced"] : []}
                onChange={(keys) => setShowUploadAdvanced(Array.isArray(keys) && keys.length > 0)}
                items={[
                  {
                    key: "upload-advanced",
                    label: "高级设置（可选）",
                    children: (
                      <Space direction="vertical" style={{ width: "100%" }} size={10}>
                        <Space wrap>
                          <Text>来源</Text>
                          <Select
                            value={uploadSource}
                            onChange={setUploadSource}
                            style={{ width: 180 }}
                            options={SOURCE_OPTIONS}
                          />
                          <Text>自动索引</Text>
                          <Switch checked={uploadAutoIndex} onChange={setUploadAutoIndex} />
                        </Space>
                        <Input
                          value={uploadStockCodes}
                          onChange={(e) => setUploadStockCodes(e.target.value)}
                          placeholder="关联股票（可选，逗号分隔，例如 SH600000,SZ000001）"
                        />
                        <Input
                          value={uploadTags}
                          onChange={(e) => setUploadTags(e.target.value)}
                          placeholder="标签（可选，逗号分隔，例如 财报,季报,业绩）"
                        />
                      </Space>
                    ),
                  },
                ]}
              />
            </Space>
          </Card>

          <Card className="premium-card" style={{ marginTop: 10 }} title="业务看板">
            <Space direction="vertical" style={{ width: "100%" }} size={10}>
              <Text style={{ color: "#64748b" }}>
                最近重建：{dashboard?.last_reindex_at || "-"}
              </Text>
              <Row gutter={[12, 12]}>
                <Col xs={24} md={8} xl={4}><Card size="small" title="文档总数">{Number(dashboard?.doc_total ?? 0)}</Card></Col>
                <Col xs={24} md={8} xl={4}><Card size="small" title="活跃Chunk">{Number(dashboard?.active_chunks ?? 0)}</Card></Col>
                <Col xs={24} md={8} xl={4}><Card size="small" title="待治理">{Number(dashboard?.review_pending ?? 0)}</Card></Col>
                <Col xs={24} md={8} xl={4}><Card size="small" title="QA记忆">{Number(dashboard?.qa_memory_total ?? 0)}</Card></Col>
                <Col xs={24} md={8} xl={8}>
                  <Card size="small" title="7日检索命中率">
                    <Progress percent={Number((Number(dashboard?.retrieval_hit_rate_7d ?? 0) * 100).toFixed(1))} />
                    <Text style={{ color: "#64748b" }}>样本数：{Number(dashboard?.retrieval_trace_count_7d ?? 0)}</Text>
                  </Card>
                </Col>
              </Row>
            </Space>
          </Card>

          <Card className="premium-card" style={{ marginTop: 10 }} title="最近上传记录">
            <Table rowKey="upload_id" columns={uploadColumns} dataSource={uploadRows} pagination={false} />
          </Card>
        </>
      ) : null}

      {mode === "ops" ? (
        <>
          <Card className="premium-card" style={{ marginTop: 10 }}>
            <Space direction="vertical" style={{ width: "100%" }} size={8}>
              <Segmented
                value={tab}
                onChange={(v) => setTab(v as RagOpsTab)}
                options={[
                  { label: "来源策略", value: "source" },
                  { label: "文档资产", value: "chunk" },
                  { label: "问答语料", value: "memory" },
                  { label: "检索追踪", value: "trace" },
                ]}
              />
              <Text style={{ color: "#64748b" }}>{opsTabHint}</Text>
            </Space>
          </Card>

          {tab === "source" ? (
            <Card className="premium-card" style={{ marginTop: 10 }} title="来源白名单策略">
              <Space direction="vertical" style={{ width: "100%" }} size={10}>
                <Space wrap>
                  <Button onClick={loadSourcePolicy} loading={loading}>刷新策略</Button>
                  <Input value={editSource} onChange={(e) => setEditSource(e.target.value)} style={{ width: 210 }} placeholder="source" />
                  <Text>自动生效</Text>
                  <Switch checked={editAutoApprove} onChange={setEditAutoApprove} />
                  <Text>信任分</Text>
                  <InputNumber min={0} max={1} step={0.05} value={editTrustScore} onChange={(v) => setEditTrustScore(Number(v ?? 0.7))} />
                  <Text>启用</Text>
                  <Switch checked={editEnabled} onChange={setEditEnabled} />
                  <Button type="primary" onClick={saveSourcePolicy} loading={loading}>保存策略</Button>
                </Space>
                <Table rowKey="source" columns={sourceColumns} dataSource={sourceRows} pagination={false} />
              </Space>
            </Card>
          ) : null}

          {tab === "chunk" ? (
            <Card className="premium-card" style={{ marginTop: 10 }} title="文档资产治理">
              <Space direction="vertical" style={{ width: "100%" }} size={10}>
                <Space wrap>
                  <Input value={chunkDocId} onChange={(e) => setChunkDocId(e.target.value)} placeholder="doc_id" style={{ width: 180 }} />
                  <Input value={chunkSource} onChange={(e) => setChunkSource(e.target.value)} placeholder="source" style={{ width: 140 }} />
                  <Select value={chunkStatusFilter || undefined} onChange={(v) => setChunkStatusFilter(v || "")} allowClear placeholder="status" style={{ width: 170 }} options={CHUNK_STATUS_OPTIONS} />
                  <Input value={chunkStockCode} onChange={(e) => setChunkStockCode(e.target.value)} placeholder="stock_code" style={{ width: 150 }} />
                  <InputNumber min={1} max={200} value={chunkLimit} onChange={(v) => setChunkLimit(Number(v ?? 40))} />
                  <Button onClick={loadChunks} loading={loading}>查询</Button>
                </Space>
                <Space wrap>
                  <Text>行内动作</Text>
                  <Select value={chunkActionStatus} onChange={(v) => setChunkActionStatus(v as ChunkStatus)} style={{ width: 160 }} options={CHUNK_STATUS_OPTIONS} />
                  <Text style={{ color: "#64748b" }}>点击某行“应用”后，会将该 chunk 更新为当前目标状态。</Text>
                </Space>
                <Table
                  rowKey="chunk_id"
                  columns={[
                    ...chunkColumns,
                    {
                      title: "操作",
                      key: "actions",
                      width: 100,
                      render: (_: unknown, row: any) => (
                        <Button size="small" onClick={() => setChunkStatus(row.chunk_id, chunkActionStatus)} loading={loading}>应用</Button>
                      ),
                    },
                  ]}
                  dataSource={chunkRows}
                  pagination={false}
                />
              </Space>
            </Card>
          ) : null}

          {tab === "memory" ? (
            <Card className="premium-card" style={{ marginTop: 10 }} title="共享问答语料池">
              <Space direction="vertical" style={{ width: "100%" }} size={10}>
                <Space wrap>
                  <Input value={memoryStockCode} onChange={(e) => setMemoryStockCode(e.target.value)} placeholder="stock_code" style={{ width: 150 }} />
                  <InputNumber min={1} max={200} value={memoryLimit} onChange={(v) => setMemoryLimit(Number(v ?? 40))} />
                  <Segmented
                    value={String(memoryRetrievalEnabled)}
                    onChange={(v) => setMemoryRetrievalEnabled(Number(v))}
                    options={[
                      { label: "全部", value: "-1" },
                      { label: "可召回", value: "1" },
                      { label: "已禁用", value: "0" },
                    ]}
                  />
                  <Button onClick={loadQAMemory} loading={loading}>查询</Button>
                </Space>
                <Table
                  rowKey="memory_id"
                  columns={[
                    ...memoryColumns,
                    {
                      title: "开关",
                      key: "switch",
                      width: 100,
                      render: (_: unknown, row: any) => <Switch checked={Boolean(row.retrieval_enabled)} onChange={(checked) => toggleMemory(row.memory_id, checked)} />,
                    },
                  ]}
                  dataSource={memoryRows}
                  pagination={false}
                />
              </Space>
            </Card>
          ) : null}

          {tab === "trace" ? (
            <Card className="premium-card" style={{ marginTop: 10 }} title="检索追踪与索引重建">
              <Space direction="vertical" style={{ width: "100%" }} size={10}>
                <Space wrap>
                  <Input value={traceId} onChange={(e) => setTraceId(e.target.value)} placeholder="trace_id（可选）" style={{ width: 320 }} />
                  <InputNumber min={1} max={300} value={traceLimit} onChange={(v) => setTraceLimit(Number(v ?? 50))} />
                  <Button onClick={loadTrace} loading={loading}>加载追踪</Button>
                  <Button type="primary" onClick={runReindex} loading={loading}>重建向量索引</Button>
                </Space>
                <Text style={{ color: "#64748b" }}>trace_count: {traceCount}</Text>
                <Table rowKey={(row: any) => String(row.id)} columns={traceColumns} dataSource={traceRows} pagination={false} />
              </Space>
            </Card>
          ) : null}
        </>
      ) : null}

      <Drawer
        title="文档片段定位"
        placement="right"
        width={640}
        open={detailOpen}
        onClose={() => setDetailOpen(false)}
      >
        <Space direction="vertical" size={10} style={{ width: "100%" }}>
          {detailLoading ? <Text>正在加载片段详情...</Text> : null}
          {!detailLoading && !detailChunk ? <Alert type="warning" showIcon message="未找到片段详情" /> : null}
          {!detailLoading && detailChunk ? (
            <>
              <Alert
                type="info"
                showIcon
                message={`doc_id=${String(detailChunk.doc_id ?? "-")} | chunk_no=${String(detailChunk.chunk_no ?? "-")}`}
                description={`query=${detailQuery || "-"}`}
              />
              <Space wrap>
                <Tag>{String(detailChunk.source ?? "-")}</Tag>
                <Tag>{String(detailChunk.effective_status ?? "-")}</Tag>
                <Tag>quality={Number(detailChunk.quality_score ?? 0).toFixed(2)}</Tag>
              </Space>
              <Card size="small" title="当前片段">
                <Text style={{ whiteSpace: "pre-wrap", color: "#334155" }}>
                  {String(detailChunk.chunk_text_redacted || detailChunk.chunk_text || "")}
                </Text>
              </Card>
              <Card size="small" title="上文片段">
                <Space direction="vertical" size={8} style={{ width: "100%" }}>
                  {(detailContext?.prev ?? []).length === 0 ? (
                    <Text type="secondary">无上文片段</Text>
                  ) : (
                    (detailContext?.prev ?? []).map((ctx: any) => (
                      <Card key={`prev-${String(ctx?.chunk_id ?? "")}`} size="small">
                        <Text strong>#{String(ctx?.chunk_no ?? "-")}</Text>
                        <Text style={{ display: "block", color: "#475569", marginTop: 4 }}>{String(ctx?.excerpt ?? "")}</Text>
                      </Card>
                    ))
                  )}
                </Space>
              </Card>
              <Card size="small" title="下文片段">
                <Space direction="vertical" size={8} style={{ width: "100%" }}>
                  {(detailContext?.next ?? []).length === 0 ? (
                    <Text type="secondary">无下文片段</Text>
                  ) : (
                    (detailContext?.next ?? []).map((ctx: any) => (
                      <Card key={`next-${String(ctx?.chunk_id ?? "")}`} size="small">
                        <Text strong>#{String(ctx?.chunk_no ?? "-")}</Text>
                        <Text style={{ display: "block", color: "#475569", marginTop: 4 }}>{String(ctx?.excerpt ?? "")}</Text>
                      </Card>
                    ))
                  )}
                </Space>
              </Card>
              {String(detailChunk.doc_id ?? "").trim() && String(detailChunk.chunk_id ?? "").trim() ? (
                <a
                  href={`/docs-center?doc_id=${encodeURIComponent(String(detailChunk.doc_id))}&chunk_id=${encodeURIComponent(
                    String(detailChunk.chunk_id),
                  )}&q=${encodeURIComponent(detailQuery || "")}`}
                  target="_blank"
                  rel="noreferrer"
                >
                  在文档页打开此定位
                </a>
              ) : null}
            </>
          ) : null}
        </Space>
      </Drawer>
    </main>
  );
}
