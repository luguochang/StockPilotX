"use client";

import { useMemo, useState } from "react";
import {
  Alert,
  Button,
  Card,
  Input,
  InputNumber,
  Segmented,
  Select,
  Space,
  Switch,
  Table,
  Tag,
  Typography,
} from "antd";
import type { ColumnsType } from "antd/es/table";

const API_BASE = process.env.NEXT_PUBLIC_API_BASE ?? "http://127.0.0.1:8000";
const { Title, Text, Paragraph } = Typography;

type RagSourcePolicy = {
  source: string;
  auto_approve: boolean;
  trust_score: number;
  enabled: boolean;
  updated_at?: string;
};

type RagDocChunk = {
  chunk_id: string;
  doc_id: string;
  chunk_no: number;
  source: string;
  source_url: string;
  effective_status: string;
  quality_score: number;
  stock_codes: string[];
  industry_tags: string[];
  updated_at?: string;
};

type RagQAMemory = {
  memory_id: string;
  user_id: string;
  stock_code: string;
  query_text: string;
  answer_text: string;
  answer_redacted: string;
  summary_text: string;
  citations: Array<{ source_id?: string }>;
  risk_flags: string[];
  intent: string;
  quality_score: number;
  share_scope: string;
  retrieval_enabled: boolean;
  created_at?: string;
};

type RagTraceItem = {
  id: number;
  trace_id: string;
  query_text: string;
  query_type: string;
  retrieved_ids: string[];
  selected_ids: string[];
  latency_ms: number;
  created_at?: string;
};

type RagTraceResponse = {
  trace_id: string;
  count: number;
  items: RagTraceItem[];
};

type RagTab = "source" | "chunk" | "memory" | "trace";

type ChunkStatus = "active" | "review" | "rejected" | "archived";

const CHUNK_STATUS_OPTIONS: Array<{ label: string; value: ChunkStatus }> = [
  { label: "active", value: "active" },
  { label: "review", value: "review" },
  { label: "rejected", value: "rejected" },
  { label: "archived", value: "archived" },
];

export default function RagCenterPage() {
  const [tab, setTab] = useState<RagTab>("source");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [message, setMessage] = useState("");

  const [sourceRows, setSourceRows] = useState<RagSourcePolicy[]>([]);
  const [editSource, setEditSource] = useState("user_upload");
  const [editAutoApprove, setEditAutoApprove] = useState(false);
  const [editTrustScore, setEditTrustScore] = useState(0.7);
  const [editEnabled, setEditEnabled] = useState(true);

  const [chunkRows, setChunkRows] = useState<RagDocChunk[]>([]);
  const [chunkDocId, setChunkDocId] = useState("");
  const [chunkSource, setChunkSource] = useState("");
  const [chunkStatusFilter, setChunkStatusFilter] = useState("");
  const [chunkStockCode, setChunkStockCode] = useState("");
  const [chunkLimit, setChunkLimit] = useState(40);
  const [chunkActionStatus, setChunkActionStatus] = useState<ChunkStatus>("review");

  const [memoryRows, setMemoryRows] = useState<RagQAMemory[]>([]);
  const [memoryStockCode, setMemoryStockCode] = useState("");
  const [memoryLimit, setMemoryLimit] = useState(40);
  const [memoryRetrievalEnabled, setMemoryRetrievalEnabled] = useState<number>(-1);

  const [traceRows, setTraceRows] = useState<RagTraceItem[]>([]);
  const [traceId, setTraceId] = useState("");
  const [traceLimit, setTraceLimit] = useState(50);
  const [traceCount, setTraceCount] = useState(0);

  // 统一后端错误解析，避免每个请求重复写分支。
  async function parseOrThrow(resp: Response) {
    const body = await resp.json();
    if (!resp.ok) throw new Error(body?.detail ?? body?.error ?? `HTTP ${resp.status}`);
    return body;
  }

  function resetFeedback() {
    setError("");
    setMessage("");
  }

  async function loadSourcePolicy() {
    setLoading(true);
    resetFeedback();
    try {
      const resp = await fetch(`${API_BASE}/v1/rag/source-policy`);
      const rows = (await parseOrThrow(resp)) as RagSourcePolicy[];
      setSourceRows(rows);
      setMessage(`来源策略已加载，共 ${rows.length} 条。`);
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
        body: JSON.stringify({
          auto_approve: editAutoApprove,
          trust_score: Number(editTrustScore),
          enabled: editEnabled,
        }),
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
      const rows = (await parseOrThrow(resp)) as RagDocChunk[];
      setChunkRows(rows);
      setMessage(`文档资产已加载，共 ${rows.length} 条。`);
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
      const rows = (await parseOrThrow(resp)) as RagQAMemory[];
      setMemoryRows(rows);
      setMessage(`共享问答语料已加载，共 ${rows.length} 条。`);
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
      const body = (await parseOrThrow(resp)) as RagTraceResponse;
      const rows = Array.isArray(body.items) ? body.items : [];
      setTraceRows(rows);
      setTraceCount(Number(body.count ?? rows.length));
      setMessage(`检索追踪已加载，共 ${Number(body.count ?? rows.length)} 条。`);
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
    } catch (e) {
      setError(e instanceof Error ? e.message : "重建索引失败");
    } finally {
      setLoading(false);
    }
  }

  const sourceColumns: ColumnsType<RagSourcePolicy> = [
    { title: "Source", dataIndex: "source", key: "source", width: 220 },
    {
      title: "Auto Approve",
      dataIndex: "auto_approve",
      key: "auto_approve",
      width: 130,
      render: (v: boolean) => <Tag color={v ? "green" : "orange"}>{String(Boolean(v))}</Tag>,
    },
    {
      title: "Trust",
      dataIndex: "trust_score",
      key: "trust_score",
      width: 100,
      render: (v: number) => Number(v ?? 0).toFixed(2),
    },
    {
      title: "Enabled",
      dataIndex: "enabled",
      key: "enabled",
      width: 110,
      render: (v: boolean) => <Tag color={v ? "green" : "red"}>{String(Boolean(v))}</Tag>,
    },
    { title: "Updated", dataIndex: "updated_at", key: "updated_at" },
  ];

  const chunkColumns: ColumnsType<RagDocChunk> = [
    { title: "Chunk ID", dataIndex: "chunk_id", key: "chunk_id", width: 220 },
    { title: "Doc ID", dataIndex: "doc_id", key: "doc_id", width: 150 },
    { title: "No", dataIndex: "chunk_no", key: "chunk_no", width: 70 },
    { title: "Source", dataIndex: "source", key: "source", width: 120 },
    {
      title: "Status",
      dataIndex: "effective_status",
      key: "effective_status",
      width: 120,
      render: (v: string) => (
        <Tag color={v === "active" ? "green" : v === "review" ? "gold" : v === "rejected" ? "red" : "default"}>{v}</Tag>
      ),
    },
    {
      title: "Quality",
      dataIndex: "quality_score",
      key: "quality_score",
      width: 90,
      render: (v: number) => Number(v ?? 0).toFixed(2),
    },
    {
      title: "Stocks",
      dataIndex: "stock_codes",
      key: "stock_codes",
      render: (v: string[]) => <Text style={{ color: "#475569" }}>{Array.isArray(v) ? v.join(", ") : ""}</Text>,
    },
  ];

  const memoryColumns: ColumnsType<RagQAMemory> = [
    { title: "Memory ID", dataIndex: "memory_id", key: "memory_id", width: 220 },
    { title: "Stock", dataIndex: "stock_code", key: "stock_code", width: 100 },
    { title: "Intent", dataIndex: "intent", key: "intent", width: 96 },
    {
      title: "Quality",
      dataIndex: "quality_score",
      key: "quality_score",
      width: 96,
      render: (v: number) => Number(v ?? 0).toFixed(2),
    },
    {
      title: "Retrieval",
      dataIndex: "retrieval_enabled",
      key: "retrieval_enabled",
      width: 110,
      render: (v: boolean) => <Tag color={v ? "green" : "orange"}>{String(Boolean(v))}</Tag>,
    },
    {
      title: "Summary",
      dataIndex: "summary_text",
      key: "summary_text",
      render: (v: string) => <Text style={{ color: "#334155" }}>{String(v ?? "").slice(0, 120)}</Text>,
    },
  ];

  const traceColumns: ColumnsType<RagTraceItem> = [
    { title: "ID", dataIndex: "id", key: "id", width: 70 },
    { title: "Trace ID", dataIndex: "trace_id", key: "trace_id", width: 220 },
    { title: "Type", dataIndex: "query_type", key: "query_type", width: 120 },
    { title: "Latency(ms)", dataIndex: "latency_ms", key: "latency_ms", width: 120 },
    {
      title: "Retrieved -> Selected",
      key: "selected",
      render: (_, row) => (
        <Text style={{ color: "#334155" }}>
          {(row.retrieved_ids ?? []).slice(0, 3).join(", ")} {"->"} {(row.selected_ids ?? []).slice(0, 3).join(", ")}
        </Text>
      ),
    },
  ];

  const tabHint = useMemo(() => {
    if (tab === "source") return "按来源控制文档是否自动生效以及信任分。";
    if (tab === "chunk") return "查看/下线已入库文档 chunk，控制在线检索质量。";
    if (tab === "memory") return "管理共享问答语料，快速禁用污染样本。";
    return "观察检索召回轨迹，并按需触发向量索引重建。";
  }, [tab]);

  return (
    <main className="container shell-fade-in">
      <Card className="premium-card">
        <Space direction="vertical" style={{ width: "100%" }} size={8}>
          <Title level={2} style={{ margin: 0 }}>
            RAG 运营台
          </Title>
          <Paragraph style={{ margin: 0, color: "#475569", maxWidth: 920 }}>
            面向运营与研发的知识资产治理面板：统一管理来源白名单、文档片段有效性、共享问答语料和检索追踪。
          </Paragraph>
          <Segmented
            value={tab}
            onChange={(v) => setTab(v as RagTab)}
            options={[
              { label: "来源策略", value: "source" },
              { label: "文档资产", value: "chunk" },
              { label: "问答语料", value: "memory" },
              { label: "检索追踪", value: "trace" },
            ]}
          />
          <Text style={{ color: "#64748b" }}>{tabHint}</Text>
        </Space>
      </Card>

      {message ? <Alert style={{ marginTop: 10 }} type="success" showIcon message={message} /> : null}
      {error ? <Alert style={{ marginTop: 10 }} type="error" showIcon message={error} /> : null}

      {tab === "source" ? (
        <Card className="premium-card" style={{ marginTop: 10 }} title="来源白名单策略">
          <Space direction="vertical" style={{ width: "100%" }} size={10}>
            <Space wrap>
              <Button onClick={loadSourcePolicy} loading={loading}>
                刷新策略
              </Button>
              <Input value={editSource} onChange={(e) => setEditSource(e.target.value)} style={{ width: 210 }} placeholder="source" />
              <Text>自动生效</Text>
              <Switch checked={editAutoApprove} onChange={setEditAutoApprove} />
              <Text>信任分</Text>
              <InputNumber min={0} max={1} step={0.05} value={editTrustScore} onChange={(v) => setEditTrustScore(Number(v ?? 0.7))} />
              <Text>启用</Text>
              <Switch checked={editEnabled} onChange={setEditEnabled} />
              <Button type="primary" onClick={saveSourcePolicy} loading={loading}>
                保存策略
              </Button>
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
              <Select
                value={chunkStatusFilter || undefined}
                onChange={(v) => setChunkStatusFilter(v || "")}
                allowClear
                placeholder="status"
                style={{ width: 170 }}
                options={CHUNK_STATUS_OPTIONS}
              />
              <Input value={chunkStockCode} onChange={(e) => setChunkStockCode(e.target.value)} placeholder="stock_code" style={{ width: 150 }} />
              <InputNumber min={1} max={200} value={chunkLimit} onChange={(v) => setChunkLimit(Number(v ?? 40))} />
              <Button onClick={loadChunks} loading={loading}>
                查询
              </Button>
            </Space>
            <Space wrap>
              <Text>行内动作</Text>
              <Select
                value={chunkActionStatus}
                onChange={(v) => setChunkActionStatus(v as ChunkStatus)}
                style={{ width: 160 }}
                options={CHUNK_STATUS_OPTIONS}
              />
              <Text style={{ color: "#64748b" }}>点击某行“应用”按钮，会将该行状态更新为当前动作。</Text>
            </Space>
            <Table
              rowKey="chunk_id"
              columns={[
                ...chunkColumns,
                {
                  title: "操作",
                  key: "actions",
                  width: 100,
                  render: (_, row) => (
                    <Button size="small" onClick={() => setChunkStatus(row.chunk_id, chunkActionStatus)} loading={loading}>
                      应用
                    </Button>
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
              <Button onClick={loadQAMemory} loading={loading}>
                查询
              </Button>
            </Space>
            <Table
              rowKey="memory_id"
              columns={[
                ...memoryColumns,
                {
                  title: "开关",
                  key: "switch",
                  width: 100,
                  render: (_, row) => (
                    <Switch checked={Boolean(row.retrieval_enabled)} onChange={(checked) => toggleMemory(row.memory_id, checked)} />
                  ),
                },
              ]}
              dataSource={memoryRows}
              pagination={false}
              expandable={{
                expandedRowRender: (row) => (
                  <Space direction="vertical" style={{ width: "100%" }} size={6}>
                    <Text style={{ color: "#334155" }}>Q: {row.query_text}</Text>
                    <Text style={{ color: "#64748b" }}>Summary: {row.summary_text}</Text>
                    <Text style={{ color: "#64748b" }}>Risk: {(row.risk_flags ?? []).join(", ") || "-"}</Text>
                  </Space>
                ),
              }}
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
              <Button onClick={loadTrace} loading={loading}>
                加载追踪
              </Button>
              <Button type="primary" onClick={runReindex} loading={loading}>
                重建向量索引
              </Button>
            </Space>
            <Text style={{ color: "#64748b" }}>trace_count: {traceCount}</Text>
            <Table rowKey={(row) => String(row.id)} columns={traceColumns} dataSource={traceRows} pagination={false} />
            <Space direction="vertical" size={2} style={{ width: "100%" }}>
              <Text style={{ color: "#0f172a" }}>说明</Text>
              <Text style={{ color: "#64748b" }}>1. 来源策略决定文档是否自动生效，避免低质量内容污染在线检索。</Text>
              <Text style={{ color: "#64748b" }}>2. 文档资产支持逐条状态切换，可快速下线异常 chunk。</Text>
              <Text style={{ color: "#64748b" }}>3. 问答语料开关用于控制历史回答是否参与后续召回。</Text>
              <Text style={{ color: "#64748b" }}>4. 追踪记录可对比“召回候选”和“最终引用”，便于优化排序与提示词。</Text>
            </Space>
          </Space>
        </Card>
      ) : null}
    </main>
  );
}
