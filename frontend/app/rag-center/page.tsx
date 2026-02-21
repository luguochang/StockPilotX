"use client";

import { useEffect, useMemo, useState } from "react";
import {
  Alert,
  Button,
  Card,
  Col,
  Input,
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
import { SafeContainer } from "../components/SafeContainer";

const API_BASE = process.env.NEXT_PUBLIC_API_BASE ?? "http://127.0.0.1:8000";
const { Title, Text, Paragraph } = Typography;

type RagMode = "business" | "ops";

const SOURCE_OPTIONS = [
  { label: "user_upload", value: "user_upload" },
  { label: "cninfo", value: "cninfo" },
  { label: "eastmoney", value: "eastmoney" },
  { label: "research", value: "research" },
];

function parseCommaValues(raw: string): string[] {
  return raw
    .split(",")
    .map((x) => x.trim())
    .filter((x) => x.length > 0);
}

export default function RagCenterPage() {
  const [mode, setMode] = useState<RagMode>("business");
  const [loading, setLoading] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [message, setMessage] = useState("");
  const [error, setError] = useState("");

  const [dashboard, setDashboard] = useState<Record<string, any> | null>(null);
  const [uploadRows, setUploadRows] = useState<Record<string, any>[]>([]);
  const [sourceRows, setSourceRows] = useState<Record<string, any>[]>([]);

  const [retrievalPreview, setRetrievalPreview] = useState<Record<string, any> | null>(null);
  const [uploadStatusDetail, setUploadStatusDetail] = useState<Record<string, any> | null>(null);
  const [uploadVerificationDetail, setUploadVerificationDetail] = useState<Record<string, any> | null>(null);
  const [docPreviewDetail, setDocPreviewDetail] = useState<Record<string, any> | null>(null);

  const [uploadFile, setUploadFile] = useState<File | null>(null);
  const [uploadSource, setUploadSource] = useState("user_upload");
  const [uploadStockCodes, setUploadStockCodes] = useState("");
  const [uploadTags, setUploadTags] = useState("");
  const [uploadAutoIndex, setUploadAutoIndex] = useState(true);

  useEffect(() => {
    refreshBusiness().catch(() => undefined);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  async function parseOrThrow(resp: Response) {
    const body = await resp.json();
    if (!resp.ok) throw new Error(body?.detail ?? body?.error ?? `HTTP ${resp.status}`);
    return body;
  }

  function resetFeedback() {
    setMessage("");
    setError("");
  }

  async function fileToBase64(file: File): Promise<string> {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = () => {
        const value = reader.result;
        if (!(value instanceof ArrayBuffer)) return reject(new Error("Failed to read file"));
        const bytes = new Uint8Array(value);
        let binary = "";
        for (let i = 0; i < bytes.length; i += 1) binary += String.fromCharCode(bytes[i]);
        resolve(btoa(binary));
      };
      reader.onerror = () => reject(new Error("Failed to read file"));
      reader.readAsArrayBuffer(file);
    });
  }

  async function refreshBusiness() {
    setLoading(true);
    resetFeedback();
    try {
      const [dashboardResp, uploadsResp] = await Promise.all([
        fetch(`${API_BASE}/v1/rag/dashboard`),
        fetch(`${API_BASE}/v1/rag/uploads?limit=30`),
      ]);
      setDashboard((await parseOrThrow(dashboardResp)) as Record<string, any>);
      setUploadRows(((await parseOrThrow(uploadsResp)) as Record<string, any>[]) ?? []);
      setMessage("RAG dashboard refreshed");
    } catch (e) {
      setError(e instanceof Error ? e.message : "Refresh failed");
    } finally {
      setLoading(false);
    }
  }

  async function loadUploadRuntime(uploadId: string) {
    const normalized = uploadId.trim();
    if (!normalized) return;
    setLoading(true);
    resetFeedback();
    try {
      const [statusResp, verificationResp] = await Promise.all([
        fetch(`${API_BASE}/v1/rag/uploads/${encodeURIComponent(normalized)}/status`),
        fetch(`${API_BASE}/v1/rag/uploads/${encodeURIComponent(normalized)}/verification`),
      ]);
      setUploadStatusDetail((await parseOrThrow(statusResp)) as Record<string, any>);
      setUploadVerificationDetail((await parseOrThrow(verificationResp)) as Record<string, any>);
      setMessage(`Loaded upload runtime: ${normalized}`);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to load upload runtime");
    } finally {
      setLoading(false);
    }
  }

  async function loadDocPreview(docId: string, page = 1) {
    const normalized = docId.trim();
    if (!normalized) return;
    setLoading(true);
    resetFeedback();
    try {
      const resp = await fetch(`${API_BASE}/v1/rag/docs/${encodeURIComponent(normalized)}/preview?page=${Math.max(1, page)}`);
      setDocPreviewDetail((await parseOrThrow(resp)) as Record<string, any>);
      setMessage(`Loaded parsed preview: ${normalized}`);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to load parsed preview");
    } finally {
      setLoading(false);
    }
  }

  async function deleteUpload(uploadId: string, filename: string) {
    const normalized = uploadId.trim();
    if (!normalized) return;
    if (!window.confirm(`Delete upload ${filename || normalized} permanently?`)) return;
    setLoading(true);
    resetFeedback();
    try {
      const resp = await fetch(`${API_BASE}/v1/rag/uploads/${encodeURIComponent(normalized)}`, { method: "DELETE" });
      await parseOrThrow(resp);
      if (String(uploadStatusDetail?.upload_id ?? "") === normalized) {
        setUploadStatusDetail(null);
        setUploadVerificationDetail(null);
        setDocPreviewDetail(null);
      }
      setRetrievalPreview(null);
      await refreshBusiness();
      setMessage(`Deleted upload: ${normalized}`);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Delete failed");
    } finally {
      setLoading(false);
    }
  }

  async function submitUpload() {
    if (!uploadFile) return setError("Please select a file first");
    setUploading(true);
    resetFeedback();
    setRetrievalPreview(null);
    try {
      const resp = await fetch(`${API_BASE}/v1/rag/workflow/upload-and-index`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          filename: uploadFile.name,
          content_type: uploadFile.type,
          content_base64: await fileToBase64(uploadFile),
          source: uploadSource,
          stock_codes: parseCommaValues(uploadStockCodes).map((x) => x.toUpperCase()),
          tags: parseCommaValues(uploadTags),
          auto_index: uploadAutoIndex,
          user_id: "frontend-rag",
        }),
      });
      const body = (await parseOrThrow(resp)) as Record<string, any>;
      const uploadId = String(body?.result?.upload_id ?? "").trim();
      const docId = String(body?.result?.doc_id ?? "").trim();
      setRetrievalPreview((body?.retrieval_preview as Record<string, any>) ?? null);
      await refreshBusiness();
      if (uploadId) await loadUploadRuntime(uploadId);
      if (docId) await loadDocPreview(docId, 1);
      setMessage(`Upload finished: ${docId || uploadFile.name}`);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Upload failed");
    } finally {
      setUploading(false);
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
      await parseOrThrow(resp);
      await refreshBusiness();
      setMessage("Vector index rebuilt");
    } catch (e) {
      setError(e instanceof Error ? e.message : "Reindex failed");
    } finally {
      setLoading(false);
    }
  }

  async function loadSourcePolicy() {
    setLoading(true);
    resetFeedback();
    try {
      const resp = await fetch(`${API_BASE}/v1/rag/source-policy`);
      setSourceRows(((await parseOrThrow(resp)) as Record<string, any>[]) ?? []);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to load source policy");
    } finally {
      setLoading(false);
    }
  }

  const uploadColumns: any[] = [
    { title: "File", dataIndex: "filename", key: "filename", width: 220 },
    { title: "Source", dataIndex: "source", key: "source", width: 110 },
    {
      title: "Status",
      dataIndex: "status",
      key: "status",
      width: 100,
      render: (v: string) => <Tag color={v === "active" ? "green" : "blue"}>{v}</Tag>,
    },
    { title: "Job", dataIndex: "job_status", key: "job_status", width: 100 },
    { title: "Parser", dataIndex: "parser_name", key: "parser_name", width: 110, render: (v: string) => String(v || "-") },
    { title: "Doc ID", dataIndex: "doc_id", key: "doc_id", width: 170 },
    {
      title: "Action",
      key: "action",
      width: 280,
      render: (_: unknown, row: any) => (
        <Space wrap>
          <Button size="small" onClick={() => loadUploadRuntime(String(row?.upload_id ?? ""))}>Status</Button>
          <Button size="small" onClick={() => loadDocPreview(String(row?.doc_id ?? ""), 1)}>Preview</Button>
          <Button size="small" danger onClick={() => deleteUpload(String(row?.upload_id ?? ""), String(row?.filename ?? ""))}>Delete</Button>
        </Space>
      ),
    },
  ];

  const parsedChunkColumns: any[] = [
    { title: "Chunk ID", dataIndex: "chunk_id", key: "chunk_id", width: 180 },
    { title: "No", dataIndex: "chunk_no", key: "chunk_no", width: 70 },
    { title: "Status", dataIndex: "status", key: "status", width: 90 },
    { title: "Quality", dataIndex: "quality_score", key: "quality_score", width: 90 },
    { title: "Excerpt", dataIndex: "excerpt", key: "excerpt" },
  ];

  const opsHint = useMemo(() => {
    return "Ops mode provides source policy and index maintenance.";
  }, []);

  return (
    <main className="container shell-fade-in">
      <Card className="premium-card">
        <Space direction="vertical" style={{ width: "100%" }} size={8}>
          <Title level={2} style={{ margin: 0 }}>RAG Operations Center</Title>
          <Paragraph style={{ margin: 0, color: "#475569", maxWidth: 920 }}>
            Unified upload entry: upload, parse, index, verification, preview, and delete are visible in one place.
          </Paragraph>
          <Segmented
            value={mode}
            onChange={(v) => setMode(v as RagMode)}
            options={[
              { label: "Business", value: "business" },
              { label: "Ops", value: "ops" },
            ]}
          />
        </Space>
      </Card>

      {message ? <Alert style={{ marginTop: 10 }} type="success" showIcon message={message} /> : null}
      {error ? <Alert style={{ marginTop: 10 }} type="error" showIcon message={error} /> : null}

      {mode === "business" ? (
        <>
          <Card className="premium-card" style={{ marginTop: 10 }} title="Quick Upload">
            <Space direction="vertical" style={{ width: "100%" }} size={10}>
              <input type="file" onChange={(e) => setUploadFile(e.target.files?.[0] ?? null)} />
              <Space wrap>
                <Select value={uploadSource} onChange={setUploadSource} style={{ width: 180 }} options={SOURCE_OPTIONS} />
                <Input value={uploadStockCodes} onChange={(e) => setUploadStockCodes(e.target.value)} placeholder="stock_codes, comma separated" style={{ width: 220 }} />
                <Input value={uploadTags} onChange={(e) => setUploadTags(e.target.value)} placeholder="tags, comma separated" style={{ width: 220 }} />
                <Text>auto_index</Text>
                <Switch checked={uploadAutoIndex} onChange={setUploadAutoIndex} />
              </Space>
              <Space wrap>
                <Button type="primary" onClick={submitUpload} loading={uploading} disabled={!uploadFile}>Upload and Index</Button>
                <Button onClick={refreshBusiness} loading={loading}>Refresh</Button>
                <Button onClick={runReindex} loading={loading}>Rebuild Index</Button>
              </Space>
            </Space>
          </Card>

          {retrievalPreview ? (
            <Card className="premium-card" style={{ marginTop: 10 }} title="Retrieval Verification">
              <Text>
                doc_id={String(retrievalPreview.doc_id ?? "-")} | hit_rate={(Number(retrievalPreview.target_hit_rate ?? 0) * 100).toFixed(1)}%
              </Text>
            </Card>
          ) : null}

          <Card className="premium-card" style={{ marginTop: 10 }} title="Dashboard">
            <Space direction="vertical" style={{ width: "100%" }} size={8}>
              <Text style={{ color: "#64748b" }}>Last reindex: {dashboard?.last_reindex_at || "-"}</Text>
              <Row gutter={[12, 12]}>
                <Col xs={24} md={8} xl={4}><Card size="small" title="Docs">{Number(dashboard?.doc_total ?? 0)}</Card></Col>
                <Col xs={24} md={8} xl={4}><Card size="small" title="Active Chunks">{Number(dashboard?.active_chunks ?? 0)}</Card></Col>
                <Col xs={24} md={8} xl={4}><Card size="small" title="Review Pending">{Number(dashboard?.review_pending ?? 0)}</Card></Col>
                <Col xs={24} md={8} xl={4}><Card size="small" title="QA Memory">{Number(dashboard?.qa_memory_total ?? 0)}</Card></Col>
                <Col xs={24} md={8} xl={8}>
                  <Card size="small" title="Hit Rate 7d">
                    <Progress percent={Number((Number(dashboard?.retrieval_hit_rate_7d ?? 0) * 100).toFixed(1))} />
                    <Text style={{ color: "#64748b" }}>trace_count={Number(dashboard?.retrieval_trace_count_7d ?? 0)}</Text>
                  </Card>
                </Col>
              </Row>
            </Space>
          </Card>

          {uploadStatusDetail ? (
            <Card className="premium-card" style={{ marginTop: 10 }} title="Upload Runtime Status">
              <Space direction="vertical" style={{ width: "100%" }} size={8}>
                <Space wrap>
                  <Tag>upload_id: {String(uploadStatusDetail.upload_id ?? "-")}</Tag>
                  <Tag>doc_id: {String(uploadStatusDetail.doc_id ?? "-")}</Tag>
                  <Tag>job: {String(uploadStatusDetail?.asset?.job_status ?? "-")}</Tag>
                  <Tag>stage: {String(uploadStatusDetail?.asset?.current_stage ?? "-")}</Tag>
                  <Tag>parser: {String(uploadStatusDetail?.asset?.parser_name ?? "-")}</Tag>
                </Space>
                <Table
                  rowKey={(row: any) => `${String(row.phase ?? "")}-${String(row.at ?? "")}`}
                  pagination={{ pageSize: 10, simple: true }}
                  size="small"
                  dataSource={(uploadStatusDetail.timeline ?? []) as Record<string, any>[]}
                  columns={[
                    { title: "Phase", dataIndex: "phase", key: "phase", width: 160 },
                    { title: "Status", dataIndex: "status", key: "status", width: 100 },
                    { title: "At", dataIndex: "at", key: "at", width: 180 },
                    {
                      title: "Detail",
                      key: "detail",
                      render: (_: unknown, row: any) => <Text>{JSON.stringify(row?.detail ?? {}).slice(0, 220)}</Text>,
                    },
                  ]}
                />
              </Space>
            </Card>
          ) : null}

          {uploadVerificationDetail ? (
            <Card className="premium-card" style={{ marginTop: 10 }} title="Upload Verification Summary">
              <Text>
                query_count={String(uploadVerificationDetail.query_count ?? 0)} matched={String(uploadVerificationDetail.matched_query_count ?? 0)} hit_rate=
                {(Number(uploadVerificationDetail.target_hit_rate ?? 0) * 100).toFixed(1)}%
              </Text>
            </Card>
          ) : null}

          {docPreviewDetail ? (
            <Card className="premium-card" style={{ marginTop: 10 }} title="Parsed Content Preview">
              <Space direction="vertical" style={{ width: "100%" }} size={10}>
                <Space wrap>
                  <Tag>doc_id: {String(docPreviewDetail.doc_id ?? "-")}</Tag>
                  <Tag>page: {String(docPreviewDetail.page ?? 1)}/{String(docPreviewDetail.total_pages ?? 1)}</Tag>
                  <Tag>chunks: {String(docPreviewDetail.total_chunks ?? 0)}</Tag>
                  <Tag>quality: {Number(docPreviewDetail?.quality_report?.quality_score ?? 0).toFixed(2)}</Tag>
                </Space>
                <Alert
                  type={String(docPreviewDetail?.parse_verdict?.status ?? "ok") === "failed" ? "error" : String(docPreviewDetail?.parse_verdict?.status ?? "ok") === "warning" ? "warning" : "success"}
                  showIcon
                  message={`Parse verdict: ${String(docPreviewDetail?.parse_verdict?.status ?? "ok")}`}
                  description={String(docPreviewDetail?.parse_verdict?.message ?? "")}
                />
                <Space wrap>
                  <Button
                    disabled={Number(docPreviewDetail.page ?? 1) <= 1}
                    onClick={() => loadDocPreview(String(docPreviewDetail.doc_id ?? ""), Number(docPreviewDetail.page ?? 1) - 1)}
                  >
                    Prev
                  </Button>
                  <Button
                    disabled={Number(docPreviewDetail.page ?? 1) >= Number(docPreviewDetail.total_pages ?? 1)}
                    onClick={() => loadDocPreview(String(docPreviewDetail.doc_id ?? ""), Number(docPreviewDetail.page ?? 1) + 1)}
                  >
                    Next
                  </Button>
                </Space>
                <Table
                  size="small"
                  rowKey={(row: any) => String(row?.chunk_id ?? "")}
                  pagination={{ pageSize: 15, simple: true }}
                  columns={parsedChunkColumns}
                  dataSource={(docPreviewDetail.items ?? []) as Record<string, any>[]}
                />
                <Card size="small" title="Parse Trace">
                  <SafeContainer maxHeight={400}>
                    <Text style={{ whiteSpace: "pre-wrap" }}>{JSON.stringify(docPreviewDetail.parse_trace ?? {}, null, 2)}</Text>
                  </SafeContainer>
                </Card>
                <Card size="small" title="Parse Quality">
                  <SafeContainer maxHeight={400}>
                    <Text style={{ whiteSpace: "pre-wrap" }}>{JSON.stringify(docPreviewDetail.parse_quality ?? {}, null, 2)}</Text>
                  </SafeContainer>
                </Card>
              </Space>
            </Card>
          ) : null}

          <Card className="premium-card" style={{ marginTop: 10 }} title="Recent Uploads">
            <Table rowKey="upload_id" columns={uploadColumns} dataSource={uploadRows} pagination={{ pageSize: 10, simple: true }} />
          </Card>
        </>
      ) : null}

      {mode === "ops" ? (
        <Card className="premium-card" style={{ marginTop: 10 }} title="Ops Quick Actions">
          <Space direction="vertical" style={{ width: "100%" }} size={10}>
            <Text style={{ color: "#64748b" }}>{opsHint}</Text>
            <Space wrap>
              <Button onClick={loadSourcePolicy} loading={loading}>Load Source Policy</Button>
              <Button onClick={runReindex} loading={loading}>Rebuild Index</Button>
            </Space>
            <Table
              rowKey="source"
              pagination={false}
              columns={[
                { title: "Source", dataIndex: "source", key: "source" },
                { title: "Trust", dataIndex: "trust_score", key: "trust_score" },
                { title: "Enabled", dataIndex: "enabled", key: "enabled" },
              ]}
              dataSource={sourceRows}
            />
          </Space>
        </Card>
      ) : null}
    </main>
  );
}
