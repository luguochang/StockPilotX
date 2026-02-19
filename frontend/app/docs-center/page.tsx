"use client";

import { Suspense, useEffect, useState } from "react";
import { useSearchParams } from "next/navigation";
import { Alert, Button, Card, Input, Space, Table, Tag, Typography } from "antd";
import type { ColumnsType } from "antd/es/table";

const API_BASE = process.env.NEXT_PUBLIC_API_BASE ?? "http://127.0.0.1:8000";
const { Title, Text } = Typography;

type DocRow = {
  doc_id: string;
  filename: string;
  parse_confidence?: number;
  needs_review?: boolean;
  review_status?: string;
  created_at?: string;
};

function DocsCenterContent() {
  const searchParams = useSearchParams();

  const [docs, setDocs] = useState<DocRow[]>([]);
  const [queue, setQueue] = useState<DocRow[]>([]);
  const [loading, setLoading] = useState(false);
  const [message, setMessage] = useState("");
  const [error, setError] = useState("");

  const [file, setFile] = useState<File | null>(null);
  const [fileSource, setFileSource] = useState("user_upload");
  const [reviewComment, setReviewComment] = useState("人工复核通过");

  const [docId, setDocId] = useState("web-doc-demo");
  const [filename, setFilename] = useState("demo-note.txt");
  const [source, setSource] = useState("user_upload");
  const [content, setContent] = useState("这是用于测试的文本内容，可直接走旧的 JSON 上传接口。\nSH600000 纪要。\n");

  const [locatedChunk, setLocatedChunk] = useState<Record<string, any> | null>(null);
  const [locatedContext, setLocatedContext] = useState<Record<string, any> | null>(null);
  const [locating, setLocating] = useState(false);

  async function parseOrThrow(resp: Response) {
    const body = await resp.json();
    if (!resp.ok) {
      throw new Error(body?.detail ?? body?.error ?? `HTTP ${resp.status}`);
    }
    return body;
  }

  function resetFeedback() {
    setError("");
    setMessage("");
  }

  async function fileToBase64(uploadFile: File): Promise<string> {
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
      reader.readAsArrayBuffer(uploadFile);
    });
  }

  async function loadChunkLocation(chunkId: string, silent = false) {
    const normalizedChunkId = chunkId.trim();
    if (!normalizedChunkId) return;
    setLocating(true);
    if (!silent) resetFeedback();
    try {
      const resp = await fetch(`${API_BASE}/v1/rag/docs/chunks/${encodeURIComponent(normalizedChunkId)}?context_window=1`);
      const body = (await parseOrThrow(resp)) as Record<string, any>;
      setLocatedChunk((body?.chunk as Record<string, any> | undefined) ?? null);
      setLocatedContext((body?.context as Record<string, any> | undefined) ?? null);
      const currentDocId = String(body?.chunk?.doc_id ?? "").trim();
      if (currentDocId) setDocId(currentDocId);
      if (!silent) setMessage(`已定位片段：${normalizedChunkId}`);
    } catch (e) {
      setError(e instanceof Error ? e.message : "定位片段失败");
    } finally {
      setLocating(false);
    }
  }

  useEffect(() => {
    const queryDocId = String(searchParams.get("doc_id") ?? "").trim();
    const queryChunkId = String(searchParams.get("chunk_id") ?? "").trim();
    if (queryDocId) setDocId(queryDocId);
    if (queryChunkId) {
      loadChunkLocation(queryChunkId, true).catch(() => undefined);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [searchParams]);

  async function load() {
    setLoading(true);
    resetFeedback();
    try {
      const [a, b] = await Promise.all([
        fetch(`${API_BASE}/v1/docs`),
        fetch(`${API_BASE}/v1/docs/review-queue`),
      ]);
      setDocs((await parseOrThrow(a)) as DocRow[]);
      setQueue((await parseOrThrow(b)) as DocRow[]);
      setMessage("文档列表与复核队列已刷新");
    } catch (e) {
      setError(e instanceof Error ? e.message : "加载失败");
    } finally {
      setLoading(false);
    }
  }

  async function uploadFileToRag() {
    if (!file) {
      setError("请先选择附件");
      return;
    }
    setLoading(true);
    resetFeedback();
    try {
      const payload = {
        filename: file.name,
        content_type: file.type,
        content_base64: await fileToBase64(file),
        source: fileSource,
        auto_index: true,
        user_id: "frontend-docs",
      };
      await parseOrThrow(
        await fetch(`${API_BASE}/v1/rag/workflow/upload-and-index`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload),
        }),
      );
      setMessage(`附件上传并入库完成：${file.name}`);
      await load();
    } catch (e) {
      setError(e instanceof Error ? e.message : "附件上传失败");
    } finally {
      setLoading(false);
    }
  }

  async function uploadTextAndIndex() {
    setLoading(true);
    resetFeedback();
    try {
      await parseOrThrow(
        await fetch(`${API_BASE}/v1/docs/upload`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            doc_id: docId.trim(),
            filename: filename.trim(),
            content,
            source,
          }),
        }),
      );
      await parseOrThrow(await fetch(`${API_BASE}/v1/docs/${encodeURIComponent(docId.trim())}/index`, { method: "POST" }));
      setMessage(`文本上传并索引完成：${docId}`);
      await load();
    } catch (e) {
      setError(e instanceof Error ? e.message : "文本上传失败");
    } finally {
      setLoading(false);
    }
  }

  async function reviewAction(targetDocId: string, action: "approve" | "reject") {
    setLoading(true);
    resetFeedback();
    try {
      await parseOrThrow(
        await fetch(`${API_BASE}/v1/docs/${encodeURIComponent(targetDocId)}/review/${action}`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ comment: reviewComment }),
        }),
      );
      setMessage(`文档 ${targetDocId} 已${action === "approve" ? "通过" : "驳回"}复核`);
      await load();
    } catch (e) {
      setError(e instanceof Error ? e.message : "复核操作失败");
    } finally {
      setLoading(false);
    }
  }

  const columns: ColumnsType<DocRow> = [
    { title: "文档ID", dataIndex: "doc_id", key: "doc_id" },
    { title: "文件名", dataIndex: "filename", key: "filename" },
    {
      title: "解析置信度",
      dataIndex: "parse_confidence",
      key: "parse_confidence",
      render: (v: number | undefined) => (typeof v === "number" ? v.toFixed(2) : "-"),
    },
    {
      title: "状态",
      key: "status",
      render: (_: unknown, row: DocRow) => (
        <Space>
          <Tag color={row.needs_review ? "orange" : "green"}>{row.needs_review ? "待复核" : "可用"}</Tag>
          {row.review_status ? <Tag>{row.review_status}</Tag> : null}
        </Space>
      ),
    },
    { title: "创建时间", dataIndex: "created_at", key: "created_at" },
  ];

  const queryChunkId = String(searchParams.get("chunk_id") ?? "").trim();
  const queryText = String(searchParams.get("q") ?? "").trim();

  return (
    <main className="container shell-fade-in">
      <Card className="premium-card">
        <Space direction="vertical" style={{ width: "100%" }} size={10}>
          <Title level={2} style={{ margin: 0 }}>文档知识中心</Title>
          <Text type="secondary">覆盖接口：`/v1/rag/workflow/upload-and-index`、`/v1/docs/*`、`/v1/rag/docs/chunks/{'{'}chunk_id{'}'}`。</Text>
          <Space wrap>
            <Button type="primary" loading={loading} onClick={load}>刷新列表与复核队列</Button>
            {queryChunkId ? (
              <Button loading={locating} onClick={() => loadChunkLocation(queryChunkId)}>重新定位 URL 片段</Button>
            ) : null}
          </Space>
        </Space>
      </Card>

      <Card className="premium-card" style={{ marginTop: 12 }} title="附件上传（推荐）">
        <Space direction="vertical" style={{ width: "100%" }} size={10}>
          <input
            type="file"
            onChange={(e) => {
              const next = e.target.files?.[0] ?? null;
              setFile(next);
            }}
          />
          <Space wrap>
            <Input value={fileSource} onChange={(e) => setFileSource(e.target.value)} placeholder="source" style={{ width: 180 }} />
            <Button loading={loading} type="primary" onClick={uploadFileToRag}>上传并入库</Button>
          </Space>
          <Text style={{ color: "#64748b" }}>
            当前文件：{file ? `${file.name} (${(file.size / 1024).toFixed(1)} KB)` : "未选择"}
          </Text>
        </Space>
      </Card>

      <Card className="premium-card" style={{ marginTop: 12 }} title="文本上传（兼容旧流程）">
        <Space direction="vertical" style={{ width: "100%" }} size={10}>
          <Space.Compact block>
            <Input value={docId} onChange={(e) => setDocId(e.target.value)} placeholder="doc_id" />
            <Input value={filename} onChange={(e) => setFilename(e.target.value)} placeholder="filename" />
            <Input value={source} onChange={(e) => setSource(e.target.value)} placeholder="source" />
            <Button loading={loading} onClick={uploadTextAndIndex}>上传并索引</Button>
          </Space.Compact>
          <Input.TextArea rows={4} value={content} onChange={(e) => setContent(e.target.value)} placeholder="文档内容" />
          <Input value={reviewComment} onChange={(e) => setReviewComment(e.target.value)} placeholder="复核备注" />
        </Space>
      </Card>

      {message ? <Alert style={{ marginTop: 12 }} type="success" showIcon message={message} /> : null}
      {error ? <Alert style={{ marginTop: 12 }} type="error" showIcon message={error} /> : null}

      {(locatedChunk || locating || queryChunkId) ? (
        <Card className="premium-card" style={{ marginTop: 12 }} title="片段定位视图">
          <Space direction="vertical" style={{ width: "100%" }} size={10}>
            <Text type="secondary">
              来源参数：chunk_id={queryChunkId || "-"} {queryText ? `| query=${queryText}` : ""}
            </Text>
            {locating ? <Text>正在加载定位片段...</Text> : null}
            {!locating && locatedChunk ? (
              <>
                <Space wrap>
                  <Tag>doc_id: {String(locatedChunk.doc_id ?? "-")}</Tag>
                  <Tag>chunk_no: {String(locatedChunk.chunk_no ?? "-")}</Tag>
                  <Tag>{String(locatedChunk.source ?? "-")}</Tag>
                  <Tag>{String(locatedChunk.effective_status ?? "-")}</Tag>
                </Space>
                <Card size="small" title="当前片段">
                  <Text style={{ whiteSpace: "pre-wrap", color: "#334155" }}>
                    {String(locatedChunk.chunk_text_redacted || locatedChunk.chunk_text || "")}
                  </Text>
                </Card>
                <Card size="small" title="上文片段">
                  <Space direction="vertical" style={{ width: "100%" }}>
                    {(locatedContext?.prev ?? []).length === 0 ? (
                      <Text type="secondary">无上文片段</Text>
                    ) : (
                      (locatedContext?.prev ?? []).map((ctx: any) => (
                        <Card key={`prev-${String(ctx?.chunk_id ?? "")}`} size="small">
                          <Text strong>#{String(ctx?.chunk_no ?? "-")}</Text>
                          <Text style={{ display: "block", color: "#475569", marginTop: 4 }}>{String(ctx?.excerpt ?? "")}</Text>
                        </Card>
                      ))
                    )}
                  </Space>
                </Card>
                <Card size="small" title="下文片段">
                  <Space direction="vertical" style={{ width: "100%" }}>
                    {(locatedContext?.next ?? []).length === 0 ? (
                      <Text type="secondary">无下文片段</Text>
                    ) : (
                      (locatedContext?.next ?? []).map((ctx: any) => (
                        <Card key={`next-${String(ctx?.chunk_id ?? "")}`} size="small">
                          <Text strong>#{String(ctx?.chunk_no ?? "-")}</Text>
                          <Text style={{ display: "block", color: "#475569", marginTop: 4 }}>{String(ctx?.excerpt ?? "")}</Text>
                        </Card>
                      ))
                    )}
                  </Space>
                </Card>
              </>
            ) : null}
          </Space>
        </Card>
      ) : null}

      <Card className="premium-card" style={{ marginTop: 12 }} title="文档列表">
        <Table rowKey="doc_id" columns={columns} dataSource={docs} pagination={false} />
      </Card>

      <Card className="premium-card" style={{ marginTop: 12 }} title="复核队列">
        <Table
          rowKey="doc_id"
          columns={[
            ...columns,
            {
              title: "操作",
              key: "actions",
              render: (_: unknown, row: DocRow) => (
                <Space>
                  <Button size="small" type="primary" onClick={() => reviewAction(row.doc_id, "approve")}>通过</Button>
                  <Button size="small" danger onClick={() => reviewAction(row.doc_id, "reject")}>驳回</Button>
                </Space>
              ),
            },
          ]}
          dataSource={queue}
          pagination={false}
        />
      </Card>
    </main>
  );
}

export default function DocsCenterPage() {
  return (
    <Suspense fallback={<main className="container"><Card><Text>加载中...</Text></Card></main>}>
      <DocsCenterContent />
    </Suspense>
  );
}
