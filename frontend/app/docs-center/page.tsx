"use client";

import { useState } from "react";
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
  updated_at?: string;
};

export default function DocsCenterPage() {
  const [docs, setDocs] = useState<DocRow[]>([]);
  const [queue, setQueue] = useState<DocRow[]>([]);
  const [docId, setDocId] = useState("web-doc-demo");
  const [filename, setFilename] = useState("demo-report.pdf");
  const [source, setSource] = useState("user_upload");
  const [content, setContent] = useState("这是一份用于测试的财报摘要文档内容。");
  const [reviewComment, setReviewComment] = useState("人工复核通过");
  const [loading, setLoading] = useState(false);
  const [message, setMessage] = useState("");
  const [error, setError] = useState("");

  // 中文注释：统一处理 API 错误，减少页面重复代码。
  async function parseOrThrow(resp: Response) {
    const body = await resp.json();
    if (!resp.ok) {
      throw new Error(body?.detail ?? body?.error ?? `HTTP ${resp.status}`);
    }
    return body;
  }

  async function load() {
    setLoading(true);
    setError("");
    try {
      const [a, b] = await Promise.all([
        fetch(`${API_BASE}/v1/docs`),
        fetch(`${API_BASE}/v1/docs/review-queue`)
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

  async function uploadAndIndex() {
    setLoading(true);
    setError("");
    try {
      await parseOrThrow(
        await fetch(`${API_BASE}/v1/docs/upload`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ doc_id: docId.trim(), filename: filename.trim(), content, source })
        })
      );
      await parseOrThrow(await fetch(`${API_BASE}/v1/docs/${encodeURIComponent(docId.trim())}/index`, { method: "POST" }));
      setMessage(`文档 ${docId} 上传并索引完成`);
      await load();
    } catch (e) {
      setError(e instanceof Error ? e.message : "上传失败");
    } finally {
      setLoading(false);
    }
  }

  async function reviewAction(targetDocId: string, action: "approve" | "reject") {
    setLoading(true);
    setError("");
    try {
      await parseOrThrow(
        await fetch(`${API_BASE}/v1/docs/${encodeURIComponent(targetDocId)}/review/${action}`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ comment: reviewComment })
        })
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
      render: (v: number | undefined) => (typeof v === "number" ? v.toFixed(2) : "-")
    },
    {
      title: "状态",
      key: "status",
      render: (_, row) => (
        <Space>
          <Tag color={row.needs_review ? "orange" : "green"}>{row.needs_review ? "待复核" : "可用"}</Tag>
          {row.review_status ? <Tag>{row.review_status}</Tag> : null}
        </Space>
      )
    },
    { title: "更新时间", dataIndex: "updated_at", key: "updated_at" }
  ];

  return (
    <main className="container">
      <Card className="premium-card">
        <Space direction="vertical" style={{ width: "100%" }} size={12}>
          <Title level={2} style={{ margin: 0 }}>文档知识库管理台</Title>
          <Text type="secondary">覆盖接口：`/v1/docs/upload`、`/v1/docs/{`doc_id`}/index`、`/v1/docs`、`/v1/docs/review-queue`、`/review/approve|reject`</Text>
          <Button type="primary" loading={loading} onClick={load}>刷新列表与复核队列</Button>

          <Space.Compact block>
            <Input value={docId} onChange={(e) => setDocId(e.target.value)} placeholder="doc_id" />
            <Input value={filename} onChange={(e) => setFilename(e.target.value)} placeholder="filename" />
            <Input value={source} onChange={(e) => setSource(e.target.value)} placeholder="source" />
            <Button loading={loading} onClick={uploadAndIndex}>上传并索引</Button>
          </Space.Compact>
          <Input.TextArea rows={4} value={content} onChange={(e) => setContent(e.target.value)} placeholder="文档内容（demo文本）" />
          <Input value={reviewComment} onChange={(e) => setReviewComment(e.target.value)} placeholder="复核备注" />
        </Space>
      </Card>

      {message ? <Alert style={{ marginTop: 12 }} type="success" showIcon message={message} /> : null}
      {error ? <Alert style={{ marginTop: 12 }} type="error" showIcon message={error} /> : null}

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
              render: (_, row) => (
                <Space>
                  <Button size="small" type="primary" onClick={() => reviewAction(row.doc_id, "approve")}>通过</Button>
                  <Button size="small" danger onClick={() => reviewAction(row.doc_id, "reject")}>驳回</Button>
                </Space>
              )
            }
          ]}
          dataSource={queue}
          pagination={false}
        />
      </Card>
    </main>
  );
}
