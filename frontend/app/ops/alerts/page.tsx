"use client";

import { useMemo, useState } from "react";
import { Alert, Button, Card, Input, Select, Space, Table, Tag, Typography } from "antd";
import type { ColumnsType } from "antd/es/table";

const API_BASE = process.env.NEXT_PUBLIC_API_BASE ?? "http://127.0.0.1:8000";
const { Title, Text } = Typography;

type AlertRow = {
  id: number;
  alert_type: string;
  severity: string;
  message: string;
  status: string;
  created_at: string;
};

export default function OpsAlertsPage() {
  const [severity, setSeverity] = useState<"all" | "high" | "medium" | "low">("all");
  const [status, setStatus] = useState<"all" | "open" | "acked">("all");
  const [keyword, setKeyword] = useState("");
  const [rows, setRows] = useState<AlertRow[]>([]);
  const [raw, setRaw] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  async function parseOrThrow(resp: Response) {
    const body = await resp.json();
    if (!resp.ok) {
      throw new Error(body?.detail ?? `HTTP ${resp.status}`);
    }
    return body;
  }

  async function load() {
    setLoading(true);
    setError("");
    try {
      const body = await parseOrThrow(
        await fetch(`${API_BASE}/v1/alerts`)
      );
      setRows(body as AlertRow[]);
      setRaw(JSON.stringify(body, null, 2));
    } catch (e) {
      setError(e instanceof Error ? e.message : "加载失败");
      setRows([]);
    } finally {
      setLoading(false);
    }
  }

  // 中文注释：确认后自动刷新列表，避免页面状态与后端不一致。
  async function ack(id: number) {
    setLoading(true);
    setError("");
    try {
      const body = await parseOrThrow(
        await fetch(`${API_BASE}/v1/alerts/${id}/ack`, {
          method: "POST"
        })
      );
      setRaw(JSON.stringify(body, null, 2));
      await load();
    } catch (e) {
      setError(e instanceof Error ? e.message : "确认失败");
    } finally {
      setLoading(false);
    }
  }

  const filteredRows = useMemo(() => {
    return rows.filter((r) => {
      if (severity !== "all" && r.severity !== severity) return false;
      if (status !== "all" && r.status !== status) return false;
      if (keyword.trim()) {
        const q = keyword.trim().toLowerCase();
        const joined = `${r.alert_type} ${r.message}`.toLowerCase();
        if (!joined.includes(q)) return false;
      }
      return true;
    });
  }, [rows, severity, status, keyword]);

  const columns: ColumnsType<AlertRow> = [
    { title: "ID", dataIndex: "id", key: "id", width: 72 },
    { title: "类型", dataIndex: "alert_type", key: "alert_type" },
    {
      title: "级别",
      dataIndex: "severity",
      key: "severity",
      render: (v: string) => {
        const color = v === "high" ? "red" : v === "medium" ? "orange" : "blue";
        return <Tag color={color}>{v}</Tag>;
      }
    },
    { title: "内容", dataIndex: "message", key: "message" },
    {
      title: "状态",
      dataIndex: "status",
      key: "status",
      render: (v: string) => <Tag color={v === "acked" ? "green" : "volcano"}>{v}</Tag>
    },
    { title: "时间", dataIndex: "created_at", key: "created_at" },
    {
      title: "操作",
      key: "actions",
      render: (_, row) => (
        <Button size="small" disabled={row.status === "acked"} onClick={() => ack(row.id)}>
          确认
        </Button>
      )
    }
  ];

  return (
    <main className="container">
      <Card className="premium-card">
        <Space direction="vertical" style={{ width: "100%" }}>
          <Title level={2} style={{ margin: 0 }}>告警中心</Title>
          <Text type="secondary">覆盖接口：`/v1/alerts`、`/v1/alerts/{`id`}/ack`</Text>
          <Space wrap>
            <Select
              value={severity}
              onChange={(v) => setSeverity(v)}
              options={[
                { label: "全部级别", value: "all" },
                { label: "high", value: "high" },
                { label: "medium", value: "medium" },
                { label: "low", value: "low" }
              ]}
              style={{ minWidth: 120 }}
            />
            <Select
              value={status}
              onChange={(v) => setStatus(v)}
              options={[
                { label: "全部状态", value: "all" },
                { label: "open", value: "open" },
                { label: "acked", value: "acked" }
              ]}
              style={{ minWidth: 120 }}
            />
            <Input value={keyword} onChange={(e) => setKeyword(e.target.value)} placeholder="关键词过滤" style={{ width: 220 }} />
            <Button type="primary" loading={loading} onClick={load}>加载/刷新告警</Button>
          </Space>
        </Space>
      </Card>
      {error ? <Alert style={{ marginTop: 12 }} type="error" showIcon message={error} /> : null}
      <Card className="premium-card" style={{ marginTop: 12 }} title={`告警列表（${filteredRows.length}）`}>
        <Table rowKey="id" columns={columns} dataSource={filteredRows} pagination={false} />
      </Card>
      <Card className="premium-card" style={{ marginTop: 12 }} title="最近接口返回">
        <pre style={{ whiteSpace: "pre-wrap", margin: 0 }}>{raw}</pre>
      </Card>
    </main>
  );
}
