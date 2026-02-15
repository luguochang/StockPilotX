"use client";

import { useState } from "react";
import { Alert, Button, Card, Input, Space, Table, Tag, Typography } from "antd";
import type { ColumnsType } from "antd/es/table";

const API_BASE = process.env.NEXT_PUBLIC_API_BASE ?? "http://127.0.0.1:8000";
const { Title, Text } = Typography;

type CapabilityRow = { key: string; status: string; detail: string };

export default function OpsEvalsPage() {
  const [token, setToken] = useState("");
  const [evals, setEvals] = useState("");
  const [releases, setReleases] = useState("");
  const [capabilities, setCapabilities] = useState<CapabilityRow[]>([]);
  const [runtimeInfo, setRuntimeInfo] = useState("");
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
      const headers: Record<string, string> = {};
      if (token.trim()) headers.Authorization = `Bearer ${token}`;
      const [a, b, c] = await Promise.all([
        fetch(`${API_BASE}/v1/ops/evals/history`, { headers }),
        fetch(`${API_BASE}/v1/ops/prompts/releases`, { headers }),
        fetch(`${API_BASE}/v1/ops/capabilities`)
      ]);
      const evalData = await parseOrThrow(a);
      const releaseData = await parseOrThrow(b);
      const capabilityData = await parseOrThrow(c);
      setEvals(JSON.stringify(evalData, null, 2));
      setReleases(JSON.stringify(releaseData, null, 2));
      setCapabilities((capabilityData?.capabilities ?? []) as CapabilityRow[]);
      setRuntimeInfo(JSON.stringify(capabilityData?.runtime ?? {}, null, 2));
    } catch (e) {
      setError(e instanceof Error ? e.message : "加载失败");
      setCapabilities([]);
    } finally {
      setLoading(false);
    }
  }

  const columns: ColumnsType<CapabilityRow> = [
    { title: "技术点", dataIndex: "key", key: "key" },
    {
      title: "状态",
      dataIndex: "status",
      key: "status",
      render: (v: string) => {
        const color = v.includes("implemented") ? "green" : v.includes("fallback") ? "gold" : "default";
        return <Tag color={color}>{v}</Tag>;
      }
    },
    { title: "说明", dataIndex: "detail", key: "detail" }
  ];

  return (
    <main className="container">
      <Card className="premium-card">
        <Space direction="vertical" style={{ width: "100%" }}>
          <Title level={2} style={{ margin: 0 }}>门禁与技术核查控制台</Title>
          <Text type="secondary">加载评测门禁 + Prompt发布 + 技术点能力快照</Text>
          <Input.Password
            value={token}
            onChange={(e) => setToken(e.target.value)}
            placeholder="Bearer Token（ops/admin用于 eval/release）"
          />
          <Button type="primary" loading={loading} onClick={load}>加载门禁信息</Button>
        </Space>
      </Card>
      {error ? <Alert style={{ marginTop: 12 }} type="error" showIcon message={error} /> : null}
      <Card className="premium-card" style={{ marginTop: 12 }} title="技术点实现快照">
        <Table rowKey="key" columns={columns} dataSource={capabilities} pagination={false} />
      </Card>
      <Card className="premium-card" style={{ marginTop: 12 }} title="运行时信息">
        <pre style={{ whiteSpace: "pre-wrap", margin: 0 }}>{runtimeInfo}</pre>
      </Card>
      <Card className="premium-card" style={{ marginTop: 12 }} title="评测历史">
        <pre style={{ whiteSpace: "pre-wrap", margin: 0 }}>{evals}</pre>
      </Card>
      <Card className="premium-card" style={{ marginTop: 12 }} title="Prompt 发布记录">
        <pre style={{ whiteSpace: "pre-wrap", margin: 0 }}>{releases}</pre>
      </Card>
    </main>
  );
}
