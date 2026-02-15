"use client";

import { useState } from "react";
import { Alert, Button, Card, Input, Space, Table, Tag, Typography } from "antd";
import type { ColumnsType } from "antd/es/table";
import StockSelectorModal from "../../components/StockSelectorModal";

const API_BASE = process.env.NEXT_PUBLIC_API_BASE ?? "http://127.0.0.1:8000";
const { Title, Text } = Typography;

type HealthRow = {
  source_id: string;
  success_rate: number;
  circuit_open: number;
  last_error: string;
  updated_at: string;
};

export default function OpsHealthPage() {
  const [stockCodes, setStockCodes] = useState<string[]>(["SH600000", "SZ000001"]);
  const [rows, setRows] = useState<HealthRow[]>([]);
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

  function parseCodes() {
    return stockCodes
      .map((x) => x.trim().toUpperCase())
      .filter(Boolean);
  }

  async function load() {
    setLoading(true);
    setError("");
    try {
      const body = await parseOrThrow(
        await fetch(`${API_BASE}/v1/ops/data-sources/health`)
      );
      setRows(body as HealthRow[]);
      setRaw(JSON.stringify(body, null, 2));
    } catch (e) {
      setError(e instanceof Error ? e.message : "请求失败");
      setRows([]);
    } finally {
      setLoading(false);
    }
  }

  // 中文注释：运维页加入“手动补抓”，可直接触发行情/公告数据摄取并回刷健康状态。
  async function runIngest(kind: "market-daily" | "announcements") {
    setLoading(true);
    setError("");
    try {
      const payload = { stock_codes: parseCodes() };
      const body = await parseOrThrow(
        await fetch(`${API_BASE}/v1/ingest/${kind}`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload)
        })
      );
      setRaw(JSON.stringify(body, null, 2));
      await load();
    } catch (e) {
      setError(e instanceof Error ? e.message : "补抓失败");
    } finally {
      setLoading(false);
    }
  }

  const columns: ColumnsType<HealthRow> = [
    { title: "数据源", dataIndex: "source_id", key: "source_id" },
    {
      title: "成功率",
      dataIndex: "success_rate",
      key: "success_rate",
      render: (v: number) => `${Math.round((v || 0) * 100)}%`
    },
    {
      title: "熔断",
      dataIndex: "circuit_open",
      key: "circuit_open",
      render: (v: number) => <Tag color={v ? "red" : "green"}>{v ? "OPEN" : "CLOSED"}</Tag>
    },
    { title: "最近错误", dataIndex: "last_error", key: "last_error", render: (v: string) => v || "-" },
    { title: "更新时间", dataIndex: "updated_at", key: "updated_at" }
  ];

  return (
    <main className="container">
      <Card className="premium-card">
        <Space direction="vertical" style={{ width: "100%" }}>
          <Title level={2} style={{ margin: 0 }}>数据源健康与补抓</Title>
          <Text type="secondary">覆盖接口：`/v1/ops/data-sources/health`、`/v1/ingest/market-daily`、`/v1/ingest/announcements`</Text>
          <Space direction="vertical" style={{ width: "100%" }}>
            <StockSelectorModal
              value={stockCodes}
              onChange={(next) => setStockCodes(Array.isArray(next) ? next : (next ? [next] : []))}
              multiple
              title="选择补抓股票"
              placeholder="请选择一个或多个股票"
            />
            <Space>
            <Button loading={loading} onClick={() => runIngest("market-daily")}>补抓行情</Button>
            <Button loading={loading} onClick={() => runIngest("announcements")}>补抓公告</Button>
            <Button loading={loading} type="primary" onClick={load}>刷新健康状态</Button>
            </Space>
          </Space>
        </Space>
      </Card>
      {error ? <Alert style={{ marginTop: 12 }} type="error" showIcon message={error} /> : null}
      <Card className="premium-card" style={{ marginTop: 12 }} title="健康状态">
        <Table rowKey="source_id" columns={columns} dataSource={rows} pagination={false} />
      </Card>
      <Card className="premium-card" style={{ marginTop: 12 }} title="最近接口返回">
        <pre style={{ whiteSpace: "pre-wrap", margin: 0 }}>{raw}</pre>
      </Card>
    </main>
  );
}
