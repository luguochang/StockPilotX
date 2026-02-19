"use client";

import { useEffect, useMemo, useState } from "react";
import { Alert, Button, Card, Select, Space, Table, Tag, Typography } from "antd";
import type { ColumnsType } from "antd/es/table";
import StockSelectorModal from "../../components/StockSelectorModal";

const API_BASE = process.env.NEXT_PUBLIC_API_BASE ?? "http://127.0.0.1:8000";
const { Title, Text } = Typography;

type FetchCategory = "quote" | "announcement" | "financial" | "news" | "research" | "macro" | "fund" | "history";

type SourceRow = {
  source_id: string;
  category: string;
  enabled: boolean;
  reliability_score: number;
  source_url: string;
  proxy_enabled: boolean;
  used_in_ui_modules?: string[];
};

type HealthRow = {
  source_id: string;
  category: string;
  enabled: boolean;
  reliability_score: number;
  source_url: string;
  proxy_enabled: boolean;
  attempts: number;
  success_rate: number;
  failure_rate: number;
  last_error: string;
  last_latency_ms: number;
  updated_at: string;
  last_used_at?: string;
  staleness_minutes?: number | null;
  circuit_open: boolean;
  used_in_ui_modules?: string[];
};

const FETCH_CATEGORY_OPTIONS: Array<{ value: FetchCategory; label: string; needsStockCodes: boolean }> = [
  { value: "quote", label: "行情", needsStockCodes: true },
  { value: "announcement", label: "公告", needsStockCodes: true },
  { value: "financial", label: "财务", needsStockCodes: true },
  { value: "news", label: "新闻", needsStockCodes: true },
  { value: "research", label: "研报", needsStockCodes: true },
  { value: "macro", label: "宏观", needsStockCodes: false },
  { value: "fund", label: "资金", needsStockCodes: true },
  { value: "history", label: "历史K线", needsStockCodes: true },
];

export default function OpsHealthPage() {
  const [stockCodes, setStockCodes] = useState<string[]>(["SH600000", "SZ000001"]);
  const [fetchCategory, setFetchCategory] = useState<FetchCategory>("quote");
  const [rows, setRows] = useState<HealthRow[]>([]);
  const [sourceRows, setSourceRows] = useState<SourceRow[]>([]);
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

  async function loadSources() {
    const body = await parseOrThrow(await fetch(`${API_BASE}/v1/datasources/sources`));
    const items = Array.isArray(body?.items) ? body.items : [];
    setSourceRows(items as SourceRow[]);
    return body;
  }

  async function loadHealth() {
    const body = await parseOrThrow(await fetch(`${API_BASE}/v1/datasources/health?limit=300`));
    const items = Array.isArray(body?.items) ? body.items : [];
    setRows(items as HealthRow[]);
    setRaw(JSON.stringify(body, null, 2));
    return body;
  }

  async function refreshDashboard() {
    setLoading(true);
    setError("");
    try {
      const [sourcesBody, healthBody] = await Promise.all([loadSources(), loadHealth()]);
      setRaw(JSON.stringify({ sources: sourcesBody, health: healthBody }, null, 2));
    } catch (e) {
      setError(e instanceof Error ? e.message : "请求失败");
      setRows([]);
      setSourceRows([]);
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    void refreshDashboard();
    // 页面仅在首屏加载时拉取一次，后续由手动刷新触发。
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const selectedCategoryMeta = useMemo(
    () => FETCH_CATEGORY_OPTIONS.find((x) => x.value === fetchCategory),
    [fetchCategory]
  );

  const selectedSourceId = useMemo(() => {
    const fromEnabled = sourceRows.find((row) => row.category === fetchCategory && row.enabled);
    if (fromEnabled) return fromEnabled.source_id;
    const fromAny = sourceRows.find((row) => row.category === fetchCategory);
    return fromAny?.source_id ?? "";
  }, [fetchCategory, sourceRows]);

  async function runFetch() {
    setLoading(true);
    setError("");
    try {
      if (!selectedSourceId) {
        throw new Error(`未找到可用数据源（category=${fetchCategory}）`);
      }
      const payload: Record<string, unknown> = {
        source_id: selectedSourceId,
        category: fetchCategory,
        // 新闻/研报/宏观默认 limit 较小，避免运维一键补抓耗时过长。
        limit: ["news", "research", "macro"].includes(fetchCategory) ? 8 : 20,
      };
      if (selectedCategoryMeta?.needsStockCodes) {
        payload.stock_codes = parseCodes();
      }
      const body = await parseOrThrow(
        await fetch(`${API_BASE}/v1/datasources/fetch`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload),
        })
      );
      setRaw(JSON.stringify(body, null, 2));
      await loadHealth();
    } catch (e) {
      setError(e instanceof Error ? e.message : "补抓失败");
    } finally {
      setLoading(false);
    }
  }

  const healthColumns: ColumnsType<HealthRow> = [
    {
      title: "数据源",
      dataIndex: "source_id",
      key: "source_id",
      render: (v: string, row) => (
        <Space size={6} wrap>
          <Text style={{ color: "#0f172a" }}>{v}</Text>
          <Tag color="blue">{row.category}</Tag>
        </Space>
      ),
    },
    {
      title: "成功率",
      dataIndex: "success_rate",
      key: "success_rate",
      render: (v: number) => `${Math.round((v || 0) * 100)}%`,
    },
    {
      title: "熔断",
      dataIndex: "circuit_open",
      key: "circuit_open",
      render: (v: boolean) => <Tag color={v ? "red" : "green"}>{v ? "OPEN" : "CLOSED"}</Tag>,
    },
    {
      title: "新鲜度",
      dataIndex: "staleness_minutes",
      key: "staleness_minutes",
      render: (v: number | null | undefined) => (typeof v === "number" ? `${v} 分钟前` : "未知"),
    },
    {
      title: "服务模块",
      dataIndex: "used_in_ui_modules",
      key: "used_in_ui_modules",
      render: (modules: string[] | undefined) => (
        <Space wrap size={[4, 4]}>
          {(Array.isArray(modules) ? modules : []).map((m) => (
            <Tag key={`${m}`}>{m}</Tag>
          ))}
          {!Array.isArray(modules) || modules.length === 0 ? <Tag>-</Tag> : null}
        </Space>
      ),
    },
    {
      title: "最近错误",
      dataIndex: "last_error",
      key: "last_error",
      render: (v: string) => v || "-",
    },
    {
      title: "更新时间",
      dataIndex: "updated_at",
      key: "updated_at",
      render: (v: string) => v || "-",
    },
  ];

  const sourceColumns: ColumnsType<SourceRow> = [
    { title: "source_id", dataIndex: "source_id", key: "source_id" },
    { title: "类别", dataIndex: "category", key: "category" },
    {
      title: "可用",
      dataIndex: "enabled",
      key: "enabled",
      render: (v: boolean) => <Tag color={v ? "green" : "red"}>{v ? "Yes" : "No"}</Tag>,
    },
    {
      title: "可靠度",
      dataIndex: "reliability_score",
      key: "reliability_score",
      render: (v: number) => Number(v || 0).toFixed(2),
    },
    {
      title: "模块映射",
      dataIndex: "used_in_ui_modules",
      key: "used_in_ui_modules",
      render: (modules: string[] | undefined) => (
        <Space wrap size={[4, 4]}>
          {(Array.isArray(modules) ? modules : []).map((m) => (
            <Tag key={`${m}`}>{m}</Tag>
          ))}
          {!Array.isArray(modules) || modules.length === 0 ? <Tag>-</Tag> : null}
        </Space>
      ),
    },
  ];

  return (
    <main className="container">
      <Card className="premium-card">
        <Space direction="vertical" style={{ width: "100%" }}>
          <Title level={2} style={{ margin: 0 }}>数据源健康与补抓</Title>
          <Text type="secondary">
            覆盖接口：`/v1/datasources/sources`、`/v1/datasources/health`、`/v1/datasources/fetch`、`/v1/datasources/logs`
          </Text>
          <Space direction="vertical" style={{ width: "100%" }}>
            <StockSelectorModal
              value={stockCodes}
              onChange={(next) => setStockCodes(Array.isArray(next) ? next : (next ? [next] : []))}
              multiple
              title="选择补抓股票"
              placeholder="请选择一个或多个股票"
            />
            <Space wrap>
              <Select<FetchCategory>
                value={fetchCategory}
                options={FETCH_CATEGORY_OPTIONS.map((x) => ({ value: x.value, label: x.label }))}
                onChange={(value) => setFetchCategory(value)}
                style={{ minWidth: 180 }}
              />
              <Button type="primary" loading={loading} onClick={runFetch}>执行补抓</Button>
              <Button loading={loading} onClick={refreshDashboard}>刷新健康状态</Button>
            </Space>
            <Text style={{ color: "#64748b" }}>
              当前类别：{selectedCategoryMeta?.label ?? "-"} | 使用数据源：{selectedSourceId || "未匹配到可用 source_id"}
            </Text>
          </Space>
        </Space>
      </Card>
      {error ? <Alert style={{ marginTop: 12 }} type="error" showIcon message={error} /> : null}
      <Card className="premium-card" style={{ marginTop: 12 }} title="健康状态">
        <Table rowKey={(row) => `${row.source_id}-${row.category}`} columns={healthColumns} dataSource={rows} pagination={false} />
      </Card>
      <Card className="premium-card" style={{ marginTop: 12 }} title="数据源目录">
        <Table rowKey="source_id" columns={sourceColumns} dataSource={sourceRows} pagination={false} />
      </Card>
      <Card className="premium-card" style={{ marginTop: 12 }} title="最近接口返回">
        <pre style={{ whiteSpace: "pre-wrap", margin: 0 }}>{raw}</pre>
      </Card>
    </main>
  );
}
