"use client";

import { useState } from "react";
import { motion } from "framer-motion";
import { Alert, Button, Card, Col, Collapse, Empty, Input, List, Progress, Row, Select, Space, Tag, Typography } from "antd";
import MediaCarousel from "../components/MediaCarousel";
import StockSelectorModal from "../components/StockSelectorModal";

const API_BASE = process.env.NEXT_PUBLIC_API_BASE ?? "http://127.0.0.1:8000";
const { Title, Text, Paragraph } = Typography;

type ReportItem = {
  report_id: string;
  stock_code: string;
  report_type: string;
  created_at: string;
};

type BusinessModuleHealth = {
  module: string;
  status: string;
  coverage: number;
  healthy_categories: number;
  expected_categories: number;
  degrade_reasons?: string[];
};

type BusinessDataHealth = {
  status: string;
  module_health: BusinessModuleHealth[];
  stock_snapshot?: {
    has_quote?: boolean;
    history_sample_size?: number;
    has_financial?: boolean;
  };
};

function statusColor(status: string): string {
  if (status === "ok") return "success";
  if (status === "degraded") return "warning";
  if (status === "critical") return "error";
  return "default";
}

function parsePreview(raw: string): Record<string, unknown> {
  if (!raw.trim()) return {};
  try {
    const parsed = JSON.parse(raw) as Record<string, unknown>;
    return parsed && typeof parsed === "object" ? parsed : {};
  } catch {
    return {};
  }
}

export default function ReportsPage() {
  const [items, setItems] = useState<ReportItem[]>([]);
  const [selectedMarkdown, setSelectedMarkdown] = useState("");
  const [selectedRaw, setSelectedRaw] = useState("");
  const [selectedVersions, setSelectedVersions] = useState("");
  const [generateStockCode, setGenerateStockCode] = useState("SH600000");
  const [generateType, setGenerateType] = useState<"fact" | "research">("research");
  const [templateId, setTemplateId] = useState("default");
  const [runId, setRunId] = useState("");
  const [poolSnapshotId, setPoolSnapshotId] = useState("");
  const [advancedOpen, setAdvancedOpen] = useState<string[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [activeReportId, setActiveReportId] = useState("");
  const [qualityGatePreview, setQualityGatePreview] = useState("");
  const [dataPackPreview, setDataPackPreview] = useState("");
  const [generationMode, setGenerationMode] = useState("");
  const [businessHealth, setBusinessHealth] = useState<BusinessDataHealth | null>(null);

  const qualityGate = parsePreview(qualityGatePreview);
  const dataPack = parsePreview(dataPackPreview);
  const qualityStatus = String(qualityGate.status ?? "unknown");
  const qualityScore = Number(qualityGate.score ?? 0);
  const qualityReasons = Array.isArray(qualityGate.reasons) ? qualityGate.reasons.map((item) => String(item)) : [];
  const reportsHealth = (businessHealth?.module_health ?? []).find((row) => String(row.module) === "reports");

  const heroSlides = [
    { src: "/assets/images/nyse-floor-1963.jpg", alt: "Archive trading report visual", caption: "档案回溯" },
    { src: "/assets/images/nyse-floor-2014.jpg", alt: "Modern report visual", caption: "研究输出" },
    { src: "/assets/images/nyse-floor-1930.png", alt: "Long cycle report visual", caption: "长期视角" },
  ];

  async function loadBusinessHealth(stockCode: string) {
    const code = stockCode.trim().toUpperCase();
    if (!code) {
      setBusinessHealth(null);
      return;
    }
    try {
      const r = await fetch(`${API_BASE}/v1/business/data-health?stock_code=${encodeURIComponent(code)}&limit=200`);
      const body = await r.json();
      if (!r.ok) throw new Error(body?.detail ?? `HTTP ${r.status}`);
      setBusinessHealth(body as BusinessDataHealth);
    } catch {
      // Keep report flow non-blocking even if observability endpoint temporarily fails.
      setBusinessHealth(null);
    }
  }

  async function load() {
    setLoading(true);
    setError("");
    try {
      const r = await fetch(`${API_BASE}/v1/reports`);
      const body = await r.json();
      if (!r.ok) throw new Error(body?.detail ?? `HTTP ${r.status}`);
      setItems(body as ReportItem[]);
    } catch (e) {
      setError(e instanceof Error ? e.message : "请求失败");
      setItems([]);
    } finally {
      setLoading(false);
    }
  }

  async function exportReport(reportId: string) {
    setLoading(true);
    setError("");
    try {
      const r = await fetch(`${API_BASE}/v1/reports/${reportId}/export`, { method: "POST" });
      const body = await r.json();
      if (!r.ok) throw new Error(body?.detail ?? `HTTP ${r.status}`);
      setSelectedMarkdown(String(body.markdown ?? ""));
      setActiveReportId(reportId);
    } catch (e) {
      setError(e instanceof Error ? e.message : "请求失败");
      setSelectedMarkdown("");
    } finally {
      setLoading(false);
    }
  }

  async function generateReport() {
    setLoading(true);
    setError("");
    try {
      const payload = {
        user_id: "demo_user",
        stock_code: generateStockCode.trim().toUpperCase(),
        period: "1y",
        report_type: generateType,
        template_id: templateId.trim() || "default",
        run_id: runId.trim(),
        pool_snapshot_id: poolSnapshotId.trim(),
      };
      const r = await fetch(`${API_BASE}/v1/report/generate`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      const body = await r.json();
      if (!r.ok) throw new Error(body?.detail ?? `HTTP ${r.status}`);
      setSelectedMarkdown(String(body.markdown ?? ""));
      setSelectedRaw(JSON.stringify(body, null, 2));
      setQualityGatePreview(JSON.stringify((body.quality_gate as Record<string, unknown>) ?? {}, null, 2));
      setDataPackPreview(JSON.stringify((body.report_data_pack_summary as Record<string, unknown>) ?? {}, null, 2));
      setGenerationMode(String(body.generation_mode ?? ""));
      setActiveReportId(String(body.report_id ?? ""));
      await loadBusinessHealth(payload.stock_code);
      await load();
    } catch (e) {
      setError(e instanceof Error ? e.message : "请求失败");
    } finally {
      setLoading(false);
    }
  }

  async function loadReportDetail(reportId: string) {
    setLoading(true);
    setError("");
    try {
      const r = await fetch(`${API_BASE}/v1/report/${reportId}`);
      const body = await r.json();
      if (!r.ok) throw new Error(body?.detail ?? `HTTP ${r.status}`);
      setSelectedRaw(JSON.stringify(body, null, 2));
      setSelectedMarkdown(String(body.markdown ?? ""));
      setQualityGatePreview(JSON.stringify((body.quality_gate as Record<string, unknown>) ?? {}, null, 2));
      setDataPackPreview(JSON.stringify((body.report_data_pack_summary as Record<string, unknown>) ?? {}, null, 2));
      setGenerationMode(String(body.generation_mode ?? ""));
      setActiveReportId(reportId);
      const detailCode = String(body.stock_code ?? "").trim().toUpperCase();
      if (detailCode) {
        setGenerateStockCode(detailCode);
        await loadBusinessHealth(detailCode);
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : "请求失败");
    } finally {
      setLoading(false);
    }
  }

  async function loadReportVersions(reportId: string) {
    setLoading(true);
    setError("");
    try {
      const r = await fetch(`${API_BASE}/v1/reports/${reportId}/versions`);
      const body = await r.json();
      if (!r.ok) throw new Error(body?.detail ?? `HTTP ${r.status}`);
      setSelectedVersions(JSON.stringify(body, null, 2));
      setActiveReportId(reportId);
    } catch (e) {
      setError(e instanceof Error ? e.message : "请求失败");
      setSelectedVersions("");
    } finally {
      setLoading(false);
    }
  }

  return (
    <main className="container">
      <motion.section initial={{ opacity: 0, y: 12 }} whileInView={{ opacity: 1, y: 0 }} viewport={{ once: true, amount: 0.2 }} transition={{ duration: 0.45 }}>
        <MediaCarousel items={heroSlides} />
      </motion.section>

      <motion.div initial={{ opacity: 0, y: 14 }} animate={{ opacity: 1, y: 0 }}>
        <Card className="premium-card" style={{ background: "linear-gradient(132deg, rgba(255,255,255,0.96), rgba(246,249,252,0.94))" }}>
          <Space direction="vertical" style={{ width: "100%" }}>
            <Tag color="processing" style={{ width: "fit-content" }}>Research Report Center</Tag>
            <Title level={2} style={{ margin: 0, color: "#0f172a" }}>报告中心</Title>
            <Paragraph style={{ margin: 0, color: "#475569", maxWidth: 760 }}>
              主流程只保留两个输入: 股票与报告类型。高级参数折叠，减少无效输入。
            </Paragraph>
            <StockSelectorModal value={generateStockCode} onChange={(next) => setGenerateStockCode(Array.isArray(next) ? (next[0] ?? "") : next)} title="选择报告标的" placeholder="请选择股票" />
            <Space.Compact block>
              <Select value={generateType} onChange={(v) => setGenerateType(v)} options={[{ label: "事实报告", value: "fact" }, { label: "研究报告", value: "research" }]} style={{ minWidth: 180 }} />
              <Button type="primary" loading={loading} onClick={generateReport}>生成报告</Button>
              <Button loading={loading} onClick={load}>刷新列表</Button>
            </Space.Compact>
            <Collapse
              size="small"
              activeKey={advancedOpen}
              onChange={(keys) => setAdvancedOpen((Array.isArray(keys) ? keys : [keys]).map(String))}
              items={[
                {
                  key: "advanced",
                  label: "高级设置（可选）",
                  children: (
                    <Space direction="vertical" style={{ width: "100%" }}>
                      <Input value={templateId} onChange={(e) => setTemplateId(e.target.value)} placeholder="template_id" />
                      <Input value={runId} onChange={(e) => setRunId(e.target.value)} placeholder="run_id" />
                      <Input value={poolSnapshotId} onChange={(e) => setPoolSnapshotId(e.target.value)} placeholder="pool_snapshot_id" />
                    </Space>
                  ),
                },
              ]}
            />
          </Space>
        </Card>
      </motion.div>

      {error ? <Alert style={{ marginTop: 12 }} type="error" showIcon message={error} /> : null}

      {generationMode || qualityGatePreview || dataPackPreview || businessHealth ? (
        <Row gutter={[12, 12]} style={{ marginTop: 12 }}>
          <Col xs={24} lg={8}>
            <Card className="premium-card" title={<span style={{ color: "#0f172a" }}>报告质量门禁</span>}>
              <Space direction="vertical" size={6} style={{ width: "100%" }}>
                <Text style={{ color: "#475569" }}>生成模式: <Tag color={generationMode === "llm" ? "success" : "default"}>{generationMode || "unknown"}</Tag></Text>
                <Text style={{ color: "#475569" }}>质量状态: <Tag color={statusColor(qualityStatus)}>{qualityStatus}</Tag></Text>
                <Progress percent={Math.max(0, Math.min(100, Number((qualityScore * 100).toFixed(1))))} strokeColor="#2563eb" />
                <Text style={{ color: "#64748b" }}>质量得分: {qualityScore.toFixed(2)}</Text>
                <Text style={{ color: "#64748b" }}>降级原因: {qualityReasons.length ? qualityReasons.join(", ") : "none"}</Text>
              </Space>
            </Card>
          </Col>
          <Col xs={24} lg={8}>
            <Card className="premium-card" title={<span style={{ color: "#0f172a" }}>数据包摘要</span>}>
              <Space direction="vertical" size={6} style={{ width: "100%" }}>
                <Text style={{ color: "#475569" }}>历史样本数: {String(dataPack.history_sample_size ?? 0)}</Text>
                <Text style={{ color: "#475569" }}>预测质量: {String(dataPack.predict_quality ?? "unknown")}</Text>
                <Text style={{ color: "#475569" }}>情报信号: {String(dataPack.intel_signal ?? "unknown")}</Text>
                <Text style={{ color: "#475569" }}>情报置信度: {Number(dataPack.intel_confidence ?? 0).toFixed(2)}</Text>
                <Text style={{ color: "#64748b" }}>
                  新闻/研报/宏观: {String(dataPack.news_count ?? 0)} / {String(dataPack.research_count ?? 0)} / {String(dataPack.macro_count ?? 0)}
                </Text>
              </Space>
            </Card>
          </Col>
          <Col xs={24} lg={8}>
            <Card className="premium-card" title={<span style={{ color: "#0f172a" }}>业务数据健康</span>}>
              {!businessHealth ? (
                <Alert type="info" showIcon message="尚未加载业务健康快照" />
              ) : (
                <Space direction="vertical" size={6} style={{ width: "100%" }}>
                  <Text style={{ color: "#475569" }}>全局状态: <Tag color={statusColor(String(businessHealth.status))}>{String(businessHealth.status)}</Tag></Text>
                  <Text style={{ color: "#475569" }}>reports 模块: <Tag color={statusColor(String(reportsHealth?.status ?? "unknown"))}>{String(reportsHealth?.status ?? "unknown")}</Tag></Text>
                  <Text style={{ color: "#64748b" }}>覆盖率: {Number(reportsHealth?.coverage ?? 0).toFixed(2)} ({Number(reportsHealth?.healthy_categories ?? 0)}/{Number(reportsHealth?.expected_categories ?? 0)})</Text>
                  <Text style={{ color: "#64748b" }}>
                    标的快照: quote={businessHealth.stock_snapshot?.has_quote ? "yes" : "no"}, history={Number(businessHealth.stock_snapshot?.history_sample_size ?? 0)}, financial={businessHealth.stock_snapshot?.has_financial ? "yes" : "no"}
                  </Text>
                </Space>
              )}
            </Card>
          </Col>
        </Row>
      ) : null}

      <motion.div initial={{ opacity: 0, y: 14 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.08 }}>
        <Card className="premium-card" style={{ marginTop: 12 }} title={<span style={{ color: "#0f172a" }}>报告索引</span>}>
          <List
            bordered
            dataSource={items}
            locale={{ emptyText: loading ? "加载中..." : <Empty description="暂无报告" /> }}
            renderItem={(item) => (
              <List.Item
                actions={[
                  <Button key="detail" size="small" onClick={() => loadReportDetail(item.report_id)}>详情</Button>,
                  <Button key="versions" size="small" onClick={() => loadReportVersions(item.report_id)}>版本</Button>,
                  <Button key="view" size="small" onClick={() => exportReport(item.report_id)}>查看导出</Button>,
                ]}
              >
                <Space>
                  <Tag color="blue">{item.stock_code}</Tag>
                  <Tag>{item.report_type}</Tag>
                  <Text style={{ color: "#64748b" }}>{item.created_at}</Text>
                </Space>
              </List.Item>
            )}
          />
        </Card>
      </motion.div>

      {selectedMarkdown ? (
        <motion.div initial={{ opacity: 0, y: 14 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.14 }}>
          <Card className="premium-card" style={{ marginTop: 12 }} title={<span style={{ color: "#0f172a" }}>Markdown 预览</span>}>
            <pre style={{ whiteSpace: "pre-wrap", color: "#0f172a", margin: 0 }}>{selectedMarkdown}</pre>
          </Card>
        </motion.div>
      ) : null}

      {selectedVersions ? (
        <motion.div initial={{ opacity: 0, y: 14 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.18 }}>
          <Card className="premium-card" style={{ marginTop: 12 }} title={<span style={{ color: "#0f172a" }}>版本历史</span>}>
            <pre style={{ whiteSpace: "pre-wrap", color: "#0f172a", margin: 0 }}>{selectedVersions}</pre>
          </Card>
        </motion.div>
      ) : null}

      {selectedRaw ? (
        <motion.div initial={{ opacity: 0, y: 14 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.22 }}>
          <Card className="premium-card" style={{ marginTop: 12 }} title={<span style={{ color: "#0f172a" }}>接口原始返回</span>}>
            <pre style={{ whiteSpace: "pre-wrap", color: "#0f172a", margin: 0 }}>{selectedRaw}</pre>
          </Card>
        </motion.div>
      ) : null}

      {activeReportId ? <Text style={{ display: "block", marginTop: 12, color: "#94a3b8" }}>当前报告 ID: {activeReportId}</Text> : null}
    </main>
  );
}

