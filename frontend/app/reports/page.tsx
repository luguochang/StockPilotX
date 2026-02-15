"use client";

import { useState } from "react";
import { motion } from "framer-motion";
import { Alert, Button, Card, Input, List, Select, Space, Tag, Typography } from "antd";
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

export default function ReportsPage() {
  const [items, setItems] = useState<ReportItem[]>([]);
  const [selectedMarkdown, setSelectedMarkdown] = useState("");
  const [selectedRaw, setSelectedRaw] = useState("");
  const [selectedVersions, setSelectedVersions] = useState("");
  const [generateUserId, setGenerateUserId] = useState("demo_user");
  const [generateStockCode, setGenerateStockCode] = useState("SH600000");
  const [generatePeriod, setGeneratePeriod] = useState("1y");
  const [generateType, setGenerateType] = useState<"fact" | "research">("research");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [activeReportId, setActiveReportId] = useState("");
  const heroSlides = [
    { src: "/assets/images/nyse-floor-1963.jpg", alt: "Archive trading report visual", caption: "档案回溯" },
    { src: "/assets/images/nyse-floor-2014.jpg", alt: "Modern report visual", caption: "研究输出" },
    { src: "/assets/images/nyse-floor-1930.png", alt: "Long cycle report visual", caption: "长期视角" }
  ];

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
      const r = await fetch(`${API_BASE}/v1/reports/${reportId}/export`, {
        method: "POST"
      });
      const body = await r.json();
      if (!r.ok) throw new Error(body?.detail ?? `HTTP ${r.status}`);
      setSelectedMarkdown(body.markdown ?? "");
      setActiveReportId(reportId);
    } catch (e) {
      setError(e instanceof Error ? e.message : "请求失败");
      setSelectedMarkdown("");
    } finally {
      setLoading(false);
    }
  }

  // 中文注释：补齐报告中心缺口，支持直接触发生成、查详情、查版本。
  async function generateReport() {
    setLoading(true);
    setError("");
    try {
      const payload = {
        user_id: generateUserId.trim(),
        stock_code: generateStockCode.trim().toUpperCase(),
        period: generatePeriod.trim(),
        report_type: generateType
      };
      const r = await fetch(`${API_BASE}/v1/report/generate`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
      });
      const body = await r.json();
      if (!r.ok) throw new Error(body?.detail ?? `HTTP ${r.status}`);
      setSelectedMarkdown(body.markdown ?? "");
      setSelectedRaw(JSON.stringify(body, null, 2));
      setActiveReportId(body.report_id ?? "");
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
      setSelectedMarkdown(body.markdown ?? "");
      setActiveReportId(reportId);
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

  // 中文注释：补齐“导出”工程化体验，支持本地下载与复制分享信息。
  async function downloadMarkdown() {
    if (!selectedMarkdown.trim()) return;
    const filename = `${(activeReportId || "report").replace(/[^\w-]/g, "_")}.md`;
    const blob = new Blob([selectedMarkdown], { type: "text/markdown;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = filename;
    a.click();
    URL.revokeObjectURL(url);
  }

  async function copyMarkdown() {
    if (!selectedMarkdown.trim()) return;
    try {
      await navigator.clipboard.writeText(selectedMarkdown);
    } catch {
      // ignore clipboard errors for unsupported env
    }
  }

  async function copyShareInfo() {
    const share = `report_id=${activeReportId || "unknown"}\napi=${API_BASE}/v1/report/${activeReportId || "{report_id}"}`;
    try {
      await navigator.clipboard.writeText(share);
    } catch {
      // ignore
    }
  }

  return (
    <main className="container">
      <motion.section
        initial={{ opacity: 0, y: 12 }}
        whileInView={{ opacity: 1, y: 0 }}
        viewport={{ once: true, amount: 0.2 }}
        transition={{ duration: 0.45 }}
      >
        <MediaCarousel items={heroSlides} />
      </motion.section>

      <motion.div initial={{ opacity: 0, y: 14 }} animate={{ opacity: 1, y: 0 }}>
        <Card className="premium-card" style={{ background: "linear-gradient(132deg, rgba(255,255,255,0.96), rgba(246,249,252,0.94))" }}>
          <Space direction="vertical" style={{ width: "100%" }}>
            <Tag color="processing" style={{ width: "fit-content" }}>Research Report Center</Tag>
            <Title level={2} style={{ margin: 0, color: "#0f172a" }}>研究报告中心</Title>
            <Paragraph style={{ margin: 0, color: "#475569", maxWidth: 760 }}>
              按用户权限读取历史报告索引，支持一键导出 Markdown 原文，便于复盘与对外共享。
            </Paragraph>
            <Button type="primary" loading={loading} onClick={load}>加载报告</Button>
            <StockSelectorModal
              value={generateStockCode}
              onChange={(next) => setGenerateStockCode(Array.isArray(next) ? (next[0] ?? "") : next)}
              title="选择报告标的"
              placeholder="请选择报告股票"
            />
            <Space.Compact block>
              <Input value={generateUserId} onChange={(e) => setGenerateUserId(e.target.value)} placeholder="user_id" />
              <Input value={generatePeriod} onChange={(e) => setGeneratePeriod(e.target.value)} placeholder="period" />
              <Select
                value={generateType}
                onChange={(v) => setGenerateType(v)}
                options={[
                  { label: "fact", value: "fact" },
                  { label: "research", value: "research" }
                ]}
                style={{ minWidth: 128 }}
              />
              <Button loading={loading} onClick={generateReport}>生成报告</Button>
            </Space.Compact>
          </Space>
        </Card>
      </motion.div>

      {error ? <Alert style={{ marginTop: 12 }} type="error" showIcon message={error} /> : null}

      <motion.div initial={{ opacity: 0, y: 14 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.08 }}>
        <Card className="premium-card" style={{ marginTop: 12 }} title={<span style={{ color: "#0f172a" }}>报告索引</span>}>
          <List
            bordered
            dataSource={items}
            locale={{ emptyText: "暂无报告" }}
            renderItem={(item) => (
              <List.Item
                actions={[
                  <Button key="detail" size="small" onClick={() => loadReportDetail(item.report_id)}>
                    详情
                  </Button>,
                  <Button key="versions" size="small" onClick={() => loadReportVersions(item.report_id)}>
                    版本
                  </Button>,
                  <Button key="view" size="small" onClick={() => exportReport(item.report_id)}>
                    查看导出
                  </Button>
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
            <Space style={{ marginBottom: 12 }}>
              <Button onClick={downloadMarkdown}>下载 .md</Button>
              <Button onClick={copyMarkdown}>复制 Markdown</Button>
              <Button onClick={copyShareInfo}>复制分享信息</Button>
            </Space>
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
    </main>
  );
}

