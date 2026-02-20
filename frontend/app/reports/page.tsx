"use client";

import { useEffect, useRef, useState } from "react";
import { motion } from "framer-motion";
import { Alert, Button, Card, Col, Collapse, Empty, Input, List, Progress, Row, Select, Space, Tabs, Tag, Typography } from "antd";
import MediaCarousel from "../components/MediaCarousel";
import StockSelectorModal from "../components/StockSelectorModal";
import { fetchJson } from "../lib/api";

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

type ReportTask = {
  task_id: string;
  status: string;
  progress: number;
  current_stage: string;
  stage_message: string;
  result_level: "none" | "partial" | "full" | string;
  error_code?: string;
  error_message?: string;
  // Backend quality snapshot for task-level transparency while polling.
  data_pack_status?: "ready" | "partial" | "failed" | string;
  data_pack_missing?: string[];
  quality_gate_detail?: Record<string, unknown>;
};

type TaskResult = {
  task_id: string;
  status: string;
  result_level: "none" | "partial" | "full" | string;
  result: Record<string, unknown> | null;
};

type ReportModule = {
  module_id: string;
  title: string;
  content: string;
  evidence_refs?: string[];
  coverage?: {
    status?: string;
    data_points?: number;
  };
  confidence?: number;
  degrade_reason?: string[];
};

type FinalDecision = {
  signal?: string;
  confidence?: number;
  rationale?: string;
  invalidation_conditions?: string[];
  execution_plan?: string[];
};

type CommitteeNotes = {
  research_note?: string;
  risk_note?: string;
};

function statusColor(status: string): string {
  if (status === "ok" || status === "completed") return "success";
  if (status === "degraded" || status === "partial_ready" || status === "running" || status === "queued") return "warning";
  if (status === "failed" || status === "cancelled") return "error";
  return "default";
}

function parseObject(value: unknown): Record<string, unknown> {
  if (!value || typeof value !== "object") return {};
  return value as Record<string, unknown>;
}

function taskRunning(status: string): boolean {
  return ["queued", "running", "partial_ready", "cancelling"].includes(status);
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
  const [qualityGatePreview, setQualityGatePreview] = useState<Record<string, unknown>>({});
  const [dataPackPreview, setDataPackPreview] = useState<Record<string, unknown>>({});
  const [degradePreview, setDegradePreview] = useState<Record<string, unknown>>({});
  const [generationMode, setGenerationMode] = useState("");
  const [businessHealth, setBusinessHealth] = useState<BusinessDataHealth | null>(null);
  const [task, setTask] = useState<ReportTask | null>(null);
  const [selectedModules, setSelectedModules] = useState<ReportModule[]>([]);
  const [selectedDecision, setSelectedDecision] = useState<FinalDecision | null>(null);
  const [selectedCommittee, setSelectedCommittee] = useState<CommitteeNotes | null>(null);
  const [selectedMetrics, setSelectedMetrics] = useState<Record<string, unknown>>({});

  const pollTimerRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const qualityStatus = String(qualityGatePreview.status ?? "unknown");
  const qualityScore = Number(qualityGatePreview.score ?? 0);
  const qualityReasons = Array.isArray(qualityGatePreview.reasons) ? qualityGatePreview.reasons.map((item) => String(item)) : [];
  const reportsHealth = (businessHealth?.module_health ?? []).find((row) => String(row.module) === "reports");
  const taskDataPackMissing = Array.isArray(task?.data_pack_missing) ? task.data_pack_missing.map((item) => String(item)) : [];
  const taskQualityGate = parseObject(task?.quality_gate_detail);
  const taskQualityStatus = String(taskQualityGate.status ?? "unknown");
  const taskQualityScore = Number(taskQualityGate.score ?? 0);
  const taskQualityReasons = Array.isArray(taskQualityGate.reasons) ? taskQualityGate.reasons.map((item) => String(item)) : [];

  const heroSlides = [
    { src: "/assets/images/nyse-floor-1963.jpg", alt: "Archive report", caption: "档案回溯" },
    { src: "/assets/images/nyse-floor-2014.jpg", alt: "Modern report", caption: "研究输出" },
    { src: "/assets/images/nyse-floor-1930.png", alt: "Long cycle", caption: "长周期视角" },
  ];

  function stopPolling() {
    if (pollTimerRef.current) {
      clearInterval(pollTimerRef.current);
      pollTimerRef.current = null;
    }
  }

  async function loadBusinessHealth(stockCode: string) {
    const code = stockCode.trim().toUpperCase();
    if (!code) {
      setBusinessHealth(null);
      return;
    }
    try {
      const body = (await fetchJson(`/v1/business/data-health?stock_code=${encodeURIComponent(code)}&limit=200`)) as BusinessDataHealth;
      setBusinessHealth(body);
    } catch {
      // Keep this non-blocking so observability endpoint failures do not break core report UX.
      setBusinessHealth(null);
    }
  }

  async function load() {
    setLoading(true);
    setError("");
    try {
      const body = (await fetchJson("/v1/reports")) as ReportItem[];
      setItems(Array.isArray(body) ? body : []);
    } catch (e) {
      setError(e instanceof Error ? e.message : "加载报告列表失败");
      setItems([]);
    } finally {
      setLoading(false);
    }
  }

  function applyReportResult(result: Record<string, unknown>) {
    setSelectedMarkdown(String(result.markdown ?? ""));
    setSelectedRaw(JSON.stringify(result, null, 2));
    setQualityGatePreview(parseObject(result.quality_gate));
    setDataPackPreview(parseObject(result.report_data_pack_summary));
    setDegradePreview(parseObject(result.degrade));
    setGenerationMode(String(result.generation_mode ?? ""));
    setActiveReportId(String(result.report_id ?? ""));
    // Parse moduleized report payload for business-friendly rendering.
    const moduleRows = Array.isArray(result.report_modules) ? result.report_modules : [];
    const normalizedModules: ReportModule[] = moduleRows
      .filter((row): row is Record<string, unknown> => Boolean(row) && typeof row === "object")
      .map((row) => ({
        module_id: String(row.module_id ?? "").trim().toLowerCase(),
        title: String(row.title ?? row.module_id ?? "模块"),
        content: String(row.content ?? ""),
        evidence_refs: Array.isArray(row.evidence_refs) ? row.evidence_refs.map((item) => String(item)) : [],
        coverage: typeof row.coverage === "object" && row.coverage ? {
          status: String((row.coverage as Record<string, unknown>).status ?? ""),
          data_points: Number((row.coverage as Record<string, unknown>).data_points ?? 0),
        } : undefined,
        confidence: Number(row.confidence ?? 0),
        degrade_reason: Array.isArray(row.degrade_reason) ? row.degrade_reason.map((item) => String(item)) : [],
      }))
      .filter((row) => Boolean(row.module_id));
    setSelectedModules(normalizedModules);
    const decision = parseObject(result.final_decision);
    setSelectedDecision(Object.keys(decision).length ? {
      signal: String(decision.signal ?? ""),
      confidence: Number(decision.confidence ?? 0),
      rationale: String(decision.rationale ?? ""),
      invalidation_conditions: Array.isArray(decision.invalidation_conditions)
        ? decision.invalidation_conditions.map((item) => String(item))
        : [],
      execution_plan: Array.isArray(decision.execution_plan)
        ? decision.execution_plan.map((item) => String(item))
        : [],
    } : null);
    const committee = parseObject(result.committee);
    setSelectedCommittee(Object.keys(committee).length ? {
      research_note: String(committee.research_note ?? ""),
      risk_note: String(committee.risk_note ?? ""),
    } : null);
    setSelectedMetrics(parseObject(result.metric_snapshot));
  }

  async function syncTaskResult(taskId: string) {
    const body = (await fetchJson(`/v1/report/tasks/${taskId}/result`)) as TaskResult;
    if (body.result && typeof body.result === "object") {
      applyReportResult(body.result);
      const code = String((body.result as Record<string, unknown>).stock_code ?? "").trim().toUpperCase();
      if (code) {
        setGenerateStockCode(code);
        await loadBusinessHealth(code);
      }
      if (body.result_level === "full") {
        await load();
      }
    }
  }

  async function pollTask(taskId: string) {
    try {
      const body = (await fetchJson(`/v1/report/tasks/${taskId}`)) as ReportTask;
      setTask(body);
      if (body.result_level === "partial" || body.result_level === "full" || body.status === "partial_ready" || body.status === "completed") {
        await syncTaskResult(taskId);
      }
      if (!taskRunning(body.status)) {
        stopPolling();
      }
    } catch (e) {
      stopPolling();
      setError(e instanceof Error ? e.message : "轮询任务状态失败");
    }
  }

  function startPolling(taskId: string) {
    stopPolling();
    void pollTask(taskId);
    pollTimerRef.current = setInterval(() => {
      void pollTask(taskId);
    }, 1500);
  }

  async function generateReport() {
    setError("");
    setLoading(true);
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
      const created = (await fetchJson("/v1/report/tasks", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      })) as ReportTask;
      setTask(created);
      startPolling(created.task_id);
    } catch (e) {
      setError(e instanceof Error ? e.message : "创建报告任务失败");
    } finally {
      setLoading(false);
    }
  }

  async function cancelTask() {
    if (!task?.task_id) return;
    try {
      const body = (await fetchJson(`/v1/report/tasks/${task.task_id}/cancel`, { method: "POST" })) as ReportTask;
      setTask(body);
      if (!taskRunning(body.status)) stopPolling();
    } catch (e) {
      setError(e instanceof Error ? e.message : "取消任务失败");
    }
  }

  async function exportReport(reportId: string) {
    setLoading(true);
    setError("");
    try {
      const body = (await fetchJson(`/v1/reports/${reportId}/export`, { method: "POST" })) as { markdown?: string };
      setSelectedMarkdown(String(body.markdown ?? ""));
      setActiveReportId(reportId);
    } catch (e) {
      setError(e instanceof Error ? e.message : "导出报告失败");
      setSelectedMarkdown("");
    } finally {
      setLoading(false);
    }
  }

  async function loadReportDetail(reportId: string) {
    setLoading(true);
    setError("");
    try {
      const body = (await fetchJson(`/v1/report/${reportId}`)) as Record<string, unknown>;
      applyReportResult(body);
      setActiveReportId(reportId);
      const detailCode = String(body.stock_code ?? "").trim().toUpperCase();
      if (detailCode) {
        setGenerateStockCode(detailCode);
        await loadBusinessHealth(detailCode);
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : "加载报告详情失败");
    } finally {
      setLoading(false);
    }
  }

  async function loadReportVersions(reportId: string) {
    setLoading(true);
    setError("");
    try {
      const body = await fetchJson(`/v1/reports/${reportId}/versions`);
      setSelectedVersions(JSON.stringify(body, null, 2));
      setActiveReportId(reportId);
    } catch (e) {
      setError(e instanceof Error ? e.message : "加载版本历史失败");
      setSelectedVersions("");
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    void load();
    return () => {
      stopPolling();
    };
  }, []);

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
              主流程仅保留两个输入项：股票与报告类型。高级参数可选展开，默认不打扰普通用户。
            </Paragraph>
            <StockSelectorModal value={generateStockCode} onChange={(next) => setGenerateStockCode(Array.isArray(next) ? (next[0] ?? "") : next)} title="选择报告标的" placeholder="请选择股票" />
            <Space.Compact block>
              <Select value={generateType} onChange={(v) => setGenerateType(v)} options={[{ label: "事实报告", value: "fact" }, { label: "研究报告", value: "research" }]} style={{ minWidth: 180 }} />
              <Button type="primary" loading={loading || taskRunning(task?.status ?? "")} onClick={generateReport}>生成报告</Button>
              <Button loading={loading} onClick={load}>刷新列表</Button>
              {task && taskRunning(task.status) ? <Button danger onClick={cancelTask}>取消任务</Button> : null}
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

      {task ? (
        <Card className="premium-card" style={{ marginTop: 12 }} title="生成任务进度">
          <Space direction="vertical" style={{ width: "100%" }}>
            <Text>任务ID: <Text code>{task.task_id}</Text></Text>
            <Text>状态: <Tag color={statusColor(task.status)}>{task.status}</Tag></Text>
            <Text>阶段: {task.current_stage || "-"}</Text>
            <Text type="secondary">{task.stage_message || "等待中"}</Text>
            <Text>
              数据包状态: <Tag color={statusColor(String(task.data_pack_status ?? "unknown"))}>{String(task.data_pack_status ?? "unknown")}</Tag>
            </Text>
            <Text>
              任务质量: <Tag color={statusColor(taskQualityStatus)}>{taskQualityStatus}</Tag>
            </Text>
            {taskQualityScore > 0 ? <Text type="secondary">任务质量得分: {taskQualityScore.toFixed(2)}</Text> : null}
            {taskQualityReasons.length ? <Text type="secondary">任务质量原因: {taskQualityReasons.join("；")}</Text> : null}
            <Progress percent={Math.max(0, Math.min(100, Number((Number(task.progress || 0) * 100).toFixed(1))))} strokeColor="#2563eb" />
            {taskDataPackMissing.length ? <Alert type="warning" showIcon message={`数据缺口: ${taskDataPackMissing.join("；")}`} /> : null}
            {task.error_message ? <Alert type="error" showIcon message={task.error_message} /> : null}
            {task.result_level === "partial" ? <Alert type="warning" showIcon message="已返回最小可用结果，系统仍在补全完整报告。" /> : null}
          </Space>
        </Card>
      ) : null}

      {error ? <Alert style={{ marginTop: 12 }} type="error" showIcon message={error} /> : null}

      {(generationMode || Object.keys(qualityGatePreview).length || Object.keys(dataPackPreview).length || businessHealth) ? (
        <Row gutter={[12, 12]} style={{ marginTop: 12 }}>
          <Col xs={24} lg={8}>
            <Card className="premium-card" title={<span style={{ color: "#0f172a" }}>报告质量闸门</span>}>
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
                <Text style={{ color: "#475569" }}>历史样本: {String(dataPackPreview.history_sample_size ?? 0)}</Text>
                <Text style={{ color: "#475569" }}>预测质量: {String(dataPackPreview.predict_quality ?? "unknown")}</Text>
                <Text style={{ color: "#475569" }}>情报信号: {String(dataPackPreview.intel_signal ?? "unknown")}</Text>
                <Text style={{ color: "#475569" }}>情报置信度: {Number(dataPackPreview.intel_confidence ?? 0).toFixed(2)}</Text>
                <Text style={{ color: "#64748b" }}>
                  新闻/研报/宏观: {String(dataPackPreview.news_count ?? 0)} / {String(dataPackPreview.research_count ?? 0)} / {String(dataPackPreview.macro_count ?? 0)}
                </Text>
              </Space>
            </Card>
          </Col>
          <Col xs={24} lg={8}>
            <Card className="premium-card" title={<span style={{ color: "#0f172a" }}>降级信息</span>}>
              {Object.keys(degradePreview).length === 0 ? (
                <Alert type="info" showIcon message="当前无降级信息" />
              ) : (
                <Space direction="vertical" size={6} style={{ width: "100%" }}>
                  <Text style={{ color: "#475569" }}>active: <Tag color={Boolean(degradePreview.active) ? "warning" : "success"}>{String(degradePreview.active)}</Tag></Text>
                  <Text style={{ color: "#475569" }}>code: {String(degradePreview.code ?? "") || "none"}</Text>
                  <Text style={{ color: "#64748b" }}>message: {String(degradePreview.user_message ?? "") || "none"}</Text>
                </Space>
              )}
            </Card>
          </Col>
        </Row>
      ) : null}

      {(selectedDecision || selectedCommittee || selectedModules.length > 0 || Object.keys(selectedMetrics).length > 0) ? (
        <Row gutter={[12, 12]} style={{ marginTop: 12 }}>
          <Col xs={24} lg={8}>
            <Card className="premium-card" title={<span style={{ color: "#0f172a" }}>最终决策与委员会</span>}>
              <Space direction="vertical" size={8} style={{ width: "100%" }}>
                {selectedDecision ? (
                  <>
                    <Text>信号: <Tag color={statusColor(String(selectedDecision.signal ?? "hold"))}>{String(selectedDecision.signal ?? "hold")}</Tag></Text>
                    <Text>置信度: {Number(selectedDecision.confidence ?? 0).toFixed(2)}</Text>
                    <Text type="secondary">决策理由: {selectedDecision.rationale || "-"}</Text>
                    {Array.isArray(selectedDecision.invalidation_conditions) && selectedDecision.invalidation_conditions.length > 0 ? (
                      <Text type="secondary">失效条件: {selectedDecision.invalidation_conditions.join("；")}</Text>
                    ) : null}
                  </>
                ) : (
                  <Alert type="info" showIcon message="暂无最终决策信息" />
                )}
                {selectedCommittee ? (
                  <>
                    <Text type="secondary">研究汇总: {selectedCommittee.research_note || "-"}</Text>
                    <Text type="secondary">风险仲裁: {selectedCommittee.risk_note || "-"}</Text>
                  </>
                ) : null}
              </Space>
            </Card>
          </Col>
          <Col xs={24} lg={8}>
            <Card className="premium-card" title={<span style={{ color: "#0f172a" }}>模块化报告</span>}>
              {selectedModules.length === 0 ? (
                <Alert type="info" showIcon message="当前结果尚未返回模块化报告。" />
              ) : (
                <Tabs
                  size="small"
                  items={selectedModules.map((module) => ({
                    key: module.module_id,
                    label: module.title,
                    children: (
                      <Space direction="vertical" size={6} style={{ width: "100%" }}>
                        <Text type="secondary">{module.content || "该模块暂无内容。"}</Text>
                        <Text type="secondary">
                          覆盖状态: {String(module.coverage?.status ?? "unknown")} / 数据点: {Number(module.coverage?.data_points ?? 0)}
                        </Text>
                        <Text type="secondary">模块置信度: {Number(module.confidence ?? 0).toFixed(2)}</Text>
                        {Array.isArray(module.degrade_reason) && module.degrade_reason.length > 0 ? (
                          <Text type="secondary">降级原因: {module.degrade_reason.join("；")}</Text>
                        ) : null}
                      </Space>
                    ),
                  }))}
                />
              )}
            </Card>
          </Col>
          <Col xs={24} lg={8}>
            <Card className="premium-card" title={<span style={{ color: "#0f172a" }}>指标快照</span>}>
              {Object.keys(selectedMetrics).length === 0 ? (
                <Alert type="info" showIcon message="暂无指标快照" />
              ) : (
                <Space direction="vertical" size={4} style={{ width: "100%" }}>
                  {Object.entries(selectedMetrics)
                    .slice(0, 16)
                    .map(([key, value]) => (
                      <Text key={key} type="secondary">{key}: {String(value)}</Text>
                    ))}
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
                  <Button key="detail" size="small" onClick={() => void loadReportDetail(item.report_id)}>详情</Button>,
                  <Button key="versions" size="small" onClick={() => void loadReportVersions(item.report_id)}>版本</Button>,
                  <Button key="view" size="small" onClick={() => void exportReport(item.report_id)}>查看导出</Button>,
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
