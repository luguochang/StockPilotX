"use client";

import { useEffect, useMemo } from "react";
import { motion } from "framer-motion";
import ReactECharts from "echarts-for-react";
import { Alert, Button, Card, Col, Collapse, Empty, List, Progress, Row, Select, Skeleton, Space, Statistic, Switch, Table, Tag, Typography, message } from "antd";
import MediaCarousel from "../components/MediaCarousel";
import StockSelectorModal from "../components/StockSelectorModal";
import ContextRibbon from "../components/ContextRibbon";
import ModuleGuideBanner from "../components/ModuleGuideBanner";
import ModuleWorkflow from "../components/ModuleWorkflow";
import { useBackendHealth } from "../hooks/useBackendHealth";
import { usePredict } from "../hooks/usePredict";

const { Title, Text, Paragraph } = Typography;

function riskColor(tier: string): string {
  if (tier === "high") return "red";
  if (tier === "medium") return "gold";
  return "green";
}

function signalColor(signal: string): string {
  if (signal === "strong_buy" || signal === "buy") return "green";
  if (signal === "strong_reduce" || signal === "reduce") return "red";
  return "blue";
}

function signalLabel(signal: string): string {
  const mapping: Record<string, string> = {
    strong_buy: "强增配",
    buy: "增配",
    hold: "持有",
    reduce: "减配",
    strong_reduce: "强减配",
  };
  return mapping[String(signal ?? "")] ?? String(signal ?? "-");
}

function gateColor(status: string): string {
  if (status === "pass") return "green";
  if (status === "watch") return "gold";
  return "red";
}

function gateLabel(status: string): string {
  if (status === "pass") return "通过";
  if (status === "watch") return "观察";
  return "降级";
}

function dimLabel(dim: string): string {
  const mapping: Record<string, string> = {
    history_quality: "历史质量",
    evidence_quality: "证据质量",
    metric_quality: "评测质量",
  };
  return mapping[dim] ?? dim;
}

export default function PredictPage() {
  const [messageApi, contextHolder] = message.useMessage();
  const backendStatus = useBackendHealth();
  const {
    selectedCodes,
    setSelectedCodes,
    pools,
    selectedPoolId,
    setSelectedPoolId,
    poolStocks,
    manualMode,
    setManualMode,
    loading,
    initialLoading,
    error,
    run,
    evalData,
    explain,
    explainLoading,
    explainError,
    ctxStock,
    ctxLastRunAt,
    loadPools,
    runPredict
  } = usePredict();

  useEffect(() => {
    if (!error) return;
    messageApi.error(error);
  }, [error, messageApi]);

  const heroSlides = [
    { src: "/assets/images/nyse-floor-2014.jpg", alt: "Trading floor visual", caption: "实时交易数据层" },
    { src: "/assets/images/nyse-floor-1963.jpg", alt: "Historic trading floor visual", caption: "历史周期回溯" },
    { src: "/assets/images/nyse-floor-1930.png", alt: "Long cycle visual", caption: "长期风险观察" }
  ];

  const stockNameMap = useMemo(() => {
    const out: Record<string, string> = {};
    for (const row of poolStocks) {
      const code = String(row.stock_code ?? "").trim().toUpperCase();
      const name = String(row.stock_name ?? "").trim();
      if (code && name) out[code] = name;
    }
    return out;
  }, [poolStocks]);

  const tableData = (run?.results ?? []).flatMap((item) =>
    item.horizons.map((h) => ({
      key: `${item.stock_code}-${h.horizon}`,
      stock_code: item.stock_code,
      stock_name: stockNameMap[String(item.stock_code ?? "").toUpperCase()] ?? "",
      horizon: h.horizon,
      expected_excess_return: h.expected_excess_return,
      up_probability: h.up_probability,
      risk_tier: h.risk_tier,
      signal: h.signal,
      rationale: h.rationale ?? "",
      history_data_mode: item.source?.history_data_mode ?? "unknown",
      history_sample_size: item.source?.history_sample_size ?? 0,
      history_source_id: item.source?.history_source_id ?? "unknown",
      data_quality: item.data_quality ?? "unknown",
      degrade_reasons: (item.degrade_reasons ?? []).join(",")
    }))
  );

  const barOption = useMemo(() => {
    const rows = run?.results ?? [];
    return {
      grid: { top: 30, left: 44, right: 12, bottom: 24 },
      tooltip: { trigger: "axis" },
      legend: { data: ["5日", "20日"], textStyle: { color: "#334155" } },
      xAxis: { type: "category", data: rows.map((r) => r.stock_code), axisLabel: { color: "#475569" }, axisLine: { lineStyle: { color: "rgba(100,116,139,0.3)" } } },
      yAxis: { type: "value", axisLabel: { color: "#475569" }, splitLine: { lineStyle: { color: "rgba(100,116,139,0.16)" } } },
      series: [
        { name: "5日", type: "bar", itemStyle: { color: "#2563eb" }, data: rows.map((r) => Number(((r.horizons.find((x) => x.horizon === "5d")?.expected_excess_return ?? 0) * 100).toFixed(2))) },
        { name: "20日", type: "bar", itemStyle: { color: "#0ea5e9" }, data: rows.map((r) => Number(((r.horizons.find((x) => x.horizon === "20d")?.expected_excess_return ?? 0) * 100).toFixed(2))) }
      ]
    };
  }, [run]);

  const workflowStep = !manualMode && !selectedPoolId ? 0 : !run ? 1 : 2;

  const metrics =
    run?.metric_mode === "live"
      ? run?.metrics_live ?? {}
      : run?.metric_mode === "backtest_proxy"
        ? (run?.metrics_backtest && Object.keys(run.metrics_backtest).length
            ? run.metrics_backtest
            : run?.metrics_simulated ?? evalData?.metrics ?? {})
        : run?.metrics_simulated ?? evalData?.metrics ?? {};

  const metricMode = run?.metric_mode ?? evalData?.metric_mode ?? "simulated";
  const qualitySummary = run?.source_coverage ?? {};
  const evalProvenance = run?.eval_provenance ?? {
    coverage_rows: Number(evalData?.metrics?.coverage ?? 0),
    evaluated_stocks: Number(evalData?.evaluated_stocks ?? 0),
    skipped_stocks: evalData?.skipped_stocks ?? [],
    history_modes: evalData?.history_modes ?? {},
    fallback_reason: evalData?.fallback_reason ?? "",
    run_data_quality: run?.data_quality ?? "unknown",
  };

  const skippedPreview = (evalProvenance.skipped_stocks ?? [])
    .slice(0, 3)
    .map((row) => `${String(row.stock_code ?? "unknown")}:${String(row.reason ?? "unknown")}`)
    .join("; ");

  const qualityGate = run?.quality_gate;
  const gateStatus = String(qualityGate?.overall_status ?? (run?.data_quality === "real" ? "watch" : "degraded"));
  const gateMessage = String(
    qualityGate?.user_message
    ?? (run?.data_quality === "real" ? "当前可用于研究参考。" : "当前存在降级项，请先补齐缺失数据。")
  );
  const gateReasonDetails = Array.isArray(qualityGate?.reason_details) ? qualityGate?.reason_details ?? [] : [];

  const featuredItem = run?.results?.[0];
  const featuredHorizon = featuredItem?.horizons?.find((x) => String(x.horizon).toLowerCase() === "20d") ?? featuredItem?.horizons?.[0];
  const featuredName = featuredItem ? stockNameMap[String(featuredItem.stock_code ?? "").toUpperCase()] ?? "" : "";

  const signalSummary = useMemo(() => {
    const counter: Record<string, number> = {};
    for (const row of tableData) {
      const key = String(row.signal || "hold");
      counter[key] = (counter[key] ?? 0) + 1;
    }
    const sorted = Object.entries(counter).sort((a, b) => b[1] - a[1]);
    return {
      topSignal: sorted[0]?.[0] ?? "hold",
      topCount: sorted[0]?.[1] ?? 0,
      total: tableData.length,
    };
  }, [tableData]);

  return (
    <main className="container">
      {contextHolder}
      <motion.section initial={{ opacity: 0, y: 12 }} whileInView={{ opacity: 1, y: 0 }} viewport={{ once: true, amount: 0.2 }} transition={{ duration: 0.45 }}>
        <MediaCarousel items={heroSlides} />
      </motion.section>

      <ContextRibbon
        poolName={pools.find((p) => p.pool_id === selectedPoolId)?.pool_name ?? ""}
        stockCode={manualMode ? (selectedCodes[0] ?? "") : ctxStock}
        lastRunAt={ctxLastRunAt}
        backendStatus={backendStatus}
      />

      <ModuleGuideBanner
        moduleKey="predict"
        title="预测研究台怎么用"
        steps={["优先选择关注池", "点击运行研究信号", "先看结论与门禁，再按需展开明细"]}
      />

      <ModuleWorkflow
        title="预测研究流程"
        items={["选择输入范围", "运行研究", "查看结论与门禁"]}
        current={workflowStep}
        hint="默认聚焦可行动信息，技术明细按需展开"
      />

      <motion.div initial={{ opacity: 0, y: 14 }} animate={{ opacity: 1, y: 0 }}>
        <Card className="premium-card" style={{ background: "linear-gradient(132deg, rgba(255,255,255,0.96), rgba(246,249,252,0.94))" }}>
          <Space direction="vertical" style={{ width: "100%" }} size={8}>
            <Tag color="processing" style={{ width: "fit-content" }}>Predict Research Workspace</Tag>
            <Title level={2} style={{ margin: 0, color: "#0f172a" }}>预测研究台</Title>
            <Paragraph style={{ margin: 0, color: "#475569", maxWidth: 780 }}>
              量化引擎先快速给出信号，再由解释层补充可读摘要与风险说明。该页面用于研究决策参考，不构成投资建议。
            </Paragraph>
            <Space.Compact block>
              <Select
                value={selectedPoolId || undefined}
                onChange={(v) => setSelectedPoolId(String(v ?? ""))}
                options={pools.map((p) => ({ label: `${p.pool_name} (${p.stock_count})`, value: p.pool_id }))}
                placeholder="选择关注池"
                style={{ minWidth: 280 }}
                disabled={manualMode}
                showSearch
                optionFilterProp="label"
                filterOption={(input, option) => String(option?.label ?? "").toLowerCase().includes(input.toLowerCase())}
              />
              <Button onClick={loadPools}>刷新关注池</Button>
              <Space style={{ marginLeft: 12 }}>
                <Text style={{ color: "#475569" }}>手动选股</Text>
                <Switch checked={manualMode} onChange={setManualMode} />
              </Space>
            </Space.Compact>

            {!manualMode ? (
              <Card size="small" style={{ background: "#f8fafc", borderColor: "#cbd5e1" }}>
                <Space direction="vertical" style={{ width: "100%" }} size={6}>
                  <Text strong style={{ color: "#0f172a" }}>
                    当前关注池成分预览（前8）
                  </Text>
                  {poolStocks.length === 0 ? (
                    <Text style={{ color: "#64748b" }}>当前关注池暂无可展示标的，请先在关注池页面补充。</Text>
                  ) : (
                    <List
                      size="small"
                      dataSource={poolStocks.slice(0, 8)}
                      renderItem={(row) => (
                        <List.Item style={{ padding: "6px 0" }}>
                          <Space wrap>
                            <Tag color="geekblue">{row.stock_code}</Tag>
                            {row.stock_name ? <Text>{row.stock_name}</Text> : null}
                            {row.exchange ? <Tag>{row.exchange}</Tag> : null}
                            {row.industry_l1 ? <Tag color="purple">{row.industry_l1}</Tag> : null}
                          </Space>
                        </List.Item>
                      )}
                    />
                  )}
                </Space>
              </Card>
            ) : null}

            {manualMode ? (
              <Collapse
                size="small"
                items={[
                  {
                    key: "manual",
                    label: "手动模式：选择研究标的",
                    children: (
                      <StockSelectorModal
                        value={selectedCodes}
                        onChange={(next) => setSelectedCodes(Array.isArray(next) ? next : (next ? [next] : []))}
                        multiple
                        title="选择预测标的"
                        placeholder="请选择一只或多只股票"
                      />
                    )
                  }
                ]}
              />
            ) : null}

            <Button type="primary" size="large" loading={loading} onClick={runPredict}>运行研究信号</Button>
          </Space>
        </Card>
      </motion.div>

      {initialLoading ? (
        <Card className="premium-card" style={{ marginTop: 12 }}>
          <Skeleton active paragraph={{ rows: 6 }} />
        </Card>
      ) : null}

      {!run && !error && !initialLoading ? (
        <Card className="premium-card" style={{ marginTop: 12 }}>
          <Empty description="尚未运行研究，选择关注池后点击运行研究信号" />
        </Card>
      ) : null}

      {run ? (
        <motion.div initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.08 }}>
          <Row gutter={[14, 14]} style={{ marginTop: 12 }}>
            <Col xs={24} lg={8}>
              <Card className="premium-card" title={<span style={{ color: "#0f172a" }}>研究结论</span>}>
                <Space direction="vertical" style={{ width: "100%" }}>
                  <Tag color={signalColor(signalSummary.topSignal)}>
                    主导信号: {signalLabel(signalSummary.topSignal)} ({signalSummary.topCount}/{signalSummary.total})
                  </Tag>
                  {featuredItem && featuredHorizon ? (
                    <>
                      <Text style={{ color: "#334155" }}>
                        代表标的: <Text strong>{featuredItem.stock_code}{featuredName ? ` ${featuredName}` : ""}</Text> · {featuredHorizon.horizon}
                      </Text>
                      <Text style={{ color: "#334155" }}>
                        预期超额收益: {(Number(featuredHorizon.expected_excess_return ?? 0) * 100).toFixed(2)}%
                      </Text>
                      <Text style={{ color: "#334155" }}>
                        上涨概率: {(Number(featuredHorizon.up_probability ?? 0) * 100).toFixed(2)}%
                      </Text>
                      <Tag color={riskColor(String(featuredHorizon.risk_tier ?? "medium"))}>风险: {String(featuredHorizon.risk_tier ?? "medium")}</Tag>
                    </>
                  ) : (
                    <Text style={{ color: "#64748b" }}>暂无代表信号。</Text>
                  )}

                  <div style={{ marginTop: 4 }}>
                    <Text strong style={{ color: "#0f172a" }}>解释摘要</Text>
                    <div style={{ marginTop: 4 }}>
                      {explainLoading ? (
                        <Text style={{ color: "#64748b" }}>解释生成中...</Text>
                      ) : explain ? (
                        <Text style={{ color: "#334155" }}>{explain.summary}</Text>
                      ) : (
                        <Text style={{ color: "#64748b" }}>暂无解释结果。</Text>
                      )}
                    </div>
                  </div>
                </Space>
              </Card>
            </Col>

            <Col xs={24} lg={8}>
              <Card className="premium-card" title={<span style={{ color: "#0f172a" }}>质量门禁</span>}>
                <Space direction="vertical" style={{ width: "100%" }}>
                  <Tag color={gateColor(gateStatus)}>状态: {gateLabel(gateStatus)}</Tag>
                  <Text style={{ color: "#334155" }}>{gateMessage}</Text>

                  {Object.entries(qualityGate?.dimensions ?? {}).map(([key, val]) => (
                    <Space key={key} style={{ width: "100%", justifyContent: "space-between" }}>
                      <Text style={{ color: "#475569" }}>{dimLabel(key)}</Text>
                      <Tag color={gateColor(String(val?.status ?? "pass"))}>
                        {gateLabel(String(val?.status ?? "pass"))} · {Number(val?.reason_count ?? 0)}
                      </Tag>
                    </Space>
                  ))}

                  {gateReasonDetails.slice(0, 2).map((row, idx) => (
                    <Text key={`reason-${idx}`} style={{ color: "#64748b" }}>
                      {String(row?.title ?? "")}：{String(row?.impact ?? "")}
                    </Text>
                  ))}
                </Space>
              </Card>
            </Col>

            <Col xs={24} lg={8}>
              <Card className="premium-card" title={<span style={{ color: "#0f172a" }}>运行元信息</span>}>
                <Space direction="vertical" style={{ width: "100%" }}>
                  <Statistic title="Run ID" value={run.run_id.slice(0, 8)} valueStyle={{ color: "#0f172a" }} />
                  <Statistic title="Trace ID" value={run.trace_id.slice(0, 8)} valueStyle={{ color: "#64748b" }} />
                  <Tag color="blue">Metric Mode: {metricMode}</Tag>
                  <Text style={{ color: "#475569" }}>引擎: {run.engine_profile?.prediction_engine ?? "quant_rule_v1"}</Text>
                  <Text style={{ color: "#475569" }}>打分是否用LLM: {run.engine_profile?.llm_used_in_scoring ? "是" : "否"}</Text>
                  <Text style={{ color: "#475569" }}>延迟模式: {run.engine_profile?.latency_mode ?? "fast_local_compute"}</Text>
                  {explain ? (
                    <Tag color={explain.llm_used ? "green" : "gold"}>
                      解释层: {explain.llm_used ? `LLM(${explain.provider || "provider"})` : "模板降级"}
                    </Tag>
                  ) : null}
                </Space>
              </Card>
            </Col>

            {explainError ? (
              <Col span={24}>
                <Alert type="info" showIcon message={`解释层降级: ${explainError}`} />
              </Col>
            ) : null}

            <Col span={24}>
              <Collapse
                defaultActiveKey={["summary"]}
                items={[
                  {
                    key: "summary",
                    label: "评测与覆盖（核心）",
                    children: (
                      <Space direction="vertical" style={{ width: "100%" }}>
                        <Text style={{ color: "#475569" }}>IC: {Number(metrics?.ic ?? 0).toFixed(4)}</Text>
                        <Progress percent={Math.max(0, Math.min(100, Number(metrics?.hit_rate ?? 0) * 100))} strokeColor="#2563eb" />
                        <Text style={{ color: "#475569" }}>Top-Bottom Spread: {Number(metrics?.top_bottom_spread ?? 0).toFixed(4)}</Text>
                        <Text style={{ color: "#475569" }}>Max Drawdown: {Number(metrics?.max_drawdown ?? 0).toFixed(4)}</Text>
                        <Text style={{ color: "#475569" }}>Coverage Rows: {Number(evalProvenance.coverage_rows ?? 0)}</Text>
                        <Text style={{ color: "#475569" }}>Evaluated Stocks: {Number(evalProvenance.evaluated_stocks ?? 0)}</Text>
                        <Text style={{ color: "#475569" }}>
                          Real History Ratio: {(Number(qualitySummary.real_history_ratio ?? 0) * 100).toFixed(1)}%
                        </Text>
                        {evalProvenance.fallback_reason ? <Text style={{ color: "#b45309" }}>Fallback: {String(evalProvenance.fallback_reason)}</Text> : null}
                        {skippedPreview ? <Text style={{ color: "#94a3b8" }}>Skipped: {skippedPreview}</Text> : null}
                        {run?.metrics_note ? <Text style={{ color: "#94a3b8" }}>Note: {run.metrics_note}</Text> : null}
                      </Space>
                    ),
                  },
                  {
                    key: "chart",
                    label: "收益对比图（高级）",
                    children: <ReactECharts option={barOption} style={{ height: 286 }} />,
                  },
                  {
                    key: "segments",
                    label: "分层聚合（高级）",
                    children: (
                      <Space direction="vertical" style={{ width: "100%" }}>
                        {(Object.entries(run.segment_metrics ?? {}) as Array<[string, Array<{ segment: string; count: number; avg_expected_excess_return: number; avg_up_probability: number }>]>).map(([k, rows]) => (
                          <div key={k}>
                            <Text strong style={{ color: "#0f172a" }}>{k}</Text>
                            <Table
                              size="small"
                              style={{ marginTop: 8 }}
                              dataSource={(rows ?? []).map((r) => ({ ...r, key: `${k}-${r.segment}` }))}
                              pagination={{ pageSize: 10, simple: true }}
                              columns={[
                                { title: "分组", dataIndex: "segment", key: "segment" },
                                { title: "数量", dataIndex: "count", key: "count" },
                                { title: "平均超额收益", dataIndex: "avg_expected_excess_return", key: "avg_expected_excess_return", render: (v: number) => `${(v * 100).toFixed(2)}%` },
                                { title: "平均上涨概率", dataIndex: "avg_up_probability", key: "avg_up_probability", render: (v: number) => `${(v * 100).toFixed(2)}%` }
                              ]}
                            />
                          </div>
                        ))}
                      </Space>
                    ),
                  },
                  {
                    key: "details",
                    label: "预测明细（高级）",
                    children: (
                      <Table
                        dataSource={tableData}
                        pagination={{ pageSize: 20, showSizeChanger: true, pageSizeOptions: ['10', '20', '50', '100'] }}
                        scroll={{ x: 980, y: 500 }}
                        columns={[
                          {
                            title: "股票",
                            dataIndex: "stock_code",
                            key: "stock_code",
                            render: (_: unknown, row: { stock_code: string; stock_name?: string }) =>
                              row.stock_name ? `${row.stock_code} ${row.stock_name}` : row.stock_code,
                          },
                          { title: "周期", dataIndex: "horizon", key: "horizon" },
                          { title: "超额收益", dataIndex: "expected_excess_return", key: "expected_excess_return", render: (v: number) => `${(v * 100).toFixed(2)}%` },
                          { title: "上涨概率", dataIndex: "up_probability", key: "up_probability", render: (v: number) => `${(v * 100).toFixed(2)}%` },
                          { title: "风险", dataIndex: "risk_tier", key: "risk_tier", render: (v: string) => <Tag color={riskColor(v)}>{v}</Tag> },
                          { title: "信号", dataIndex: "signal", key: "signal", render: (v: string) => <Tag color={signalColor(v)}>{signalLabel(v)}</Tag> },
                          { title: "历史数据", dataIndex: "history_data_mode", key: "history_data_mode" },
                          { title: "样本数", dataIndex: "history_sample_size", key: "history_sample_size" },
                          { title: "历史源", dataIndex: "history_source_id", key: "history_source_id" },
                          { title: "质量", dataIndex: "data_quality", key: "data_quality" },
                          { title: "降级原因", dataIndex: "degrade_reasons", key: "degrade_reasons", width: 220 },
                          { title: "解释", dataIndex: "rationale", key: "rationale", width: 320 }
                        ]}
                      />
                    ),
                  },
                ]}
              />
            </Col>
          </Row>
        </motion.div>
      ) : null}
    </main>
  );
}
