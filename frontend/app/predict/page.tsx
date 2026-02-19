"use client";

import { useMemo, useEffect } from "react";
import { motion } from "framer-motion";
import ReactECharts from "echarts-for-react";
import { Button, Card, Col, Collapse, Empty, Progress, Row, Select, Skeleton, Space, Statistic, Switch, Table, Tag, Typography, message } from "antd";
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

export default function PredictPage() {
  const [messageApi, contextHolder] = message.useMessage();
  const backendStatus = useBackendHealth();
  const {
    selectedCodes,
    setSelectedCodes,
    pools,
    selectedPoolId,
    setSelectedPoolId,
    manualMode,
    setManualMode,
    loading,
    initialLoading,
    error,
    run,
    evalData,
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

  const tableData = (run?.results ?? []).flatMap((item) =>
    item.horizons.map((h) => ({
      key: `${item.stock_code}-${h.horizon}`,
      stock_code: item.stock_code,
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
  const workflowStep = !manualMode && !selectedPoolId ? 0 : !run ? 1 : 2;
  // Metric source is now explicit: live/backtest_proxy/simulated should never be mixed.
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
    .join(";");

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
        title="策略预测驾驶舱怎么用"
        steps={["默认先选择关注池", "点击运行预测直接产出结果", "需要精细控制时再打开手动选股"]}
      />
      <ModuleWorkflow
        title="预测流程"
        items={["选择输入范围", "运行预测", "查看分层与明细"]}
        current={workflowStep}
        hint="默认关注池模式最简，手动模式仅用于特定标的精调"
      />

      <motion.div initial={{ opacity: 0, y: 14 }} animate={{ opacity: 1, y: 0 }}>
        <Card className="premium-card" style={{ background: "linear-gradient(132deg, rgba(255,255,255,0.96), rgba(246,249,252,0.94))" }}>
          <Space direction="vertical" style={{ width: "100%" }} size={8}>
            <Tag color="processing" style={{ width: "fit-content" }}>Predictive Intelligence Console</Tag>
            <Title level={2} style={{ margin: 0, color: "#0f172a" }}>策略预测驾驶舱</Title>
            <Paragraph style={{ margin: 0, color: "#475569", maxWidth: 760 }}>
              默认使用关注池运行策略，减少用户配置负担。手动选股仅在需要时展开。
            </Paragraph>
            <Space.Compact block>
              <Select
                value={selectedPoolId || undefined}
                onChange={(v) => setSelectedPoolId(String(v ?? ""))}
                options={pools.map((p) => ({ label: `${p.pool_name} (${p.stock_count})`, value: p.pool_id }))}
                placeholder="选择关注池"
                style={{ minWidth: 260 }}
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
            {!manualMode && pools.length === 0 ? (
              <Card size="small" style={{ background: "#fff8f1", borderColor: "#fed7aa" }}>
                <Space direction="vertical">
                  <Text style={{ color: "#9a3412" }}>暂无可用关注池，建议先在关注池页面创建后再运行预测。</Text>
                  <Button onClick={() => { window.location.href = "/watchlist"; }}>去关注池创建</Button>
                </Space>
              </Card>
            ) : null}
            {manualMode ? (
              <Collapse
                size="small"
                items={[
                  {
                    key: "manual",
                    label: "手动模式: 选择股票",
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
            <Button type="primary" size="large" loading={loading} onClick={runPredict}>运行预测</Button>
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
          <Empty description="尚未运行策略，选择关注池后点击运行预测" />
        </Card>
      ) : null}

      {run ? (
        <motion.div initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.08 }}>
          <Row gutter={[14, 14]} style={{ marginTop: 12 }}>
            {run.data_quality !== "real" ? (
              <Col span={24}>
                <Card className="premium-card" style={{ borderColor: "#f59e0b", background: "#fffaf0" }}>
                  <Space direction="vertical" size={4}>
                    <Text strong style={{ color: "#92400e" }}>当前预测处于降级模式</Text>
                    <Text style={{ color: "#78350f" }}>
                      触发原因: {(run.degrade_reasons ?? []).join(",") || "unknown"}，请补齐真实历史样本后再使用高置信结论。
                    </Text>
                    <Text style={{ color: "#78350f" }}>
                      覆盖率: real_history_ratio={(Number(qualitySummary.real_history_ratio ?? 0) * 100).toFixed(1)}%
                    </Text>
                  </Space>
                </Card>
              </Col>
            ) : null}
            {metricMode === "simulated" ? (
              <Col span={24}>
                <Card className="premium-card" style={{ borderColor: "#cbd5e1", background: "#f8fafc" }}>
                  <Space direction="vertical" size={4}>
                    <Text strong style={{ color: "#334155" }}>当前评测为模拟模式</Text>
                    <Text style={{ color: "#475569" }}>
                      说明: 缺少足量真实回测样本时，系统回退为 simulated 指标，仅用于相对排序，不用于绝对收益承诺。
                    </Text>
                  </Space>
                </Card>
              </Col>
            ) : null}
            <Col xs={24} lg={8}>
              <Card className="premium-card" title={<span style={{ color: "#0f172a" }}>运行元信息</span>}>
                <Statistic title="Run ID" value={run.run_id.slice(0, 8)} valueStyle={{ color: "#0f172a" }} />
                <Statistic title="Trace ID" value={run.trace_id.slice(0, 8)} valueStyle={{ color: "#64748b", marginTop: 10 }} />
                <Statistic title="覆盖标的" value={run.results.length} valueStyle={{ color: "#059669", marginTop: 10 }} />
              </Card>
            </Col>
            <Col xs={24} lg={16}>
              <Card className="premium-card" title={<span style={{ color: "#0f172a" }}>超额收益对比(%)</span>}>
                <ReactECharts option={barOption} style={{ height: 286 }} />
              </Card>
            </Col>
            <Col xs={24} lg={12}>
              <Card className="premium-card" title={<span style={{ color: "#0f172a" }}>评测门禁</span>}>
                <Space direction="vertical" style={{ width: "100%" }}>
                  <Text style={{ color: "#475569" }}>Metric Mode: {metricMode}</Text>
                  <Text style={{ color: "#475569" }}>IC: {Number(metrics?.ic ?? 0).toFixed(4)}</Text>
                  <Progress percent={Math.max(0, Math.min(100, Number(metrics?.hit_rate ?? 0) * 100))} strokeColor="#2563eb" />
                  <Text style={{ color: "#475569" }}>Top-Bottom Spread: {Number(metrics?.top_bottom_spread ?? 0).toFixed(4)}</Text>
                  <Text style={{ color: "#475569" }}>Max Drawdown: {Number(metrics?.max_drawdown ?? 0).toFixed(4)}</Text>
                  <Text style={{ color: "#475569" }}>
                    Coverage Rows: {Number(evalProvenance.coverage_rows ?? metrics?.coverage ?? 0)}
                  </Text>
                  <Text style={{ color: "#475569" }}>
                    Evaluated Stocks: {Number(evalProvenance.evaluated_stocks ?? 0)}
                  </Text>
                  {evalProvenance.fallback_reason ? (
                    <Text style={{ color: "#b45309" }}>Fallback: {String(evalProvenance.fallback_reason)}</Text>
                  ) : null}
                  {skippedPreview ? (
                    <Text style={{ color: "#94a3b8" }}>Skipped: {skippedPreview}</Text>
                  ) : null}
                  {run?.metrics_note ? <Text style={{ color: "#94a3b8" }}>Note: {run.metrics_note}</Text> : null}
                </Space>
              </Card>
            </Col>
            <Col xs={24} lg={12}>
              <Card className="premium-card" title={<span style={{ color: "#0f172a" }}>信号概览</span>}>
                <Space wrap>
                  {tableData.slice(0, 8).map((x) => (
                    <Tag key={x.key} color={riskColor(x.risk_tier)}>{x.stock_code} {x.horizon} {x.signal}</Tag>
                  ))}
                </Space>
              </Card>
            </Col>
            <Col span={24}>
              <Card className="premium-card" title={<span style={{ color: "#0f172a" }}>分层聚合(交易所/市场层级/行业)</span>}>
                <Space direction="vertical" style={{ width: "100%" }}>
                  {(Object.entries(run.segment_metrics ?? {}) as Array<[string, Array<{ segment: string; count: number; avg_expected_excess_return: number; avg_up_probability: number }>]>).map(([k, rows]) => (
                    <div key={k}>
                      <Text strong style={{ color: "#0f172a" }}>{k}</Text>
                      <Table
                        size="small"
                        style={{ marginTop: 8 }}
                        dataSource={(rows ?? []).map((r) => ({ ...r, key: `${k}-${r.segment}` }))}
                        pagination={false}
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
              </Card>
            </Col>
            <Col span={24}>
              <Card className="premium-card" title={<span style={{ color: "#0f172a" }}>预测明细</span>}>
                <Table
                  dataSource={tableData}
                  pagination={false}
                  scroll={{ x: 900 }}
                  columns={[
                    { title: "股票", dataIndex: "stock_code", key: "stock_code" },
                    { title: "周期", dataIndex: "horizon", key: "horizon" },
                    { title: "超额收益", dataIndex: "expected_excess_return", key: "expected_excess_return", render: (v: number) => `${(v * 100).toFixed(2)}%` },
                    { title: "上涨概率", dataIndex: "up_probability", key: "up_probability", render: (v: number) => `${(v * 100).toFixed(2)}%` },
                    { title: "风险", dataIndex: "risk_tier", key: "risk_tier", render: (v: string) => <Tag color={riskColor(v)}>{v}</Tag> },
                    { title: "信号", dataIndex: "signal", key: "signal" },
                    { title: "历史数据", dataIndex: "history_data_mode", key: "history_data_mode" },
                    { title: "样本数", dataIndex: "history_sample_size", key: "history_sample_size" },
                    { title: "历史源", dataIndex: "history_source_id", key: "history_source_id" },
                    { title: "质量", dataIndex: "data_quality", key: "data_quality" },
                    { title: "降级原因", dataIndex: "degrade_reasons", key: "degrade_reasons", width: 220 },
                    { title: "解释", dataIndex: "rationale", key: "rationale", width: 320 }
                  ]}
                />
              </Card>
            </Col>
          </Row>
        </motion.div>
      ) : null}
    </main>
  );
}
