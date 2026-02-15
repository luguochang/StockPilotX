"use client";

import { useMemo, useState } from "react";
import { motion } from "framer-motion";
import ReactECharts from "echarts-for-react";
import { Alert, Button, Card, Col, Progress, Row, Space, Statistic, Table, Tag, Typography } from "antd";
import MediaCarousel from "../components/MediaCarousel";
import StockSelectorModal from "../components/StockSelectorModal";

const API_BASE = process.env.NEXT_PUBLIC_API_BASE ?? "http://127.0.0.1:8000";
const { Title, Text, Paragraph } = Typography;

type HorizonResult = {
  horizon: string;
  score: number;
  expected_excess_return: number;
  up_probability: number;
  risk_tier: string;
  signal: string;
  rationale?: string;
};

type PredictItem = {
  stock_code: string;
  horizons: HorizonResult[];
  factors: Record<string, number>;
  source?: {
    source_id?: string;
    history_data_mode?: string;
    history_source_id?: string;
    history_sample_size?: number;
  };
};

type PredictRunResponse = {
  run_id: string;
  trace_id: string;
  results: PredictItem[];
};

type EvalResponse = {
  status: string;
  metrics?: Record<string, number>;
};

function riskColor(tier: string): string {
  if (tier === "high") return "red";
  if (tier === "medium") return "gold";
  return "green";
}

export default function PredictPage() {
  const [selectedCodes, setSelectedCodes] = useState<string[]>(["SH600000", "SZ000001"]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [run, setRun] = useState<PredictRunResponse | null>(null);
  const [evalData, setEvalData] = useState<EvalResponse | null>(null);

  async function ensureStockInUniverse(codes: string[]) {
    for (const code of codes) {
      const normalized = code.trim().toUpperCase();
      const resp = await fetch(`${API_BASE}/v1/stocks/search?keyword=${encodeURIComponent(normalized)}&limit=30`);
      if (!resp.ok) throw new Error("股票库检索失败，请稍后重试");
      const rows = await resp.json();
      const hit = Array.isArray(rows) && rows.some((x: any) => String(x?.stock_code ?? "").toUpperCase() === normalized);
      if (!hit) throw new Error(`股票 ${normalized} 不在已同步股票库中，请重新选择`);
    }
  }

  async function runPredict() {
    setLoading(true);
    setError("");
    try {
      if (!selectedCodes.length) {
        throw new Error("请先选择至少一只股票");
      }
      await ensureStockInUniverse(selectedCodes);

      // 中文注释：一次请求拉取预测结果，再补充最近评测摘要用于质量解释。
      const resp = await fetch(`${API_BASE}/v1/predict/run`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ stock_codes: selectedCodes, horizons: ["5d", "20d"] })
      });
      const body = await resp.json();
      if (!resp.ok) throw new Error(body?.detail ?? `HTTP ${resp.status}`);

      setRun(body as PredictRunResponse);

      const evalResp = await fetch(`${API_BASE}/v1/predict/evals/latest`);
      const evalBody = await evalResp.json();
      if (evalResp.ok) setEvalData(evalBody as EvalResponse);
    } catch (e) {
      setError(e instanceof Error ? e.message : "请求失败");
      setRun(null);
    } finally {
      setLoading(false);
    }
  }

  const barOption = useMemo(() => {
    const rows = run?.results ?? [];
    return {
      grid: { top: 30, left: 44, right: 12, bottom: 24 },
      tooltip: { trigger: "axis" },
      legend: { data: ["5日", "20日"], textStyle: { color: "#334155" } },
      xAxis: {
        type: "category",
        data: rows.map((r) => r.stock_code),
        axisLabel: { color: "#475569" },
        axisLine: { lineStyle: { color: "rgba(100,116,139,0.3)" } }
      },
      yAxis: {
        type: "value",
        axisLabel: { color: "#475569" },
        splitLine: { lineStyle: { color: "rgba(100,116,139,0.16)" } }
      },
      series: [
        {
          name: "5日",
          type: "bar",
          itemStyle: { color: "#2563eb" },
          data: rows.map((r) => Number(((r.horizons.find((x) => x.horizon === "5d")?.expected_excess_return ?? 0) * 100).toFixed(2)))
        },
        {
          name: "20日",
          type: "bar",
          itemStyle: { color: "#0ea5e9" },
          data: rows.map((r) => Number(((r.horizons.find((x) => x.horizon === "20d")?.expected_excess_return ?? 0) * 100).toFixed(2)))
        }
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
      history_source_id: item.source?.history_source_id ?? "unknown"
    }))
  );
  const heroSlides = [
    { src: "/assets/images/nyse-floor-2014.jpg", alt: "Trading floor visual", caption: "实时交易数据屏" },
    { src: "/assets/images/nyse-floor-1963.jpg", alt: "Historic trading floor visual", caption: "历史周期回溯" },
    { src: "/assets/images/nyse-floor-1930.png", alt: "Long cycle visual", caption: "长期风险观察" }
  ];

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
          <Space direction="vertical" style={{ width: "100%" }} size={8}>
            <Tag color="processing" style={{ width: "fit-content" }}>Predictive Intelligence Console</Tag>
            <Title level={2} style={{ margin: 0, color: "#0f172a" }}>策略预测驾驶舱</Title>
            <Paragraph style={{ margin: 0, color: "#475569", maxWidth: 760 }}>
              聚合多周期收益预测、风险分层、胜率估计和评测门禁，输出可解释的量化观察结果。
            </Paragraph>
            <StockSelectorModal
              value={selectedCodes}
              onChange={(next) => setSelectedCodes(Array.isArray(next) ? next : (next ? [next] : []))}
              multiple
              title="选择预测标的"
              placeholder="请选择一只或多只股票"
            />
            <Button type="primary" size="large" loading={loading} onClick={runPredict}>运行预测</Button>
          </Space>
        </Card>
      </motion.div>

      {error ? <Alert style={{ marginTop: 12 }} type="error" showIcon message={error} /> : null}

      {run ? (
        <motion.div initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.08 }}>
          <Row gutter={[14, 14]} style={{ marginTop: 12 }}>
            <Col xs={24} lg={8}>
              <Card className="premium-card" title={<span style={{ color: "#0f172a" }}>运行元信息</span>}>
                <Statistic title="Run ID" value={run.run_id.slice(0, 8)} valueStyle={{ color: "#0f172a" }} />
                <Statistic title="Trace ID" value={run.trace_id.slice(0, 8)} valueStyle={{ color: "#64748b", marginTop: 10 }} />
                <Statistic title="覆盖标的" value={run.results.length} valueStyle={{ color: "#059669", marginTop: 10 }} />
              </Card>
            </Col>
            <Col xs={24} lg={16}>
              <Card className="premium-card" title={<span style={{ color: "#0f172a" }}>超额收益对比（%）</span>}>
                <ReactECharts option={barOption} style={{ height: 286 }} />
              </Card>
            </Col>
            <Col xs={24} lg={12}>
              <Card className="premium-card" title={<span style={{ color: "#0f172a" }}>评测门禁</span>}>
                <Space direction="vertical" style={{ width: "100%" }}>
                  <Text style={{ color: "#475569" }}>IC: {(evalData?.metrics?.ic ?? 0).toFixed(4)}</Text>
                  <Progress percent={Math.max(0, Math.min(100, (evalData?.metrics?.hit_rate ?? 0) * 100))} strokeColor="#2563eb" />
                  <Text style={{ color: "#475569" }}>Top-Bottom Spread: {(evalData?.metrics?.top_bottom_spread ?? 0).toFixed(4)}</Text>
                  <Text style={{ color: "#475569" }}>Max Drawdown: {(evalData?.metrics?.max_drawdown ?? 0).toFixed(4)}</Text>
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

