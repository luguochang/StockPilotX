"use client";

import { useState } from "react";
import { Alert, Button, Card, Input, Select, Space, Table, Tag, Typography } from "antd";
import type { ColumnsType } from "antd/es/table";

const API_BASE = process.env.NEXT_PUBLIC_API_BASE ?? "http://127.0.0.1:8000";
const { Title, Text } = Typography;

type CapabilityRow = { key: string; status: string; detail: string };
type DebateOpinion = { agent: string; signal: string; confidence: number; reason: string };
type RagCase = {
  query: string;
  positive_source_ids: string[];
  predicted_source_ids: string[];
  hit_source_ids: string[];
  recall_at_k: number;
  mrr: number;
  ndcg_at_k: number;
};
type PromptVersion = { prompt_id: string; version: string; scenario: string; status: string };

export default function OpsEvalsPage() {
  const [token, setToken] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const [capabilities, setCapabilities] = useState<CapabilityRow[]>([]);
  const [runtimeInfo, setRuntimeInfo] = useState("");
  const [evals, setEvals] = useState("");
  const [releases, setReleases] = useState("");

  const [debateRaw, setDebateRaw] = useState("");
  const [debateOpinions, setDebateOpinions] = useState<DebateOpinion[]>([]);
  const [ragRaw, setRagRaw] = useState("");
  const [ragCases, setRagCases] = useState<RagCase[]>([]);
  const [promptCompareRaw, setPromptCompareRaw] = useState("");
  const [promptVersions, setPromptVersions] = useState<PromptVersion[]>([]);
  const [baseVersion, setBaseVersion] = useState("1.0.0");
  const [candidateVersion, setCandidateVersion] = useState("1.1.0");

  async function parseOrThrow(resp: Response) {
    const body = await resp.json();
    if (!resp.ok) throw new Error(body?.detail ?? `HTTP ${resp.status}`);
    return body;
  }

  async function load() {
    setLoading(true);
    setError("");
    try {
      const headers: Record<string, string> = {};
      if (token.trim()) headers.Authorization = `Bearer ${token}`;
      const [a, b, c, d, e, f] = await Promise.all([
        fetch(`${API_BASE}/v1/ops/evals/history`, { headers }),
        fetch(`${API_BASE}/v1/ops/prompts/releases`, { headers }),
        fetch(`${API_BASE}/v1/ops/capabilities`),
        fetch(`${API_BASE}/v1/ops/agent/debate?stock_code=SH600000&question=${encodeURIComponent("请给出短中期观点")}`),
        fetch(`${API_BASE}/v1/ops/rag/quality`),
        fetch(`${API_BASE}/v1/ops/prompts/fact_qa/versions`)
      ]);
      const evalData = await parseOrThrow(a);
      const releaseData = await parseOrThrow(b);
      const capabilityData = await parseOrThrow(c);
      const debateData = await parseOrThrow(d);
      const ragData = await parseOrThrow(e);
      const versionData = await parseOrThrow(f);
      setEvals(JSON.stringify(evalData, null, 2));
      setReleases(JSON.stringify(releaseData, null, 2));
      setCapabilities((capabilityData?.capabilities ?? []) as CapabilityRow[]);
      setRuntimeInfo(JSON.stringify(capabilityData?.runtime ?? {}, null, 2));
      setDebateRaw(JSON.stringify(debateData, null, 2));
      setDebateOpinions((debateData?.opinions ?? []) as DebateOpinion[]);
      setRagRaw(JSON.stringify(ragData?.metrics ?? {}, null, 2));
      setRagCases((ragData?.offline?.cases ?? []) as RagCase[]);
      setPromptVersions((versionData ?? []) as PromptVersion[]);

      const compareResp = await fetch(`${API_BASE}/v1/ops/prompts/compare`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          prompt_id: "fact_qa",
          base_version: baseVersion,
          candidate_version: candidateVersion,
          variables: { question: "请分析SH600000", stock_codes: ["SH600000"], evidence: "source:cninfo" }
        })
      });
      const compareData = await parseOrThrow(compareResp);
      setPromptCompareRaw(JSON.stringify(compareData, null, 2));
    } catch (e) {
      setError(e instanceof Error ? e.message : "加载失败");
      setCapabilities([]);
      setDebateOpinions([]);
      setRagCases([]);
    } finally {
      setLoading(false);
    }
  }

  const capabilityColumns: ColumnsType<CapabilityRow> = [
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

  const debateColumns: ColumnsType<DebateOpinion> = [
    { title: "Agent", dataIndex: "agent", key: "agent" },
    {
      title: "Signal",
      dataIndex: "signal",
      key: "signal",
      render: (v: string) => <Tag color={v === "buy" ? "green" : v === "reduce" ? "red" : "blue"}>{v}</Tag>
    },
    { title: "Confidence", dataIndex: "confidence", key: "confidence" },
    { title: "Reason", dataIndex: "reason", key: "reason" }
  ];

  const ragColumns: ColumnsType<RagCase> = [
    { title: "Query", dataIndex: "query", key: "query", width: 260 },
    { title: "Recall@5", dataIndex: "recall_at_k", key: "recall_at_k" },
    { title: "MRR", dataIndex: "mrr", key: "mrr" },
    { title: "nDCG@5", dataIndex: "ndcg_at_k", key: "ndcg_at_k" },
    { title: "Hit Sources", dataIndex: "hit_source_ids", key: "hit_source_ids", render: (v: string[]) => (v || []).join(", ") || "-" }
  ];

  return (
    <main className="container">
      <Card className="premium-card">
        <Space direction="vertical" style={{ width: "100%" }}>
          <Title level={2} style={{ margin: 0 }}>技术核查与评测控制台</Title>
          <Text type="secondary">包含能力快照、多 Agent 分歧、RAG 质量与 Prompt 版本回放。</Text>
          <Input.Password
            value={token}
            onChange={(e) => setToken(e.target.value)}
            placeholder="Bearer Token（ops/admin 用于 eval/release）"
          />
          <Button type="primary" loading={loading} onClick={load}>加载面板</Button>
        </Space>
      </Card>

      {error ? <Alert style={{ marginTop: 12 }} type="error" showIcon message={error} /> : null}

      <Card className="premium-card" style={{ marginTop: 12 }} title="技术能力快照">
        <Table rowKey="key" columns={capabilityColumns} dataSource={capabilities} pagination={false} />
      </Card>
      <Card className="premium-card" style={{ marginTop: 12 }} title="运行时信息">
        <pre style={{ whiteSpace: "pre-wrap", margin: 0 }}>{runtimeInfo}</pre>
      </Card>

      <Card className="premium-card" style={{ marginTop: 12 }} title="多 Agent 分歧分析">
        <Table rowKey={(r) => r.agent} columns={debateColumns} dataSource={debateOpinions} pagination={false} />
        <pre style={{ whiteSpace: "pre-wrap", marginTop: 12 }}>{debateRaw}</pre>
      </Card>

      <Card className="premium-card" style={{ marginTop: 12 }} title="RAG 质量评测">
        <Text>聚合指标</Text>
        <pre style={{ whiteSpace: "pre-wrap", marginTop: 8 }}>{ragRaw}</pre>
        <Table rowKey={(r) => r.query} columns={ragColumns} dataSource={ragCases} pagination={false} />
      </Card>

      <Card className="premium-card" style={{ marginTop: 12 }} title="Prompt 版本对比回放">
        <Space style={{ marginBottom: 8 }}>
          <Select
            style={{ minWidth: 160 }}
            value={baseVersion}
            onChange={setBaseVersion}
            options={promptVersions.map((x) => ({ label: `${x.version} (${x.status})`, value: x.version }))}
          />
          <Select
            style={{ minWidth: 160 }}
            value={candidateVersion}
            onChange={setCandidateVersion}
            options={promptVersions.map((x) => ({ label: `${x.version} (${x.status})`, value: x.version }))}
          />
          <Button onClick={load}>重新对比</Button>
        </Space>
        <pre style={{ whiteSpace: "pre-wrap", margin: 0 }}>{promptCompareRaw}</pre>
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
