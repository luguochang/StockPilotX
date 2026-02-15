"use client";

import { useState } from "react";
import { Alert, Button, Card, Input, Space, Table, Tag, Typography } from "antd";
import type { ColumnsType } from "antd/es/table";

const API_BASE = process.env.NEXT_PUBLIC_API_BASE ?? "http://127.0.0.1:8000";
const { Title, Text } = Typography;

type JobStatusRow = {
  job_name: string;
  failure_count: number;
  last_status: string;
  last_error: string;
  last_run_at: string;
  circuit_open_until: string;
  paused: boolean;
  history: Array<Record<string, unknown>>;
};

export default function OpsSchedulerPage() {
  const [job, setJob] = useState("intraday_quote_ingest");
  const [rows, setRows] = useState<JobStatusRow[]>([]);
  const [rawResult, setRawResult] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  // 中文注释：调度相关接口统一错误处理。

  async function parseOrThrow(resp: Response) {
    const body = await resp.json();
    if (!resp.ok) {
      throw new Error(body?.detail ?? `HTTP ${resp.status}`);
    }
    return body;
  }

  async function loadStatus() {
    setLoading(true);
    setError("");
    try {
      const status = await parseOrThrow(await fetch(`${API_BASE}/v1/scheduler/status`));
      const entries = Object.entries(status || {}).map(([job_name, data]) => ({ job_name, ...(data as object) })) as JobStatusRow[];
      setRows(entries);
      setRawResult(JSON.stringify(status, null, 2));
    } catch (e) {
      setError(e instanceof Error ? e.message : "读取状态失败");
      setRows([]);
    } finally {
      setLoading(false);
    }
  }

  async function runJob(targetJobName?: string) {
    setLoading(true);
    setError("");
    try {
      const selectedJob = (targetJobName ?? job).trim();
      const body = await parseOrThrow(
        await fetch(`${API_BASE}/v1/scheduler/run`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ job_name: selectedJob })
        })
      );
      setRawResult(JSON.stringify(body, null, 2));
      await loadStatus();
    } catch (e) {
      setError(e instanceof Error ? e.message : "触发失败");
    } finally {
      setLoading(false);
    }
  }

  async function pauseOrResume(action: "pause" | "resume", name: string) {
    setLoading(true);
    setError("");
    try {
      const body = await parseOrThrow(
        await fetch(`${API_BASE}/v1/scheduler/${action}`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ job_name: name })
        })
      );
      setRawResult(JSON.stringify(body, null, 2));
      await loadStatus();
    } catch (e) {
      setError(e instanceof Error ? e.message : "操作失败");
    } finally {
      setLoading(false);
    }
  }

  const columns: ColumnsType<JobStatusRow> = [
    { title: "任务名", dataIndex: "job_name", key: "job_name" },
    {
      title: "状态",
      key: "state",
      render: (_, row) => (
        <Space>
          <Tag color={row.paused ? "orange" : "green"}>{row.paused ? "Paused" : "Running"}</Tag>
          <Tag color={row.last_status === "ok" || row.last_status === "never" ? "blue" : "red"}>{row.last_status}</Tag>
        </Space>
      )
    },
    { title: "失败次数", dataIndex: "failure_count", key: "failure_count" },
    { title: "最近运行", dataIndex: "last_run_at", key: "last_run_at" },
    { title: "熔断到", dataIndex: "circuit_open_until", key: "circuit_open_until" },
    {
      title: "操作",
      key: "actions",
      render: (_, row) => (
        <Space>
          <Button size="small" onClick={() => runJob()}>Run</Button>
          <Button size="small" onClick={() => runJob(row.job_name)}>RunThis</Button>
          <Button size="small" onClick={() => pauseOrResume("pause", row.job_name)}>Pause</Button>
          <Button size="small" type="primary" onClick={() => pauseOrResume("resume", row.job_name)}>Resume</Button>
        </Space>
      )
    }
  ];

  return (
    <main className="container">
      <Card className="premium-card">
        <Space direction="vertical" style={{ width: "100%" }}>
          <Title level={2} style={{ margin: 0 }}>调度管理台</Title>
          <Text type="secondary">覆盖接口：`/v1/scheduler/status`、`/v1/scheduler/run`、`/v1/scheduler/pause`、`/v1/scheduler/resume`</Text>
          <Space.Compact block>
            <Input value={job} onChange={(e) => setJob(e.target.value)} placeholder="job_name" />
            <Button loading={loading} onClick={() => runJob()}>触发任务</Button>
            <Button loading={loading} onClick={loadStatus}>刷新状态</Button>
          </Space.Compact>
        </Space>
      </Card>

      {error ? <Alert style={{ marginTop: 12 }} type="error" showIcon message={error} /> : null}

      <Card className="premium-card" style={{ marginTop: 12 }} title="任务状态">
        <Table rowKey="job_name" columns={columns} dataSource={rows} pagination={false} />
      </Card>

      <Card className="premium-card" style={{ marginTop: 12 }} title="最近操作返回">
        <pre style={{ whiteSpace: "pre-wrap", margin: 0 }}>{rawResult}</pre>
      </Card>
    </main>
  );
}
