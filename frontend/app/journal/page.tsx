"use client";

import { useEffect, useMemo, useState } from "react";
import { motion } from "framer-motion";
import { Alert, Button, Card, Col, Flex, Form, Input, Row, Select, Table, Tabs, Tag, Typography, message } from "antd";
import type { ColumnsType } from "antd/es/table";

const API_BASE = process.env.NEXT_PUBLIC_API_BASE ?? "http://127.0.0.1:8000";
const { Title, Paragraph, Text } = Typography;

type JournalType = "decision" | "reflection" | "learning";

type JournalItem = {
  journal_id: number;
  journal_type: string;
  title: string;
  content: string;
  stock_code: string;
  decision_type: string;
  tags: string[];
  sentiment: string;
  created_at: string;
  updated_at: string;
  reflection_count?: number;
  has_ai_reflection?: boolean;
};

type JournalReflection = {
  reflection_id: number;
  journal_id: number;
  reflection_content: string;
  ai_insights: string;
  lessons_learned: string;
  created_at: string;
};

type JournalAIReflection = {
  ai_reflection_id: number;
  journal_id: number;
  status: string;
  summary: string;
  insights: string[];
  lessons: string[];
  confidence: number;
  provider: string;
  model: string;
  trace_id: string;
  generated_at: string;
  latency_ms?: number;
};

type InsightBreakdown = {
  key: string;
  count: number;
  ratio: number;
};

type InsightKeyword = {
  keyword: string;
  count: number;
};

type InsightTimeline = {
  day: string;
  journal_count: number;
  reflection_count: number;
  ai_reflection_count: number;
};

type JournalInsights = {
  status: string;
  window_days: number;
  timeline_days: number;
  total_journals: number;
  type_distribution: InsightBreakdown[];
  decision_distribution: InsightBreakdown[];
  stock_activity: InsightBreakdown[];
  reflection_coverage: {
    with_reflection: number;
    with_ai_reflection: number;
    reflection_coverage_rate: number;
    ai_reflection_coverage_rate: number;
    total_reflection_records: number;
    avg_reflections_per_journal: number;
  };
  keyword_profile: InsightKeyword[];
  timeline: InsightTimeline[];
};

type CreateJournalForm = {
  journal_type: JournalType;
  title: string;
  stock_code: string;
  decision_type: string;
  tags: string;
  sentiment: string;
  content: string;
};

type FilterForm = {
  journal_type: string;
  stock_code: string;
  limit: number;
};

function parseCommaItems(raw: string): string[] {
  return raw
    .split(",")
    .map((item) => item.trim())
    .filter((item) => item.length > 0);
}

async function parseOrThrow<T>(resp: Response): Promise<T> {
  const body = (await resp.json()) as T | { detail?: string; error?: string };
  if (!resp.ok) {
    const msg = typeof body === "object" && body ? ((body as { detail?: string; error?: string }).detail ?? (body as { detail?: string; error?: string }).error) : "";
    throw new Error(msg || `HTTP ${resp.status}`);
  }
  return body as T;
}

export default function JournalPage() {
  const [messageApi, contextHolder] = message.useMessage();
  const [createForm] = Form.useForm<CreateJournalForm>();
  const [filterForm] = Form.useForm<FilterForm>();
  const [listLoading, setListLoading] = useState(false);
  const [createLoading, setCreateLoading] = useState(false);
  const [reflectionLoading, setReflectionLoading] = useState(false);
  const [aiLoading, setAiLoading] = useState(false);
  const [insightsLoading, setInsightsLoading] = useState(false);
  const [journals, setJournals] = useState<JournalItem[]>([]);
  const [selectedJournalId, setSelectedJournalId] = useState<number | null>(null);
  const [reflections, setReflections] = useState<JournalReflection[]>([]);
  const [reflectionInput, setReflectionInput] = useState("");
  const [aiFocus, setAiFocus] = useState("优先给出可验证改进和失效条件");
  const [aiReflection, setAiReflection] = useState<JournalAIReflection | null>(null);
  const [insights, setInsights] = useState<JournalInsights | null>(null);

  const selectedJournal = useMemo(() => journals.find((item) => item.journal_id === selectedJournalId) ?? null, [journals, selectedJournalId]);

  const journalColumns: ColumnsType<JournalItem> = [
    {
      title: "日志",
      dataIndex: "title",
      key: "title",
      render: (_, record) => (
        <Flex vertical gap={2}>
          <Text strong>{record.title}</Text>
          <Text type="secondary" style={{ fontSize: 12 }}>
            {record.stock_code || "未绑定标的"} · {record.created_at}
          </Text>
        </Flex>
      ),
    },
    {
      title: "类型",
      key: "journal_type",
      width: 140,
      render: (_, record) => (
        <Flex gap={6} wrap>
          <Tag color="blue">{record.journal_type}</Tag>
          <Tag>{record.decision_type || "none"}</Tag>
        </Flex>
      ),
    },
    {
      title: "覆盖",
      key: "coverage",
      width: 140,
      render: (_, record) => (
        <Flex vertical gap={2}>
          <Text style={{ fontSize: 12 }}>复盘: {Number(record.reflection_count ?? 0)}</Text>
          <Text style={{ fontSize: 12 }}>AI: {record.has_ai_reflection ? "yes" : "no"}</Text>
        </Flex>
      ),
    },
  ];

  const timelineColumns: ColumnsType<InsightTimeline> = [
    { title: "日期", dataIndex: "day", key: "day", width: 130 },
    { title: "日志", dataIndex: "journal_count", key: "journal_count", width: 90 },
    { title: "复盘", dataIndex: "reflection_count", key: "reflection_count", width: 90 },
    { title: "AI复盘", dataIndex: "ai_reflection_count", key: "ai_reflection_count", width: 100 },
  ];

  async function loadJournals(values?: Partial<FilterForm>) {
    const current = { ...filterForm.getFieldsValue(), ...(values ?? {}) };
    const query = new URLSearchParams();
    if (current.journal_type) query.set("journal_type", current.journal_type);
    if (current.stock_code) query.set("stock_code", current.stock_code.trim().toUpperCase());
    query.set("limit", String(current.limit || 40));
    setListLoading(true);
    try {
      const resp = await fetch(`${API_BASE}/v1/journal?${query.toString()}`);
      const body = await parseOrThrow<JournalItem[]>(resp);
      setJournals(Array.isArray(body) ? body : []);
      if (selectedJournalId && !body.find((item) => item.journal_id === selectedJournalId)) {
        setSelectedJournalId(null);
      }
    } catch (error) {
      messageApi.error(error instanceof Error ? error.message : "加载日志失败");
    } finally {
      setListLoading(false);
    }
  }

  async function loadReflections(journalId: number) {
    setReflectionLoading(true);
    try {
      const resp = await fetch(`${API_BASE}/v1/journal/${journalId}/reflections?limit=80`);
      const body = await parseOrThrow<JournalReflection[]>(resp);
      setReflections(Array.isArray(body) ? body : []);
    } catch (error) {
      messageApi.error(error instanceof Error ? error.message : "加载复盘失败");
    } finally {
      setReflectionLoading(false);
    }
  }

  async function loadAIReflection(journalId: number) {
    setAiLoading(true);
    try {
      const resp = await fetch(`${API_BASE}/v1/journal/${journalId}/ai-reflection`);
      const body = await parseOrThrow<JournalAIReflection | Record<string, never>>(resp);
      if (body && typeof body === "object" && "journal_id" in body) {
        setAiReflection(body as JournalAIReflection);
      } else {
        setAiReflection(null);
      }
    } catch (error) {
      messageApi.error(error instanceof Error ? error.message : "加载 AI 复盘失败");
    } finally {
      setAiLoading(false);
    }
  }

  async function loadInsights() {
    setInsightsLoading(true);
    try {
      const resp = await fetch(`${API_BASE}/v1/journal/insights?window_days=180&timeline_days=60&limit=600`);
      const body = await parseOrThrow<JournalInsights>(resp);
      setInsights(body);
    } catch (error) {
      messageApi.error(error instanceof Error ? error.message : "加载洞察失败");
    } finally {
      setInsightsLoading(false);
    }
  }

  async function handleCreate(values: CreateJournalForm) {
    setCreateLoading(true);
    try {
      const payload = {
        journal_type: values.journal_type,
        title: values.title.trim(),
        content: values.content.trim(),
        stock_code: values.stock_code.trim().toUpperCase(),
        decision_type: values.decision_type.trim().toLowerCase(),
        tags: parseCommaItems(values.tags || ""),
        sentiment: values.sentiment.trim().toLowerCase(),
      };
      const resp = await fetch(`${API_BASE}/v1/journal`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      const created = await parseOrThrow<JournalItem>(resp);
      messageApi.success("日志创建成功");
      createForm.resetFields();
      await Promise.all([loadJournals(), loadInsights()]);
      setSelectedJournalId(Number(created.journal_id));
    } catch (error) {
      messageApi.error(error instanceof Error ? error.message : "创建日志失败");
    } finally {
      setCreateLoading(false);
    }
  }

  async function handleAddReflection() {
    if (!selectedJournalId) {
      messageApi.warning("请先选择一条日志");
      return;
    }
    if (!reflectionInput.trim()) {
      messageApi.warning("请输入复盘内容");
      return;
    }
    setReflectionLoading(true);
    try {
      const resp = await fetch(`${API_BASE}/v1/journal/${selectedJournalId}/reflections`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ reflection_content: reflectionInput.trim() }),
      });
      await parseOrThrow<JournalReflection>(resp);
      setReflectionInput("");
      messageApi.success("复盘已记录");
      await Promise.all([loadReflections(selectedJournalId), loadJournals(), loadInsights()]);
    } catch (error) {
      messageApi.error(error instanceof Error ? error.message : "写入复盘失败");
    } finally {
      setReflectionLoading(false);
    }
  }

  async function handleGenerateAIReflection() {
    if (!selectedJournalId) {
      messageApi.warning("请先选择一条日志");
      return;
    }
    setAiLoading(true);
    try {
      const resp = await fetch(`${API_BASE}/v1/journal/${selectedJournalId}/ai-reflection/generate`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ focus: aiFocus.trim() }),
      });
      const generated = await parseOrThrow<JournalAIReflection>(resp);
      setAiReflection(generated);
      messageApi.success(generated.status === "ready" ? "AI复盘生成完成" : "AI复盘已回退到本地模板");
      await Promise.all([loadJournals(), loadInsights()]);
    } catch (error) {
      messageApi.error(error instanceof Error ? error.message : "生成 AI 复盘失败");
    } finally {
      setAiLoading(false);
    }
  }

  useEffect(() => {
    // 页面初始化同时加载日志列表和洞察总览。
    void Promise.all([loadJournals({ limit: 40 }), loadInsights()]);
    filterForm.setFieldsValue({ journal_type: "", stock_code: "", limit: 40 });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    if (!selectedJournalId) {
      setReflections([]);
      setAiReflection(null);
      return;
    }
    // 选中日志后并行加载手工复盘与 AI 复盘。
    void Promise.all([loadReflections(selectedJournalId), loadAIReflection(selectedJournalId)]);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedJournalId]);

  return (
    <main className="container">
      {contextHolder}
      <motion.section initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.4 }}>
        <Card className="premium-card" style={{ marginBottom: 12 }}>
          <Flex vertical gap={10}>
            <Tag color="processing" style={{ width: "fit-content" }}>
              Investment Journal Workspace
            </Tag>
            <Title level={2} style={{ margin: 0 }}>
              投资日志工作台
            </Title>
            <Paragraph style={{ margin: 0, color: "#475569", maxWidth: 820 }}>
              这里统一承接日志创建、手工复盘、AI复盘和洞察看板。用户只需要先写一条日志，再逐步补充复盘，即可形成可追踪的研究闭环。
            </Paragraph>
          </Flex>
        </Card>
      </motion.section>

      <Row gutter={[12, 12]}>
        <Col xs={24} xl={14}>
          <Card className="premium-card" title="新建日志" style={{ marginBottom: 12 }}>
            <Form<CreateJournalForm>
              layout="vertical"
              form={createForm}
              initialValues={{
                journal_type: "decision",
                decision_type: "hold",
                sentiment: "neutral",
                stock_code: "SH600000",
                tags: "deepthink,复盘",
              }}
              onFinish={handleCreate}
            >
              <Row gutter={10}>
                <Col xs={24} md={8}>
                  <Form.Item name="journal_type" label="日志类型" rules={[{ required: true }]}>
                    <Select
                      options={[
                        { label: "决策记录", value: "decision" },
                        { label: "复盘记录", value: "reflection" },
                        { label: "学习记录", value: "learning" },
                      ]}
                    />
                  </Form.Item>
                </Col>
                <Col xs={24} md={8}>
                  <Form.Item name="stock_code" label="股票代码" rules={[{ required: true, message: "请输入股票代码" }]}>
                    <Input placeholder="例如 SH600000" />
                  </Form.Item>
                </Col>
                <Col xs={24} md={8}>
                  <Form.Item name="decision_type" label="决策倾向">
                    <Select
                      options={[
                        { label: "买入 buy", value: "buy" },
                        { label: "持有 hold", value: "hold" },
                        { label: "减仓 reduce", value: "reduce" },
                      ]}
                    />
                  </Form.Item>
                </Col>
              </Row>
              <Form.Item name="title" label="标题" rules={[{ required: true, message: "请输入标题" }]}>
                <Input placeholder="例如: 估值修复阶段的仓位调整计划" />
              </Form.Item>
              <Form.Item name="tags" label="标签（逗号分隔）">
                <Input placeholder="估值,风险,仓位" />
              </Form.Item>
              <Form.Item name="sentiment" label="情绪标签">
                <Select
                  options={[
                    { label: "中性 neutral", value: "neutral" },
                    { label: "积极 positive", value: "positive" },
                    { label: "谨慎 negative", value: "negative" },
                  ]}
                />
              </Form.Item>
              <Form.Item name="content" label="正文" rules={[{ required: true, message: "请输入日志正文" }]}>
                <Input.TextArea rows={5} placeholder="记录触发条件、预期路径和失效边界..." />
              </Form.Item>
              <Button type="primary" htmlType="submit" loading={createLoading}>
                创建日志
              </Button>
            </Form>
          </Card>

          <Card className="premium-card" title="日志列表">
            <Form<FilterForm> form={filterForm} layout="inline" onFinish={(values) => void loadJournals(values)}>
              <Form.Item name="journal_type" label="类型">
                <Select
                  style={{ width: 140 }}
                  options={[
                    { label: "全部", value: "" },
                    { label: "决策", value: "decision" },
                    { label: "复盘", value: "reflection" },
                    { label: "学习", value: "learning" },
                  ]}
                />
              </Form.Item>
              <Form.Item name="stock_code" label="股票">
                <Input placeholder="SH600000" />
              </Form.Item>
              <Form.Item name="limit" label="条数">
                <Select
                  style={{ width: 100 }}
                  options={[
                    { label: "20", value: 20 },
                    { label: "40", value: 40 },
                    { label: "80", value: 80 },
                  ]}
                />
              </Form.Item>
              <Form.Item>
                <Button htmlType="submit" loading={listLoading}>
                  查询
                </Button>
              </Form.Item>
              <Form.Item>
                <Button
                  onClick={() => {
                    filterForm.setFieldsValue({ journal_type: "", stock_code: "", limit: 40 });
                    void loadJournals({ journal_type: "", stock_code: "", limit: 40 });
                  }}
                >
                  重置
                </Button>
              </Form.Item>
            </Form>
            <Table<JournalItem>
              style={{ marginTop: 12 }}
              rowKey="journal_id"
              loading={listLoading}
              dataSource={journals}
              columns={journalColumns}
              pagination={{ pageSize: 8, showSizeChanger: false }}
              rowClassName={(record) => (record.journal_id === selectedJournalId ? "ant-table-row-selected" : "")}
              onRow={(record) => ({
                onClick: () => {
                  setSelectedJournalId(record.journal_id);
                },
              })}
            />
          </Card>
        </Col>

        <Col xs={24} xl={10}>
          <Card
            className="premium-card"
            title="Journal 洞察看板"
            extra={
              <Button size="small" loading={insightsLoading} onClick={() => void loadInsights()}>
                刷新
              </Button>
            }
          >
            {!insights ? (
              <Alert type="info" showIcon message="暂无洞察数据，请先创建日志。" />
            ) : (
              <Flex vertical gap={10}>
                <Row gutter={10}>
                  <Col span={8}>
                    <Card size="small">
                      <Text type="secondary">日志总数</Text>
                      <Title level={4} style={{ margin: 0 }}>
                        {insights.total_journals}
                      </Title>
                    </Card>
                  </Col>
                  <Col span={8}>
                    <Card size="small">
                      <Text type="secondary">手工复盘覆盖</Text>
                      <Title level={4} style={{ margin: 0 }}>
                        {(insights.reflection_coverage.reflection_coverage_rate * 100).toFixed(1)}%
                      </Title>
                    </Card>
                  </Col>
                  <Col span={8}>
                    <Card size="small">
                      <Text type="secondary">AI复盘覆盖</Text>
                      <Title level={4} style={{ margin: 0 }}>
                        {(insights.reflection_coverage.ai_reflection_coverage_rate * 100).toFixed(1)}%
                      </Title>
                    </Card>
                  </Col>
                </Row>
                <Flex vertical gap={6}>
                  <Text strong>高频标的</Text>
                  <Flex gap={6} wrap>
                    {insights.stock_activity.slice(0, 6).map((item) => (
                      <Tag key={item.key}>
                        {item.key} · {item.count}
                      </Tag>
                    ))}
                  </Flex>
                </Flex>
                <Flex vertical gap={6}>
                  <Text strong>关键词画像</Text>
                  <Flex gap={6} wrap>
                    {insights.keyword_profile.slice(0, 10).map((item) => (
                      <Tag key={item.keyword} color="blue">
                        {item.keyword} · {item.count}
                      </Tag>
                    ))}
                  </Flex>
                </Flex>
                <Table<InsightTimeline>
                  size="small"
                  rowKey={(row) => row.day}
                  dataSource={insights.timeline.slice(-10)}
                  columns={timelineColumns}
                  pagination={false}
                />
              </Flex>
            )}
          </Card>
        </Col>
      </Row>

      <Card className="premium-card" style={{ marginTop: 12 }} title={selectedJournal ? `日志详情 #${selectedJournal.journal_id}` : "日志详情"}>
        {!selectedJournal ? (
          <Alert type="info" showIcon message="请在左侧日志列表中选择一条记录后继续。" />
        ) : (
          <Tabs
            items={[
              {
                key: "manual-reflection",
                label: "手工复盘",
                children: (
                  <Flex vertical gap={10}>
                    <Paragraph style={{ margin: 0, color: "#475569" }}>
                      已选日志：<Text strong>{selectedJournal.title}</Text>
                    </Paragraph>
                    <Input.TextArea rows={4} value={reflectionInput} onChange={(e) => setReflectionInput(e.target.value)} placeholder="记录本次复盘：预期偏差、执行问题、修正计划..." />
                    <Flex gap={8}>
                      <Button type="primary" loading={reflectionLoading} onClick={() => void handleAddReflection()}>
                        提交手工复盘
                      </Button>
                      <Button loading={reflectionLoading} onClick={() => void loadReflections(selectedJournal.journal_id)}>
                        刷新复盘列表
                      </Button>
                    </Flex>
                    <Table<JournalReflection>
                      rowKey="reflection_id"
                      size="small"
                      loading={reflectionLoading}
                      dataSource={reflections}
                      pagination={{ pageSize: 5, showSizeChanger: false }}
                      columns={[
                        {
                          title: "内容",
                          dataIndex: "reflection_content",
                          key: "reflection_content",
                          render: (value: string) => <Text>{value}</Text>,
                        },
                        { title: "时间", dataIndex: "created_at", key: "created_at", width: 180 },
                      ]}
                    />
                  </Flex>
                ),
              },
              {
                key: "ai-reflection",
                label: "AI复盘",
                children: (
                  <Flex vertical gap={10}>
                    <Input value={aiFocus} onChange={(e) => setAiFocus(e.target.value)} placeholder="AI复盘重点，例如：触发条件和失效条件" />
                    <Flex gap={8}>
                      <Button type="primary" loading={aiLoading} onClick={() => void handleGenerateAIReflection()}>
                        生成 AI 复盘
                      </Button>
                      <Button loading={aiLoading} onClick={() => void loadAIReflection(selectedJournal.journal_id)}>
                        刷新 AI 结果
                      </Button>
                    </Flex>
                    {!aiReflection ? (
                      <Alert type="info" showIcon message="当前日志尚未生成 AI 复盘。" />
                    ) : (
                      <Card size="small">
                        <Flex vertical gap={8}>
                          <Flex gap={8} wrap>
                            <Tag color={aiReflection.status === "ready" ? "green" : "orange"}>{aiReflection.status}</Tag>
                            <Tag>confidence {aiReflection.confidence.toFixed(2)}</Tag>
                            <Tag>{aiReflection.generated_at}</Tag>
                          </Flex>
                          <Paragraph style={{ margin: 0 }}>{aiReflection.summary}</Paragraph>
                          <Flex vertical gap={4}>
                            <Text strong>洞察</Text>
                            {aiReflection.insights.map((item) => (
                              <Text key={`insight-${item}`}>• {item}</Text>
                            ))}
                          </Flex>
                          <Flex vertical gap={4}>
                            <Text strong>教训</Text>
                            {aiReflection.lessons.map((item) => (
                              <Text key={`lesson-${item}`}>• {item}</Text>
                            ))}
                          </Flex>
                        </Flex>
                      </Card>
                    )}
                  </Flex>
                ),
              },
            ]}
          />
        )}
      </Card>
    </main>
  );
}
