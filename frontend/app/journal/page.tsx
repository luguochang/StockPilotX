"use client";

import { useEffect, useMemo, useState } from "react";
import { motion } from "framer-motion";
import { Alert, Button, Card, Col, Collapse, Flex, Form, Input, Row, Select, Table, Tabs, Tag, Typography, message } from "antd";
import type { ColumnsType } from "antd/es/table";

const API_BASE = process.env.NEXT_PUBLIC_API_BASE ?? "http://127.0.0.1:8000";
const { Title, Paragraph, Text } = Typography;

type JournalType = "decision" | "reflection" | "learning";
type TemplateId = "decision" | "risk" | "review";

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
  reflection_count?: number;
  has_ai_reflection?: boolean;
};

type ReflectionItem = {
  reflection_id: number;
  reflection_content: string;
  created_at: string;
};

type AiReflection = {
  status: string;
  summary: string;
  insights: string[];
  lessons: string[];
  confidence: number;
  generated_at: string;
};

type Insights = {
  total_journals: number;
  stock_activity: Array<{ key: string; count: number }>;
  keyword_profile: Array<{ keyword: string; count: number }>;
  reflection_coverage: { reflection_coverage_rate: number; ai_reflection_coverage_rate: number };
};

type CreateForm = {
  template_id: TemplateId;
  stock_code: string;
  thesis: string;
  custom_title: string;
  custom_tags: string;
  journal_type: JournalType;
  decision_type: "buy" | "hold" | "reduce";
  sentiment: "positive" | "neutral" | "negative";
};

type FilterForm = {
  journal_type: string;
  stock_code: string;
  limit: number;
};

const TEMPLATES: Record<TemplateId, { label: string; hint: string; journal_type: JournalType; decision_type: "buy" | "hold" | "reduce"; sentiment: "positive" | "neutral" | "negative"; tags: string[]; ai_focus: string; title_prefix: string }> = {
  decision: {
    label: "交易决策简报",
    hint: "写核心观点、触发条件、失效条件",
    journal_type: "decision",
    decision_type: "hold",
    sentiment: "neutral",
    tags: ["decision", "thesis"],
    ai_focus: "优先检查触发和失效条件是否可验证",
    title_prefix: "交易决策",
  },
  risk: {
    label: "风险防守记录",
    hint: "写风险来源、风控阈值和执行动作",
    journal_type: "reflection",
    decision_type: "reduce",
    sentiment: "negative",
    tags: ["risk", "drawdown"],
    ai_focus: "优先检查风险识别盲区和风控阈值",
    title_prefix: "风险防守",
  },
  review: {
    label: "复盘改进记录",
    hint: "写结果偏差、原因和下次改进动作",
    journal_type: "learning",
    decision_type: "hold",
    sentiment: "neutral",
    tags: ["review", "improvement"],
    ai_focus: "优先输出可执行改进清单",
    title_prefix: "交易复盘",
  },
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
  const [createForm] = Form.useForm<CreateForm>();
  const [filterForm] = Form.useForm<FilterForm>();
  const [advancedOpen, setAdvancedOpen] = useState<string[]>([]);
  const [listLoading, setListLoading] = useState(false);
  const [createLoading, setCreateLoading] = useState(false);
  const [reflectionLoading, setReflectionLoading] = useState(false);
  const [aiLoading, setAiLoading] = useState(false);
  const [insightsLoading, setInsightsLoading] = useState(false);
  const [journals, setJournals] = useState<JournalItem[]>([]);
  const [selectedJournalId, setSelectedJournalId] = useState<number | null>(null);
  const [reflections, setReflections] = useState<ReflectionItem[]>([]);
  const [reflectionInput, setReflectionInput] = useState("");
  const [aiFocus, setAiFocus] = useState("");
  const [aiReflection, setAiReflection] = useState<AiReflection | null>(null);
  const [insights, setInsights] = useState<Insights | null>(null);

  const selectedTemplateId = Form.useWatch("template_id", createForm) as TemplateId | undefined;
  const selectedTemplate = useMemo(() => TEMPLATES[selectedTemplateId ?? "decision"], [selectedTemplateId]);
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
            {record.stock_code || "未绑定标的"} | {record.created_at}
          </Text>
        </Flex>
      ),
    },
    {
      title: "类型",
      key: "type",
      width: 150,
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
          <Text style={{ fontSize: 12 }}>手工: {Number(record.reflection_count ?? 0)}</Text>
          <Text style={{ fontSize: 12 }}>AI: {record.has_ai_reflection ? "yes" : "no"}</Text>
        </Flex>
      ),
    },
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
      if (selectedJournalId && !body.find((item) => item.journal_id === selectedJournalId)) setSelectedJournalId(null);
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
      const body = await parseOrThrow<ReflectionItem[]>(resp);
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
      const body = await parseOrThrow<AiReflection | Record<string, never>>(resp);
      if (body && typeof body === "object" && "status" in body) setAiReflection(body as AiReflection);
      else setAiReflection(null);
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
      const body = await parseOrThrow<Insights>(resp);
      setInsights(body);
    } catch (error) {
      messageApi.error(error instanceof Error ? error.message : "加载洞察失败");
    } finally {
      setInsightsLoading(false);
    }
  }

  async function handleCreate(values: CreateForm) {
    setCreateLoading(true);
    try {
      const tpl = TEMPLATES[values.template_id];
      const stockCode = values.stock_code.trim().toUpperCase();
      const tags = Array.from(new Set([...tpl.tags, ...parseCommaItems(values.custom_tags || "")])).slice(0, 12);
      // Template-first payload: default flow only requires template + stock + thesis.
      const payload = {
        journal_type: values.journal_type || tpl.journal_type,
        title: values.custom_title.trim() || `${tpl.title_prefix} ${stockCode}`,
        stock_code: stockCode,
        decision_type: values.decision_type || tpl.decision_type,
        tags,
        sentiment: values.sentiment || tpl.sentiment,
        content: [`模板: ${tpl.label}`, `核心观点: ${values.thesis.trim()}`, "触发条件: （可补充）", "失效条件: （可补充）", "执行计划: （可补充）"].join("\n"),
      };
      const resp = await fetch(`${API_BASE}/v1/journal`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      const created = await parseOrThrow<JournalItem>(resp);
      messageApi.success("日志创建成功");
      createForm.setFieldValue("thesis", "");
      createForm.setFieldValue("custom_title", "");
      createForm.setFieldValue("custom_tags", "");
      setAiFocus(tpl.ai_focus);
      await Promise.all([loadJournals(), loadInsights()]);
      setSelectedJournalId(Number(created.journal_id));
    } catch (error) {
      messageApi.error(error instanceof Error ? error.message : "创建日志失败");
    } finally {
      setCreateLoading(false);
    }
  }

  async function handleAddReflection() {
    if (!selectedJournalId) return void messageApi.warning("请先选择一条日志");
    if (!reflectionInput.trim()) return void messageApi.warning("请输入复盘内容");
    setReflectionLoading(true);
    try {
      const resp = await fetch(`${API_BASE}/v1/journal/${selectedJournalId}/reflections`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ reflection_content: reflectionInput.trim() }),
      });
      await parseOrThrow<ReflectionItem>(resp);
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
    if (!selectedJournalId) return void messageApi.warning("请先选择一条日志");
    const focus = aiFocus.trim() || selectedTemplate.ai_focus;
    setAiLoading(true);
    try {
      const resp = await fetch(`${API_BASE}/v1/journal/${selectedJournalId}/ai-reflection/generate`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ focus }),
      });
      const generated = await parseOrThrow<AiReflection>(resp);
      setAiReflection(generated);
      messageApi.success(generated.status === "ready" ? "AI复盘生成完成" : "AI复盘回退到本地模板");
      await Promise.all([loadJournals(), loadInsights()]);
    } catch (error) {
      messageApi.error(error instanceof Error ? error.message : "生成 AI 复盘失败");
    } finally {
      setAiLoading(false);
    }
  }

  useEffect(() => {
    createForm.setFieldsValue({
      template_id: "decision",
      stock_code: "SH600000",
      thesis: "",
      custom_title: "",
      custom_tags: "",
      journal_type: "decision",
      decision_type: "hold",
      sentiment: "neutral",
    });
    filterForm.setFieldsValue({ journal_type: "", stock_code: "", limit: 40 });
    setAiFocus(TEMPLATES.decision.ai_focus);
    void Promise.all([loadJournals({ limit: 40 }), loadInsights()]);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    createForm.setFieldsValue({
      journal_type: selectedTemplate.journal_type,
      decision_type: selectedTemplate.decision_type,
      sentiment: selectedTemplate.sentiment,
    });
    setAiFocus(selectedTemplate.ai_focus);
  }, [createForm, selectedTemplate]);

  useEffect(() => {
    if (!selectedJournalId) {
      setReflections([]);
      setAiReflection(null);
      return;
    }
    void Promise.all([loadReflections(selectedJournalId), loadAIReflection(selectedJournalId)]);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedJournalId]);

  return (
    <main className="container">
      {contextHolder}
      <motion.section initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.4 }}>
        <Card className="premium-card" style={{ marginBottom: 12 }}>
          <Flex vertical gap={10}>
            <Tag color="processing" style={{ width: "fit-content" }}>Investment Journal Workspace</Tag>
            <Title level={2} style={{ margin: 0 }}>投资日志工作台</Title>
            <Paragraph style={{ margin: 0, color: "#475569", maxWidth: 820 }}>
              默认只保留三项输入: 模板、股票、核心观点。其余参数进入可选高级设置，降低输入负担。
            </Paragraph>
          </Flex>
        </Card>
      </motion.section>

      <Row gutter={[12, 12]}>
        <Col xs={24} xl={14}>
          <Card className="premium-card" title="快速记录（模板优先）" style={{ marginBottom: 12 }}>
            <Form<CreateForm> layout="vertical" form={createForm} onFinish={handleCreate}>
              <Row gutter={10}>
                <Col xs={24} md={12}>
                  <Form.Item name="template_id" label="模板" rules={[{ required: true, message: "请选择模板" }]}>
                    <Select options={Object.entries(TEMPLATES).map(([key, val]) => ({ value: key, label: `${val.label} - ${val.hint}` }))} />
                  </Form.Item>
                </Col>
                <Col xs={24} md={12}>
                  <Form.Item name="stock_code" label="股票代码" rules={[{ required: true, message: "请输入股票代码" }]}>
                    <Input placeholder="例如 SH600000" />
                  </Form.Item>
                </Col>
              </Row>
              <Form.Item name="thesis" label="核心观点" rules={[{ required: true, message: "请输入观点" }]}>
                <Input.TextArea rows={5} placeholder={selectedTemplate.hint} />
              </Form.Item>
              <Collapse
                size="small"
                activeKey={advancedOpen}
                onChange={(keys) => setAdvancedOpen((Array.isArray(keys) ? keys : [keys]).map(String))}
                items={[{ key: "advanced", label: "高级设置（可选）", children: (
                  <Flex vertical gap={8}>
                    <Form.Item name="custom_title" label="自定义标题"><Input placeholder="可留空，系统自动生成标题" /></Form.Item>
                    <Form.Item name="custom_tags" label="补充标签（逗号分隔）"><Input placeholder="估值,风险,仓位" /></Form.Item>
                    <Row gutter={10}>
                      <Col xs={24} md={8}>
                        <Form.Item name="journal_type" label="日志类型">
                          <Select options={[{ label: "决策", value: "decision" }, { label: "复盘", value: "reflection" }, { label: "学习", value: "learning" }]} />
                        </Form.Item>
                      </Col>
                      <Col xs={24} md={8}>
                        <Form.Item name="decision_type" label="决策方向">
                          <Select options={[{ label: "买入", value: "buy" }, { label: "持有", value: "hold" }, { label: "减仓", value: "reduce" }]} />
                        </Form.Item>
                      </Col>
                      <Col xs={24} md={8}>
                        <Form.Item name="sentiment" label="情绪标签">
                          <Select options={[{ label: "积极", value: "positive" }, { label: "中性", value: "neutral" }, { label: "谨慎", value: "negative" }]} />
                        </Form.Item>
                      </Col>
                    </Row>
                  </Flex>
                ) }]}
              />
              <Button type="primary" htmlType="submit" loading={createLoading} style={{ marginTop: 12 }}>创建日志</Button>
            </Form>
          </Card>

          <Card className="premium-card" title="日志列表">
            <Form<FilterForm> form={filterForm} layout="inline" onFinish={(values) => void loadJournals(values)}>
              <Form.Item name="journal_type" label="类型">
                <Select style={{ width: 140 }} options={[{ label: "全部", value: "" }, { label: "决策", value: "decision" }, { label: "复盘", value: "reflection" }, { label: "学习", value: "learning" }]} />
              </Form.Item>
              <Form.Item name="stock_code" label="股票"><Input placeholder="SH600000" /></Form.Item>
              <Form.Item name="limit" label="条数"><Select style={{ width: 100 }} options={[{ label: "20", value: 20 }, { label: "40", value: 40 }, { label: "80", value: 80 }]} /></Form.Item>
              <Form.Item><Button htmlType="submit" loading={listLoading}>查询</Button></Form.Item>
            </Form>
            <Table<JournalItem>
              style={{ marginTop: 12 }}
              rowKey="journal_id"
              loading={listLoading}
              dataSource={journals}
              columns={journalColumns}
              pagination={{ pageSize: 8, showSizeChanger: false }}
              rowClassName={(record) => (record.journal_id === selectedJournalId ? "ant-table-row-selected" : "")}
              onRow={(record) => ({ onClick: () => setSelectedJournalId(record.journal_id) })}
            />
          </Card>
        </Col>

        <Col xs={24} xl={10}>
          <Card className="premium-card" title="Journal 洞察看板" extra={<Button size="small" loading={insightsLoading} onClick={() => void loadInsights()}>刷新</Button>}>
            {!insights ? (
              <Alert type="info" showIcon message="暂无洞察数据，请先创建日志。" />
            ) : (
              <Flex vertical gap={10}>
                <Row gutter={10}>
                  <Col span={8}><Card size="small"><Text type="secondary">日志总数</Text><Title level={4} style={{ margin: 0 }}>{insights.total_journals}</Title></Card></Col>
                  <Col span={8}><Card size="small"><Text type="secondary">手工覆盖</Text><Title level={4} style={{ margin: 0 }}>{(insights.reflection_coverage.reflection_coverage_rate * 100).toFixed(1)}%</Title></Card></Col>
                  <Col span={8}><Card size="small"><Text type="secondary">AI覆盖</Text><Title level={4} style={{ margin: 0 }}>{(insights.reflection_coverage.ai_reflection_coverage_rate * 100).toFixed(1)}%</Title></Card></Col>
                </Row>
                <Flex gap={6} wrap>{insights.stock_activity.slice(0, 6).map((item) => <Tag key={item.key}>{item.key} | {item.count}</Tag>)}</Flex>
                <Flex gap={6} wrap>{insights.keyword_profile.slice(0, 8).map((item) => <Tag key={item.keyword} color="blue">{item.keyword} | {item.count}</Tag>)}</Flex>
              </Flex>
            )}
          </Card>
        </Col>
      </Row>

      <Card className="premium-card" style={{ marginTop: 12 }} title={selectedJournal ? `日志详情 #${selectedJournal.journal_id}` : "日志详情"}>
        {!selectedJournal ? (
          <Alert type="info" showIcon message="请先在日志列表选择一条记录。" />
        ) : (
          <Tabs
            items={[
              {
                key: "manual",
                label: "手工复盘",
                children: (
                  <Flex vertical gap={10}>
                    <Paragraph style={{ margin: 0, color: "#475569" }}>已选日志: <Text strong>{selectedJournal.title}</Text></Paragraph>
                    <Input.TextArea rows={4} value={reflectionInput} onChange={(e) => setReflectionInput(e.target.value)} placeholder="记录偏差原因、执行问题、修正动作..." />
                    <Flex gap={8}>
                      <Button type="primary" loading={reflectionLoading} onClick={() => void handleAddReflection()}>提交手工复盘</Button>
                      <Button loading={reflectionLoading} onClick={() => void loadReflections(selectedJournal.journal_id)}>刷新复盘列表</Button>
                    </Flex>
                    <Table<ReflectionItem>
                      rowKey="reflection_id"
                      size="small"
                      loading={reflectionLoading}
                      dataSource={reflections}
                      pagination={{ pageSize: 5, showSizeChanger: false }}
                      columns={[{ title: "内容", dataIndex: "reflection_content", key: "reflection_content", render: (value: string) => <Text>{value}</Text> }, { title: "时间", dataIndex: "created_at", key: "created_at", width: 180 }]}
                    />
                  </Flex>
                ),
              },
              {
                key: "ai",
                label: "AI复盘",
                children: (
                  <Flex vertical gap={10}>
                    <Input value={aiFocus} onChange={(e) => setAiFocus(e.target.value)} placeholder="AI复盘重点（可自定义）" />
                    <Flex gap={8}>
                      <Button type="primary" loading={aiLoading} onClick={() => void handleGenerateAIReflection()}>生成 AI 复盘</Button>
                      <Button loading={aiLoading} onClick={() => void loadAIReflection(selectedJournal.journal_id)}>刷新 AI 结果</Button>
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
                          <Flex vertical gap={4}><Text strong>洞察</Text>{aiReflection.insights.map((item) => <Text key={`insight-${item}`}>- {item}</Text>)}</Flex>
                          <Flex vertical gap={4}><Text strong>改进动作</Text>{aiReflection.lessons.map((item) => <Text key={`lesson-${item}`}>- {item}</Text>)}</Flex>
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

