"use client";

import { useEffect, useMemo, useState } from "react";
import { motion } from "framer-motion";
import { Alert, Button, Card, Col, Flex, Form, Input, Progress, Row, Select, Switch, Table, Tag, Typography, message } from "antd";
import type { ColumnsType } from "antd/es/table";

const API_BASE = process.env.NEXT_PUBLIC_API_BASE ?? "http://127.0.0.1:8000";
const { Title, Paragraph, Text } = Typography;

type JournalStatus = "open" | "review_due" | "closed";
type OutcomeRating = "" | "good" | "neutral" | "bad";

type JournalItem = {
  journal_id: number;
  journal_type: string;
  title: string;
  content: string;
  stock_code: string;
  decision_type: string;
  related_research_id: string;
  related_portfolio_id: number | null;
  tags: string[];
  sentiment: string;
  source_type: string;
  source_ref_id: string;
  status: JournalStatus;
  review_due_at: string;
  executed_as_planned: boolean;
  outcome_rating: OutcomeRating;
  outcome_note: string;
  deviation_reason: string;
  closed_at: string;
  created_at: string;
  updated_at: string;
  is_overdue?: boolean;
};

type PortfolioItem = {
  portfolio_id: number;
  portfolio_name: string;
};

type TransactionItem = {
  transaction_id: number;
  stock_code: string;
  transaction_type: string;
  quantity: number;
  price: number;
  fee: number;
  amount: number;
  transaction_date: string;
};

type ExecutionBoard = {
  status: string;
  window_days: number;
  new_logs_7d: number;
  review_due_count: number;
  overdue_count: number;
  closed_count_30d: number;
  execution_rate: number;
  close_rate: number;
  top_deviation_reasons: Array<{ reason: string; count: number }>;
};

type QuickCreateForm = {
  stock_code: string;
  event_type: "buy" | "sell" | "rebalance" | "watch";
  review_days: number;
  thesis: string;
  tags: string;
};

type TransactionCreateForm = {
  portfolio_id: number;
  transaction_id: number;
  review_days: number;
};

type QueueFilterForm = {
  status: "" | JournalStatus;
  stock_code: string;
  limit: number;
};

type ListFilterForm = {
  stock_code: string;
  journal_type: string;
  limit: number;
};

type OutcomeForm = {
  executed_as_planned: boolean;
  outcome_rating: OutcomeRating;
  outcome_note: string;
  deviation_reason: string;
  close: boolean;
};

function parseCommaItems(raw: string): string[] {
  return raw
    .split(",")
    .map((item) => item.trim())
    .filter((item) => item.length > 0)
    .slice(0, 12);
}

function statusColor(status: string): string {
  if (status === "closed") return "green";
  if (status === "review_due") return "orange";
  return "blue";
}

function sourceColor(sourceType: string): string {
  if (sourceType === "transaction") return "cyan";
  if (sourceType === "deepthink") return "purple";
  return "default";
}

function friendlyError(messageText: string, fallback: string): string {
  const lower = messageText.toLowerCase();
  if (!messageText) return fallback;
  if (lower.includes("stock_code")) return "股票代码必填。";
  if (lower.includes("portfolio_id")) return "请先选择组合。";
  if (lower.includes("transaction_id")) return "请先选择交易记录。";
  if (lower.includes("outcome_rating")) return "结果评级只支持 good / neutral / bad。";
  if (lower.includes("http 404")) return "接口未找到，请确认后端已更新。";
  if (lower.includes("http 500")) return "服务暂时异常，请稍后重试。";
  return messageText;
}

async function parseOrThrow<T>(resp: Response, fallbackError: string): Promise<T> {
  const body = (await resp.json()) as T | { detail?: string; error?: string };
  if (!resp.ok) {
    const msg = typeof body === "object" && body ? ((body as { detail?: string; error?: string }).detail ?? (body as { detail?: string; error?: string }).error) : "";
    throw new Error(friendlyError(msg || `HTTP ${resp.status}`, fallbackError));
  }
  return body as T;
}

export default function JournalPage() {
  const [messageApi, contextHolder] = message.useMessage();
  const [quickForm] = Form.useForm<QuickCreateForm>();
  const [transactionForm] = Form.useForm<TransactionCreateForm>();
  const [queueFilterForm] = Form.useForm<QueueFilterForm>();
  const [listFilterForm] = Form.useForm<ListFilterForm>();
  const [outcomeForm] = Form.useForm<OutcomeForm>();

  const [quickCreating, setQuickCreating] = useState(false);
  const [transactionCreating, setTransactionCreating] = useState(false);
  const [queueLoading, setQueueLoading] = useState(false);
  const [listLoading, setListLoading] = useState(false);
  const [boardLoading, setBoardLoading] = useState(false);
  const [portfolioLoading, setPortfolioLoading] = useState(false);
  const [transactionLoading, setTransactionLoading] = useState(false);
  const [outcomeLoading, setOutcomeLoading] = useState(false);

  const [journals, setJournals] = useState<JournalItem[]>([]);
  const [reviewQueue, setReviewQueue] = useState<JournalItem[]>([]);
  const [portfolios, setPortfolios] = useState<PortfolioItem[]>([]);
  const [transactions, setTransactions] = useState<TransactionItem[]>([]);
  const [board, setBoard] = useState<ExecutionBoard | null>(null);
  const [boardWindowDays, setBoardWindowDays] = useState(90);
  const [selectedJournalId, setSelectedJournalId] = useState<number | null>(null);

  const watchedPortfolioId = Form.useWatch("portfolio_id", transactionForm) as number | undefined;

  const selectedJournal = useMemo(() => {
    if (!selectedJournalId) return null;
    const fromQueue = reviewQueue.find((item) => item.journal_id === selectedJournalId);
    if (fromQueue) return fromQueue;
    return journals.find((item) => item.journal_id === selectedJournalId) ?? null;
  }, [journals, reviewQueue, selectedJournalId]);

  const queueColumns: ColumnsType<JournalItem> = [
    {
      title: "日志",
      dataIndex: "title",
      key: "title",
      render: (_, record) => (
        <Flex vertical gap={2}>
          <Text strong>{record.title}</Text>
          <Text type="secondary" style={{ fontSize: 12 }}>
            {record.stock_code || "N/A"} | {record.created_at}
          </Text>
        </Flex>
      ),
    },
    {
      title: "状态",
      key: "status",
      width: 180,
      render: (_, record) => (
        <Flex vertical gap={4}>
          <Flex gap={6} wrap>
            <Tag color={statusColor(record.status)}>{record.status}</Tag>
            {record.is_overdue ? <Tag color="red">overdue</Tag> : null}
          </Flex>
          <Text type="secondary" style={{ fontSize: 12 }}>
            due: {record.review_due_at || "--"}
          </Text>
        </Flex>
      ),
    },
    {
      title: "来源",
      key: "source",
      width: 190,
      render: (_, record) => (
        <Flex vertical gap={2}>
          <Tag color={sourceColor(record.source_type)}>{record.source_type}</Tag>
          <Text type="secondary" style={{ fontSize: 12 }} ellipsis>
            {record.source_ref_id || "manual"}
          </Text>
        </Flex>
      ),
    },
  ];

  const listColumns: ColumnsType<JournalItem> = [
    {
      title: "日志",
      dataIndex: "title",
      key: "title",
      render: (_, record) => (
        <Flex vertical gap={2}>
          <Text strong>{record.title}</Text>
          <Text type="secondary" style={{ fontSize: 12 }}>
            {record.stock_code || "N/A"} | {record.created_at}
          </Text>
        </Flex>
      ),
    },
    {
      title: "执行",
      key: "outcome",
      width: 220,
      render: (_, record) => (
        <Flex vertical gap={4}>
          <Flex gap={6} wrap>
            <Tag color={statusColor(record.status)}>{record.status}</Tag>
            <Tag>{record.decision_type || "hold"}</Tag>
            {record.executed_as_planned ? <Tag color="green">on-plan</Tag> : null}
          </Flex>
          <Text type="secondary" style={{ fontSize: 12 }}>
            rating: {record.outcome_rating || "--"}
          </Text>
        </Flex>
      ),
    },
  ];

  async function loadPortfolios() {
    setPortfolioLoading(true);
    try {
      const resp = await fetch(`${API_BASE}/v1/portfolio`);
      const rows = await parseOrThrow<PortfolioItem[]>(resp, "加载组合失败");
      const nextRows = Array.isArray(rows) ? rows : [];
      setPortfolios(nextRows);
      const current = Number(transactionForm.getFieldValue("portfolio_id") || 0);
      if (!current && nextRows.length > 0) {
        transactionForm.setFieldValue("portfolio_id", Number(nextRows[0].portfolio_id));
      }
    } catch (error) {
      messageApi.error(error instanceof Error ? error.message : "加载组合失败");
    } finally {
      setPortfolioLoading(false);
    }
  }

  async function loadTransactions(portfolioId: number, keepValue = false) {
    setTransactionLoading(true);
    try {
      const resp = await fetch(`${API_BASE}/v1/portfolio/${portfolioId}/transactions?limit=200`);
      const rows = await parseOrThrow<TransactionItem[]>(resp, "加载交易记录失败");
      const nextRows = Array.isArray(rows) ? rows : [];
      setTransactions(nextRows);
      if (!keepValue) {
        transactionForm.setFieldValue("transaction_id", undefined);
      }
    } catch (error) {
      setTransactions([]);
      messageApi.error(error instanceof Error ? error.message : "加载交易记录失败");
    } finally {
      setTransactionLoading(false);
    }
  }

  async function loadJournals(values?: Partial<ListFilterForm>) {
    const current = { ...listFilterForm.getFieldsValue(), ...(values ?? {}) };
    const query = new URLSearchParams();
    if (current.stock_code) query.set("stock_code", current.stock_code.trim().toUpperCase());
    if (current.journal_type) query.set("journal_type", current.journal_type);
    query.set("limit", String(current.limit || 60));

    setListLoading(true);
    try {
      const resp = await fetch(`${API_BASE}/v1/journal?${query.toString()}`);
      const rows = await parseOrThrow<JournalItem[]>(resp, "加载日志列表失败");
      setJournals(Array.isArray(rows) ? rows : []);
    } catch (error) {
      messageApi.error(error instanceof Error ? error.message : "加载日志列表失败");
    } finally {
      setListLoading(false);
    }
  }

  async function loadReviewQueue(values?: Partial<QueueFilterForm>) {
    const current = { ...queueFilterForm.getFieldsValue(), ...(values ?? {}) };
    const query = new URLSearchParams();
    if (current.status) query.set("status", current.status);
    if (current.stock_code) query.set("stock_code", current.stock_code.trim().toUpperCase());
    query.set("limit", String(current.limit || 80));

    setQueueLoading(true);
    try {
      const resp = await fetch(`${API_BASE}/v1/journal/review-queue?${query.toString()}`);
      const rows = await parseOrThrow<JournalItem[]>(resp, "加载复盘队列失败");
      setReviewQueue(Array.isArray(rows) ? rows : []);
    } catch (error) {
      messageApi.error(error instanceof Error ? error.message : "加载复盘队列失败");
    } finally {
      setQueueLoading(false);
    }
  }

  async function loadExecutionBoard(windowDays: number) {
    setBoardLoading(true);
    try {
      const resp = await fetch(`${API_BASE}/v1/journal/execution-board?window_days=${windowDays}`);
      const payload = await parseOrThrow<ExecutionBoard>(resp, "加载执行看板失败");
      setBoard(payload);
    } catch (error) {
      messageApi.error(error instanceof Error ? error.message : "加载执行看板失败");
    } finally {
      setBoardLoading(false);
    }
  }

  async function refreshJournalWorkspace() {
    await Promise.all([
      loadJournals(),
      loadReviewQueue(),
      loadExecutionBoard(boardWindowDays),
    ]);
  }

  async function handleQuickCreate(values: QuickCreateForm) {
    setQuickCreating(true);
    try {
      const stockCode = values.stock_code.trim().toUpperCase();
      if (!stockCode) throw new Error("股票代码必填");
      const payload = {
        stock_code: stockCode,
        event_type: values.event_type,
        review_days: Number(values.review_days || 5),
        thesis: values.thesis?.trim() ?? "",
        tags: parseCommaItems(values.tags ?? ""),
      };
      const resp = await fetch(`${API_BASE}/v1/journal/quick`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      const created = await parseOrThrow<JournalItem>(resp, "快速创建失败");
      messageApi.success("日志已创建");
      quickForm.setFieldValue("thesis", "");
      quickForm.setFieldValue("tags", "");
      setSelectedJournalId(Number(created.journal_id));
      await refreshJournalWorkspace();
    } catch (error) {
      messageApi.error(error instanceof Error ? error.message : "快速创建失败");
    } finally {
      setQuickCreating(false);
    }
  }

  async function handleCreateFromTransaction(values: TransactionCreateForm) {
    setTransactionCreating(true);
    try {
      const payload = {
        portfolio_id: Number(values.portfolio_id),
        transaction_id: Number(values.transaction_id),
        review_days: Number(values.review_days || 5),
      };
      const resp = await fetch(`${API_BASE}/v1/journal/from-transaction`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      const created = await parseOrThrow<JournalItem & { action?: string }>(resp, "交易联动创建失败");
      const action = String(created.action || "created");
      messageApi.success(action === "reused" ? "已复用已有日志" : "已按交易生成日志");
      setSelectedJournalId(Number(created.journal_id));
      await refreshJournalWorkspace();
    } catch (error) {
      messageApi.error(error instanceof Error ? error.message : "交易联动创建失败");
    } finally {
      setTransactionCreating(false);
    }
  }

  async function handleOutcomeSubmit(values: OutcomeForm) {
    if (!selectedJournalId) {
      messageApi.warning("请先从列表或队列选择一条日志");
      return;
    }

    setOutcomeLoading(true);
    try {
      const payload = {
        executed_as_planned: Boolean(values.executed_as_planned),
        outcome_rating: String(values.outcome_rating || "").trim().toLowerCase(),
        outcome_note: String(values.outcome_note || "").trim(),
        deviation_reason: String(values.deviation_reason || "").trim(),
        close: Boolean(values.close),
      };
      const resp = await fetch(`${API_BASE}/v1/journal/${selectedJournalId}/outcome`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      const updated = await parseOrThrow<JournalItem>(resp, "回填执行结果失败");
      messageApi.success(updated.status === "closed" ? "结果已保存并关闭" : "结果已保存");
      await refreshJournalWorkspace();
    } catch (error) {
      messageApi.error(error instanceof Error ? error.message : "回填执行结果失败");
    } finally {
      setOutcomeLoading(false);
    }
  }

  useEffect(() => {
    quickForm.setFieldsValue({
      stock_code: "SH600000",
      event_type: "watch",
      review_days: 5,
      thesis: "",
      tags: "",
    });
    transactionForm.setFieldsValue({ review_days: 5 });
    queueFilterForm.setFieldsValue({ status: "", stock_code: "", limit: 80 });
    listFilterForm.setFieldsValue({ stock_code: "", journal_type: "", limit: 60 });
    void Promise.all([
      loadPortfolios(),
      loadJournals({ limit: 60 }),
      loadReviewQueue({ limit: 80 }),
      loadExecutionBoard(90),
    ]);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    if (!watchedPortfolioId) {
      setTransactions([]);
      return;
    }
    void loadTransactions(Number(watchedPortfolioId));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [watchedPortfolioId]);

  useEffect(() => {
    if (!selectedJournal) {
      outcomeForm.resetFields();
      return;
    }
    outcomeForm.setFieldsValue({
      executed_as_planned: Boolean(selectedJournal.executed_as_planned),
      outcome_rating: (selectedJournal.outcome_rating || "") as OutcomeRating,
      outcome_note: selectedJournal.outcome_note || "",
      deviation_reason: selectedJournal.deviation_reason || "",
      close: selectedJournal.status === "closed",
    });
  }, [selectedJournal, outcomeForm]);

  return (
    <main className="container">
      {contextHolder}
      <motion.section initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.35 }}>
        <Card className="premium-card" style={{ marginBottom: 12 }}>
          <Flex vertical gap={8}>
            <Tag color="processing" style={{ width: "fit-content" }}>
              Journal Execution Ledger
            </Tag>
            <Title level={2} style={{ margin: 0 }}>
              投资日志执行复盘账本
            </Title>
            <Paragraph style={{ margin: 0, color: "#475569", maxWidth: 860 }}>
              主链路聚焦执行闭环：快速创建日志、从交易自动建日志、进入复盘队列、回填执行结果、在执行看板上观察偏差和完成率。
            </Paragraph>
          </Flex>
        </Card>
      </motion.section>

      <Row gutter={[12, 12]}>
        <Col xs={24} xl={12}>
          <Card className="premium-card" title="快速创建日志（观点可选）">
            <Form<QuickCreateForm> form={quickForm} layout="vertical" onFinish={(values) => void handleQuickCreate(values)}>
              <Row gutter={10}>
                <Col xs={24} md={10}>
                  <Form.Item name="stock_code" label="股票代码" rules={[{ required: true, message: "请输入股票代码" }]}>
                    <Input placeholder="SH600000" />
                  </Form.Item>
                </Col>
                <Col xs={24} md={7}>
                  <Form.Item name="event_type" label="事件类型" rules={[{ required: true, message: "请选择事件类型" }]}>
                    <Select
                      options={[
                        { label: "买入", value: "buy" },
                        { label: "卖出", value: "sell" },
                        { label: "调仓", value: "rebalance" },
                        { label: "观察", value: "watch" },
                      ]}
                    />
                  </Form.Item>
                </Col>
                <Col xs={24} md={7}>
                  <Form.Item name="review_days" label="复盘天数" rules={[{ required: true, message: "请选择复盘天数" }]}>
                    <Select options={[3, 5, 7, 14, 30].map((day) => ({ label: `${day}天`, value: day }))} />
                  </Form.Item>
                </Col>
              </Row>
              <Form.Item name="thesis" label="一句观点（可选）">
                <Input.TextArea rows={3} placeholder="可留空，后续在结果回填时补充" />
              </Form.Item>
              <Form.Item name="tags" label="标签（可选，逗号分隔）">
                <Input placeholder="risk, earnings, breakout" />
              </Form.Item>
              <Flex justify="space-between" align="center">
                <Text type="secondary">不依赖大模型，先把执行动作落表，后续再复盘。</Text>
                <Button type="primary" htmlType="submit" loading={quickCreating}>
                  创建日志
                </Button>
              </Flex>
            </Form>
          </Card>
        </Col>

        <Col xs={24} xl={12}>
          <Card
            className="premium-card"
            title="从交易自动建日志"
            extra={
              <Button size="small" loading={portfolioLoading} onClick={() => void loadPortfolios()}>
                刷新组合
              </Button>
            }
          >
            <Form<TransactionCreateForm> form={transactionForm} layout="vertical" onFinish={(values) => void handleCreateFromTransaction(values)}>
              <Row gutter={10}>
                <Col xs={24} md={10}>
                  <Form.Item name="portfolio_id" label="组合" rules={[{ required: true, message: "请选择组合" }]}>
                    <Select
                      loading={portfolioLoading}
                      options={portfolios.map((item) => ({ label: `${item.portfolio_name} (#${item.portfolio_id})`, value: item.portfolio_id }))}
                      placeholder={portfolios.length > 0 ? "请选择组合" : "暂无组合"}
                    />
                  </Form.Item>
                </Col>
                <Col xs={24} md={9}>
                  <Form.Item name="transaction_id" label="交易记录" rules={[{ required: true, message: "请选择交易记录" }]}>
                    <Select
                      loading={transactionLoading}
                      options={transactions.map((item) => ({
                        label: `${item.stock_code} ${item.transaction_type.toUpperCase()} qty ${Number(item.quantity).toFixed(2)} @ ${Number(item.price).toFixed(2)} | ${item.transaction_date}`,
                        value: item.transaction_id,
                      }))}
                      placeholder={watchedPortfolioId ? "请选择交易" : "先选择组合"}
                      showSearch
                      optionFilterProp="label"
                    />
                  </Form.Item>
                </Col>
                <Col xs={24} md={5}>
                  <Form.Item name="review_days" label="复盘天数" rules={[{ required: true, message: "请选择" }]}>
                    <Select options={[3, 5, 7, 14, 30].map((day) => ({ label: `${day}天`, value: day }))} />
                  </Form.Item>
                </Col>
              </Row>
              <Flex justify="space-between" align="center">
                <Text type="secondary">同一组合 + 同一交易会自动幂等复用同一条日志。</Text>
                <Button type="primary" htmlType="submit" loading={transactionCreating}>
                  生成/复用日志
                </Button>
              </Flex>
            </Form>
          </Card>
        </Col>
      </Row>

      <Row gutter={[12, 12]} style={{ marginTop: 2 }}>
        <Col xs={24} xl={15}>
          <Card
            className="premium-card"
            title="复盘队列"
            extra={
              <Button size="small" loading={queueLoading} onClick={() => void loadReviewQueue()}>
                刷新队列
              </Button>
            }
          >
            <Form<QueueFilterForm> form={queueFilterForm} layout="inline" onFinish={(values) => void loadReviewQueue(values)}>
              <Form.Item name="status" label="状态">
                <Select
                  style={{ width: 150 }}
                  options={[
                    { label: "全部（open+review_due）", value: "" },
                    { label: "open", value: "open" },
                    { label: "review_due", value: "review_due" },
                    { label: "closed", value: "closed" },
                  ]}
                />
              </Form.Item>
              <Form.Item name="stock_code" label="股票">
                <Input placeholder="SH600000" style={{ width: 140 }} />
              </Form.Item>
              <Form.Item name="limit" label="条数">
                <Select style={{ width: 100 }} options={[40, 80, 120, 200].map((n) => ({ label: String(n), value: n }))} />
              </Form.Item>
              <Form.Item>
                <Button htmlType="submit" loading={queueLoading}>
                  查询
                </Button>
              </Form.Item>
            </Form>

            <Table<JournalItem>
              style={{ marginTop: 12 }}
              rowKey="journal_id"
              size="small"
              loading={queueLoading}
              dataSource={reviewQueue}
              columns={queueColumns}
              pagination={{ pageSize: 8, showSizeChanger: false }}
              rowClassName={(record) => (record.journal_id === selectedJournalId ? "ant-table-row-selected" : "")}
              onRow={(record) => ({ onClick: () => setSelectedJournalId(record.journal_id) })}
            />
          </Card>
        </Col>

        <Col xs={24} xl={9}>
          <Card
            className="premium-card"
            title="执行看板"
            extra={
              <Flex gap={8}>
                <Select
                  size="small"
                  value={boardWindowDays}
                  onChange={(value) => {
                    setBoardWindowDays(value);
                    void loadExecutionBoard(value);
                  }}
                  options={[30, 60, 90, 180, 365].map((d) => ({ label: `${d}天`, value: d }))}
                />
                <Button size="small" loading={boardLoading} onClick={() => void loadExecutionBoard(boardWindowDays)}>
                  刷新
                </Button>
              </Flex>
            }
          >
            {!board ? (
              <Alert type="info" showIcon message="暂无看板数据" />
            ) : (
              <Flex vertical gap={10}>
                <Row gutter={8}>
                  <Col span={12}>
                    <Card size="small">
                      <Text type="secondary">近7天新增</Text>
                      <Title level={4} style={{ margin: 0 }}>{board.new_logs_7d}</Title>
                    </Card>
                  </Col>
                  <Col span={12}>
                    <Card size="small">
                      <Text type="secondary">待复盘</Text>
                      <Title level={4} style={{ margin: 0 }}>{board.review_due_count}</Title>
                    </Card>
                  </Col>
                </Row>
                <Row gutter={8}>
                  <Col span={12}>
                    <Card size="small">
                      <Text type="secondary">超期</Text>
                      <Title level={4} style={{ margin: 0 }}>{board.overdue_count}</Title>
                    </Card>
                  </Col>
                  <Col span={12}>
                    <Card size="small">
                      <Text type="secondary">已关闭</Text>
                      <Title level={4} style={{ margin: 0 }}>{board.closed_count_30d}</Title>
                    </Card>
                  </Col>
                </Row>
                <Card size="small">
                  <Flex vertical gap={6}>
                    <Text>按计划执行率 {(board.execution_rate * 100).toFixed(1)}%</Text>
                    <Progress percent={Math.round(board.execution_rate * 100)} size="small" />
                    <Text>复盘闭环率 {(board.close_rate * 100).toFixed(1)}%</Text>
                    <Progress percent={Math.round(board.close_rate * 100)} size="small" status="active" />
                  </Flex>
                </Card>
                <Flex vertical gap={6}>
                  <Text strong>主要偏差原因</Text>
                  {board.top_deviation_reasons.length === 0 ? (
                    <Text type="secondary">暂无偏差原因记录</Text>
                  ) : (
                    <Flex gap={6} wrap>
                      {board.top_deviation_reasons.map((item) => (
                        <Tag key={`${item.reason}-${item.count}`} color="orange">
                          {item.reason} | {item.count}
                        </Tag>
                      ))}
                    </Flex>
                  )}
                </Flex>
              </Flex>
            )}
          </Card>
        </Col>
      </Row>

      <Row gutter={[12, 12]} style={{ marginTop: 2 }}>
        <Col xs={24} xl={14}>
          <Card
            className="premium-card"
            title="日志列表"
            extra={
              <Button size="small" loading={listLoading} onClick={() => void loadJournals()}>
                刷新日志
              </Button>
            }
          >
            <Form<ListFilterForm> form={listFilterForm} layout="inline" onFinish={(values) => void loadJournals(values)}>
              <Form.Item name="stock_code" label="股票">
                <Input placeholder="SH600000" style={{ width: 140 }} />
              </Form.Item>
              <Form.Item name="journal_type" label="类型">
                <Select
                  style={{ width: 150 }}
                  options={[
                    { label: "全部", value: "" },
                    { label: "decision", value: "decision" },
                    { label: "reflection", value: "reflection" },
                    { label: "learning", value: "learning" },
                  ]}
                />
              </Form.Item>
              <Form.Item name="limit" label="条数">
                <Select style={{ width: 100 }} options={[20, 40, 60, 100, 200].map((n) => ({ label: String(n), value: n }))} />
              </Form.Item>
              <Form.Item>
                <Button htmlType="submit" loading={listLoading}>
                  查询
                </Button>
              </Form.Item>
            </Form>

            <Table<JournalItem>
              style={{ marginTop: 12 }}
              rowKey="journal_id"
              size="small"
              loading={listLoading}
              dataSource={journals}
              columns={listColumns}
              pagination={{ pageSize: 8, showSizeChanger: false }}
              rowClassName={(record) => (record.journal_id === selectedJournalId ? "ant-table-row-selected" : "")}
              onRow={(record) => ({ onClick: () => setSelectedJournalId(record.journal_id) })}
            />
          </Card>
        </Col>

        <Col xs={24} xl={10}>
          <Card className="premium-card" title={selectedJournal ? `结果回填 #${selectedJournal.journal_id}` : "结果回填"}>
            {!selectedJournal ? (
              <Alert type="info" showIcon message="请先在复盘队列或日志列表选择一条记录" />
            ) : (
              <Flex vertical gap={10}>
                <Card size="small">
                  <Flex vertical gap={6}>
                    <Paragraph style={{ margin: 0 }}>
                      <Text strong>{selectedJournal.title}</Text>
                    </Paragraph>
                    <Flex gap={6} wrap>
                      <Tag>{selectedJournal.stock_code || "N/A"}</Tag>
                      <Tag color={statusColor(selectedJournal.status)}>{selectedJournal.status}</Tag>
                      <Tag color={sourceColor(selectedJournal.source_type)}>{selectedJournal.source_type}</Tag>
                      <Tag>{selectedJournal.decision_type || "hold"}</Tag>
                    </Flex>
                    <Text type="secondary">review_due_at: {selectedJournal.review_due_at || "--"}</Text>
                  </Flex>
                </Card>

                <Form<OutcomeForm> form={outcomeForm} layout="vertical" onFinish={(values) => void handleOutcomeSubmit(values)}>
                  <Row gutter={10}>
                    <Col xs={24} md={12}>
                      <Form.Item name="executed_as_planned" label="是否按计划执行" valuePropName="checked">
                        <Switch checkedChildren="是" unCheckedChildren="否" />
                      </Form.Item>
                    </Col>
                    <Col xs={24} md={12}>
                      <Form.Item name="outcome_rating" label="结果评级">
                        <Select
                          options={[
                            { label: "未评级", value: "" },
                            { label: "good", value: "good" },
                            { label: "neutral", value: "neutral" },
                            { label: "bad", value: "bad" },
                          ]}
                        />
                      </Form.Item>
                    </Col>
                  </Row>
                  <Form.Item name="deviation_reason" label="偏差原因（可选）">
                    <Input placeholder="例如：入场时点过早 / 仓位过重 / 风险响应滞后" />
                  </Form.Item>
                  <Form.Item name="outcome_note" label="复盘结论（可选）">
                    <Input.TextArea rows={4} placeholder="记录实际执行、偏差、下次改进动作" />
                  </Form.Item>
                  <Form.Item name="close" label="提交后关闭该日志" valuePropName="checked">
                    <Switch checkedChildren="关闭" unCheckedChildren="保持开启" />
                  </Form.Item>
                  <Flex justify="space-between" align="center">
                    <Button onClick={() => outcomeForm.resetFields()}>重置</Button>
                    <Button type="primary" htmlType="submit" loading={outcomeLoading}>
                      保存结果
                    </Button>
                  </Flex>
                </Form>
              </Flex>
            )}
          </Card>
        </Col>
      </Row>
    </main>
  );
}
