"use client";

import Link from "next/link";
import { motion } from "framer-motion";
import { Button, Card, Col, Row, Space, Tag, Typography } from "antd";

const { Title, Paragraph, Text } = Typography;

// 首页只做导航与模块边界说明，避免把业务分析逻辑塞在同一页面导致认知负担。
const featureCards = [
  {
    title: "DeepThink 高级分析",
    desc: "多 Agent 协作、按轮推进、流式输出与冲突复核，适合复杂问题和深度研判。",
    href: "/deep-think",
    tag: "核心入口",
  },
  {
    title: "预测与因子分析",
    desc: "查看预测信号、关键因子和评测摘要，支持多标的横向对比。",
    href: "/predict",
    tag: "量化视角",
  },
  {
    title: "Journal Workspace",
    desc: "Record decisions, add reflections, generate AI reflection, and review insights.",
    href: "/journal",
    tag: "Research Loop",
  },
  {
    title: "报告中心",
    desc: "生成、检索和回放报告版本，支持业务交付与归档复盘。",
    href: "/reports",
    tag: "研究资产",
  },
  {
    title: "文档知识中心",
    desc: "上传、索引和复核文档，管理可进入在线检索链路的语料来源。",
    href: "/docs-center",
    tag: "知识管理",
  },
  {
    title: "RAG 运营台",
    desc: "管理来源策略、文档 chunk、共享问答语料和检索追踪，支撑知识治理。",
    href: "/rag-center",
    tag: "语料治理",
  },
  {
    title: "运维与评测",
    desc: "查看调度状态、告警和评测结果，对比 Prompt 版本并跟踪质量波动。",
    href: "/ops/evals",
    tag: "系统治理",
  },
  {
    title: "关注池与监控",
    desc: "维护关注标的并快速进入日常观察流程，提升重复分析效率。",
    href: "/watchlist",
    tag: "日常操作",
  },
];

export default function LandingPage() {
  return (
    <main className="container shell-fade-in">
      <motion.section
        initial={{ opacity: 0, y: 12 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.45 }}
        className="landing-hero"
      >
        <div className="landing-hero-mask" />
        <div className="landing-hero-content">
          <Tag color="processing">StockPilotX Agent Platform</Tag>
          <Title level={1} style={{ margin: 0, color: "#f8fafc" }}>
            A股多 Agent 研究系统
          </Title>
          <Paragraph style={{ margin: 0, color: "rgba(241,245,249,0.9)", maxWidth: 760 }}>
            首页聚焦模块导航与能力总览。深度分析、预测、报告、知识治理和运维能力分路由承载，避免单页信息过载。
          </Paragraph>
          <Space>
            <Link href="/deep-think">
              <Button type="primary" size="large">
                进入 DeepThink
              </Button>
            </Link>
            <Link href="/rag-center">
              <Button size="large">进入 RAG 运营台</Button>
            </Link>
          </Space>
        </div>
      </motion.section>

      <motion.section
        initial={{ opacity: 0, y: 18 }}
        whileInView={{ opacity: 1, y: 0 }}
        viewport={{ once: true, amount: 0.2 }}
        transition={{ duration: 0.45 }}
      >
        <Card className="premium-card" style={{ marginBottom: 10 }}>
          <Space direction="vertical" size={4}>
            <Text style={{ color: "#2563eb", fontWeight: 600 }}>系统入口导航</Text>
            <Title level={3} style={{ margin: 0 }}>
              按业务场景拆分功能，减少切换成本
            </Title>
            <Paragraph style={{ margin: 0, color: "#475569" }}>
              分析链路集中在 `DeepThink` 页面；首页只提供清晰入口与模块职责说明，便于新用户快速建立正确使用路径。
            </Paragraph>
          </Space>
        </Card>
      </motion.section>

      <motion.section
        initial={{ opacity: 0, y: 18 }}
        whileInView={{ opacity: 1, y: 0 }}
        viewport={{ once: true, amount: 0.2 }}
        transition={{ duration: 0.45 }}
      >
        <Row gutter={[14, 14]}>
          {featureCards.map((item) => (
            <Col xs={24} md={12} xl={8} key={item.href}>
              <Card className="premium-card landing-card">
                <Space direction="vertical" style={{ width: "100%" }} size={10}>
                  <Tag color="blue" style={{ width: "fit-content" }}>
                    {item.tag}
                  </Tag>
                  <Title level={4} style={{ margin: 0 }}>
                    {item.title}
                  </Title>
                  <Paragraph style={{ margin: 0, color: "#475569", minHeight: 48 }}>
                    {item.desc}
                  </Paragraph>
                  <Link href={item.href}>
                    <Button type="primary">进入模块</Button>
                  </Link>
                </Space>
              </Card>
            </Col>
          ))}
        </Row>
      </motion.section>
    </main>
  );
}
