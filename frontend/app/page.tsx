"use client";

import Link from "next/link";
import { motion } from "framer-motion";
import { Button, Card, Col, Row, Space, Tag, Typography } from "antd";

const { Title, Paragraph, Text } = Typography;

const featureCards = [
  {
    title: "DeepThink 多Agent研判",
    desc: "多角色协商、轮次裁决、流式事件回放，适合复杂问题与深度推理。",
    href: "/deep-think",
    tag: "核心入口",
  },
  {
    title: "预测与因子分析",
    desc: "查看预测信号、关键因子与评测摘要，支持多标的对比观察。",
    href: "/predict",
    tag: "量化视角",
  },
  {
    title: "报告中心",
    desc: "生成、检索、回放报告版本，支持分享与导出。",
    href: "/reports",
    tag: "研究资产",
  },
  {
    title: "文档知识中心",
    desc: "上传、索引、复核文档，接入知识检索链路。",
    href: "/docs-center",
    tag: "知识管理",
  },
  {
    title: "运维与评测",
    desc: "查看调度、告警、RAG质量与Prompt版本差异。",
    href: "/ops/evals",
    tag: "治理面板",
  },
  {
    title: "关注池与监控",
    desc: "维护关注股票并快速进入日常观察流程。",
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
            A股多Agent研究系统
          </Title>
          <Paragraph style={{ margin: 0, color: "rgba(241,245,249,0.9)", maxWidth: 760 }}>
            首页只做导航与能力总览。深度分析、预测、报告、知识管理和运维面板分路由承载，避免单页面信息过载。
          </Paragraph>
          <Space>
            <Link href="/deep-think">
              <Button type="primary" size="large">
                进入 DeepThink
              </Button>
            </Link>
            <Link href="/ops/evals">
              <Button size="large">查看评测面板</Button>
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
              功能按场景拆分，减少认知负担
            </Title>
            <Paragraph style={{ margin: 0, color: "#475569" }}>
              分析链路集中在 `DeepThink` 页面，首页提供清晰入口与模块说明，便于新用户快速理解系统边界。
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
