"use client";

import Link from "next/link";
import { motion } from "framer-motion";
import {
  BulbOutlined,
  LineChartOutlined,
  EditOutlined,
  FileTextOutlined,
  DatabaseOutlined,
  SettingOutlined,
  EyeOutlined
} from "@ant-design/icons";
import { Tag, Typography } from "antd";
import styles from "./styles/minimalist.module.css";
import "./styles/tokens.css";

const { Title, Paragraph } = Typography;

const HERO_VIDEO_URL = process.env.NEXT_PUBLIC_HERO_VIDEO_URL ?? "/assets/media/hero-stock-analysis.mp4";

const bannerItems = [
  "实时市场数据",
  "公告事件时间线",
  "多Agent研判",
  "证据可追溯引用",
  "策略预测评测"
];

const featureCards = [
  {
    title: "DeepThink 高级分析",
    desc: "多 Agent 协作、按轮推进、流式输出与冲突复核，适合复杂问题和深度研判。",
    href: "/deep-think",
    icon: <BulbOutlined />,
  },
  {
    title: "预测研究台",
    desc: "先看结论与质量门禁，再按需展开预测明细、分层聚合和解释层结果。",
    href: "/predict",
    icon: <LineChartOutlined />,
  },
  {
    title: "投资日志工作台",
    desc: "记录决策、补充复盘、生成 AI 复盘并查看聚合洞察，形成研究闭环。",
    href: "/journal",
    icon: <EditOutlined />,
  },
  {
    title: "报告中心",
    desc: "生成、检索和回放报告版本，支持业务交付与归档复盘。",
    href: "/reports",
    icon: <FileTextOutlined />,
  },
  {
    title: "RAG 运营台",
    desc: "管理来源策略、文档 chunk、共享问答语料和检索追踪，支撑知识治理。",
    href: "/rag-center",
    icon: <DatabaseOutlined />,
  },
  {
    title: "运维与评测",
    desc: "查看调度状态、告警和评测结果，对比 Prompt 版本并跟踪质量波动。",
    href: "/ops/evals",
    icon: <SettingOutlined />,
  },
  {
    title: "关注池与监控",
    desc: "维护关注标的并快速进入日常观察流程，提升重复分析效率。",
    href: "/watchlist",
    icon: <EyeOutlined />,
  },
];

export default function LandingPage() {
  return (
    <main>
      {/* Hero Video Section */}
      <motion.section
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
        className="hero-video-section"
      >
        <video className="hero-video-media" autoPlay muted loop playsInline preload="metadata">
          <source src={HERO_VIDEO_URL} type="video/mp4" />
        </video>
        <div className="hero-video-mask" />
        <div className="hero-video-content">
          <Title level={1} style={{ margin: 0, color: "#ffffff", fontSize: "72px", fontWeight: 700, letterSpacing: "-0.02em" }}>
            StockPilotX
          </Title>
          <Paragraph style={{ margin: 0, color: "rgba(255,255,255,0.86)", maxWidth: 640, fontSize: "18px", marginTop: "16px" }}>
            A股多 Agent 研究系统 · 极简主义设计
          </Paragraph>
        </div>
      </motion.section>

      {/* Scrolling Banner */}
      <motion.section
        initial={{ opacity: 0, y: 14 }}
        whileInView={{ opacity: 1, y: 0 }}
        viewport={{ once: true, amount: 0.3 }}
        transition={{ duration: 0.45 }}
        className="banner-marquee"
      >
        <div className="banner-track">
          {[...bannerItems, ...bannerItems].map((item, idx) => (
            <span key={`${item}-${idx}`} className="banner-chip">
              {item}
            </span>
          ))}
        </div>
      </motion.section>

      {/* Feature Grid */}
      <section className={styles.featureGrid}>
        {featureCards.map((card, index) => (
          <motion.div
            key={index}
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true, amount: 0.2 }}
            transition={{ duration: 0.4, delay: index * 0.05 }}
          >
            <Link href={card.href} style={{ textDecoration: "none" }}>
              <div className={styles.featureCard}>
                <div style={{ fontSize: "32px", color: "var(--color-primary)", marginBottom: "16px" }}>
                  {card.icon}
                </div>
                <h3 className={styles.featureTitle}>{card.title}</h3>
                <p className={styles.featureDesc}>{card.desc}</p>
                <span className={styles.featureButton}>进入 →</span>
              </div>
            </Link>
          </motion.div>
        ))}
      </section>

      {/* Footer Spacing */}
      <div style={{ height: "80px" }} />
    </main>
  );
}
