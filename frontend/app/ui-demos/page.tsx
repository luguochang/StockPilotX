"use client";

import { useMemo, useState } from "react";
import { Button, Col, Row, Space, Tag, Typography } from "antd";
import { ArrowRightOutlined, BarChartOutlined, DashboardOutlined, ThunderboltOutlined } from "@ant-design/icons";
import styles from "./ui-demos.module.css";

const { Title, Paragraph, Text } = Typography;

type DemoId = "quant_terminal" | "editorial_pro" | "signal_lab";

const demoMeta: Record<DemoId, { name: string; slogan: string; icon: React.ReactNode }> = {
  quant_terminal: {
    name: "A. Quant Terminal",
    slogan: "高密度实时决策界面",
    icon: <DashboardOutlined />
  },
  editorial_pro: {
    name: "B. Editorial Pro",
    slogan: "轻奢内容型研究台",
    icon: <BarChartOutlined />
  },
  signal_lab: {
    name: "C. Signal Lab",
    slogan: "实验室风格策略中枢",
    icon: <ThunderboltOutlined />
  }
};

export default function UiDemosPage() {
  const [selected, setSelected] = useState<DemoId>("quant_terminal");
  const selectedMeta = useMemo(() => demoMeta[selected], [selected]);

  return (
    <main className={styles.stage}>
      <section className={styles.hero}>
        <Tag color="processing">UI Direction Sprint</Tag>
        <Title level={2} style={{ margin: 0 }}>
          选择你要的专业潮流风格
        </Title>
        <Paragraph className={styles.heroText}>
          这不是“套模板”。我给你三套完整设计语言，分别对应不同产品定位。你选一套，我下一步把全站页面统一改造。
        </Paragraph>
        <Space wrap>
          {(Object.keys(demoMeta) as DemoId[]).map((id) => (
            <Button key={id} type={selected === id ? "primary" : "default"} onClick={() => setSelected(id)}>
              {demoMeta[id].name}
            </Button>
          ))}
        </Space>
      </section>

      <section className={styles.grid}>
        <DemoA active={selected === "quant_terminal"} onPick={() => setSelected("quant_terminal")} />
        <DemoB active={selected === "editorial_pro"} onPick={() => setSelected("editorial_pro")} />
        <DemoC active={selected === "signal_lab"} onPick={() => setSelected("signal_lab")} />
      </section>

      <section className={styles.footerBar}>
        <Space size="middle" align="center">
          <Text className={styles.footerLabel}>已选方案</Text>
          <Tag color="success" icon={selectedMeta.icon}>
            {selectedMeta.name}
          </Tag>
          <Text className={styles.footerDesc}>{selectedMeta.slogan}</Text>
        </Space>
        <Button type="primary" size="large" icon={<ArrowRightOutlined />}>
          确认用这个方案重构全站
        </Button>
      </section>
    </main>
  );
}

function DemoA({ active, onPick }: { active: boolean; onPick: () => void }) {
  return (
    <article className={`${styles.card} ${styles.a} ${active ? styles.active : ""}`}>
      <header className={styles.head}>
        <div>
          <Title level={4}>A. Quant Terminal</Title>
          <Text>暗色高对比 | 交易中台感 | 指标优先</Text>
        </div>
        <Button type={active ? "primary" : "default"} onClick={onPick}>
          选这个
        </Button>
      </header>
      <Row gutter={[12, 12]}>
        <Col span={8}>
          <div className={styles.metric}>
            <Text>收益率</Text>
            <strong>+12.84%</strong>
          </div>
        </Col>
        <Col span={8}>
          <div className={styles.metric}>
            <Text>胜率</Text>
            <strong>61.2%</strong>
          </div>
        </Col>
        <Col span={8}>
          <div className={styles.metric}>
            <Text>最大回撤</Text>
            <strong>-7.9%</strong>
          </div>
        </Col>
      </Row>
      <div className={styles.chartMockA} />
      <Paragraph className={styles.tip}>适合你要的“像专业终端”的第一印象，信息密度高，适配量化/Agent联动看板。</Paragraph>
    </article>
  );
}

function DemoB({ active, onPick }: { active: boolean; onPick: () => void }) {
  return (
    <article className={`${styles.card} ${styles.b} ${active ? styles.active : ""}`}>
      <header className={styles.head}>
        <div>
          <Title level={4}>B. Editorial Pro</Title>
          <Text>明亮高级感 | 研究报告优先 | 品牌感</Text>
        </div>
        <Button type={active ? "primary" : "default"} onClick={onPick}>
          选这个
        </Button>
      </header>
      <div className={styles.timelineMock}>
        <div>08:30 财报更新</div>
        <div>09:45 盘口异动</div>
        <div>11:20 多因子评分上调</div>
      </div>
      <Row gutter={[12, 12]}>
        <Col span={12}>
          <div className={styles.infoBlock}>PM Agent: 优先保障“证据可追溯”叙事链。</div>
        </Col>
        <Col span={12}>
          <div className={styles.infoBlock}>Dev Manager: 强制展示数据时间戳与来源可靠度。</div>
        </Col>
      </Row>
      <Paragraph className={styles.tip}>适合偏“投研产品”定位，给人严肃、可信、可付费的视觉感受。</Paragraph>
    </article>
  );
}

function DemoC({ active, onPick }: { active: boolean; onPick: () => void }) {
  return (
    <article className={`${styles.card} ${styles.c} ${active ? styles.active : ""}`}>
      <header className={styles.head}>
        <div>
          <Title level={4}>C. Signal Lab</Title>
          <Text>实验室风格 | Agent协作轨迹可视化</Text>
        </div>
        <Button type={active ? "primary" : "default"} onClick={onPick}>
          选这个
        </Button>
      </header>
      <div className={styles.flow}>
        <span>Data Agent</span>
        <span>RAG Agent</span>
        <span>Risk Agent</span>
        <span>PM Agent</span>
      </div>
      <Row gutter={[12, 12]}>
        <Col span={8}>
          <div className={styles.signal}>信号强度 87</div>
        </Col>
        <Col span={8}>
          <div className={styles.signal}>一致性 0.79</div>
        </Col>
        <Col span={8}>
          <div className={styles.signal}>置信区间 ±2.1%</div>
        </Col>
      </Row>
      <Paragraph className={styles.tip}>适合强调“前沿技术”形象，把多 Agent 推理过程变成产品差异化卖点。</Paragraph>
    </article>
  );
}
