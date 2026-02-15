"use client";

import { useMemo, useState } from "react";
import { motion } from "framer-motion";
import { Alert, Button, Card, Input, List, Space, Statistic, Tag, Typography } from "antd";
import MediaCarousel from "../components/MediaCarousel";
import StockSelectorModal from "../components/StockSelectorModal";

const API_BASE = process.env.NEXT_PUBLIC_API_BASE ?? "http://127.0.0.1:8000";
const { Title, Text, Paragraph } = Typography;

type WatchItem = { stock_code: string; created_at: string };

export default function WatchlistPage() {
  const [stock, setStock] = useState("SH600000");
  const [items, setItems] = useState<WatchItem[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const latest = useMemo(() => items.slice().sort((a, b) => String(b.created_at).localeCompare(String(a.created_at)))[0], [items]);
  const heroSlides = [
    { src: "/assets/images/nyse-floor-1930.png", alt: "Historic market cycle visual", caption: "周期观察" },
    { src: "/assets/images/nyse-floor-2014.jpg", alt: "Modern market tracking visual", caption: "现代监控" },
    { src: "/assets/images/nyse-floor-1963.jpg", alt: "Archive market visual", caption: "策略样本档案" }
  ];

  async function refresh() {
    const r = await fetch(`${API_BASE}/v1/watchlist`);
    const body = await r.json();
    if (!r.ok) throw new Error(body?.detail ?? `HTTP ${r.status}`);
    setItems(body as WatchItem[]);
  }

  async function add() {
    setLoading(true);
    setError("");
    try {
      const r = await fetch(`${API_BASE}/v1/watchlist`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ stock_code: stock })
      });
      const body = await r.json();
      if (!r.ok) throw new Error(body?.detail ?? `HTTP ${r.status}`);
      await refresh();
    } catch (e) {
      setError(e instanceof Error ? e.message : "请求失败");
    } finally {
      setLoading(false);
    }
  }

  async function remove(code: string) {
    setLoading(true);
    setError("");
    try {
      const r = await fetch(`${API_BASE}/v1/watchlist/${code}`, {
        method: "DELETE"
      });
      const body = await r.json();
      if (!r.ok) throw new Error(body?.detail ?? `HTTP ${r.status}`);
      await refresh();
    } catch (e) {
      setError(e instanceof Error ? e.message : "请求失败");
    } finally {
      setLoading(false);
    }
  }

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
          <Space direction="vertical" style={{ width: "100%" }}>
            <Tag color="processing" style={{ width: "fit-content" }}>Portfolio Monitoring Hub</Tag>
            <Title level={2} style={{ margin: 0, color: "#0f172a" }}>关注池管理台</Title>
            <Paragraph style={{ margin: 0, color: "#475569", maxWidth: 760 }}>
              管理你重点跟踪的股票清单，作为后续定时抓取、预警触发和报告生成的目标集合。
            </Paragraph>
            <Space direction="vertical" style={{ width: "100%" }}>
              <StockSelectorModal
                value={stock}
                onChange={(next) => setStock(Array.isArray(next) ? (next[0] ?? "") : next)}
                title="选择关注股票"
                placeholder="请先选择要加入关注池的股票"
              />
              <Space>
              <Button type="primary" loading={loading} onClick={add}>添加</Button>
              <Button onClick={refresh}>刷新</Button>
              </Space>
            </Space>
          </Space>
        </Card>
      </motion.div>

      {error ? <Alert style={{ marginTop: 12 }} type="error" showIcon message={error} /> : null}

      <motion.div initial={{ opacity: 0, y: 14 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.08 }}>
        <Space style={{ width: "100%", marginTop: 12 }} size={12}>
          <Card className="premium-card" style={{ flex: 1 }}>
            <Statistic title="关注总数" value={items.length} valueStyle={{ color: "#0f172a" }} />
          </Card>
          <Card className="premium-card" style={{ flex: 1 }}>
            <Statistic title="最近新增" value={latest?.stock_code ?? "--"} valueStyle={{ color: "#059669" }} />
            <Text style={{ color: "#64748b" }}>{latest?.created_at ?? "--"}</Text>
          </Card>
        </Space>
      </motion.div>

      <motion.div initial={{ opacity: 0, y: 14 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.14 }}>
        <Card className="premium-card" style={{ marginTop: 12 }} title={<span style={{ color: "#0f172a" }}>关注池列表</span>}>
          <List
            bordered
            dataSource={items}
            locale={{ emptyText: "暂无关注股票" }}
            renderItem={(item) => (
              <List.Item
                actions={[
                  <Button key="delete" danger size="small" onClick={() => remove(item.stock_code)}>
                    移除
                  </Button>
                ]}
              >
                <Space>
                  <Tag color="cyan">{item.stock_code}</Tag>
                  <Text style={{ color: "#64748b" }}>{item.created_at}</Text>
                </Space>
              </List.Item>
            )}
          />
        </Card>
      </motion.div>
    </main>
  );
}

