"use client";

import { useEffect } from "react";
import { motion } from "framer-motion";
import { Button, Card, Empty, List, Select, Skeleton, Space, Statistic, Tag, Typography, message } from "antd";
import MediaCarousel from "../components/MediaCarousel";
import StockSelectorModal from "../components/StockSelectorModal";
import ContextRibbon from "../components/ContextRibbon";
import ModuleGuideBanner from "../components/ModuleGuideBanner";
import ModuleWorkflow from "../components/ModuleWorkflow";
import { useBackendHealth } from "../hooks/useBackendHealth";
import { useWatchlist } from "../hooks/useWatchlist";

const { Title, Text, Paragraph } = Typography;

export default function WatchlistPage() {
  const [messageApi, contextHolder] = message.useMessage();
  const backendStatus = useBackendHealth();
  const {
    stock,
    setStock,
    items,
    pools,
    selectedPoolId,
    setSelectedPoolId,
    poolStocks,
    loading,
    initialLoading,
    error,
    ctxLastRunAt,
    latest,
    createPool,
    addStock,
    removeFromPool,
    refreshAll,
    refresh,
    refreshPools
  } = useWatchlist();

  useEffect(() => {
    if (!error) return;
    messageApi.error(error);
  }, [error, messageApi]);

  const heroSlides = [
    { src: "/assets/images/nyse-floor-1930.png", alt: "Historic market cycle visual", caption: "周期观察" },
    { src: "/assets/images/nyse-floor-2014.jpg", alt: "Modern market tracking visual", caption: "现代监控" },
    { src: "/assets/images/nyse-floor-1963.jpg", alt: "Archive market visual", caption: "策略样本档案" }
  ];
  const workflowStep = !selectedPoolId ? 0 : poolStocks.length === 0 ? 1 : 2;

  return (
    <main className="container">
      {contextHolder}
      <motion.section initial={{ opacity: 0, y: 12 }} whileInView={{ opacity: 1, y: 0 }} viewport={{ once: true, amount: 0.2 }} transition={{ duration: 0.45 }}>
        <MediaCarousel items={heroSlides} />
      </motion.section>

      <ContextRibbon
        poolName={pools.find((p) => p.pool_id === selectedPoolId)?.pool_name ?? ""}
        stockCode={stock}
        lastRunAt={ctxLastRunAt}
        backendStatus={backendStatus}
      />

      <ModuleGuideBanner
        moduleKey="watchlist"
        title="关注池模块怎么用"
        steps={["先选或新建一个关注池", "点击搜索选择股票", "点击加入关注池完成入池"]}
      />
      <ModuleWorkflow
        title="关注池操作流程"
        items={["选择池", "选择股票", "完成入池"]}
        current={workflowStep}
        hint="按步骤完成后，池内持仓会自动更新"
      />

      <motion.div initial={{ opacity: 0, y: 14 }} animate={{ opacity: 1, y: 0 }}>
        <Card className="premium-card" style={{ background: "linear-gradient(132deg, rgba(255,255,255,0.96), rgba(246,249,252,0.94))" }}>
          <Space direction="vertical" style={{ width: "100%" }}>
            <Tag color="processing" style={{ width: "fit-content" }}>Portfolio Monitoring Hub</Tag>
            <Title level={2} style={{ margin: 0, color: "#0f172a" }}>关注池</Title>
            <Paragraph style={{ margin: 0, color: "#475569", maxWidth: 760 }}>
              只保留必要动作: 选池、选股、加入。避免让用户面对过多输入框。
            </Paragraph>
            <Space.Compact block>
              <Select
                value={selectedPoolId || undefined}
                onChange={(v) => setSelectedPoolId(String(v ?? ""))}
                options={pools.map((p) => ({ label: `${p.pool_name} (${p.stock_count})`, value: p.pool_id }))}
                placeholder="选择关注池"
                style={{ minWidth: 260 }}
                showSearch
                optionFilterProp="label"
                filterOption={(input, option) => String(option?.label ?? "").toLowerCase().includes(input.toLowerCase())}
              />
              <Button onClick={createPool} loading={loading}>新建池</Button>
              <Button onClick={() => { void refresh(); void refreshPools(); }} loading={loading}>重试</Button>
            </Space.Compact>
            <StockSelectorModal value={stock} onChange={(next) => setStock(Array.isArray(next) ? (next[0] ?? "") : next)} title="选择关注股票" placeholder="请选择股票" />
            <Button type="primary" loading={loading} onClick={addStock}>加入关注池</Button>
          </Space>
        </Card>
      </motion.div>

      {initialLoading ? (
        <Card className="premium-card" style={{ marginTop: 12 }}>
          <Skeleton active paragraph={{ rows: 5 }} />
        </Card>
      ) : (
        <>
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
            <Card className="premium-card" style={{ marginTop: 12 }} title={<span style={{ color: "#0f172a" }}>池内持仓</span>}>
              <List
                bordered
                dataSource={poolStocks}
                locale={{ emptyText: <Empty description="当前关注池暂无股票" /> }}
                loading={loading}
                pagination={{ pageSize: 20, showSizeChanger: true, pageSizeOptions: ['10', '20', '50', '100'] }}
                renderItem={(item) => (
                  <List.Item
                    actions={[
                      <Button key="remove-pool" danger size="small" onClick={() => removeFromPool(item.stock_code)}>
                        移除
                      </Button>
                    ]}
                  >
                    <Space wrap>
                      <Tag color="geekblue">{item.stock_code}</Tag>
                      {item.stock_name ? <Text>{item.stock_name}</Text> : null}
                      {item.exchange ? <Tag>{item.exchange}</Tag> : null}
                      {item.market_tier ? <Tag color="cyan">{item.market_tier}</Tag> : null}
                      {item.industry_l1 ? <Tag color="purple">{item.industry_l1}</Tag> : null}
                    </Space>
                  </List.Item>
                )}
              />
            </Card>
          </motion.div>
        </>
      )}

      {!initialLoading && !poolStocks.length && !error ? (
        <div style={{ marginTop: 12 }}>
          <Button onClick={() => void refreshAll()}>刷新数据</Button>
        </div>
      ) : null}
    </main>
  );
}
