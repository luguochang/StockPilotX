"use client";

import { Card, Space, Tag } from "antd";

type Props = {
  poolName?: string;
  stockCode?: string;
  lastRunAt?: string;
  backendStatus?: "online" | "offline" | "checking";
};

export default function ContextRibbon({ poolName = "", stockCode = "", lastRunAt = "", backendStatus = "checking" }: Props) {
  const statusColor = backendStatus === "online" ? "green" : backendStatus === "offline" ? "red" : "gold";
  const statusText = backendStatus === "online" ? "在线" : backendStatus === "offline" ? "离线" : "检测中";

  return (
    <Card className="premium-card" style={{ marginTop: 12 }}>
      <Space wrap>
        <Tag color="blue">当前池: {poolName || "-"}</Tag>
        <Tag color="cyan">当前标的: {stockCode || "-"}</Tag>
        <Tag>最近运行: {lastRunAt || "-"}</Tag>
        <Tag color={statusColor}>后端: {statusText}</Tag>
      </Space>
    </Card>
  );
}
