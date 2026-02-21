"use client";

import { Card, Steps, Typography } from "antd";

const { Text } = Typography;

type Props = {
  title: string;
  items: string[];
  current: number;
  hint?: string;
};

export default function ModuleWorkflow({ title, items, current, hint = "" }: Props) {
  const safeCurrent = Math.max(0, Math.min(items.length - 1, current));

  return (
    <Card className="premium-card" style={{ marginTop: 12 }}>
      <Text strong style={{ color: "#0f172a" }}>{title}</Text>
      {hint ? <Text style={{ color: "#64748b", marginLeft: 10 }}>{hint}</Text> : null}
      <div style={{ marginTop: 10 }}>
        <Steps
          size="small"
          current={safeCurrent}
          responsive
          items={items.map((label) => ({ title: label }))}
        />
      </div>
    </Card>
  );
}
