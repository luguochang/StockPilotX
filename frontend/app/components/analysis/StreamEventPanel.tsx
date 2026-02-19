"use client";

import { Card, Collapse, List, Space, Tag, Typography } from "antd";

const { Text } = Typography;

export type StreamEvent = { event: string; data: Record<string, any>; emitted_at: string };

export default function StreamEventPanel({ events }: { events: StreamEvent[] }) {
  return (
    <Card className="premium-card" title="流式事件回放" style={{ marginTop: 12 }}>
      <Collapse
        items={[
          {
            key: "events",
            label: `查看事件流 (${events.length})`,
            children: (
              <List
                size="small"
                locale={{ emptyText: "暂无事件" }}
                dataSource={events.slice().reverse()}
                renderItem={(item) => (
                  <List.Item>
                    <Space direction="vertical" size={1}>
                      <Space>
                        <Tag color="processing">{item.event}</Tag>
                        <Text style={{ color: "#64748b" }}>{item.emitted_at}</Text>
                      </Space>
                      <Text style={{ color: "#475569" }}>{JSON.stringify(item.data).slice(0, 200)}</Text>
                    </Space>
                  </List.Item>
                )}
              />
            )
          }
        ]}
      />
    </Card>
  );
}

