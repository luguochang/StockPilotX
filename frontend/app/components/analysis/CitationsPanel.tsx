"use client";

import { Card, List, Space, Tag, Typography } from "antd";

const { Text } = Typography;

export type Citation = { source_id: string; source_url: string; excerpt: string };

export default function CitationsPanel({ citations }: { citations: Citation[] }) {
  return (
    <Card className="premium-card" title="引用来源与证据" style={{ marginTop: 12 }}>
      <List
        dataSource={citations}
        locale={{ emptyText: "本轮暂无引用" }}
        renderItem={(item) => {
          const sourceId = String(item.source_id ?? "");
          const hitType = sourceId.includes("report:") ? "历史问答摘要" : "原始来源";
          return (
            <List.Item>
              <Space direction="vertical" size={1}>
                <Space wrap>
                  <Tag color={hitType === "历史问答摘要" ? "purple" : "cyan"}>{hitType}</Tag>
                  <Text style={{ color: "#0f172a" }}>{sourceId || "-"}</Text>
                </Space>
                <Text style={{ color: "#64748b" }}>{String(item.excerpt ?? "").slice(0, 180)}</Text>
                {item.source_url ? (
                  <a href={item.source_url} target="_blank" rel="noreferrer" style={{ color: "#2563eb" }}>
                    {item.source_url}
                  </a>
                ) : null}
              </Space>
            </List.Item>
          );
        }}
      />
    </Card>
  );
}

