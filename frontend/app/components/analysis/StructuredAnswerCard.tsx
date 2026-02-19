"use client";

import { useMemo, useState } from "react";
import { Button, Card, Collapse, Space, Tag, Typography } from "antd";
import { extractKeywords, parseAnswerBlocks } from "./answer-format";

const { Paragraph, Text, Title } = Typography;

type Props = {
  answer: string;
  traceId: string;
  streaming: boolean;
  progressText: string;
  streamSource: string;
  model: string;
  provider: string;
  apiStyle: string;
  runtime: string;
};

export default function StructuredAnswerCard(props: Props) {
  const {
    answer,
    traceId,
    streaming,
    progressText,
    streamSource,
    model,
    provider,
    apiStyle,
    runtime
  } = props;
  const blocks = useMemo(() => parseAnswerBlocks(answer), [answer]);
  const keywords = useMemo(() => extractKeywords(answer), [answer]);
  const [expanded, setExpanded] = useState(false);
  const visibleBlocks = expanded ? blocks : blocks.slice(0, 8);

  return (
    <Card title="分析主报告（结构化）" className="premium-card">
      <Space direction="vertical" style={{ width: "100%" }} size={10}>
        <Space wrap>
          <Tag color={streaming ? "processing" : "default"}>{streaming ? "流式输出中" : "流式输出结束"}</Tag>
          {progressText ? <Tag color="blue">阶段：{progressText}</Tag> : null}
          <Tag color={streamSource === "external_llm_stream" ? "green" : streamSource === "local_fallback_stream" ? "gold" : "default"}>
            来源：{streamSource || "unknown"}
          </Tag>
          <Tag>模型：{model || "unknown"}</Tag>
          <Tag>Provider：{provider || "unknown"}</Tag>
          <Tag>API：{apiStyle || "unknown"}</Tag>
          <Tag>Engine：{runtime || "unknown"}</Tag>
        </Space>

        {keywords.length ? (
          <Space wrap>
            {keywords.map((key) => (
              <Tag key={key} color="cyan">{key}</Tag>
            ))}
          </Space>
        ) : null}

        {!blocks.length ? <Text style={{ color: "#64748b" }}>等待流式内容...</Text> : null}

        <Space direction="vertical" style={{ width: "100%" }} size={8}>
          {visibleBlocks.map((block, idx) => {
            if (block.kind === "heading") {
              return <Title key={`h-${idx}`} level={5} style={{ margin: 0 }}>{block.title || block.lines[0]}</Title>;
            }
            if (block.kind === "list") {
              return (
                <ul key={`l-${idx}`} style={{ margin: 0, paddingLeft: 18, color: "#0f172a" }}>
                  {block.lines.map((line, i) => (
                    <li key={`li-${idx}-${i}`} style={{ marginBottom: 4 }}>
                      {line.replace(/^(\-|\*|\d+\.)\s+/, "")}
                    </li>
                  ))}
                </ul>
              );
            }
            return (
              <Paragraph key={`p-${idx}`} style={{ margin: 0, color: "#0f172a", lineHeight: 1.8 }}>
                {block.lines.join(" ")}
              </Paragraph>
            );
          })}
        </Space>

        {blocks.length > 8 ? (
          <Button onClick={() => setExpanded((x) => !x)}>{expanded ? "收起" : "展开全文"}</Button>
        ) : null}

        <Collapse
          items={[
            {
              key: "raw",
              label: "原始全文",
              children: <pre style={{ whiteSpace: "pre-wrap", color: "#0f172a", margin: 0 }}>{answer || "-"}</pre>
            }
          ]}
        />

        <Text style={{ color: "#64748b" }}>trace_id: {traceId || "-"}</Text>
      </Space>
    </Card>
  );
}

