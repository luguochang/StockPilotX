"use client";

import { Card, Progress, Space, Tag, Typography } from "antd";

const { Text } = Typography;

type DeepThinkOpinion = {
  agent_id: string;
  signal: "buy" | "hold" | "reduce";
  confidence: number;
  reason: string;
};

type DeepThinkRound = {
  round_id: string;
  round_no: number;
  consensus_signal: string;
  disagreement_score: number;
  conflict_sources: string[];
  stop_reason: string;
  opinions: DeepThinkOpinion[];
};

type DeepThinkSession = {
  session_id: string;
  status: string;
  current_round: number;
  max_rounds: number;
  rounds: DeepThinkRound[];
};

function getSignalLabel(signal: string): string {
  if (signal === "buy") return "增配";
  if (signal === "reduce") return "减配";
  if (signal === "hold") return "持有";
  return signal || "未知";
}

export default function DeepRoundSummaryPanel({
  session,
  stageLabel,
  stagePercent,
  running,
  deepError
}: {
  session: DeepThinkSession | null;
  stageLabel: string;
  stagePercent: number;
  running: boolean;
  deepError: string;
}) {
  const latestRound = session?.rounds?.length ? session.rounds[session.rounds.length - 1] : null;
  return (
    <Card className="premium-card" title="DeepThink 轮次洞察">
      <Space direction="vertical" style={{ width: "100%" }} size={10}>
        <Space wrap>
          <Tag color={session ? "blue" : "default"}>会话：{session?.session_id ?? "未创建"}</Tag>
          <Tag color={running ? "processing" : deepError ? "red" : "green"}>{running ? "执行中" : deepError ? "失败" : "完成/待执行"}</Tag>
          <Tag>轮次：{session?.current_round ?? 0}/{session?.max_rounds ?? 0}</Tag>
        </Space>

        <Text style={{ color: "#334155" }}>阶段：{stageLabel}</Text>
        <Progress percent={stagePercent} status={deepError ? "exception" : running ? "active" : "normal"} />

        {latestRound ? (
          <Space direction="vertical" style={{ width: "100%" }} size={6}>
            <Space wrap>
              <Tag color={latestRound.consensus_signal === "buy" ? "green" : latestRound.consensus_signal === "reduce" ? "red" : "blue"}>
                共识：{getSignalLabel(String(latestRound.consensus_signal))}
              </Tag>
              <Tag>分歧度：{Number(latestRound.disagreement_score ?? 0).toFixed(3)}</Tag>
              {latestRound.stop_reason ? <Tag color="red">{latestRound.stop_reason}</Tag> : null}
            </Space>
            <Text style={{ color: "#334155" }}>
              {(latestRound.conflict_sources ?? []).length
                ? `冲突源：${latestRound.conflict_sources.join(" / ")}`
                : "当前无显著冲突源"}
            </Text>
            {latestRound.opinions?.slice(0, 3).map((op) => (
              <Text key={`${latestRound.round_id}-${op.agent_id}`} style={{ color: "#64748b" }}>
                {op.agent_id}: {getSignalLabel(op.signal)} / {Number(op.confidence ?? 0).toFixed(3)} / {String(op.reason ?? "").slice(0, 50)}
              </Text>
            ))}
          </Space>
        ) : (
          <Text style={{ color: "#64748b" }}>执行后展示最新轮次结论、冲突和角色观点。</Text>
        )}
      </Space>
    </Card>
  );
}

