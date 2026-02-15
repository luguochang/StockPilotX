# 多 Agent 评测体系：离线、在线、LangSmith

日期：2026-02-15

## 1. 为什么要单独评测多 Agent
- 多 agent 的失败不只在“答案错”，还可能是“协商过程错”。
- 必须同时评估结果质量与过程质量。

## 2. 评测维度
- 结果维度：
  - signal 准确性
  - 风险提示有效性
  - 证据引用覆盖率
- 过程维度：
  - 分歧是否被识别
  - 仲裁是否引用有效证据
  - 回退与降级是否按策略执行

## 3. 指标建议
- Outcome：
  - hit_rate@horizon
  - downside_alert_precision
  - citation_coverage
- Trajectory：
  - disagreement_detection_recall
  - invalid_tool_call_rate
  - unresolved_conflict_rate

## 4. 离线评测
- 数据集：历史行情 + 财报事件 + 已标注结论样本
- 执行：固定 prompt 版本 + 固定 agent 配置回放
- 输出：版本对比与回归门禁结论

## 5. 在线评测
- 对真实请求抽样，记录：
  - 输入
  - agent 轨迹
  - 最终结论
  - 用户反馈
- 与离线指标合并为质量看板

## 6. LangSmith 集成建议
- 每轮 deep think 都输出独立 trace span：
  - planner span
  - subagent span
  - critic span
  - arbitration span
- 记录 prompt version、model provider、tool calls、latency、errors

## 7. 上线门禁（建议）
- citation_coverage >= 0.95
- unresolved_conflict_rate <= 0.05
- invalid_tool_call_rate <= 0.01
- 任何合规红线违规：阻断发布

