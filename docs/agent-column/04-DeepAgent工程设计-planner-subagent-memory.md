# DeepAgent 工程设计：Planner、Subagent、Memory、Tool Budget

日期：2026-02-15

## 1. DeepAgent 的工程定义
- 不是“多回答几轮”，而是“任务分解-执行-反思-重规划”闭环。
- 核心组件：
  - Planner
  - Subagent Pool
  - Shared Memory
  - Tool Budget Controller
  - Critic / Verifier

## 2. 当前系统差距
- 已有：固定三子任务并行检索。
- 缺失：
  - 动态任务树（todo graph）
  - 失败重规划（re-plan）
  - 预算与停止条件（token/time/tool calls）

## 3. 推荐执行循环
1. Planner 生成任务树（含优先级、依赖、预算）。
2. Supervisor 派发到 subagent。
3. Subagent 调工具 + 产出证据。
4. Critic 检查证据质量与冲突。
5. 若未达标，触发 re-plan 并继续。
6. 达标后生成最终报告并封存轨迹。

## 4. Memory 设计
- Working Memory（会话内）：
  - round context、未决问题、临时证据
- Long-term Memory（会话外）：
  - 用户风险偏好
  - 历史分歧处理结果
  - 高价值研究卡片
- 写入策略：
  - 仅写“高价值确认事实”，拒绝把低置信临时推断写入长期记忆

## 5. Tool Budget 设计
- 预算维度：
  - token budget
  - time budget
  - tool call budget
- 预算超限策略：
  - 降级检索范围
  - 切换低成本模型
  - 输出“证据不足”而非强结论

## 6. 与 LangChain Deep Agents 的对齐点
- 使用 planner 驱动的多步执行。
- 使用受控工具调用和结构化结果。
- 使用记忆与任务清单增强复杂任务稳定性。

## 7. 参考
- Deep Agents Overview:
  - https://docs.langchain.com/oss/python/deepagents/overview

