# 多 Agent 架构模式：监督者、网络、分层团队

日期：2026-02-15

## 1. 三种模式
- 监督者（Supervisor）：一个总控 agent 调度多个专家 agent。
- 网络（Network）：agent 之间点对点转发任务。
- 分层团队（Hierarchical）：上层策略、下层执行，层级清晰。

## 2. 在股票分析系统的推荐模式
- 首选“监督者 + 分层团队”混合模式：
  - Supervisor：管理目标、预算、停止条件
  - 专家层：PM/Quant/Risk/RAG/Critic
  - 工具层：行情、公告、财报、文档检索

## 3. 当前系统映射
- 当前更接近“轻量监督者”：
  - `service.ops_agent_debate` 负责聚合意见
  - `workflow` 负责固定阶段编排
- 尚未支持：
  - agent 间显式转派（handoff）
  - 跨轮会话共享决策上下文

## 4. 推荐决策协议（内部）
- 每个 agent 输出统一结构：
  - `signal`: buy|hold|reduce
  - `confidence`: 0~1
  - `reason`: 文本
  - `evidence_ids`: 证据列表
  - `risk_tags`: 风险标签

## 5. 仲裁器设计
- 输入：多 agent 输出 + 历史表现权重 + 当前市场波动等级
- 输出：
  - 共识信号
  - 分歧分数
  - 冲突来源（证据冲突/模型冲突/时效冲突）
- 要求：强制输出“反方观点”

## 6. 工程建议
- 先保留当前三角色，再增加：
  - `macro_agent`（宏观）
  - `execution_agent`（交易执行建议表达层）
  - `compliance_agent`（合规审查）

## 7. 参考
- LangGraph Multi-Agent Collaboration:
  - https://langchain-ai.github.io/langgraph/tutorials/multi_agent/multi-agent-collaboration/
- LangGraph Hierarchical Teams:
  - https://langchain-ai.github.io/langgraph/tutorials/multi_agent/hierarchical_agent_teams/

