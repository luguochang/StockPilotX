# 总览：多 Agent、A2A、DeepAgent 的分层方法

日期：2026-02-15  
适用项目：`StockPilotX`

## 1. 这套专栏要解决什么
- 把“多 Agent”“A2A”“DeepAgent”从概念口号变成可落地工程分层。
- 解释当前系统已经做到什么、还缺什么、下一步怎么升级。
- 给后续编码提供决策完备规范，减少“边做边猜”。

## 2. 三层架构（建议作为专栏主线）
- 编排层（Orchestration）：LangGraph 负责节点状态机、分支、并行、恢复。
- 协议层（Interoperability）：A2A 负责 agent 与 agent 之间跨系统通信标准。
- 认知层（DeepAgent）：Planner + Subagent + Memory + Tool Budget 的深度任务求解。

## 3. 当前项目结论（简版）
- 已实现“单服务多角色编排”：`backend/app/agents/workflow.py`
- 已实现“并行辩论（可选 LLM）”：`backend/app/service.py`
- 已实现“LangGraph 运行时可切换”：`backend/app/agents/langgraph_runtime.py`
- 尚未实现“严格 A2A 协议互联”（当前没有独立 A2A 网关与 agent card）

## 4. 读者最容易混淆的点
- 多 Agent 编排 != A2A。
- 并行检索 != DeepAgent 全流程。
- LLM 多角色回答 != 多代理协作协议。

## 5. 本专栏目录
1. 现状审计：StockPilotX 的多 Agent 与 Deep 实现证据
2. A2A 协议：定义、状态机、在本项目的接入边界
3. 多 Agent 架构模式：监督者、网络、分层团队
4. DeepAgent 工程设计：planner/subagent/memory/tool budget
5. Deep Think 接口规范：会话、轮次、流式事件、仲裁
6. 评测体系：轨迹质量 + 结果质量 + RAG 质量
7. 中间件与治理：hook、权限、预算、合规、观测
8. 实施路线图：从当前系统升级到专业版本
9. Round-E 实现记录：DeepThink + Internal A2A MVP 落地证据

## 6. 关键参考
- A2A 规范：https://a2aproject.github.io/A2A/specification/
- A2A 开发文档：https://a2aproject.github.io/A2A/dev/
- LangChain Deep Agents：https://docs.langchain.com/oss/python/deepagents/overview
- LangGraph 多 Agent（网络）：https://langchain-ai.github.io/langgraph/tutorials/multi_agent/multi-agent-collaboration/
- LangGraph 多 Agent（分层团队）：https://langchain-ai.github.io/langgraph/tutorials/multi_agent/hierarchical_agent_teams/
