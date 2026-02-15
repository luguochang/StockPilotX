# 总览：多 Agent、A2A 与 DeepAgent 分层方法

日期：2026-02-15  
适用项目：`StockPilotX`

## 1. 专栏目标
- 把“多 Agent / A2A / DeepAgent”从概念说明转成可执行工程方案。
- 明确当前系统已经做到什么、还缺什么、下一步怎么演进。
- 为后续研发提供一致术语和决策框架，减少重复讨论。

## 2. 三层方法
- 编排层（Orchestration）：以 LangGraph/Workflow 组织状态、节点和控制流。
- 协议层（Interoperability）：以 A2A 统一 agent 间任务派发、状态回传和可观测结构。
- 认知层（DeepAgent）：以 planner/subagent/memory/budget 进行多轮深度求解与治理。

## 3. 当前项目结论（简版）
- 已实现同进程多角色编排：`backend/app/agents/workflow.py`
- 已实现可选 LLM 并行辩论与回退：`backend/app/service.py`
- 已实现运行时切换（langgraph/direct）：`backend/app/agents/langgraph_runtime.py`
- 尚未实现跨进程或跨服务的标准 A2A 互联网关（当前为内部 A2A 适配层）

## 4. 常见误区
- 多 Agent 编排 ≠ A2A 协议互联
- 并行观点生成 ≠ DeepAgent 全流程治理
- 多角色回复 ≠ 多智能体自治协作

## 5. 专栏目录
1. 现状审计：StockPilotX 多 Agent 与 Deep 实现证据  
2. A2A 协议：定义、状态机与项目接入边界  
3. 多 Agent 架构模式：监督者网络与分层团队  
4. DeepAgent 设计：planner/subagent/memory/tool budget  
5. DeepThink 接口规范：会话、轮次、流式事件与裁决  
6. 评测体系：轨迹质量 + 结果质量 + RAG 质量  
7. 中间件治理：权限、预算、合规、观测  
8. 演进路线图：从当前实现到专业版  
9. Round-E：DeepThink + Internal A2A MVP 实现记录  
10. Round-F：DeepThink Planner + Budget + Replan 实现记录  
11. Round-G：首页导航化与 DeepThink 独立页面实现记录  
12. Round-H：DeepThink 轮次可视化与治理看板实现记录  
13. Round-I：DeepThink 跨轮差分、冲突下钻与事件存档实现记录  
14. Round-J：DeepThink 事件过滤与归档保留治理实现记录  
15. Round-K：DeepThink 归档分页、时间过滤与导出实现记录

## 6. 参考资料
- A2A Specification: https://a2aproject.github.io/A2A/specification/
- A2A Developer Docs: https://a2aproject.github.io/A2A/dev/
- LangChain Deep Agents: https://docs.langchain.com/oss/python/deepagents/overview
- LangGraph Multi-Agent Collaboration: https://langchain-ai.github.io/langgraph/tutorials/multi_agent/multi-agent-collaboration/
- LangGraph Hierarchical Agent Teams: https://langchain-ai.github.io/langgraph/tutorials/multi_agent/hierarchical_agent_teams/
