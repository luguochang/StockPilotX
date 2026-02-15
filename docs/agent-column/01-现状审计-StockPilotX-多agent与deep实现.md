# 现状审计：StockPilotX 的多 Agent 与 Deep 实现

日期：2026-02-15  
结论级摘要：当前是“单进程编排 + 可选 LLM 并行辩论 + 轻量 deep 检索”，不是严格 A2A。

## 1. 多 Agent 现状（已实现）
- 工作流主线：`before_agent -> router -> retrieval -> analysis -> model -> critic -> after_agent`
- 代码位置：
  - `backend/app/agents/workflow.py:59`
  - `backend/app/agents/workflow.py:154`
  - `backend/app/agents/workflow.py:230`
- 说明：
  - 角色分工体现在逻辑阶段，不是独立进程代理。
  - 由同一 runtime 统一驱动执行。

## 2. LangGraph 现状（已实现）
- 支持 `direct/langgraph` 双运行时切换。
- 代码位置：
  - `backend/app/agents/langgraph_runtime.py:64`
  - `backend/app/service.py:95`
  - `backend/app/service.py:133`
- 说明：
  - 已有图节点编排，但节点内部仍是本地 workflow 方法调用。

## 3. Deep 现状（部分实现）
- 触发条件：`intent in (deep, compare)` 或命中关键词。
- 子任务策略：固定 3 子问题并行检索后合并。
- 代码位置：
  - `backend/app/agents/workflow.py:237`
  - `backend/app/agents/workflow.py:256`
- 差距：
  - 还没有任务树动态重规划（re-plan）。
  - 还没有显式 todo graph 与步骤预算控制。

## 4. 多代理辩论现状（已实现）
- 接口：`GET /v1/ops/agent/debate`
- 角色：`pm_agent / quant_agent / risk_agent`
- 模式：`llm_parallel`（可用时）或 `rule_fallback`
- 代码位置：
  - `backend/app/http_api.py:302`
  - `backend/app/service.py:822`
  - `backend/app/service.py:979`
- 差距：
  - 仍是“中心函数并行调用”，不是 agent-to-agent 协议对话。

## 5. A2A 现状（未实现）
- 当前无 A2A 网关、无 agent card、无标准 task lifecycle endpoint。
- 当前是“系统内编排”，不是“跨系统代理互操作”。

## 6. 现状能力边界
- 已有：
  - 生产可运行查询链路（含 SSE）
  - 可追踪 runtime 与部分评测面板
  - 多角色观点融合
- 尚缺：
  - 协议层互联（A2A）
  - 深度推理任务会话化（Deep Think Session）
  - 轨迹级评测标准化（step correctness、reasoning consistency）

## 7. 给专栏读者的一句话
- 这个项目已经从“单问答”升级到“多角色协作”，但还处于“编排型多 Agent”，下一步应补协议层与深推理治理层。

