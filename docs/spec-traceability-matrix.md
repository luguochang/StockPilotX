# 规格追踪矩阵（Spec Traceability Matrix）

## 1. 目的
- 确保 `docs/a-share-agent-system-executable-spec.md` 与辅助文档要求可映射到可执行任务。
- 防止“实现遗漏、记录遗漏、验收遗漏”。

## 2. 映射规则
- `Spec Ref` 命名：
  - 主文档章节：`S2`~`S17.x`
  - Prompt 文档：`P3`、`P7.2` 等
  - 免费源文档：`D2.1`、`D6.2` 等
- 每个映射项必须绑定 `Task ID`（见 `docs/implementation-checklist.md`）。

## 3. 主文档映射（核心）
| Spec Ref | 要求摘要 | Task ID | 状态 |
|---|---|---|---|
| S2 | 产品目标与边界 | GOV-001 | [x] |
| S3 | 端到端流程图 | GOV-002 | [x] |
| S4 | 分层架构（前端/编排/数据/评测） | API-001, FRONT-001, OBS-001 | [x] |
| S5 | 免费数据源分层 + 回退策略 | DATA-001, DATA-002 | [x] |
| S6 | 三层 RAG + 模式切换 + 初始参数 | RAG-001, RAG-002 | [x] |
| S7 | Prompt 资产化与门禁 | PROMPT-001, PROMPT-002, PROMPT-003 | [x] |
| S8 | 多 Agent 主流程 + Deep Agents + 权限控制 | BASE-001, AGT-001, AGT-002 | [x] |
| S9 | 文档处理工程 + 质量门禁 + 版本治理 | RAG-003 | [x] |
| S10 | 评测体系与发布门禁 | PROMPT-003, OBS-001, API-002 | [x] |
| S11 | 数据模型首批必须项 | DATA-003 | [x] |
| S12 | API 契约首批 | BASE-004, API-001, API-002 | [x] |
| S13 | 8周实施路线图 | GOV-001 | [x] |
| S14 | 上线前工程化检查清单 | GOV-003, OPS-001 | [x] |
| S15 | DoD 验收标准 | GOV-003, OPS-001 | [x] |
| S16 | 当前版本结论与拍板项 | GOV-001 | [x] |
| S17.1 | 状态管理 | BASE-001 | [x] |
| S17.2 | LangSmith | OBS-001 | [x] |
| S17.3 | 多 Agent 协作 | BASE-001 | [x] |
| S17.4 | Long-term Memory | BASE-003 | [x] |
| S17.5 | LangGraph 集成 | AGT-001 | [x] |
| S17.6 | Middleware 工程化 | BASE-002, AGT-002 | [x] |
| S17.7 | Deep Agents | AGT-001 | [x] |
| S17.8 | 文档处理工程 | RAG-003 | [x] |
| S17.9 | GraphRAG + AgenticRAG + 测评 | RAG-001, RAG-002 | [x] |
| S17.10 | LangChain 1.0 必用点 | AGT-001 | [x] |
| S17.11~S17.13 | 14阶段映射与交付物 | GOV-002 | [x] |

## 3.1 预测模块新增映射（2026-02-13）
| Spec Ref | 要求摘要 | Task ID | 状态 |
|---|---|---|---|
| S2 | 研究辅助能力扩展（预测/风险分层） | PRED-001, PRED-004 | [x] |
| S4 | 分层架构扩展到预测域（后端+前端） | PRED-001, PRED-004 | [x] |
| S5 | 免费行情源支撑预测输入 | PRED-002, PRED-005 | [x] |
| S10 | 预测评测摘要门禁化（IC/命中率/分层收益/回撤） | PRED-003, PRED-005 | [x] |
| S11 | 预测状态与因子快照存储 | PRED-002 | [x] |
| S12 | 新增预测接口契约 | PRED-001, PRED-005 | [x] |
| S15 | 可验证验收（自动化测试+真实源烟测） | PRED-005 | [x] |
| S17.1 | 预测运行状态管理 | PRED-002 | [x] |
| S17.2 | 预测链路 trace 对齐 | PRED-003 | [x] |
| S17.9 | Agentic 预测评测闭环 | PRED-003 | [x] |
| S17.10 | LangChain/LangGraph 既有架构复用 | PRED-001 | [x] |

## 3.2 DeepThink + Internal A2A 映射（2026-02-15 Round-E）
| Spec Ref | 要求摘要 | Task ID | 状态 |
|---|---|---|---|
| S8 | 多 Agent 协作与深度任务执行 | AGT-009, AGT-010 | [x] |
| S12 | API 契约扩展（DeepThink/A2A） | AGT-009, AGT-010 | [x] |
| S16 | 实施治理与执行可追溯 | GOV-004 | [x] |
| S17.7 | Deep Agents 会话化与轮次化 | AGT-009 | [x] |
| S17.11 | 多阶段技术映射（多 Agent 协议化） | AGT-010 | [x] |

## 3.3 DeepThink Planner/Budget/Replan 映射（2026-02-15 Round-F）
| Spec Ref | 要求摘要 | Task ID | 状态 |
|---|---|---|---|
| S8 | 多 Agent 深度协商的任务化执行 | AGT-011 | [x] |
| S15 | 超预算可降级并可解释 | AGT-012 | [x] |
| S17.6 | 治理链路预算控制与事件观测 | AGT-012 | [x] |
| S17.7 | Deep Agents 的计划-执行-重规划闭环 | AGT-011 | [x] |
| S17.11 | 阶段化工程映射持续迭代 | AGT-011, AGT-012 | [x] |

## 3.4 首页导航化与 DeepThink 页面拆分映射（2026-02-15 Round-G）
| Spec Ref | 要求摘要 | Task ID | 状态 |
|---|---|---|---|
| S4 | 前端分层与路由职责清晰 | FRONT-003 | [x] |
| S12 | 入口路由与能力边界稳定 | FRONT-003 | [x] |
| S14 | 发布前构建/类型检查可执行 | FRONT-004 | [x] |
| S15 | 交付可验证（build + tsc + regression） | FRONT-004 | [x] |
| S16 | 实施治理与交付记录完整 | GOV-004 | [x] |

## 3.5 DeepThink 治理看板可视化映射（2026-02-15 Round-H）
| Spec Ref | 要求摘要 | Task ID | 状态 |
|---|---|---|---|
| S4 | 前端分层与深度分析视图职责明确 | FRONT-005 | [x] |
| S12 | 已有 DeepThink/A2A 接口在前端可观测、可操作 | FRONT-005 | [x] |
| S15 | 可验证交付（build + tsc + backend regression） | FRONT-005 | [x] |
| S17.7 | Deep Agents 多轮治理信号（conflict/budget/replan）可视化 | FRONT-005 | [x] |
| S16 | 实施治理文档与轮次记录完整可追溯 | GOV-004 | [x] |

## 3.6 DeepThink 差分/下钻/事件存档映射（2026-02-15 Round-I）
| Spec Ref | 要求摘要 | Task ID | 状态 |
|---|---|---|---|
| S8 | Deep Agents 多轮裁决过程可复盘 | AGT-013 | [x] |
| S12 | DeepThink 事件查询契约与前端消费闭环 | AGT-013, FRONT-006 | [x] |
| S15 | 可验证交付（pytest + build + typecheck） | AGT-013, FRONT-006 | [x] |
| S17.7 | Deep Agents 轮次差分与冲突证据视图增强 | FRONT-006 | [x] |
| S16 | 实施治理文档与轮次追踪完整可审计 | GOV-004 | [x] |

## 4. Prompt 文档映射
| Spec Ref | 要求摘要 | Task ID | 状态 |
|---|---|---|---|
| P3.1 | `prompt_registry/prompt_release/prompt_eval_result` | PROMPT-002 | [x] |
| P4 | System/Policy/Task 三层模板 | PROMPT-001 | [x] |
| P5 | 运行时 schema 注入 + token 预算 | PROMPT-001 | [x] |
| P6 | 金融 Guardrails | BASE-002, PROMPT-001 | [x] |
| P7.1 | unit/regression/redteam 必测 | PROMPT-003 | [x] |
| P7.2 | 门禁阈值强制 | PROMPT-002 | [x] |
| P8 | 样本字段与分层 | PROMPT-003 | [x] |
| P9 | 发布流程 + 回滚 | PROMPT-002 | [x] |
| P10 | 审计字段完整性 | OBS-001 | [x] |

## 5. 免费数据源文档映射
| Spec Ref | 要求摘要 | Task ID | 状态 |
|---|---|---|---|
| D2.1 | 一级事实源（巨潮/交易所） | DATA-002 | [x] |
| D2.2 | 二级行情源（腾讯/网易/新浪/雪球） | DATA-001 | [x] |
| D2.3 | 三级资讯线索层 | DATA-001 | [x] |
| D3.2 | 统一适配器接口 | DATA-001 | [x] |
| D4 | Canonical Schema 规范 | DATA-003 | [x] |
| D5 | 调度任务与频率 | DATA-004 | [x] |
| D6 | 回退/重试/熔断/缓存 | DATA-001, DATA-004 | [x] |
| D7 | 质量校验与可靠性评分 | DATA-003 | [x] |
| D8 | 与 RAG 衔接 + 引用规则 | DATA-003, RAG-001 | [x] |
| D9 | 合规抓取与审计 | GOV-003, OPS-001 | [x] |

## 6. 覆盖率结论
- 主文档 2~17 节：已建立任务映射（覆盖率 100%）。
- Prompt 文档关键条目：已建立任务映射（覆盖率 100%）。
- 免费源文档关键条目：已建立任务映射（覆盖率 100%）。

## 7. 维护规则
- 新增需求时，先改本矩阵，再改执行清单。
- 未在矩阵中登记的需求，禁止进入“执行完成”。
