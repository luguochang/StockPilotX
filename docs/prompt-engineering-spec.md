# Prompt Engineering Spec（企业级执行规范）

## 1. 文档目标
- 把 Prompt 从“经验写法”变成“可管理的工程资产”。
- 支撑金融分析场景的稳定输出、合规约束、可观测与可回溯。
- 绑定治理文档：
  - 执行清单：`docs/implementation-checklist.md`
  - 追踪矩阵：`docs/spec-traceability-matrix.md`
  - 全局门禁：`docs/global-constraints.md`

## 2. 适用范围
- 适用服务：`agent-gateway`、`report-service`、`eval-service`
- 适用场景：
  - `fact_qa`（事实问答）
  - `deep_analysis`（深度分析）
  - `doc_qa`（文档问答）
  - `compare_research`（对比研究）

## 3. Prompt 资产模型
## 3.1 核心实体
1. `prompt_registry`
- `prompt_id`：唯一ID
- `name`：名称
- `scenario`：场景
- `version`：语义版本（如 `1.3.0`）
- `status`：`draft/canary/stable/deprecated`
- `model_family`：模型族（如 `gpt-5`）
- `template_system`：系统层模板
- `template_policy`：策略层模板
- `template_task`：任务层模板
- `variables_schema`：变量 JSON Schema
- `guardrails`：规则集
- `owner`：责任人
- `created_at/updated_at`

2. `prompt_release`
- `release_id`
- `prompt_id/version`
- `target_env`：`staging/prod`
- `canary_ratio`
- `gate_result`
- `released_by/released_at`

3. `prompt_eval_result`
- `eval_run_id`
- `prompt_id/version`
- `suite_id`
- `metrics_json`
- `pass_gate`
- `created_at`

## 3.2 版本策略
- 规则变更：`minor` 或 `major`
- 仅文案修正：`patch`
- 每次发布必须保留“上一稳定版”用于快速回滚

## 4. Prompt 分层设计
## 4.1 System Layer（全局不可绕过）
- 角色声明
- 合规声明
- 输出结构要求
- 禁止表达（保证收益、确定性买卖建议）

## 4.2 Policy Layer（场景策略）
- `fact_qa`：事实优先、简明回答、强制引用
- `deep_analysis`：分维度分析、风险路径、冲突说明
- `doc_qa`：仅基于可用证据、缺证据时拒答或降级
- `compare_research`：统一口径对比、时间窗一致

## 4.3 Task Layer（请求级）
- 用户问题
- 可用工具清单
- 检索证据摘要
- 输出格式约束（JSON/Markdown）

## 5. 运行时编排规范
1. 输入预处理
- 清洗输入、注入意图标签、识别风险请求
2. 上下文构建
- `system` -> `policy` -> `task` 顺序拼装
3. 变量注入
- 必须通过 `variables_schema` 校验
4. Token 预算
- system: 15%
- policy: 15%
- evidence: 50%
- history: 20%
5. 动态裁剪
- 先裁剪低可信/低相关内容，再裁剪历史轮次

## 6. Prompt Guardrails（金融专用）
1. 强制规则
- 核心结论必须附引用
- 无法验证时必须标记“不确定”
- 不得输出确定性收益承诺
2. 冲突规则
- 多源冲突必须输出冲突摘要与置信度
3. 时效规则
- 超过时效窗口的数据必须加“可能过期”标签
4. 安全规则
- 检测 prompt injection，并忽略越权指令

## 7. 质量门禁（Release Gate）
## 7.1 必测项
1. `prompt_unit_tests`
- 固定输入 -> 固定结构断言
2. `prompt_regression_tests`
- 新旧版本同集对比
3. `prompt_redteam_tests`
- 越权、诱导、幻觉、违规表达攻击

## 7.2 通过阈值（MVP）
- 事实正确率 `>= 85%`
- 引用覆盖率 `>= 95%`
- 幻觉率 `<= 5%`
- 高风险违规率 `= 0`
- 成本增幅（对比 baseline）`<= 15%`

任一不达标：禁止发布到 `stable`。

## 8. 评测数据集规范
## 8.1 样本字段
- `case_id`
- `scenario`
- `input`
- `expected_facts`
- `expected_citations`
- `risk_level`
- `gold_answer`（可选）

## 8.2 样本分层
- `golden`：核心样本（稳定回归）
- `adversarial`：对抗样本（注入、误导、冲突）
- `freshness`：时效样本（新公告、新事件）

## 9. 发布流程（标准）
1. 提交 PR（含变更说明）
2. 自动跑 unit/regression/redteam
3. 人工审阅高风险样本输出
4. `staging` 灰度（canary 5%-20%）
5. 观察指标达标后晋升 `stable`
6. 失败自动回滚到上一稳定版本

## 10. 可观测与审计字段
- `prompt_id`
- `prompt_version`
- `model`
- `input_hash`
- `context_hash`
- `tool_calls`
- `citations`
- `output_hash`
- `risk_flags`
- `trace_id`（LangSmith）
- `latency_ms`
- `token_usage`
- `cost`

## 11. 最小落地清单（两周）
## 第1周
1. 建 `prompt_registry` 与 `prompt_release` 表
2. 打通版本加载与 runtime 组装
3. 实现 1 套 `fact_qa` 模板与 20 条单测样本

## 第2周
1. 增加 `deep_analysis` 模板
2. 接入 LangSmith 评测结果回写
3. 加入发布门禁与回滚机制

## 12. 不做项（当前阶段）
- 不做自动自我改写 Prompt 并自动上线
- 不做跨模型无约束自动迁移
- 不做无人工审核的高风险模板发布

## 13. 与执行清单绑定（新增强约束）
1. 任务绑定
- 每次 Prompt 变更必须绑定 `Task ID` 与 `Spec Ref`。
- 未绑定任务的 Prompt 变更禁止发布。

2. 评测绑定
- 每次发布必须绑定 `prompt_eval_result` 记录。
- 未通过回归门禁（含 RedTeam）禁止晋升 `stable`。

3. 证据绑定
- 必须在执行清单中记录命令、关键结果、日期。
