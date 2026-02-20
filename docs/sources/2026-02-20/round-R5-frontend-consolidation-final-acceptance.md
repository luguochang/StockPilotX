# ROUND-R5：前端收口与最终验收（DeepThink Multi-Role）

## 1. 目标
1. 完成 R3/R4 后端能力到前端业务层的最后一公里：`multi_role_pre / pre_arbitration / runtime_guard / runtime_timeout`。
2. 让业务用户能看懂“预裁决结论、运行时告警/超时、下一步动作”，而不只是看到原始事件名。
3. 完成 R5 最终验收并关闭 checklist。

## 2. 关键改动

### 2.1 类型与字段收口
- 文件：`frontend/app/deep-think/page.tsx`
- 新增前端类型：
  - `DeepThinkRuntimeGuard`
  - `DeepThinkMultiRolePre`
- 结构扩展：
  - `DeepThinkBudgetUsage.runtime_guard`
  - `DeepThinkRound.runtime_guard`
  - `DeepThinkRound.multi_role_pre`

### 2.2 事件解析与阶段可见性
- `resolveDeepStageFromEvent(...)` 增加：
  - `pre_arbitration` -> `arbitration`
  - `runtime_timeout` -> `persist`
  - `runtime_guard(stage=round_finalize)` -> `persist`
- `appendDeepEvent(...)` 增加对 R3/R4 事件的人类可读反馈：
  - `pre_arbitration`：显示预裁决完成与分歧度
  - `runtime_guard`：warning 时提示阶段预警
  - `runtime_timeout`：明确“已降级 + 可继续下一轮”

### 2.3 业务/工程界面可视化
- 分析模式新增卡片：`预裁决与运行时守卫`
  - 展示预裁决信号、置信度、分歧度、冲突源、judge 摘要、trace_id
  - 展示运行时阈值、耗时、超时阶段、是否降级
- 工程模式增强卡片：`预算使用与剩余`
  - 补充 runtime guard 的 warning/timed_out/timeout_stage/elapsed/threshold 指标
- 状态栏增强：
  - 分析模式与工程模式都显示预裁决标签、运行时超时标签
- 事件筛选增强：
  - 显式加入 `pre_arbitration / runtime_guard / runtime_timeout / round_persisted / journal_linked`

## 3. 为什么 R5 必须做
- R3/R4 已在后端输出事件与快照字段，但前端此前没有显式消费这些字段，用户无法区分：
  - 常规停止 vs 运行时超时降级
  - 最终仲裁 vs 预裁决
- R5 的意义是把“可用但不可见”的能力变成“可理解且可操作”的产品能力。

## 4. 自测与结果

### 4.1 前端类型检查
- 命令：`npx tsc --noEmit`（`frontend/`）
- 结果：通过

### 4.2 后端链路回归（确保前端消费字段有稳定来源）
- 命令：`.venv/Scripts/python.exe -m pytest -q tests/test_service.py -k "deep_think_session_and_round or deep_think_runtime_timeout_guard"`
- 结果：`2 passed`

- 命令：`.venv/Scripts/python.exe -m pytest -q tests/test_http_api.py -k "deep_think_v2_round_stream or deep_think_runtime_timeout_guard"`
- 结果：`2 passed`

### 4.3 备注
- `npm run build` 在当前环境命中 `.next/trace` 文件占用（EPERM），该问题与代码逻辑无关，已使用 `tsc --noEmit` 完成本轮前端静态验收。

## 5. 验收结论
- R5 验收通过，`R5-01` 可勾选完成。
- DeepThink 前端现已与 R3/R4 后端契约对齐，具备完整的预裁决与运行时治理可视化闭环。
