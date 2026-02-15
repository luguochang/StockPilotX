# 全局工程约束（Global Constraints）

## 1. 目标
- 将规范中的“建议”升级为“可执行强约束”，防止实现偏离主文档。

## 2. 约束优先级
1. 本文件（全局约束）
2. `docs/specs/a-share-agent-system-executable-spec.md`
3. `docs/specs/prompt-engineering-spec.md` 与 `docs/specs/free-data-source-implementation.md`
4. 其他说明文档

若冲突，按优先级高者执行。

## 3. 硬门禁（Hard Gates）
1. 任务完成门禁
- 无 `Evidence`（命令 + 关键输出 + 日期）禁止从 `[-]` 改为 `[x]`。
- 无 `Spec Ref` 的任务禁止入清单。

2. 引用与合规门禁
- 关键结论必须有引用；引用覆盖率不足时禁止标记“可发布”。
- 出现确定性投资建议违规时，相关任务必须标 ` [!] ` 并修复后重测。

3. 评测门禁
- 以下阈值任一未达标，禁止发布：
  - 事实正确率 `>= 85%`
  - 引用覆盖率 `>= 95%`
  - 幻觉率 `<= 5%`
  - 高风险违规率 `= 0`

4. 变更门禁
- 未写入 `docs/governance/spec-traceability-matrix.md` 的需求禁止执行。
- 未登记到 `docs/governance/implementation-checklist.md` 的执行项禁止开始。

## 4. 软门禁（Soft Gates）
- 推荐每次提交后执行最小回归测试并记录结果。
- 推荐每个里程碑完成后生成一次风险清单与回滚预案。

## 5. 任务状态机
- `[ ] -> [-] -> [x]`：标准路径
- `[ ] -> [!] -> [-] -> [x]`：阻塞恢复路径
- 禁止直接 `[ ] -> [x]`

## 6. 审计字段最小集合
- `Task ID`
- `Spec Ref`
- `Owner`
- `Status`
- `Evidence`
- `Updated At`

## 7. 发布前检查（最小清单）
1. 覆盖检查
- 追踪矩阵覆盖率为 100%。

2. 质量检查
- 回归集执行完成且达门禁阈值。

3. 合规检查
- 无高风险违规项未关闭。

4. 运维检查
- 关键告警、回滚路径、运行手册可用。

## 8. 例外处理
- 如需临时豁免，必须在执行清单中记录：
  - 豁免项
  - 风险说明
  - 责任人
  - 到期时间

