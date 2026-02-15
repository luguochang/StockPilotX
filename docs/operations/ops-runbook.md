# OPS Runbook（OPS-001）

## 1. SLO 目标
- Query 成功率 >= 99%
- P95 响应 <= 3s（本地压测环境）
- 评测门禁通过率：100%（发布前）

## 2. 关键告警
- `api_error_rate_high`：5 分钟错误率 > 2%
- `eval_gate_failed`：门禁失败
- `data_ingest_failed`：抓取任务失败且熔断开启

## 3. 故障处理
1. API 5xx 激增
- 检查 `TraceStore` 最新 trace 事件
- 降级到 mock 数据源（已内置兜底）

2. 数据源不可达
- 查看 scheduler 状态
- 若 `circuit_open`，等待冷却后重试

3. 评测不通过
- 禁止 stable 发布
- 回退到上一稳定 prompt 版本

## 4. 回滚策略
- Prompt 回滚：切换 `prompt_release` 最近 `gate_result=pass` 版本
- 服务回滚：回退到上一个测试通过的镜像标签

## 5. 发布前检查
- `python -m unittest discover -s tests -p "test_*.py" -v` 全通过
- `/v1/evals/run` pass_gate = true
- 清单任务证据完整（命令/输出/日期）

