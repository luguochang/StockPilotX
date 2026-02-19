# ROUND-4 技术记录：DeepThink 业务视图重构

## 1. 本轮目标

将 DeepThink 从“工程信号优先”改为“业务结论优先”展示，分析模式先让用户看到可执行信息，再按需查看工程细节。

本轮围绕计划中的 ROUND-4 完成：

1. 结论卡  
2. 证据卡  
3. 事件日历卡  
4. 情景矩阵卡  
5. 交互联动与回归

---

## 2. 前端实现

文件：`frontend/app/deep-think/page.tsx`

## 2.1 新增业务数据模型

新增类型：
- `IntelCardSnapshot`
- `IntelCardEvidence`
- `IntelCardScenario`
- `IntelCardEvent`

并扩展 `Citation`：
- `retrieval_track`
- `rerank_score`
- `reliability_score`
- `event_time`

用于承接 ROUND-3 的归因字段。

## 2.2 新增状态与加载逻辑

新增状态：
- `intelCard`
- `intelCardLoading`
- `intelCardError`
- `intelCardHorizon`（7d/30d/90d）
- `intelCardRiskProfile`（conservative/neutral/aggressive）

新增函数：
- `loadIntelCard(options?)`
  - 调用 `/v1/analysis/intel-card`
  - 支持参数化 horizon/risk_profile
  - 支持跳过股票库重复校验（用于 runAnalysis 并行加载）

在 `runAnalysis` 内并行触发 intel-card 刷新，减少用户等待。

## 2.3 新增业务卡片区域

新增模块：

1. 参数控制区  
- 时间窗选择  
- 风险偏好选择  
- 刷新业务卡片按钮

2. 结论摘要区  
- 建议动作  
- 置信度  
- 风险等级  
- 仓位建议  
- 复核时间

3. 触发/失效条件区  
- trigger_conditions  
- invalidation_conditions

4. 催化与风险区  
- key_catalysts  
- risk_watch

5. 事件日历区  
- event_calendar

6. 情景矩阵区  
- scenario_matrix（场景、预期收益、概率）

7. 证据链路区  
- evidence（source、retrieval_track、reliability、summary、source_url）

8. 数据新鲜度区  
- data_freshness（分钟级）

---

## 3. 设计取舍

1. 分析模式优先展示业务卡片，工程信息仍保留在 DeepThink 控制台中。  
2. 股票切换后先清空卡片状态，防止跨股票误读。  
3. 业务卡片支持手动刷新，避免仅依赖一次分析流程触发。

---

## 4. 自测记录

执行时间：2026-02-19

1. 前端类型检查  
命令：
```powershell
cd frontend
npx tsc --noEmit
```
结果：通过（exit code 0）

2. 关键接口回归  
命令：
```powershell
.\.venv\Scripts\python.exe -m pytest -q tests/test_http_api.py -k "analysis_intel_card"
```
结果：`1 passed, 34 deselected`

---

## 5. 风险与后续

1. 目前业务卡片已接入，但与 DeepThink 轮次仲裁结论尚未完全合流展示。  
2. 下一轮（ROUND-5）建议将仓位节奏与风控阈值进一步结构化，形成真正的执行建议模板。  
3. 后续可在业务卡片中增加“本轮 vs 上轮变化解释”，提升可复盘性。
