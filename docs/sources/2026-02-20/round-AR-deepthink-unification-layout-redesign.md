# ROUND-AR：DeepThink 一体化执行流与业务布局重构

## 目标
- 下线“分析润色”独立模块，统一到 DeepThink。
- 将业务页执行方式改成“一键全流程”（流式分析 + DeepThink 下一轮）。
- 优化页面布局，避免流式回答内容把页面整体撑开。
- 将“最近5交易日明细/证据/样本”放在与分析结果同一业务区域联动展示。

## 关键改动

### 1) 模块收敛与路由兼容
- `frontend/app/analysis-studio/page.tsx`
  - 改为重定向到 `/deep-think`。
  - 语义：分析润色模块下线，保留历史链接兼容。
- `frontend/app/layout.tsx`
  - 移除导航中的“分析润色”入口。
- `frontend/app/page.tsx`
  - 移除首页“分析润色台”卡片。
- 删除 `frontend/app/market-quick/page.tsx` 与 `frontend/app/market-quick/market-quick.module.css`。

### 2) 一键全流程编排
- `frontend/app/deep-think/page.tsx`
  - 新增 `DeepFullFlowStage` 与全流程状态标签。
  - 新增 `runDeepThinkFullFlow()`：
    1. 先执行 `runAnalysis()`
    2. 再执行 `runDeepThinkRound()`（自动建会话）
  - `runAnalysis()` 与 `runDeepThinkRound()` 改为返回 `Promise<boolean>`，便于编排控制。

### 3) 业务页入口去重
- Hero 区按钮从“启动深度分析”改为“进入分析面板”，只做滚动定位，不再触发执行。
- 业务输入卡保留唯一执行按钮：`开始全流程分析`。
- 轮次控制台（业务模式）不再展示“新建会话/执行下一轮”等工程型按钮，避免与上方入口冲突。

### 4) 布局与可读性优化
- 分析主报告 `<pre>` 增加 `maxHeight + overflowY:auto`，避免流式内容扩展导致页面抖动。
- 右栏设为粘性（桌面）以稳定证据阅读区。
- 新增“最近5个交易日明细”卡片（表格），与“最近三个月连续样本/证据引用”集中在右栏。

## 业务价值
- 用户点击一次即可跑完整链路，不再理解“先开始分析还是先新建会话”。
- 主报告输出稳定，不会因为持续流式输出破坏整体布局。
- 证据、样本、交易日明细和主结论贴近展示，减少跳读成本。

## 风险与说明
- 工程模式（`/deep-think/console`）仍保留完整会话/归档/导出操作，业务模式仅做收敛。
- 右栏粘性布局主要针对桌面体验，移动端会自然堆叠为单列。
