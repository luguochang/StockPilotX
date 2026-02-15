# Round-R: 高级分析三个月连续样本固定卡片

Date: 2026-02-15

## 1. 目标
- 将“最近三个月连续样本摘要”从长回答文本中抽离，固定展示在高级分析页面右侧区域。
- 让用户无需阅读整段回答，也能直接看到样本覆盖、区间与涨跌信息。

## 2. 实现
- 文件：`frontend/app/deep-think/page.tsx`
- 新增 `history3mSummary` 计算逻辑：
  - 从 `overview.history` 取最近 90 条连续样本。
  - 计算：
    - `sampleCount`
    - `startDate/endDate`
    - `startClose/endClose`
    - `pctChange`
- 新增固定卡片“最近三个月连续样本”：
  - 展示样本覆盖（90窗口/总样本）
  - 展示区间涨跌（红绿标识）
  - 展示区间与收盘首末值
  - 无数据时提示“先执行高级分析刷新历史数据”

## 3. 自测
- 执行：`npm --prefix frontend run -s build`
- 结果：通过（Next.js 构建成功）

## 4. 结果
- 用户可以在 UI 结构化看到三个月窗口结论，不再依赖回答正文里零散描述。
