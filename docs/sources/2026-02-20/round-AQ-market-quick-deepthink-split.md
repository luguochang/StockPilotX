# ROUND-AQ：市场快析与 DeepThink 模块拆分实施记录

## 背景
现状中“高级分析”和“DeepThink 轮次控制台”在同一工作台串联执行，导致：
1. 用户误以为必须先做高级分析才能执行下一轮。
2. 单次流式问答与多轮任务图治理混在一个页面，认知成本高。
3. 入口命名不清，用户难以判断该去哪个模块。

## 本轮目标
- 把“单次高级流式分析”独立为 `市场快析` 模块。
- 保留 DeepThink 的多轮研判和控制台能力，不做功能删减。
- 保留历史链接可访问（兼容旧路由）。

## 具体改动

### 1) 新建 `market-quick`（单次流式分析）
- 文件：`frontend/app/market-quick/page.tsx`
- 文件：`frontend/app/market-quick/market-quick.module.css`

设计要点：
- 仅调用 `/v1/query/stream` + `/v1/market/overview/{stock}`。
- 不再在该页面触发 `/v1|v2/deep-think/*` 轮次接口。
- 页面内加入“模块边界说明”卡片，明确：
  - 市场快析 = 单次流式分析
  - DeepThink = 多 Agent 按轮研判 + 控制台
- 增加快捷提问模板按钮，减少用户自由输入成本。

关键注释：
- 在 `runMarketQuickAnalysis` 前增加注释，明确职责边界，防止后续维护再次把 DeepThink 轮次逻辑塞回该页面。

### 2) 旧路由兼容跳转
- 文件：`frontend/app/analysis-studio/page.tsx`

改造方式：
- 使用 Next.js `redirect("/market-quick")`。
- 目的：保留历史入口和外部书签，不中断使用。

### 3) DeepThink 文案边界修正
- 文件：`frontend/app/deep-think/page.tsx`

本轮仅做文案层优化，不改核心逻辑：
- “开始高级分析” -> “开始分析”。
- “无需先执行高级分析...” -> “无需先执行上方流式分析...”。
- 样本不足提示引导到“市场快析”补充一次分析。
- 保留“执行下一轮自动建会话”逻辑。

## 为什么这样拆分
- 市场快析强调“快速得到高质量单次回答”。
- DeepThink强调“多轮任务编排与冲突治理”。
- 二者共享后端能力，但前端职责分离后，用户路径更清晰，且不影响已有 DeepThink 工作流。

## 自测结果
1. 编译级检查（通过）
- 命令：
  `npx tsc --noEmit --pretty false --skipLibCheck --jsx preserve --moduleResolution bundler --module esnext --target es2022 --lib dom,dom.iterable,es2022 --types node,react,react-dom next-env.d.ts app/market-quick/page.tsx app/analysis-studio/page.tsx app/deep-think/page.tsx`

2. 生产构建（受环境影响未通过）
- 命令：`npm run build`
- 错误：`EPERM: operation not permitted, open '...frontend\\.next\\trace'`
- 判断：本地 `.next` 文件占用导致，非本轮代码逻辑错误。

## 影响范围
- 受影响页面：
  - `/market-quick`（新增）
  - `/analysis-studio`（重定向）
  - `/deep-think`（文案更新）
- 对后端接口无新增改动。

## 后续工作建议
1. 释放 `.next\\trace` 文件占用后重跑 `npm run build`。
2. 若确认本轮稳定，再把导航与首页文案显式改为“市场快析”，降低“分析润色”旧命名残留带来的认知偏差。
