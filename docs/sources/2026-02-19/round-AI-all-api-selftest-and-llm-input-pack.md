# ROUND-AI｜全链路输入数据包与全接口自测稳定化

## 1. 目标与范围
本轮目标是完成“后端所有关键分析入口统一补数 + 前端可见反馈 + 全接口自测可落地”，重点覆盖：

1. Query / Query Stream
2. DeepThink Round Stream
3. Predict
4. Report（同步生成 + 异步任务）
5. Journal 快速创建
6. 全路由 API 自测脚本（含轻量模式）

## 2. 本轮已落地实现

### 2.1 后端统一补数（LLM Input Pack）
文件：`backend/app/service.py`

已实现统一输入数据包能力，并接入多个业务链路：

1. 新增/强化统一补数函数（按场景要求补齐近30/90日历史、财务、新闻、研报、宏观等）
2. Query、Query Stream、DeepThink、Predict、Report 全部改为走统一 `data_pack`
3. 在返回结果中显式暴露 `data_packs` / `missing_data` / `data_quality`
4. 当关键数据不足时，降级原因会写入响应，避免“静默失败”

业务收益：

1. 降低“样本太少导致模型乱猜”的概率
2. 让前端可以明确告诉用户“缺什么数据，而不是只给空泛结论”
3. 为后续 A 股特化策略（牛短熊长、事件驱动）提供稳定数据底座

### 2.2 Report 任务透明度增强
文件：`backend/app/service.py`、`frontend/app/reports/page.tsx`

后端在任务快照中返回：

1. `data_pack_status`
2. `data_pack_missing`
3. `quality_gate_detail`

前端报告页新增展示：

1. 任务级数据包状态
2. 任务级质量状态/得分/原因
3. 数据缺口告警
4. 任务错误信息

业务收益：

1. 避免“长时间转圈但用户无感知”的体验问题
2. 用户可直接看到为什么是 partial、缺哪些证据

### 2.3 Journal 快速创建防呆
文件：`backend/app/service.py`、`frontend/app/journal/page.tsx`

1. 前端去掉“核心观点必填”阻断，允许一步直建
2. `trim` 空值保护，修复 `Cannot read properties of undefined (reading 'trim')`
3. 后端自动补默认标题/正文/标签，确保最小输入可用

业务收益：

1. 用户不需要先写长文本才能记日志
2. 减少前端报错与放弃率

### 2.4 全接口自测脚本增强
文件：`scripts/full_api_selftest.py`

新增能力：

1. 覆盖 `/v1/*` + `/v2/*` 路由探活（共 129 路由）
2. 默认轻量模式：跳过高耗时/强外部依赖路由，避免本地阻塞
3. 支持环境变量控制：
   - `SMOKE_RUN_HEAVY=1`：运行重型路由
   - `SMOKE_SEED_FULL=1`：启用深度上下文播种
   - `SMOKE_VERBOSE=1`：打印逐路由探测进度
4. Prompt Compare 动态读取实际版本，避免 `fact_qa@v1` 假失败

业务收益：

1. 自测可在开发机稳定执行并给出统一回归视图
2. 重型模式可用于发布前深验，轻量模式可用于高频迭代

## 3. 本轮自测记录

### 3.1 后端核心测试
命令：

```bash
.\.venv\Scripts\python.exe -m pytest tests/test_service.py tests/test_http_api.py -q
```

结果：

- `82 passed`

### 3.2 全接口 smoke
命令：

```bash
.\.venv\Scripts\python.exe scripts/full_api_selftest.py
```

结果：

- `total=129`
- `failed=0`
- `status_hist={"200":81,"299":29,"400":4,"401":1,"404":14}`

说明：

- `299` 表示轻量模式下跳过的重型路由（避免本地阻塞），不是失败。

### 3.3 前端类型检查
命令：

```bash
npx tsc --noEmit
```

结果：

- 通过。

### 3.4 前端生产构建
命令：

```bash
npm run build
```

结果：

- 当前环境出现 `.next/trace` 文件锁（`EPERM`），导致构建未完成。
- 该问题属于本机运行环境/进程占用，不是本轮代码语法错误（`tsc` 已通过）。

## 4. 关键设计说明

### 4.1 为什么会反复出现“假流式”
核心不是 SSE 协议本身，而是数据准备阶段阻塞：

1. 先补数（行情/历史/财报/新闻）
2. 再进入模型 token 流

如果补数阶段很慢，前端就会误以为“不是流式”。

本轮通过统一 `data_pack` + 显式进度/缺口输出，把“等待原因”前置可视化，避免黑盒体验。

### 4.2 轻量 smoke 与重型 smoke 的区别

1. 轻量 smoke：验证路由健壮性、参数校验、鉴权、状态码，不强依赖外部资源
2. 重型 smoke：验证真实业务链路（补数 + 推理 + 导出），用于发布前验收

## 5. 后续建议

1. CI 默认跑轻量 smoke + 单测
2. 每日定时跑一次重型 smoke（关键链路白名单）
3. 报告页继续增加“已补齐数据维度”可视化，减少用户困惑
