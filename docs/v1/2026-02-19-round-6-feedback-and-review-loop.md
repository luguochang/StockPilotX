# ROUND-6 技术记录：采纳反馈与偏差复盘闭环

## 1. 本轮目标

建立 Intel Card 的“建议 -> 用户反馈 -> 后验复盘”闭环，支撑后续策略校准。

目标对应：

1. 用户采纳记录  
2. T+1 / T+5 / T+20 偏差统计  
3. 复盘指标落库与前端可视化

---

## 2. 后端实现

## 2.1 数据落库

文件：`backend/app/web/store.py`

新增表：
- `analysis_intel_feedback`
  - stock_code
  - trace_id
  - signal
  - confidence
  - position_hint
  - feedback（adopt/watch/reject）
  - baseline_trade_date
  - baseline_price
  - created_at

新增索引：
- `idx_analysis_intel_feedback_stock_time`

## 2.2 Web 服务层

文件：`backend/app/web/service.py`

新增方法：
- `analysis_intel_feedback_add(...)`
- `analysis_intel_feedback_list(...)`

## 2.3 业务服务层

文件：`backend/app/service.py`

新增方法：
- `analysis_intel_feedback(payload)`
  - 写入反馈记录，保存基准日与基准价格
- `_analysis_forward_return(...)`
  - 计算指定反馈样本的 T+n 收益偏差
- `analysis_intel_review(stock_code, limit)`
  - 汇总 T+1/T+5/T+20 平均收益与命中率
  - 返回样本明细供前端展示

## 2.4 API 暴露

文件：`backend/app/http_api.py`

新增接口：
- `POST /v1/analysis/intel-card/feedback`
- `GET /v1/analysis/intel-card/review`

---

## 3. 前端实现

文件：`frontend/app/deep-think/page.tsx`

在业务卡片新增：

1. 反馈动作按钮  
- 采纳本次建议  
- 继续观察  
- 拒绝本次建议

2. 复盘统计卡  
- T+1/T+5/T+20 样本数  
- 平均收益  
- 命中率  
- 最新反馈样本列表（含区间偏差）

新增页面状态：
- `intelFeedbackLoading`
- `intelFeedbackMessage`
- `intelReviewLoading`
- `intelReview`

---

## 4. 测试与结果

## 4.1 后端服务测试

命令：
```powershell
.\.venv\Scripts\python.exe -m pytest -q tests/test_service.py -k "analysis_intel_card_contract or analysis_intel_feedback_and_review"
```
结果：`2 passed, 39 deselected`

## 4.2 API 契约测试

命令：
```powershell
.\.venv\Scripts\python.exe -m pytest -q tests/test_http_api.py -k "analysis_intel_card or analysis_intel_feedback_and_review"
```
结果：`2 passed, 34 deselected`

## 4.3 前端编译检查

命令：
```powershell
cd frontend
npx tsc --noEmit
```
结果：通过（exit code 0）

---

## 5. 后续建议

1. 目前复盘以规则命中定义为主，后续可引入按行业/波动分层的命中口径。  
2. 建议增加“采纳后自动提醒复盘时间窗”的通知机制。  
3. ROUND-7 可进行全链路回归、稳定性压测与交付收口。
