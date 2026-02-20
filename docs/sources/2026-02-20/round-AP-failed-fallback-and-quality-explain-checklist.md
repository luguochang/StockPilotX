# ROUND-AP 失败降级结果 + 质量解释模板 + 阶段文案优化

## 目标
- 在报告任务失败时，前端仍能拿到可展示结果，避免“失败后空白”。
- 将报告/降级原因统一映射为可读解释与补救动作，减少泛化报错。
- 报告页任务阶段改为业务可读文案，提升用户可理解性。

## Checklist

### A. 任务失败可展示结果（Backend）
- [x] 新增 `_build_report_task_failed_result`，在超时/停滞/失败时构建轻量降级报告。
- [x] `_report_task_mark_failed` 自动注入失败降级结果（若无 partial/full）。
- [x] 失败状态下 `report_task_snapshot` 改为 `display_ready=true`（有 fallback 时可直接展示）。
- [x] 失败状态下 `report_task_result` 返回 `partial_reason=failed_fallback`。

### B. 统一质量解释模板（Backend）
- [x] 新增 `_build_quality_explain`，将 reason code 映射为「影响 + 补救动作」。
- [x] 报告主流程 `degrade.user_message` 改为解释模板摘要。
- [x] `degrade.explain` 新增结构化解释对象（summary/items/actions）。
- [x] partial/failure 降级结果统一接入解释模板。
- [x] DeepThink business_summary 注入 `quality_explain`（来自 data gap）。

### C. 阶段业务文案与展示策略（Frontend）
- [x] 新增任务阶段标签映射：`queued/partial/full_report/done/failed...` -> 中文业务文案。
- [x] 新增阶段提示文案（stage hint）。
- [x] 任务失败且 result=partial 时默认展示降级结果（不受临时预览开关限制）。
- [x] 失败场景增加显式提示：已自动展示降级结果用于排障。

### D. 测试与回归
- [x] 单测新增：`test_report_task_runtime_guard_returns_failed_fallback`。
- [x] 后端回归：`tests/test_service.py` + `tests/test_http_api.py`。
- [x] 全接口 smoke：`scripts/full_api_selftest.py`。
- [x] 前端生产构建：`npm run build`。

## 自测记录
1) 后端单测
```bash
.venv\Scripts\python -m pytest tests/test_service.py tests/test_http_api.py -q
```
结果：`84 passed`

2) 全接口 smoke
```bash
.venv\Scripts\python scripts/full_api_selftest.py
```
结果：`total=130, failed=0`

3) 前端构建
```bash
cd frontend
npm run build
```
结果：通过，静态页面 `19/19`。

4) 前端类型检查
```bash
cd frontend
npx tsc --noEmit
```
结果：通过。

## 变更文件
- `backend/app/service.py`
- `frontend/app/reports/page.tsx`
- `tests/test_service.py`
- `tests/test_http_api.py`
- `docs/sources/2026-02-20/round-AP-failed-fallback-and-quality-explain-checklist.md`

## 提交
- Commit: 见本轮提交记录（`git log --oneline`）
