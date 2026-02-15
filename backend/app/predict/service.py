from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import math
import uuid
from typing import Any

from backend.app.data.sources import QuoteService
from backend.app.observability.tracing import TraceStore


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalize_stock_code(stock_code: str) -> str:
    code = (stock_code or "").upper().replace(".", "")
    if code.startswith(("SH", "SZ")):
        return code
    if code.startswith("6"):
        return "SH" + code
    return "SZ" + code


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


@dataclass(slots=True)
class PredictionStore:
    """预测域存储（MVP 内存实现）。"""

    runs: dict[str, dict[str, Any]] = field(default_factory=dict)
    latest_eval: dict[str, Any] = field(default_factory=dict)


class PredictionService:
    """量化预测服务。

    首版目标：
    - 日频输入
    - 双周期预测（5d/20d）
    - 输出分层分数、上涨概率、风险等级、信号方向
    """

    def __init__(self, quote_service: QuoteService, traces: TraceStore, store: PredictionStore | None = None) -> None:
        self.quote_service = quote_service
        self.traces = traces
        self.store = store or PredictionStore()

    def run_prediction(self, stock_codes: list[str], horizons: list[str] | None = None, as_of_date: str | None = None) -> dict[str, Any]:
        """执行一次预测任务并保存运行结果。"""
        selected_horizons = horizons or ["5d", "20d"]
        run_id = str(uuid.uuid4())
        trace_id = self.traces.new_trace()
        as_of = as_of_date or _now_iso()

        items: list[dict[str, Any]] = []
        for code in stock_codes:
            normalized = _normalize_stock_code(code)
            quote = self.quote_service.get_quote(normalized)
            # 使用行情快照生成稳定的日频序列，确保本地可重复验证。
            synthetic_close = self._build_synthetic_close_series(normalized, quote.price)
            factor = self._compute_factors(synthetic_close, quote.turnover)
            forecast = self._forecast(factor, selected_horizons)
            items.append(
                {
                    "stock_code": normalized,
                    "as_of_date": as_of,
                    "horizons": forecast,
                    "factors": factor,
                    "source": {
                        "source_id": quote.source_id,
                        "source_url": quote.source_url,
                        "reliability_score": quote.reliability_score,
                    },
                }
            )

        result = {
            "run_id": run_id,
            "trace_id": trace_id,
            "as_of_date": as_of,
            "horizons": selected_horizons,
            "results": items,
        }
        self.store.runs[run_id] = result
        # 每次预测后刷新一份评测摘要，供运营页直接展示。
        self.store.latest_eval = self._build_eval_summary(run_id)
        return result

    def get_prediction(self, run_id: str) -> dict[str, Any]:
        return self.store.runs.get(run_id, {"error": "not_found", "run_id": run_id})

    def get_factor_snapshot(self, stock_code: str) -> dict[str, Any]:
        normalized = _normalize_stock_code(stock_code)
        quote = self.quote_service.get_quote(normalized)
        series = self._build_synthetic_close_series(normalized, quote.price)
        factor = self._compute_factors(series, quote.turnover)
        return {
            "stock_code": normalized,
            "as_of_date": _now_iso(),
            "factors": factor,
            "source": {
                "source_id": quote.source_id,
                "source_url": quote.source_url,
                "reliability_score": quote.reliability_score,
            },
        }

    def eval_latest(self) -> dict[str, Any]:
        if self.store.latest_eval:
            return self.store.latest_eval
        return {
            "status": "empty",
            "message": "no prediction run yet",
            "metrics": {},
        }

    def _build_synthetic_close_series(self, stock_code: str, anchor_price: float, days: int = 60) -> list[float]:
        """构建可重复的日频收盘序列。

        说明：
        - 不引入外部依赖，保证单测稳定。
        - 用股票代码做种子，避免每次随机值变化。
        """
        seed = sum(ord(c) for c in stock_code)
        base = max(anchor_price, 1.0)
        closes: list[float] = []
        price = base
        for idx in range(days):
            noise = (((seed * (idx + 7)) % 21) - 10) / 1000.0
            drift = (((seed + idx) % 9) - 4) / 2000.0
            price = max(0.5, price * (1 + drift + noise))
            closes.append(round(price, 4))
        return closes

    def _compute_factors(self, closes: list[float], turnover: float) -> dict[str, float]:
        """计算首版技术+风险因子。"""
        ret = self._returns(closes)
        ma5 = self._mean(closes[-5:])
        ma20 = self._mean(closes[-20:])
        current = closes[-1]
        momentum_5 = (current / closes[-6] - 1) if len(closes) >= 6 and closes[-6] else 0.0
        momentum_20 = (current / closes[-21] - 1) if len(closes) >= 21 and closes[-21] else 0.0
        vol_20 = self._std(ret[-20:])
        drawdown_20 = self._max_drawdown(closes[-20:])
        rsi_14 = self._rsi(closes, 14)
        liquidity = min(1.0, max(0.0, turnover / 100000000.0))
        risk_score = min(1.0, max(0.0, vol_20 * 8 + drawdown_20 * 4 + (1.0 - liquidity) * 0.5))

        return {
            "ma5_bias": round((current / ma5 - 1) if ma5 else 0.0, 6),
            "ma20_bias": round((current / ma20 - 1) if ma20 else 0.0, 6),
            "momentum_5": round(momentum_5, 6),
            "momentum_20": round(momentum_20, 6),
            "volatility_20": round(vol_20, 6),
            "drawdown_20": round(drawdown_20, 6),
            "rsi_14": round(rsi_14, 4),
            "liquidity_score": round(liquidity, 6),
            "risk_score": round(risk_score, 6),
        }

    def _forecast(self, factors: dict[str, float], horizons: list[str]) -> list[dict[str, Any]]:
        """把因子映射为双周期预测输出。"""
        # 线性组合用于首版可解释输出，后续可替换为训练模型。
        alpha_score = (
            factors["momentum_5"] * 0.35
            + factors["momentum_20"] * 0.35
            + factors["ma5_bias"] * 0.15
            + factors["ma20_bias"] * 0.15
            - factors["volatility_20"] * 0.2
        )
        up_prob = _sigmoid(alpha_score * 15)
        risk_score = factors["risk_score"]

        if risk_score >= 0.67:
            risk_tier = "high"
        elif risk_score >= 0.34:
            risk_tier = "medium"
        else:
            risk_tier = "low"

        rows: list[dict[str, Any]] = []
        for h in horizons:
            horizon_scale = 1.0 if h == "5d" else 1.8 if h == "20d" else 1.2
            expected_excess = alpha_score * horizon_scale
            if expected_excess >= 0.02:
                signal = "strong_buy"
            elif expected_excess >= 0.005:
                signal = "buy"
            elif expected_excess <= -0.02:
                signal = "strong_reduce"
            elif expected_excess <= -0.005:
                signal = "reduce"
            else:
                signal = "hold"
            rows.append(
                {
                    "horizon": h,
                    "score": round(alpha_score, 6),
                    "expected_excess_return": round(expected_excess, 6),
                    "up_probability": round(up_prob, 6),
                    "risk_tier": risk_tier,
                    "signal": signal,
                }
            )
        return rows

    def _build_eval_summary(self, run_id: str) -> dict[str, Any]:
        """生成最近一次预测的评测摘要。"""
        run = self.store.runs[run_id]
        all_rows: list[tuple[float, float]] = []
        for item in run["results"]:
            code = item["stock_code"]
            seed = sum(ord(c) for c in code)
            for horizon in item["horizons"]:
                pred = float(horizon["expected_excess_return"])
                # 用稳定噪声构建“准未来收益”，便于本地回归验证。
                future = pred * 0.6 + (((seed + len(horizon["horizon"])) % 11) - 5) / 500.0
                all_rows.append((pred, future))

        if not all_rows:
            return {"status": "empty", "metrics": {}, "run_id": run_id}

        preds = [x for x, _ in all_rows]
        futs = [y for _, y in all_rows]
        ic = self._corr(preds, futs)
        hit_rate = sum(1 for p, f in all_rows if (p >= 0 and f >= 0) or (p < 0 and f < 0)) / len(all_rows)
        spread = self._top_bottom_spread(all_rows)
        max_dd = self._mock_max_drawdown(all_rows)
        return {
            "status": "ok",
            "run_id": run_id,
            "generated_at": _now_iso(),
            "metrics": {
                "ic": round(ic, 6),
                "hit_rate": round(hit_rate, 6),
                "top_bottom_spread": round(spread, 6),
                "max_drawdown": round(max_dd, 6),
                "coverage": float(len(all_rows)),
            },
        }

    @staticmethod
    def _returns(closes: list[float]) -> list[float]:
        rows: list[float] = []
        for idx in range(1, len(closes)):
            prev = closes[idx - 1]
            if prev <= 0:
                rows.append(0.0)
            else:
                rows.append(closes[idx] / prev - 1)
        return rows

    @staticmethod
    def _mean(values: list[float]) -> float:
        return sum(values) / len(values) if values else 0.0

    @staticmethod
    def _std(values: list[float]) -> float:
        if not values:
            return 0.0
        avg = sum(values) / len(values)
        var = sum((v - avg) * (v - avg) for v in values) / len(values)
        return math.sqrt(var)

    @staticmethod
    def _max_drawdown(closes: list[float]) -> float:
        if not closes:
            return 0.0
        peak = closes[0]
        max_dd = 0.0
        for c in closes:
            peak = max(peak, c)
            dd = (peak - c) / peak if peak else 0.0
            max_dd = max(max_dd, dd)
        return max_dd

    @staticmethod
    def _rsi(closes: list[float], n: int) -> float:
        if len(closes) < n + 1:
            return 50.0
        gains = 0.0
        losses = 0.0
        for i in range(-n, 0):
            diff = closes[i] - closes[i - 1]
            if diff >= 0:
                gains += diff
            else:
                losses -= diff
        if losses == 0:
            return 100.0
        rs = gains / losses
        return 100 - (100 / (1 + rs))

    @staticmethod
    def _corr(xs: list[float], ys: list[float]) -> float:
        if not xs or len(xs) != len(ys):
            return 0.0
        mx = sum(xs) / len(xs)
        my = sum(ys) / len(ys)
        num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
        den_x = math.sqrt(sum((x - mx) ** 2 for x in xs))
        den_y = math.sqrt(sum((y - my) ** 2 for y in ys))
        if den_x == 0 or den_y == 0:
            return 0.0
        return num / (den_x * den_y)

    @staticmethod
    def _top_bottom_spread(rows: list[tuple[float, float]]) -> float:
        ordered = sorted(rows, key=lambda x: x[0], reverse=True)
        n = max(1, len(ordered) // 3)
        top = [x[1] for x in ordered[:n]]
        bottom = [x[1] for x in ordered[-n:]]
        return (sum(top) / len(top)) - (sum(bottom) / len(bottom))

    @staticmethod
    def _mock_max_drawdown(rows: list[tuple[float, float]]) -> float:
        equity = 1.0
        peak = 1.0
        max_dd = 0.0
        for _, future_ret in rows:
            equity *= 1 + future_ret
            peak = max(peak, equity)
            dd = (peak - equity) / peak if peak else 0.0
            max_dd = max(max_dd, dd)
        return max_dd
