"use client";

import { useEffect, useRef, useState } from "react";
import { fetchJson } from "../lib/api";
import { CTX_LAST_RUN, CTX_POOL_ID, CTX_POOL_NAME, CTX_STOCK_CODE, readCtx, writeCtx } from "../lib/context";

export type HorizonResult = {
  horizon: string;
  score: number;
  expected_excess_return: number;
  up_probability: number;
  risk_tier: string;
  signal: string;
  rationale?: string;
};

export type PredictItem = {
  stock_code: string;
  horizons: HorizonResult[];
  data_quality?: string;
  degrade_reasons?: string[];
  source?: {
    history_data_mode?: string;
    history_source_id?: string;
    history_sample_size?: number;
    history_degraded?: boolean;
    history_degrade_reason?: string;
  };
};

export type PredictRunResponse = {
  run_id: string;
  trace_id: string;
  results: PredictItem[];
  pool_id?: string;
  segment_metrics?: Record<string, Array<{ segment: string; count: number; avg_expected_excess_return: number; avg_up_probability: number }>>;
  data_quality?: string;
  degrade_reasons?: string[];
  source_coverage?: {
    total?: number;
    real_history_count?: number;
    synthetic_count?: number;
    real_history_ratio?: number;
  };
  metric_mode?: string;
  metrics_note?: string;
  metrics_live?: Record<string, number>;
  metrics_backtest?: Record<string, number>;
  metrics_simulated?: Record<string, number>;
  eval_provenance?: {
    coverage_rows?: number;
    evaluated_stocks?: number;
    skipped_stocks?: Array<{ stock_code?: string; reason?: string }>;
    history_modes?: Record<string, number>;
    fallback_reason?: string;
    run_data_quality?: string;
  };
  quality_gate?: {
    overall_status?: "pass" | "watch" | "degraded";
    dimensions?: Record<string, { status?: "pass" | "watch" | "degraded"; reason_count?: number; reasons?: string[] }>;
    reasons?: string[];
    reason_details?: Array<{ code?: string; dimension?: string; title?: string; impact?: string; action?: string }>;
    user_message?: string;
    actions?: string[];
  };
  engine_profile?: {
    prediction_engine?: string;
    llm_used_in_scoring?: boolean;
    llm_used_in_explain?: boolean;
    latency_mode?: string;
  };
};

export type EvalResponse = {
  status: string;
  metrics?: Record<string, number>;
  metric_mode?: string;
  metrics_note?: string;
  evaluated_stocks?: number;
  skipped_stocks?: Array<{ stock_code?: string; reason?: string }>;
  history_modes?: Record<string, number>;
  fallback_reason?: string;
};

type PoolItem = { pool_id: string; pool_name: string; stock_count: number; is_default?: boolean };
type PoolStock = { stock_code: string; stock_name?: string; exchange?: string; market_tier?: string; industry_l1?: string };

export type PredictExplainResponse = {
  run_id: string;
  trace_id: string;
  stock_code: string;
  horizon: string;
  signal: string;
  risk_tier: string;
  expected_excess_return: number;
  up_probability: number;
  summary: string;
  drivers: string[];
  risks: string[];
  actions: string[];
  llm_used: boolean;
  provider?: string;
  model?: string;
  degraded_reason?: string;
  generated_at: string;
};

export function usePredict() {
  const [selectedCodes, setSelectedCodes] = useState<string[]>(["SH600000"]);
  const [pools, setPools] = useState<PoolItem[]>([]);
  const [selectedPoolId, setSelectedPoolId] = useState("");
  const [poolStocks, setPoolStocks] = useState<PoolStock[]>([]);
  const [manualMode, setManualMode] = useState(false);
  const [loading, setLoading] = useState(false);
  const [initialLoading, setInitialLoading] = useState(true);
  const [error, setError] = useState("");
  const [run, setRun] = useState<PredictRunResponse | null>(null);
  const [evalData, setEvalData] = useState<EvalResponse | null>(null);
  const [explain, setExplain] = useState<PredictExplainResponse | null>(null);
  const [explainLoading, setExplainLoading] = useState(false);
  const [explainError, setExplainError] = useState("");
  const [ctxStock, setCtxStock] = useState("");
  const [ctxLastRunAt, setCtxLastRunAt] = useState("");
  const explainRunRef = useRef("");

  async function loadPools() {
    const body = (await fetchJson("/v1/watchlist/pools")) as PoolItem[];
    const rows = Array.isArray(body) ? body : [];
    setPools(rows);
    if (!selectedPoolId && rows.length) {
      const fallback = rows.find((x) => Boolean(x.is_default))?.pool_id ?? rows[0].pool_id;
      setSelectedPoolId(String(fallback ?? ""));
    }
  }

  async function loadPoolStocks(poolId: string) {
    if (!poolId) {
      setPoolStocks([]);
      return;
    }
    const rows = (await fetchJson(`/v1/watchlist/pools/${poolId}/stocks`)) as PoolStock[];
    setPoolStocks(Array.isArray(rows) ? rows : []);
  }

  async function runPredict() {
    setLoading(true);
    setError("");
    setExplain(null);
    setExplainError("");
    try {
      if (!manualMode && !selectedPoolId) throw new Error("请先选择关注池");
      if (manualMode && !selectedCodes.length) throw new Error("手动模式下请至少选择一只股票");

      const body = (await fetchJson("/v1/predict/run", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(
          manualMode ? { stock_codes: selectedCodes, horizons: ["5d", "20d"] } : { pool_id: selectedPoolId, horizons: ["5d", "20d"] }
        )
      })) as PredictRunResponse;

      setRun(body);
      explainRunRef.current = body.run_id;
      const now = new Date().toLocaleString();
      setCtxLastRunAt(now);
      writeCtx(CTX_LAST_RUN, now);

      if (manualMode && selectedCodes[0]) {
        writeCtx(CTX_STOCK_CODE, selectedCodes[0]);
      }
      if (!manualMode && selectedPoolId) {
        writeCtx(CTX_POOL_ID, selectedPoolId);
        const name = pools.find((p) => p.pool_id === selectedPoolId)?.pool_name ?? "";
        if (name) writeCtx(CTX_POOL_NAME, name);
      }

      const explainStock = String(body.results?.[0]?.stock_code ?? "").trim().toUpperCase();
      const explainHorizon =
        String(body.results?.[0]?.horizons?.find((x) => String(x?.horizon ?? "").toLowerCase() === "20d")?.horizon ?? "")
        || String(body.results?.[0]?.horizons?.[0]?.horizon ?? "20d");
      if (body.run_id && explainStock) {
        setExplainLoading(true);
        const currentRunId = body.run_id;
        void (async () => {
          try {
            const explainBody = (await fetchJson("/v1/predict/explain", {
              method: "POST",
              headers: { "Content-Type": "application/json" },
              body: JSON.stringify({ run_id: currentRunId, stock_code: explainStock, horizon: explainHorizon }),
            })) as PredictExplainResponse;
            if (explainRunRef.current !== currentRunId) return;
            setExplain(explainBody);
            setExplainError("");
          } catch (e) {
            if (explainRunRef.current !== currentRunId) return;
            setExplainError(e instanceof Error ? e.message : "解释生成失败");
          } finally {
            if (explainRunRef.current === currentRunId) setExplainLoading(false);
          }
        })();
      } else {
        setExplainLoading(false);
      }

      const evalBody = (await fetchJson("/v1/predict/evals/latest")) as EvalResponse;
      setEvalData(evalBody);
    } catch (e) {
      setError(e instanceof Error ? e.message : "请求失败");
      setRun(null);
      setExplainLoading(false);
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    if (manualMode) {
      setPoolStocks([]);
      return;
    }
    if (!selectedPoolId) {
      setPoolStocks([]);
      return;
    }
    void loadPoolStocks(selectedPoolId).catch((e) => setError(e instanceof Error ? e.message : "加载关注池明细失败"));
  }, [selectedPoolId, manualMode]);

  useEffect(() => {
    let alive = true;
    (async () => {
      try {
        await loadPools();
      } catch (e) {
        if (alive) setError(e instanceof Error ? e.message : "加载关注池失败");
      } finally {
        if (alive) setInitialLoading(false);
      }
    })();
    setCtxStock(readCtx(CTX_STOCK_CODE));
    setCtxLastRunAt(readCtx(CTX_LAST_RUN));

    return () => {
      alive = false;
    };
  }, []);

  return {
    selectedCodes,
    setSelectedCodes,
    pools,
    selectedPoolId,
    setSelectedPoolId,
    poolStocks,
    manualMode,
    setManualMode,
    loading,
    initialLoading,
    error,
    setError,
    run,
    evalData,
    explain,
    explainLoading,
    explainError,
    ctxStock,
    ctxLastRunAt,
    loadPools,
    loadPoolStocks,
    runPredict
  };
}
