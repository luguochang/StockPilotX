"use client";

import { useEffect, useState } from "react";
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

export function usePredict() {
  const [selectedCodes, setSelectedCodes] = useState<string[]>(["SH600000"]);
  const [pools, setPools] = useState<PoolItem[]>([]);
  const [selectedPoolId, setSelectedPoolId] = useState("");
  const [manualMode, setManualMode] = useState(false);
  const [loading, setLoading] = useState(false);
  const [initialLoading, setInitialLoading] = useState(true);
  const [error, setError] = useState("");
  const [run, setRun] = useState<PredictRunResponse | null>(null);
  const [evalData, setEvalData] = useState<EvalResponse | null>(null);
  const [ctxStock, setCtxStock] = useState("");
  const [ctxLastRunAt, setCtxLastRunAt] = useState("");

  async function loadPools() {
    const body = (await fetchJson("/v1/watchlist/pools")) as PoolItem[];
    const rows = Array.isArray(body) ? body : [];
    setPools(rows);
    if (!selectedPoolId && rows.length) {
      const fallback = rows.find((x) => Boolean(x.is_default))?.pool_id ?? rows[0].pool_id;
      setSelectedPoolId(String(fallback ?? ""));
    }
  }

  async function runPredict() {
    setLoading(true);
    setError("");
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

      const evalBody = (await fetchJson("/v1/predict/evals/latest")) as EvalResponse;
      setEvalData(evalBody);
    } catch (e) {
      setError(e instanceof Error ? e.message : "请求失败");
      setRun(null);
    } finally {
      setLoading(false);
    }
  }

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
    manualMode,
    setManualMode,
    loading,
    initialLoading,
    error,
    setError,
    run,
    evalData,
    ctxStock,
    ctxLastRunAt,
    loadPools,
    runPredict
  };
}

