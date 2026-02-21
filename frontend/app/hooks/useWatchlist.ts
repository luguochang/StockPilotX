"use client";

import { useEffect, useMemo, useState } from "react";
import { fetchJson } from "../lib/api";
import { CTX_LAST_RUN, CTX_POOL_ID, CTX_POOL_NAME, CTX_STOCK_CODE, readCtx, writeCtx } from "../lib/context";

export type WatchItem = { stock_code: string; created_at: string };
export type WatchPool = { pool_id: string; pool_name: string; description: string; stock_count: number; is_default: boolean };
export type PoolStock = { stock_code: string; stock_name?: string; exchange?: string; market_tier?: string; industry_l1?: string };

export function useWatchlist() {
  const [stock, setStock] = useState("SH600000");
  const [items, setItems] = useState<WatchItem[]>([]);
  const [pools, setPools] = useState<WatchPool[]>([]);
  const [selectedPoolId, setSelectedPoolId] = useState("");
  const [poolStocks, setPoolStocks] = useState<PoolStock[]>([]);
  const [loading, setLoading] = useState(false);
  const [initialLoading, setInitialLoading] = useState(true);
  const [error, setError] = useState("");
  const [ctxLastRunAt, setCtxLastRunAt] = useState("");

  const latest = useMemo(
    () => items.slice().sort((a, b) => String(b.created_at).localeCompare(String(a.created_at)))[0],
    [items]
  );

  async function refresh() {
    const body = (await fetchJson("/v1/watchlist")) as WatchItem[];
    setItems(Array.isArray(body) ? body : []);
  }

  async function refreshPools() {
    const rows = (await fetchJson("/v1/watchlist/pools")) as WatchPool[];
    const next = Array.isArray(rows) ? rows : [];
    setPools(next);
    if (!selectedPoolId && next.length) {
      const fallback = next.find((x) => Boolean(x.is_default))?.pool_id ?? next[0].pool_id;
      setSelectedPoolId(fallback);
    }
  }

  async function refreshPoolStocks(poolId: string) {
    if (!poolId) {
      setPoolStocks([]);
      return;
    }
    const rows = (await fetchJson(`/v1/watchlist/pools/${poolId}/stocks`)) as PoolStock[];
    setPoolStocks(Array.isArray(rows) ? rows : []);
  }

  async function refreshAll() {
    await Promise.all([refresh(), refreshPools()]);
  }

  async function createPool() {
    const name = window.prompt("输入关注池名称");
    if (!name?.trim()) return;
    const created = (await fetchJson("/v1/watchlist/pools", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ pool_name: name.trim(), description: "", is_default: pools.length === 0 })
    })) as { pool_id?: string };

    await refreshPools();
    if (created?.pool_id) {
      setSelectedPoolId(created.pool_id);
      await refreshPoolStocks(created.pool_id);
    }
  }

  async function addStock() {
    setLoading(true);
    setError("");
    try {
      await fetchJson("/v1/watchlist", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ stock_code: stock })
      });
      if (selectedPoolId) {
        await fetchJson(`/v1/watchlist/pools/${selectedPoolId}/stocks`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ stock_code: stock, source_filters: {} })
        });
      }
      await refreshAll();
      if (selectedPoolId) await refreshPoolStocks(selectedPoolId);
    } catch (e) {
      setError(e instanceof Error ? e.message : "请求失败");
    } finally {
      setLoading(false);
    }
  }

  async function removeFromPool(stockCode: string) {
    if (!selectedPoolId) return;
    setLoading(true);
    setError("");
    try {
      await fetchJson(`/v1/watchlist/pools/${selectedPoolId}/stocks/${stockCode}`, { method: "DELETE" });
      await refreshPools();
      await refreshPoolStocks(selectedPoolId);
    } catch (e) {
      setError(e instanceof Error ? e.message : "请求失败");
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    let alive = true;
    (async () => {
      try {
        await refreshAll();
      } catch (e) {
        if (alive) setError(e instanceof Error ? e.message : "初始化失败");
      } finally {
        if (alive) setInitialLoading(false);
      }
    })();
    setCtxLastRunAt(readCtx(CTX_LAST_RUN));
    return () => {
      alive = false;
    };
  }, []);

  useEffect(() => {
    if (!selectedPoolId) return;
    void refreshPoolStocks(selectedPoolId).catch((e) => setError(e instanceof Error ? e.message : "加载失败"));
  }, [selectedPoolId]);

  useEffect(() => {
    writeCtx(CTX_STOCK_CODE, stock);
    if (!selectedPoolId) return;
    writeCtx(CTX_POOL_ID, selectedPoolId);
    const name = pools.find((p) => p.pool_id === selectedPoolId)?.pool_name ?? "";
    if (name) writeCtx(CTX_POOL_NAME, name);
  }, [stock, selectedPoolId, pools]);

  return {
    stock,
    setStock,
    items,
    pools,
    selectedPoolId,
    setSelectedPoolId,
    poolStocks,
    loading,
    initialLoading,
    error,
    setError,
    ctxLastRunAt,
    latest,
    createPool,
    addStock,
    removeFromPool,
    refresh,
    refreshPools,
    refreshAll
  };
}

