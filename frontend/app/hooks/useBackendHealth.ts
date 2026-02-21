"use client";

import { useEffect, useState } from "react";
import { API_BASE } from "../lib/api";

export function useBackendHealth(pollMs: number = 15000) {
  const [status, setStatus] = useState<"online" | "offline" | "checking">("checking");

  useEffect(() => {
    let alive = true;
    let timer: ReturnType<typeof setTimeout> | null = null;

    const probe = async () => {
      try {
        const ctrl = new AbortController();
        const timeout = setTimeout(() => ctrl.abort(), 3000);
        await fetch(`${API_BASE}/v1/watchlist/pools`, { method: "GET", signal: ctrl.signal });
        clearTimeout(timeout);
        if (alive) setStatus("online");
      } catch {
        if (alive) setStatus("offline");
      } finally {
        if (alive) timer = setTimeout(probe, pollMs);
      }
    };

    void probe();
    return () => {
      alive = false;
      if (timer) clearTimeout(timer);
    };
  }, [pollMs]);

  return status;
}
