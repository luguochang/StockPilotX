"use client";

import { useEffect, useState } from "react";
import { fetchJson } from "../lib/api";
import { CTX_LAST_RUN, CTX_POOL_NAME, CTX_STOCK_CODE, readCtx, writeCtx } from "../lib/context";

export type ReportItem = {
  report_id: string;
  stock_code: string;
  report_type: string;
  run_id?: string;
  pool_snapshot_id?: string;
  template_id?: string;
  created_at: string;
};

export type BusinessModuleHealth = {
  module: string;
  status: "ok" | "degraded" | "critical" | string;
  coverage: number;
  healthy_categories: number;
  expected_categories: number;
  degrade_reasons: string[];
};

export type BusinessDataHealth = {
  status: "ok" | "degraded" | "critical" | string;
  module_health: BusinessModuleHealth[];
  degrade_reasons: string[];
  category_health_count: number;
  generated_at: string;
  stock_snapshot?: {
    stock_code?: string;
    has_quote?: boolean;
    history_sample_size?: number;
    has_financial?: boolean;
    news_count?: number;
    research_count?: number;
  };
};

export function useReports() {
  const [items, setItems] = useState<ReportItem[]>([]);
  const [selectedMarkdown, setSelectedMarkdown] = useState("");
  const [selectedRaw, setSelectedRaw] = useState("");
  const [selectedVersions, setSelectedVersions] = useState("");
  const [generateStockCode, setGenerateStockCode] = useState("SH600000");
  const [generateType, setGenerateType] = useState<"fact" | "research">("research");
  const [loading, setLoading] = useState(false);
  const [initialLoading, setInitialLoading] = useState(true);
  const [error, setError] = useState("");
  const [activeReportId, setActiveReportId] = useState("");
  const [advancedOpen, setAdvancedOpen] = useState<string[]>([]);
  const [templateId, setTemplateId] = useState("default");
  const [runId, setRunId] = useState("");
  const [poolSnapshotId, setPoolSnapshotId] = useState("");
  const [sectionPreview, setSectionPreview] = useState("");
  const [evidencePreview, setEvidencePreview] = useState("");
  const [qualityGatePreview, setQualityGatePreview] = useState("");
  const [dataPackPreview, setDataPackPreview] = useState("");
  const [generationMode, setGenerationMode] = useState("");
  const [businessHealth, setBusinessHealth] = useState<BusinessDataHealth | null>(null);
  const [ctxPoolName, setCtxPoolName] = useState("");
  const [ctxStock, setCtxStock] = useState("");
  const [ctxLastRunAt, setCtxLastRunAt] = useState("");

  async function load() {
    setLoading(true);
    setError("");
    try {
      const body = (await fetchJson("/v1/reports")) as ReportItem[];
      setItems(Array.isArray(body) ? body : []);
    } catch (e) {
      setError(e instanceof Error ? e.message : "请求失败");
      setItems([]);
    } finally {
      setLoading(false);
    }
  }

  async function exportReport(reportId: string) {
    setLoading(true);
    setError("");
    try {
      const body = (await fetchJson(`/v1/reports/${reportId}/export`, { method: "POST" })) as { markdown?: string };
      setSelectedMarkdown(body.markdown ?? "");
      setActiveReportId(reportId);
    } catch (e) {
      setError(e instanceof Error ? e.message : "请求失败");
      setSelectedMarkdown("");
    } finally {
      setLoading(false);
    }
  }

  async function loadBusinessHealth(stockCode: string = generateStockCode) {
    const normalized = String(stockCode ?? "").trim().toUpperCase();
    if (!normalized) {
      setBusinessHealth(null);
      return;
    }
    try {
      // Keep this request non-blocking; report generation should not fail because health check is temporarily unavailable.
      const body = (await fetchJson(`/v1/business/data-health?stock_code=${encodeURIComponent(normalized)}&limit=200`)) as BusinessDataHealth;
      setBusinessHealth(body);
    } catch {
      setBusinessHealth(null);
    }
  }

  async function generateReport() {
    setLoading(true);
    setError("");
    try {
      const payload = {
        user_id: "demo_user",
        stock_code: generateStockCode.trim().toUpperCase(),
        period: "1y",
        report_type: generateType,
        template_id: templateId.trim() || "default",
        run_id: runId.trim(),
        pool_snapshot_id: poolSnapshotId.trim()
      };
      const body = (await fetchJson("/v1/report/generate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
      })) as Record<string, unknown>;

      setSelectedMarkdown(String(body.markdown ?? ""));
      setSelectedRaw(JSON.stringify(body, null, 2));
      setSectionPreview(JSON.stringify((body.report_sections as unknown[]) ?? [], null, 2));
      setEvidencePreview(JSON.stringify((body.evidence_refs as unknown[]) ?? [], null, 2));
      setQualityGatePreview(JSON.stringify((body.quality_gate as Record<string, unknown>) ?? {}, null, 2));
      setDataPackPreview(JSON.stringify((body.report_data_pack_summary as Record<string, unknown>) ?? {}, null, 2));
      setGenerationMode(String(body.generation_mode ?? ""));
      setActiveReportId(String(body.report_id ?? ""));

      writeCtx(CTX_STOCK_CODE, payload.stock_code);
      writeCtx(CTX_LAST_RUN, new Date().toISOString());
      setCtxStock(payload.stock_code);
      setCtxLastRunAt(readCtx(CTX_LAST_RUN));
      await loadBusinessHealth(payload.stock_code);
      await load();
    } catch (e) {
      setError(e instanceof Error ? e.message : "请求失败");
    } finally {
      setLoading(false);
    }
  }

  async function loadReportDetail(reportId: string) {
    setLoading(true);
    setError("");
    try {
      const body = (await fetchJson(`/v1/report/${reportId}`)) as Record<string, unknown>;
      setSelectedRaw(JSON.stringify(body, null, 2));
      setSelectedMarkdown(String(body.markdown ?? ""));
      setSectionPreview(JSON.stringify((body.report_sections as unknown[]) ?? [], null, 2));
      setEvidencePreview(JSON.stringify((body.evidence_refs as unknown[]) ?? [], null, 2));
      setQualityGatePreview(JSON.stringify((body.quality_gate as Record<string, unknown>) ?? {}, null, 2));
      setDataPackPreview(JSON.stringify((body.report_data_pack_summary as Record<string, unknown>) ?? {}, null, 2));
      setGenerationMode(String(body.generation_mode ?? ""));
      setActiveReportId(reportId);
      const detailStock = String(body.stock_code ?? "").trim().toUpperCase();
      if (detailStock) {
        setGenerateStockCode(detailStock);
        writeCtx(CTX_STOCK_CODE, detailStock);
        setCtxStock(detailStock);
        await loadBusinessHealth(detailStock);
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : "请求失败");
    } finally {
      setLoading(false);
    }
  }

  async function loadReportVersions(reportId: string) {
    setLoading(true);
    setError("");
    try {
      const body = await fetchJson(`/v1/reports/${reportId}/versions`);
      setSelectedVersions(JSON.stringify(body, null, 2));
      setActiveReportId(reportId);
    } catch (e) {
      setError(e instanceof Error ? e.message : "请求失败");
      setSelectedVersions("");
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    let alive = true;
    (async () => {
      try {
        await Promise.all([load(), loadBusinessHealth(generateStockCode)]);
      } finally {
        if (alive) setInitialLoading(false);
      }
    })();
    setCtxPoolName(readCtx(CTX_POOL_NAME));
    setCtxStock(readCtx(CTX_STOCK_CODE));
    setCtxLastRunAt(readCtx(CTX_LAST_RUN));

    return () => {
      alive = false;
    };
  }, []);

  useEffect(() => {
    // Refresh business health when user switches target stock; this keeps quality hints aligned with current context.
    if (!generateStockCode.trim()) return;
    void loadBusinessHealth(generateStockCode);
  }, [generateStockCode]);

  return {
    items,
    selectedMarkdown,
    selectedRaw,
    selectedVersions,
    generateStockCode,
    setGenerateStockCode,
    generateType,
    setGenerateType,
    loading,
    initialLoading,
    error,
    setError,
    activeReportId,
    advancedOpen,
    setAdvancedOpen,
    templateId,
    setTemplateId,
    runId,
    setRunId,
    poolSnapshotId,
    setPoolSnapshotId,
    sectionPreview,
    evidencePreview,
    qualityGatePreview,
    dataPackPreview,
    generationMode,
    businessHealth,
    ctxPoolName,
    ctxStock,
    ctxLastRunAt,
    load,
    loadBusinessHealth,
    exportReport,
    generateReport,
    loadReportDetail,
    loadReportVersions
  };
}

