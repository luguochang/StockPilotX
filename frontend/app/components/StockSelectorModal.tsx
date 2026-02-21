"use client";

import { useEffect, useMemo, useState } from "react";
import { Alert, Button, Cascader, Checkbox, Empty, Input, List, Modal, Select, Space, Tag, Typography } from "antd";
import { STOCK_CATALOG, type StockOption } from "./stockCatalog";

const { Text } = Typography;
const HISTORY_KEY = "stockpilotx:selected_stock_history";
const TOKEN_KEY = "stockpilotx:access_token";
const FILTER_MEMORY_KEY = "stockpilotx:stock_filter_memory";
const RECENT_INDUSTRY_KEY = "stockpilotx:recent_industry_filters";
const API_BASE = process.env.NEXT_PUBLIC_API_BASE ?? "http://127.0.0.1:8000";
const COMMON_INDUSTRY_PRIORITY = [
  "银行",
  "电子",
  "医药生物",
  "计算机",
  "电力设备",
  "食品饮料",
  "非银金融",
  "通信",
  "机械设备",
  "汽车"
];

type Props = {
  value: string | string[];
  onChange: (next: string | string[]) => void;
  multiple?: boolean;
  title?: string;
  placeholder?: string;
  triggerText?: string;
  authToken?: string;
};

function normalizeCode(raw: string): string {
  return raw.trim().toUpperCase().replace(".", "");
}

function mergeCatalog(base: StockOption[], recentCodes: string[]): StockOption[] {
  const index = new Map<string, StockOption>();
  for (const item of base) {
    index.set(item.code, item);
  }
  for (const code of recentCodes) {
    if (!index.has(code)) {
      index.set(code, { code, name: "最近选择", market: code.startsWith("SZ") ? "SZ" : "SH", sector: "History" });
    }
  }
  return Array.from(index.values());
}

type RemoteStockItem = {
  stock_code: string;
  stock_name: string;
  exchange: string;
  market_tier: string;
  listing_board: string;
  industry_l1: string;
};

type FilterBucket = {
  exchange: string[];
  market_tier: string[];
  listing_board: string[];
  industry_l1: string[];
  industry_l2?: string[];
  industry_l3?: string[];
};

type CascaderNode = {
  value: string;
  label: string;
  children?: CascaderNode[];
};

function exchangeOfTier(tier: string): string {
  if (tier.includes("上证") || tier === "科创板") return "SH";
  if (tier.includes("深证") || tier === "创业板") return "SZ";
  if (tier.includes("北交")) return "BJ";
  return "";
}

function boardOfTier(tier: string): string {
  if (tier.includes("主板")) return "主板";
  if (tier === "科创板") return "科创板";
  if (tier === "创业板") return "创业板";
  if (tier.includes("北交")) return "北交所";
  return "";
}

function tierBelongsToExchange(tier: string, exchange: string): boolean {
  const inferred = exchangeOfTier(tier);
  return inferred ? inferred === exchange : true;
}

function uniqueStrings(values: string[]): string[] {
  return Array.from(new Set(values.filter(Boolean)));
}

export default function StockSelectorModal({
  value,
  onChange,
  multiple = false,
  title = "选择股票",
  placeholder = "点击搜索并选择股票",
  triggerText = "搜索选择",
  authToken = ""
}: Props) {
  const [open, setOpen] = useState(false);
  const [keyword, setKeyword] = useState("");
  const [draft, setDraft] = useState<string[]>([]);
  const [knownNameMap, setKnownNameMap] = useState<Record<string, string>>({});
  const [recentCodes, setRecentCodes] = useState<string[]>([]);
  const [remoteItems, setRemoteItems] = useState<RemoteStockItem[]>([]);
  const [remoteLoading, setRemoteLoading] = useState(false);
  const [remoteError, setRemoteError] = useState("");
  const [filters, setFilters] = useState<FilterBucket>({ exchange: [], market_tier: [], listing_board: [], industry_l1: [] });
  const [selectedMarketPath, setSelectedMarketPath] = useState<string[]>([]);
  const [selectedIndustry, setSelectedIndustry] = useState("");
  const [recentIndustryFilters, setRecentIndustryFilters] = useState<string[]>([]);
  const [filtersLoaded, setFiltersLoaded] = useState(false);
  const [useRemote, setUseRemote] = useState(false);

  const current = useMemo(() => {
    if (Array.isArray(value)) return value.map(normalizeCode).filter(Boolean);
    return [normalizeCode(value)].filter(Boolean);
  }, [value]);

  useEffect(() => {
    try {
      const raw = localStorage.getItem(HISTORY_KEY);
      if (!raw) return;
      const arr = JSON.parse(raw);
      if (Array.isArray(arr)) {
        setRecentCodes(arr.map((x) => normalizeCode(String(x))).filter(Boolean).slice(0, 50));
      }
    } catch {
      setRecentCodes([]);
    }
  }, []);

  useEffect(() => {
    try {
      const raw = localStorage.getItem(RECENT_INDUSTRY_KEY);
      if (!raw) return;
      const arr = JSON.parse(raw);
      if (Array.isArray(arr)) {
        setRecentIndustryFilters(arr.map((x) => String(x)).filter(Boolean).slice(0, 8));
      }
    } catch {
      setRecentIndustryFilters([]);
    }
  }, []);

  const effectiveToken = useMemo(() => {
    if (authToken.trim()) return authToken.trim();
    try {
      return localStorage.getItem(TOKEN_KEY) ?? "";
    } catch {
      return "";
    }
  }, [authToken]);

  const options = useMemo(() => mergeCatalog(STOCK_CATALOG, recentCodes), [recentCodes]);

  useEffect(() => {
    // Keep a local code->name cache so selected values can still show names after filters/search change.
    setKnownNameMap((prev) => {
      const next = { ...prev };
      for (const item of STOCK_CATALOG) {
        const code = normalizeCode(item.code);
        if (code && item.name) next[code] = item.name;
      }
      for (const item of options) {
        const code = normalizeCode(item.code);
        if (code && item.name) next[code] = item.name;
      }
      for (const item of remoteItems) {
        const code = normalizeCode(item.stock_code);
        if (code && item.stock_name) next[code] = item.stock_name;
      }
      return next;
    });
  }, [options, remoteItems]);

  const filtered = useMemo(() => {
    if (useRemote) {
      return remoteItems.map((x) => ({
        code: x.stock_code,
        name: x.stock_name,
        market: (x.exchange === "SH" ? "SH" : "SZ") as "SH" | "SZ",
        sector: `${x.market_tier}${x.industry_l1 ? ` / ${x.industry_l1}` : ""}`
      }));
    }
    const q = keyword.trim().toUpperCase();
    if (!q) return options;
    return options.filter((x) => x.code.includes(q) || x.name.includes(keyword.trim()));
  }, [keyword, options, remoteItems, useRemote]);

  const marketCascaderOptions = useMemo<CascaderNode[]>(() => {
    const exchanges = filters.exchange.length ? filters.exchange : uniqueStrings(filters.market_tier.map(exchangeOfTier));
    return exchanges
      .filter(Boolean)
      .map((ex) => {
        const tiers = filters.market_tier.filter((t) => tierBelongsToExchange(t, ex));
        return {
          value: ex,
          label: ex,
          children: tiers.map((tier) => {
            const board = boardOfTier(tier);
            const boardChildren = board ? [{ value: board, label: board }] : undefined;
            return { value: tier, label: tier, children: boardChildren };
          })
        };
      });
  }, [filters.exchange, filters.market_tier]);

  const selectedExchange = selectedMarketPath[0] ?? "";
  const selectedTier = selectedMarketPath[1] ?? "";
  const selectedBoard = selectedMarketPath[2] ?? "";

  const industryOptions = useMemo(() => {
    const all = filters.industry_l1 ?? [];
    const recent = recentIndustryFilters.filter((x) => all.includes(x));
    const common = COMMON_INDUSTRY_PRIORITY.filter((k) => all.includes(k));
    const top = uniqueStrings([...recent, ...common]);
    const rest = all.filter((x) => !top.includes(x));
    return [...top, ...rest].map((x) => ({ value: x, label: x }));
  }, [filters.industry_l1, recentIndustryFilters]);

  useEffect(() => {
    if (!open) return;
    try {
      localStorage.setItem(
        FILTER_MEMORY_KEY,
        JSON.stringify({
          market_path: selectedMarketPath,
          industry: selectedIndustry
        })
      );
    } catch {
      // ignore storage errors
    }
  }, [open, selectedMarketPath, selectedIndustry]);

  useEffect(() => {
    if (!open || filtersLoaded) return;
    const ctrl = new AbortController();
    (async () => {
      try {
        const headers = effectiveToken ? { Authorization: `Bearer ${effectiveToken}` } : undefined;
        const resp = await fetch(`${API_BASE}/v1/stocks/filters`, { signal: ctrl.signal, headers });
        if (!resp.ok) return;
        const data = (await resp.json()) as FilterBucket;
        setFilters({
          exchange: data.exchange ?? [],
          market_tier: data.market_tier ?? [],
          listing_board: data.listing_board ?? [],
          industry_l1: data.industry_l1 ?? [],
          industry_l2: data.industry_l2 ?? [],
          industry_l3: data.industry_l3 ?? []
        });
        setFiltersLoaded(true);
      } catch {
        // ignore filter load errors
      }
    })();
    return () => ctrl.abort();
  }, [open, effectiveToken, filtersLoaded]);

  useEffect(() => {
    if (!open) {
      return;
    }
    const timer = setTimeout(async () => {
      setRemoteLoading(true);
      setRemoteError("");
      const params = new URLSearchParams();
      if (keyword.trim()) params.set("keyword", keyword.trim());
      if (selectedExchange) params.set("exchange", selectedExchange);
      if (selectedTier) params.set("market_tier", selectedTier);
      if (selectedBoard) params.set("listing_board", selectedBoard);
      if (selectedIndustry) params.set("industry_l1", selectedIndustry);
      params.set("limit", "120");
      try {
        const headers = effectiveToken ? { Authorization: `Bearer ${effectiveToken}` } : undefined;
        const resp = await fetch(`${API_BASE}/v1/stocks/search?${params.toString()}`, { headers });
        if (!resp.ok) {
          setUseRemote(false);
          setRemoteError("远程搜索失败，已切换为本地股票池。");
          setRemoteItems([]);
          return;
        }
        const data = (await resp.json()) as RemoteStockItem[];
        setRemoteItems(Array.isArray(data) ? data : []);
        setUseRemote(true);
      } catch {
        setUseRemote(false);
        setRemoteError("远程搜索失败，已切换为本地股票池。");
        setRemoteItems([]);
      } finally {
        setRemoteLoading(false);
      }
    }, 260);
    return () => clearTimeout(timer);
  }, [open, keyword, selectedExchange, selectedTier, selectedBoard, selectedIndustry, effectiveToken]);

  function persistHistory(codes: string[]) {
    const merged = Array.from(new Set([...codes.map(normalizeCode), ...recentCodes])).slice(0, 50);
    setRecentCodes(merged);
    try {
      localStorage.setItem(HISTORY_KEY, JSON.stringify(merged));
    } catch {
      // ignore storage errors
    }
  }

  function openModal() {
    setDraft(current);
    setKeyword("");
    try {
      const raw = localStorage.getItem(FILTER_MEMORY_KEY);
      if (raw) {
        const parsed = JSON.parse(raw) as { market_path?: string[]; industry?: string };
        setSelectedMarketPath(Array.isArray(parsed?.market_path) ? parsed.market_path.map((x) => String(x)) : []);
        setSelectedIndustry(parsed?.industry ? String(parsed.industry) : "");
      } else {
        setSelectedMarketPath([]);
        setSelectedIndustry("");
      }
    } catch {
      setSelectedMarketPath([]);
      setSelectedIndustry("");
    }
    setOpen(true);
  }

  function toggleCode(code: string, checked: boolean) {
    const normalized = normalizeCode(code);
    if (!multiple) {
      setDraft(checked ? [normalized] : []);
      return;
    }
    setDraft((prev) => {
      if (checked) return Array.from(new Set([...prev, normalized]));
      return prev.filter((x) => x !== normalized);
    });
  }

  function handleConfirm() {
    const picked = multiple ? draft : draft.slice(0, 1);
    setKnownNameMap((prev) => {
      const next = { ...prev };
      for (const item of filtered) {
        const code = normalizeCode(item.code);
        if (!picked.includes(code)) continue;
        if (item.name) next[code] = item.name;
      }
      return next;
    });
    persistHistory(picked);
    if (selectedIndustry) {
      const nextRecent = uniqueStrings([selectedIndustry, ...recentIndustryFilters]).slice(0, 8);
      setRecentIndustryFilters(nextRecent);
      try {
        localStorage.setItem(RECENT_INDUSTRY_KEY, JSON.stringify(nextRecent));
      } catch {
        // ignore storage errors
      }
    }
    onChange(multiple ? picked : (picked[0] ?? ""));
    setOpen(false);
  }

  function renderCodeWithName(code: string): string {
    const normalized = normalizeCode(code);
    const name = knownNameMap[normalized];
    return name ? `${normalized} ${name}` : normalized;
  }

  const displayText = current.map((code) => renderCodeWithName(code)).join("；");

  return (
    <>
      <Space.Compact block>
        <Input readOnly value={displayText} placeholder={placeholder} />
        <Button onClick={openModal}>{triggerText}</Button>
      </Space.Compact>
      {current.length > 0 ? (
        <Space wrap style={{ marginTop: 8 }}>
          {current.map((code) => (
            <Tag key={code} color="blue">{renderCodeWithName(code)}</Tag>
          ))}
        </Space>
      ) : null}

      <Modal
        title={title}
        open={open}
        onCancel={() => setOpen(false)}
        onOk={handleConfirm}
        okText="确认选择"
        cancelText="取消"
        width={720}
      >
        <Space direction="vertical" style={{ width: "100%" }} size={10}>
          <Input
            value={keyword}
            onChange={(e) => setKeyword(e.target.value)}
            placeholder="输入股票代码或名称搜索"
            allowClear
          />
          <Space wrap>
            <Cascader
              value={selectedMarketPath.length ? selectedMarketPath : undefined}
              onChange={(vals) => setSelectedMarketPath((vals ?? []).map((x) => String(x)))}
              placeholder="交易所 / 市场层级 / 板块"
              options={marketCascaderOptions}
              allowClear
              showSearch
              style={{ width: 420 }}
            />
            <Select
              value={selectedIndustry || undefined}
              onChange={(v) => setSelectedIndustry(v)}
              placeholder="行业"
              allowClear
              showSearch
              optionFilterProp="label"
              filterOption={(input, option) => String(option?.label ?? "").toLowerCase().includes(input.toLowerCase())}
              style={{ width: 220 }}
              options={industryOptions}
            />
          </Space>
          {remoteError ? <Alert type="info" showIcon message={remoteError} /> : null}
          <Text type="secondary">{useRemote ? "数据源：后端股票主数据" : "数据源：本地兜底股票池"}{remoteLoading ? "（加载中...）" : ""}</Text>
          <Text type="secondary">已选 {draft.length} 个</Text>
          <List
            bordered
            dataSource={filtered}
            locale={{ emptyText: <Empty description="没有匹配股票" /> }}
            style={{ maxHeight: 420, overflowY: "auto" }}
            renderItem={(item) => {
              const checked = draft.includes(item.code);
              return (
                <List.Item
                  onClick={() => toggleCode(item.code, !checked)}
                  style={{
                    cursor: "pointer",
                    margin: "4px 6px",
                    padding: "10px 12px",
                    borderRadius: 10,
                    border: checked ? "1px solid rgba(22,119,255,0.55)" : "1px solid transparent",
                    background: checked ? "rgba(22,119,255,0.10)" : "transparent",
                    boxShadow: checked ? "inset 0 0 0 1px rgba(22,119,255,0.12)" : "none"
                  }}
                  actions={[
                    multiple ? (
                      <Checkbox
                        key="check"
                        checked={checked}
                        onClick={(e) => e.stopPropagation()}
                        onChange={(e) => toggleCode(item.code, e.target.checked)}
                      />
                    ) : (
                      <Tag key="mark" color={checked ? "green" : "default"}>{checked ? "已选" : "未选"}</Tag>
                    )
                  ]}
                >
                  <Space>
                    <Tag color={item.market === "SH" ? "geekblue" : "cyan"}>{item.market}</Tag>
                    <Text strong>{item.code}</Text>
                    <Text>{item.name}</Text>
                    <Text type="secondary">{item.sector}</Text>
                  </Space>
                </List.Item>
              );
            }}
          />
        </Space>
      </Modal>
    </>
  );
}
