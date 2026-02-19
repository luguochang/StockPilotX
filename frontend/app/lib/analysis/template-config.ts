export type AnalysisTemplateId =
  | "short_term_opportunity"
  | "mid_term_trend_risk"
  | "event_driven_impact"
  | "valuation_edge"
  | "position_execution";

export type HorizonOption = "7d" | "30d" | "90d";
export type RiskProfileOption = "conservative" | "neutral" | "aggressive";
export type PositionStateOption = "flat" | "holding";

export type AnalysisTemplate = {
  id: AnalysisTemplateId;
  title: string;
  description: string;
  goal: string;
};

// Static template catalog for v1. We intentionally keep it on frontend to iterate UX quickly.
export const ANALYSIS_TEMPLATES: AnalysisTemplate[] = [
  {
    id: "short_term_opportunity",
    title: "短线机会判断",
    description: "判断未来 1-2 周是否存在可执行机会。",
    goal: "给出触发条件、失效条件和分批执行节奏。",
  },
  {
    id: "mid_term_trend_risk",
    title: "中期趋势与回撤风险",
    description: "判断未来 1-3 个月趋势延续与下行风险。",
    goal: "给出趋势证据、波动区间和回撤阈值。",
  },
  {
    id: "event_driven_impact",
    title: "事件驱动影响评估",
    description: "聚焦公告、政策与行业事件的价格影响。",
    goal: "给出事件清单、影响方向和时滞假设。",
  },
  {
    id: "valuation_edge",
    title: "估值与性价比",
    description: "评估当前估值位置与赔率-胜率匹配度。",
    goal: "给出估值对比、风险补偿和策略建议。",
  },
  {
    id: "position_execution",
    title: "仓位与执行节奏建议",
    description: "围绕当前持仓状态给出仓位和节奏建议。",
    goal: "给出分批策略、风控线和复核频率。",
  },
];

export function getTemplateById(templateId: AnalysisTemplateId): AnalysisTemplate {
  return ANALYSIS_TEMPLATES.find((t) => t.id === templateId) ?? ANALYSIS_TEMPLATES[0];
}
