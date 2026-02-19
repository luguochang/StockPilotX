import {
  type AnalysisTemplateId,
  type HorizonOption,
  type PositionStateOption,
  type RiskProfileOption,
  getTemplateById,
} from "./template-config";

export type TemplateComposeInput = {
  templateId: AnalysisTemplateId;
  stockCode: string;
  horizon: HorizonOption;
  riskProfile: RiskProfileOption;
  positionState: PositionStateOption;
};

function horizonLabel(value: HorizonOption): string {
  if (value === "7d") return "7天";
  if (value === "90d") return "90天";
  return "30天";
}

function riskLabel(value: RiskProfileOption): string {
  if (value === "conservative") return "保守";
  if (value === "aggressive") return "积极";
  return "中性";
}

function positionLabel(value: PositionStateOption): string {
  return value === "holding" ? "已持仓" : "空仓";
}

// Build a structured, business-oriented question string from template and slots.
export function composeStructuredQuestion(input: TemplateComposeInput): string {
  const code = String(input.stockCode || "").trim().toUpperCase() || "待选标的";
  const horizon = horizonLabel(input.horizon);
  const risk = riskLabel(input.riskProfile);
  const position = positionLabel(input.positionState);
  const template = getTemplateById(input.templateId);

  const common =
    `请基于 ${code} 最近三个月连续日线、实时行情、公告、新闻、研报与宏观环境，` +
    `在${horizon}视角下进行分析。风险偏好=${risk}，当前状态=${position}。`;

  if (input.templateId === "short_term_opportunity") {
    return `${common}重点判断短线机会是否成立，并输出：触发条件、失效条件、分批执行节奏、止损与复核节点。`;
  }
  if (input.templateId === "mid_term_trend_risk") {
    return `${common}重点评估中期趋势与回撤风险，并输出：趋势证据、波动区间、最大回撤阈值、仓位建议。`;
  }
  if (input.templateId === "event_driven_impact") {
    return `${common}重点评估未来事件驱动影响，并输出：关键事件清单、影响方向、时滞假设、需要跟踪的验证信号。`;
  }
  if (input.templateId === "valuation_edge") {
    return `${common}重点评估估值与性价比，并输出：估值位置、赔率-胜率匹配、风险补偿是否充分、操作建议。`;
  }
  if (input.templateId === "position_execution") {
    return `${common}重点给出仓位与执行策略，并输出：目标仓位区间、分批节奏、风控阈值、复核时间。`;
  }

  return `${common}${template.goal}`;
}
