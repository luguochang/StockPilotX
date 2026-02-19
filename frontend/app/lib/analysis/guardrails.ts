import type { HorizonOption } from "./template-config";

export type PromptGuardrailInput = {
  stockCode: string;
  question: string;
  horizon: HorizonOption;
};

export type PromptGuardrailResult = {
  score: number;
  errors: string[];
  warnings: string[];
  suggestions: string[];
};

// Lightweight guardrails for reducing low-quality prompts.
export function validatePromptQuality(input: PromptGuardrailInput): PromptGuardrailResult {
  const errors: string[] = [];
  const warnings: string[] = [];
  const suggestions: string[] = [];

  const code = String(input.stockCode || "").trim().toUpperCase();
  const question = String(input.question || "").trim();

  if (!code) {
    errors.push("请先选择分析标的");
  }
  if (!question) {
    errors.push("问题不能为空");
  }
  if (question && question.length < 24) {
    warnings.push("问题较短，模型可能无法准确识别目标和约束");
    suggestions.push("建议补充：时间窗口、风险偏好、执行约束");
  }
  if (input.horizon === "7d") {
    warnings.push("7天窗口较短，结果容易受噪声影响");
    suggestions.push("可同时查看30天窗口结论做交叉验证");
  }
  if (!/风险|回撤|波动|止损|仓位/.test(question)) {
    suggestions.push("建议补充风险控制要求（如止损阈值、仓位上限）");
  }

  const scorePenalty = errors.length * 45 + warnings.length * 12;
  const score = Math.max(0, 100 - scorePenalty);

  return { score, errors, warnings, suggestions };
}
