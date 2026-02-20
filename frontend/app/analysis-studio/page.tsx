import { redirect } from "next/navigation";

// 兼容历史入口：analysis-studio 已重命名为 market-quick。
export default function AnalysisStudioRedirectPage() {
  redirect("/market-quick");
}
