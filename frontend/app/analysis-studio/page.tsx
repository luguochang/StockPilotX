import { redirect } from "next/navigation";

// 兼容历史入口：分析润色模块已下线，统一收敛到 DeepThink。
export default function AnalysisStudioRedirectPage() {
  redirect("/deep-think");
}
