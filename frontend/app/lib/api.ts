export const API_BASE = process.env.NEXT_PUBLIC_API_BASE ?? "http://127.0.0.1:8000";

export async function fetchJson(path: string, init?: RequestInit) {
  try {
    const r = await fetch(`${API_BASE}${path}`, init);
    const text = await r.text();
    const body = text ? JSON.parse(text) : {};
    if (!r.ok) throw new Error(String((body as { detail?: string })?.detail ?? `HTTP ${r.status}`));
    return body;
  } catch (e) {
    throw new Error(e instanceof Error ? e.message : "后端服务不可用，请确认 http://127.0.0.1:8000 已启动");
  }
}
