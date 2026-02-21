export const CTX_POOL_ID = "stockpilotx:ctx_pool_id";
export const CTX_POOL_NAME = "stockpilotx:ctx_pool_name";
export const CTX_STOCK_CODE = "stockpilotx:ctx_stock_code";
export const CTX_LAST_RUN = "stockpilotx:ctx_last_run_at";

export function readCtx(key: string): string {
  try {
    return localStorage.getItem(key) ?? "";
  } catch {
    return "";
  }
}

export function writeCtx(key: string, value: string) {
  try {
    localStorage.setItem(key, value);
  } catch {
    // ignore
  }
}
