export async function readSSEAndConsume(
  resp: Response,
  onEvent: (event: string, payload: Record<string, any>) => void
) {
  if (!resp.body) throw new Error("浏览器不支持流式响应读取");
  const reader = resp.body.getReader();
  const decoder = new TextDecoder("utf-8");
  let buffer = "";
  const nextEventSplitAt = (text: string): { at: number; len: number } => {
    const lf = text.indexOf("\n\n");
    const crlf = text.indexOf("\r\n\r\n");
    if (lf < 0 && crlf < 0) return { at: -1, len: 0 };
    if (lf >= 0 && (crlf < 0 || lf <= crlf)) return { at: lf, len: 2 };
    return { at: crlf, len: 4 };
  };
  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    while (true) {
      const split = nextEventSplitAt(buffer);
      if (split.at < 0) break;
      const rawEvent = buffer.slice(0, split.at);
      buffer = buffer.slice(split.at + split.len);
      const lines = rawEvent.split("\n");
      let eventName = "message";
      const dataLines: string[] = [];
      for (const line of lines) {
        const normalized = line.replace(/\r$/, "");
        if (normalized.startsWith("event:")) eventName = normalized.slice(6).trim();
        if (normalized.startsWith("data:")) dataLines.push(normalized.slice(5).trim());
      }
      if (!dataLines.length) continue;
      try {
        const payload = JSON.parse(dataLines.join("\n"));
        onEvent(eventName, payload as Record<string, any>);
      } catch {
        // ignore malformed payload
      }
    }
  }
  const tail = buffer.trim();
  if (!tail) return;
  const lines = tail.split("\n");
  let eventName = "message";
  const dataLines: string[] = [];
  for (const line of lines) {
    const normalized = line.replace(/\r$/, "");
    if (normalized.startsWith("event:")) eventName = normalized.slice(6).trim();
    if (normalized.startsWith("data:")) dataLines.push(normalized.slice(5).trim());
  }
  if (!dataLines.length) return;
  try {
    const payload = JSON.parse(dataLines.join("\n"));
    onEvent(eventName, payload as Record<string, any>);
  } catch {
    // ignore malformed tail payload
  }
}

