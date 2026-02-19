export type AnswerBlock = {
  kind: "heading" | "list" | "paragraph";
  title?: string;
  lines: string[];
};

function isHeading(line: string): string {
  const v = line.trim();
  if (!v) return "";
  const md = v.match(/^#{1,3}\s+(.+)$/);
  if (md) return md[1].trim();
  const cn = v.match(/^[一二三四五六七八九十]+[、.]\s*(.+)$/);
  if (cn) return cn[1].trim();
  const bracket = v.match(/^【(.+)】$/);
  if (bracket) return bracket[1].trim();
  if (v === "结论" || v === "建议" || v === "风险提示") return v;
  return "";
}

function isListLine(line: string): boolean {
  return /^(\-|\*|\d+\.)\s+/.test(line.trim());
}

export function parseAnswerBlocks(raw: string): AnswerBlock[] {
  const lines = String(raw ?? "")
    .replace(/\r\n/g, "\n")
    .split("\n")
    .map((x) => x.trimEnd());
  const blocks: AnswerBlock[] = [];
  let i = 0;
  while (i < lines.length) {
    const line = lines[i].trim();
    if (!line) {
      i += 1;
      continue;
    }
    const heading = isHeading(line);
    if (heading) {
      blocks.push({ kind: "heading", title: heading, lines: [line] });
      i += 1;
      continue;
    }
    if (isListLine(line)) {
      const listLines: string[] = [];
      while (i < lines.length && isListLine(lines[i])) {
        listLines.push(lines[i].trim());
        i += 1;
      }
      blocks.push({ kind: "list", lines: listLines });
      continue;
    }
    const paragraph: string[] = [];
    while (i < lines.length) {
      const v = lines[i].trim();
      if (!v) break;
      if (isHeading(v) || isListLine(v)) break;
      paragraph.push(v);
      i += 1;
    }
    blocks.push({ kind: "paragraph", lines: paragraph });
    while (i < lines.length && !lines[i].trim()) i += 1;
  }
  return blocks;
}

export function extractKeywords(raw: string): string[] {
  const text = String(raw ?? "");
  const keys = ["结论", "建议", "风险", "触发条件", "失效条件", "仓位", "置信度"];
  return keys.filter((x) => text.includes(x));
}

