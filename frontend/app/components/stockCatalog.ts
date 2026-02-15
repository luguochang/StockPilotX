export type StockOption = {
  code: string;
  name: string;
  market: "SH" | "SZ";
  sector: string;
};

export const STOCK_CATALOG: StockOption[] = [
  { code: "SH600000", name: "浦发银行", market: "SH", sector: "银行" },
  { code: "SH600036", name: "招商银行", market: "SH", sector: "银行" },
  { code: "SH600519", name: "贵州茅台", market: "SH", sector: "白酒" },
  { code: "SH601318", name: "中国平安", market: "SH", sector: "保险" },
  { code: "SH601888", name: "中国中免", market: "SH", sector: "消费" },
  { code: "SH603259", name: "药明康德", market: "SH", sector: "医药" },
  { code: "SZ000001", name: "平安银行", market: "SZ", sector: "银行" },
  { code: "SZ000333", name: "美的集团", market: "SZ", sector: "家电" },
  { code: "SZ000651", name: "格力电器", market: "SZ", sector: "家电" },
  { code: "SZ000858", name: "五粮液", market: "SZ", sector: "白酒" },
  { code: "SZ002415", name: "海康威视", market: "SZ", sector: "安防" },
  { code: "SZ002594", name: "比亚迪", market: "SZ", sector: "新能源车" },
  { code: "SZ300059", name: "东方财富", market: "SZ", sector: "金融科技" },
  { code: "SZ300308", name: "中际旭创", market: "SZ", sector: "光模块" },
  { code: "SZ300750", name: "宁德时代", market: "SZ", sector: "新能源" }
];
