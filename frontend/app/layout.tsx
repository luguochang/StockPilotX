import "./globals.css";
import type { ReactNode } from "react";
import "antd/dist/reset.css";
import ShellHeader from "./components/ShellHeader";

export const metadata = {
  title: "StockPilotX",
  description: "A-share Agent Analysis System"
};

const navItems = [
  { href: "/", label: "首页" },
  { href: "/deep-think", label: "DeepThink" },
  { href: "/predict", label: "预测" },
  { href: "/watchlist", label: "关注池" },
  { href: "/reports", label: "报告" },
  { href: "/docs-center", label: "文档" },
  { href: "/ui-demos", label: "UI Lab" }
];

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="zh-CN">
      <body>
        <div className="bg-orb bg-orb-a" />
        <div className="bg-orb bg-orb-b" />
        <ShellHeader navItems={navItems} />
        {children}
        <footer className="site-footer">
          <div className="site-footer-inner">
            <div className="site-footer-col">
              <div className="site-footer-brand">StockPilotX</div>
              <p className="site-footer-desc">A-share Agent Intelligence Workspace</p>
            </div>
            <div className="site-footer-col">
              <div className="site-footer-title">开源与专栏</div>
              <div className="site-footer-links">
                <a href="https://github.com/luguochang" target="_blank" rel="noreferrer">
                  GitHub / luguochang
                </a>
                <a href="https://blog.csdn.net/luguochang" target="_blank" rel="noreferrer">
                  CSDN / luguochang
                </a>
              </div>
            </div>
            <div className="site-footer-col">
              <div className="site-footer-title">联系方式</div>
              <div className="site-footer-links">
                <a href="https://github.com/luguochang" target="_blank" rel="noreferrer">
                  Open Source Profile
                </a>
                <a href="https://blog.csdn.net/luguochang" target="_blank" rel="noreferrer">
                  Technical Articles
                </a>
              </div>
            </div>
          </div>
          <div className="site-footer-bottom">© {new Date().getFullYear()} StockPilotX</div>
        </footer>
      </body>
    </html>
  );
}
