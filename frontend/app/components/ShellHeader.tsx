"use client";

import { useEffect, useState } from "react";
import Link from "next/link";

type NavItem = { href: string; label: string };

export default function ShellHeader({ navItems }: { navItems: NavItem[] }) {
  const [scrolled, setScrolled] = useState(false);

  useEffect(() => {
    const onScroll = () => setScrolled(window.scrollY > 24);
    onScroll();
    window.addEventListener("scroll", onScroll, { passive: true });
    return () => window.removeEventListener("scroll", onScroll);
  }, []);

  return (
    <header className={`shell-header shell-fade-in ${scrolled ? "scrolled" : ""}`}>
      <div className="brand-wrap">
        <div className="shell-brand">StockPilotX</div>
        <div className="shell-sub">Agent-native Equity Intelligence</div>
      </div>
      <nav className="shell-nav">
        {navItems.map((item) => (
          <Link key={item.href} href={item.href} className="nav-link">
            {item.label}
          </Link>
        ))}
      </nav>
    </header>
  );
}
