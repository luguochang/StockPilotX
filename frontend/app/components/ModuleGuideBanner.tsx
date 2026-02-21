"use client";

import { useEffect, useState } from "react";
import { Alert } from "antd";

type Props = {
  moduleKey: string;
  title: string;
  steps: string[];
};

export default function ModuleGuideBanner({ moduleKey, title, steps }: Props) {
  const storageKey = `stockpilotx:guide:dismissed:${moduleKey}`;
  const [visible, setVisible] = useState(false);

  useEffect(() => {
    try {
      const dismissed = localStorage.getItem(storageKey) === "1";
      setVisible(!dismissed);
    } catch {
      setVisible(true);
    }
  }, [storageKey]);

  if (!visible) return null;

  return (
    <Alert
      style={{ marginTop: 12 }}
      type="info"
      showIcon
      closable
      onClose={() => {
        setVisible(false);
        try {
          localStorage.setItem(storageKey, "1");
        } catch {
          // ignore
        }
      }}
      message={title}
      description={steps.map((s, i) => `${i + 1}. ${s}`).join("  ")}
    />
  );
}
