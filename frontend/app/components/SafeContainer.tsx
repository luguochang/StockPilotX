import { Skeleton } from "antd";
import React from "react";

interface SafeContainerProps {
  maxHeight?: number;
  children: React.ReactNode;
  loading?: boolean;
}

export function SafeContainer({
  maxHeight = 600,
  children,
  loading
}: SafeContainerProps) {
  return (
    <div style={{
      maxHeight,
      overflowY: "auto",
      overflowX: "hidden"
    }}>
      {loading ? <Skeleton active /> : children}
    </div>
  );
}
