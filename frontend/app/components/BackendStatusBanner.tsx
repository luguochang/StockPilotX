"use client";

import { Alert, Button, Space, Tag } from "antd";
import { useBackendHealth } from "../hooks/useBackendHealth";

export default function BackendStatusBanner() {
  const status = useBackendHealth(12000);

  if (status === "online") return null;

  const checking = status === "checking";

  return (
    <div style={{ width: "min(1240px, calc(100% - 22px))", margin: "6px auto 0" }}>
      <Alert
        showIcon
        type={checking ? "info" : "warning"}
        message={checking ? "正在检测后端连接..." : "后端连接不可用，部分功能将无法请求数据"}
        description={
          <Space>
            <Tag color={checking ? "processing" : "error"}>{checking ? "检查中" : "离线"}</Tag>
            <span>请确认 API 服务已启动：`http://127.0.0.1:8000`</span>
          </Space>
        }
        action={!checking ? <Button size="small" onClick={() => window.location.reload()}>刷新页面</Button> : undefined}
      />
    </div>
  );
}

