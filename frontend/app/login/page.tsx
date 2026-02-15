"use client";

import { useEffect, useState } from "react";
import { Alert, Button, Card, Input, Space, Typography } from "antd";

const API_BASE = process.env.NEXT_PUBLIC_API_BASE ?? "http://127.0.0.1:8000";
const TOKEN_KEY = "stockpilotx:access_token";
const { Title, Text } = Typography;

export default function LoginPage() {
  const [username, setUsername] = useState("demo_user");
  const [password, setPassword] = useState("pw123456");
  const [token, setToken] = useState("");
  const [msg, setMsg] = useState("");
  const [me, setMe] = useState("");
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    try {
      const saved = localStorage.getItem(TOKEN_KEY);
      if (saved) setToken(saved);
    } catch {
      // ignore storage errors
    }
  }, []);

  async function registerAndLogin() {
    setLoading(true);
    setMsg("处理中...");
    try {
      await fetch(`${API_BASE}/v1/auth/register`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ username, password, tenant_name: `${username}_tenant` })
      });
      const resp = await fetch(`${API_BASE}/v1/auth/login`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ username, password })
      });
      const data = await resp.json();
      setToken(data.access_token ?? "");
      if (data.access_token) {
        try {
          localStorage.setItem(TOKEN_KEY, String(data.access_token));
        } catch {
          // ignore storage errors
        }
      }
      setMsg(resp.ok ? "登录成功" : "登录失败");
      setMe("");
    } catch {
      setMsg("请求失败");
    } finally {
      setLoading(false);
    }
  }

  // 中文注释：直接把 me/refresh 接口暴露在页面，便于调试 token 生命周期。
  async function loadMe() {
    setLoading(true);
    try {
      const resp = await fetch(`${API_BASE}/v1/auth/me`, {
        headers: { Authorization: `Bearer ${token}` }
      });
      const body = await resp.json();
      if (!resp.ok) throw new Error(body?.detail ?? `HTTP ${resp.status}`);
      setMe(JSON.stringify(body, null, 2));
      setMsg("读取当前用户成功");
    } catch (e) {
      setMsg(e instanceof Error ? e.message : "请求失败");
      setMe("");
    } finally {
      setLoading(false);
    }
  }

  return (
    <main className="container">
      <Card className="premium-card">
        <Space direction="vertical" style={{ width: "100%" }}>
          <Title level={2} style={{ margin: 0 }}>登录 / 注册</Title>
          <Text type="secondary">覆盖接口：`/v1/auth/register`、`/v1/auth/login`、`/v1/auth/me`</Text>
          <Input value={username} onChange={(e) => setUsername(e.target.value)} placeholder="username" />
          <Input.Password value={password} onChange={(e) => setPassword(e.target.value)} placeholder="password" />
          <Space>
            <Button type="primary" loading={loading} onClick={registerAndLogin}>注册并登录</Button>
            <Button loading={loading} onClick={loadMe}>读取 me</Button>
          </Space>
          {msg ? <Alert type="info" showIcon message={msg} /> : null}
          <Text>当前用户 me</Text>
          <Input.TextArea rows={6} value={me} readOnly />
        </Space>
      </Card>
    </main>
  );
}
