# 03 hook与中间件机制

## 先理解一个结论
`hook` 在这里不是 React Hook，而是“流程钩子函数”。
你可以把它理解成 Java 里的 Interceptor + Around Advice。

## 中间件接口（`Middleware`）
- `before_agent(state, ctx)`：流程开始前
- `before_model(state, prompt, ctx)`：模型调用前，可改 prompt
- `after_model(state, output, ctx)`：模型调用后，可改 output
- `after_agent(state, ctx)`：流程结束后
- `wrap_model_call(...)`：包裹模型调用
- `wrap_tool_call(...)`：包裹工具调用

## 洋葱模型（Onion Model）
`MiddlewareStack.call_model()` 会把多个中间件逐层嵌套：
外层先执行 `wrap_model_call`，再调用下一个，最后回到外层。

类比 Java：
- Spring AOP 的 `@Around`
- Servlet Filter 链

## 当前内置中间件
- `GuardrailMiddleware`
  - 识别高风险投资请求
  - 在 prompt 追加安全规则
  - 结果兜底补免责声明
- `BudgetMiddleware`
  - 截断过长上下文
  - 限制模型和工具调用次数

## 适合扩展的场景
- 统一重试
- 限流和熔断
- 审计日志
- 敏感信息脱敏

## 为什么不用装饰器直接做
中间件栈更适合运行时按配置拼装顺序，便于 A/B 和灰度。
