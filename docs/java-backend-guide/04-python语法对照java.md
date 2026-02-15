# 04 python语法对照java

## 1) dataclass vs Java POJO/Lombok
`@dataclass(slots=True)`
- 自动生成构造、比较等方法
- `slots=True` 限制动态属性，减少内存开销

## 2) Pydantic BaseModel vs DTO + Bean Validation
`QueryRequest(BaseModel)`
- 定义字段
- 通过 `Field(min_length=...)` 做校验
- `model_dump()` 类似 Java 序列化输出

## 3) Protocol vs Interface
`class QuoteAdapter(Protocol)`
- 类似 Java 接口
- 结构化类型：只要实现同名方法即可，不强制继承

## 4) 类型注解（Type Hints）
`list[str]`, `dict[str, Any]`, `str | None`
- 运行时不强制
- 主要服务于静态检查和可读性

## 5) 一等函数
函数可作为参数传递，例如：
- `trace_emit`
- `model_call`

这点在 Java 中通常由接口实例或 Lambda 实现。

## 6) 上下文对象的可变更新
`AgentState` 在 workflow 中持续被修改。
等价于 Java 里传同一个 Context 引用在各节点填充字段。

## 7) SQLite 轻存储
`sqlite3` 直接执行 SQL。
可类比 Java 里用 JdbcTemplate 做轻量持久化。
