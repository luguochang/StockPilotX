# StockPilotX 生产发布与启动说明

## 1. 适用范围
- 项目根目录：`StockPilotX/`
- 后端入口：`backend.app.http_api:create_app`
- 前端框架：`Next.js 14`

## 2. 启动脚本清单
- 一键启动（Windows）：`start-all.bat`
- 后端启动（Windows）：`start-backend.bat`
- 前端启动（Windows）：`start-frontend.bat`
- 容器编排：`docker-compose.yml`

## 3. 依赖基线（已核验）
### Python
- 依赖文件：`requirements.txt`
- 已在干净虚拟环境执行：
  - `pip install -r requirements.txt`
  - `from backend.app.http_api import create_app; create_app()`
- 结论：核心后端依赖可闭环启动。

### Node.js
- 依赖文件：`frontend/package.json` + `frontend/package-lock.json`
- 已执行：`npm run build`
- 结论：前端生产构建通过。

## 4. 最小可用发布（推荐默认）
### 4.1 后端
```powershell
cd StockPilotX
python -m venv .venv
.venv\Scripts\python -m pip install --upgrade pip
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python -m uvicorn backend.app.http_api:create_app --factory --host 0.0.0.0 --port 8000
```

### 4.2 前端
```powershell
cd StockPilotX\frontend
npm install
npm run build
npm run start
```

### 4.3 前后端联调变量
- 前端：`NEXT_PUBLIC_API_BASE=http://127.0.0.1:8000`
- 后端（按需）：
  - `LLM_EXTERNAL_ENABLED=true`
  - `LLM_CONFIG_PATH=backend/config/llm_providers.local.json`
  - `LLM_FALLBACK_TO_LOCAL=true`
  - `LLM_RETRY_COUNT=1`

## 5. 增强能力依赖（可选）
- `docling`：增强文档解析（缺失时自动回退）。
- `faiss`：向量检索加速（缺失时回退 JSON 检索）。
- `neo4j`：GraphRAG 外部图数据库（缺失时回退内存图）。

建议通过独立可选依赖文件维护，例如：`requirements-optional.txt`。

## 6. 系统级依赖（非 pip）
- Tesseract OCR 可执行程序（配合 `pytesseract`）。
- LibreOffice/soffice（用于 `.doc -> .docx` 转换）。

未安装时系统可运行，但对应能力会降级或关闭。

## 7. Docker 发布
```powershell
cd StockPilotX
docker compose up -d --build
```

默认端口：
- 前端：`3000`
- 后端：`8000`

## 8. 发布后验收（Smoke Test）
- 后端健康：
```powershell
Invoke-WebRequest http://127.0.0.1:8000/docs
```
- 前端访问：
  - 打开 `http://127.0.0.1:3000`
- 核心接口最小验证：
```powershell
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/v1/query -ContentType "application/json" -Body '{"question":"测试","stock_codes":["SH600000"]}'
```

## 9. 常见问题
- 前端报接口不可达：
  - 检查 `NEXT_PUBLIC_API_BASE` 是否指向后端地址。
- 后端 OCR 不生效：
  - 检查是否安装 Tesseract，并确保命令行可执行。
- `.doc` 无法解析：
  - 检查是否安装 `soffice/libreoffice`。
- 图谱未启用：
  - 未设置 `NEO4J_URI/NEO4J_USER/NEO4J_PASSWORD` 时会自动回退内存图。
