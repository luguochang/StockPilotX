from __future__ import annotations

from pathlib import Path
import unittest


class ProjectAssetsTestCase(unittest.TestCase):
    """FRONT-001 / OPS-001 资产存在性测试。"""

    def test_frontend_files_exist(self) -> None:
        root = Path(__file__).resolve().parents[1]
        required = [
            root / "frontend" / "package.json",
            root / "frontend" / "app" / "page.tsx",
            root / "frontend" / "app" / "layout.tsx",
            root / "frontend" / "app" / "globals.css",
            root / "frontend" / "app" / "login" / "page.tsx",
            root / "frontend" / "app" / "watchlist" / "page.tsx",
            root / "frontend" / "app" / "reports" / "page.tsx",
            root / "frontend" / "app" / "predict" / "page.tsx",
            root / "frontend" / "app" / "docs-center" / "page.tsx",
            root / "frontend" / "app" / "ops" / "health" / "page.tsx",
            root / "frontend" / "app" / "ops" / "scheduler" / "page.tsx",
            root / "frontend" / "app" / "ops" / "evals" / "page.tsx",
            root / "frontend" / "app" / "ops" / "alerts" / "page.tsx",
        ]
        for p in required:
            self.assertTrue(p.exists(), f"missing file: {p}")

    def test_ops_runbook_exists(self) -> None:
        root = Path(__file__).resolve().parents[1]
        self.assertTrue((root / "docs" / "operations" / "ops-runbook.md").exists())

    def test_deployment_assets_exist(self) -> None:
        root = Path(__file__).resolve().parents[1]
        required = [
            root / "docker-compose.yml",
            root / "deploy" / "backend.Dockerfile",
            root / "deploy" / "frontend.Dockerfile",
            root / ".github" / "workflows" / "ci.yml",
        ]
        for p in required:
            self.assertTrue(p.exists(), f"missing file: {p}")


if __name__ == "__main__":
    unittest.main()
