from __future__ import annotations

import json

from backend.app.service import AShareAgentService


def main() -> None:
    """本地演示入口：快速验证 query 主链路是否可用。"""
    service = AShareAgentService()
    result = service.query(
        {
            "user_id": "demo-user",
            "question": "请分析SH600000最近风险与机会，并给出证据。",
            "stock_codes": ["SH600000"],
        }
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
