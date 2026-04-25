"""VisualHunt — Web 平台启动入口

启动 Flask 前端服务：
    uv run python main.py

然后浏览器访问 http://localhost:5000
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# 确保项目根目录在 Python 路径中
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from src.api.app import app


def _load_api_cfg() -> dict:
    cfg_path = Path(PROJECT_ROOT) / "runtime_config.json"
    if cfg_path.exists():
        try:
            with open(cfg_path, encoding="utf-8") as f:
                return json.load(f).get("api", {})
        except Exception:
            pass
    return {}


if __name__ == "__main__":
    api_cfg = _load_api_cfg()
    host = api_cfg.get("host", "0.0.0.0")
    port = api_cfg.get("port", 5000)
    debug = api_cfg.get("debug", True)

    # Windows 控制台 UTF-8 支持
    os.environ["PYTHONIOENCODING"] = "utf-8"
    app.run(host=host, port=port, debug=debug)
