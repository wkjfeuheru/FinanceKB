"""python -m src.mcp.knowledge 入口。"""

from __future__ import annotations

from src.mcp.common.server import run_server
from src.mcp.knowledge.server import create_app

if __name__ == "__main__":
    run_server(create_app())
