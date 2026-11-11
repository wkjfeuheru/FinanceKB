"""python -m src.mcp.fundamentals 入口。"""

from __future__ import annotations

from src.mcp.common.server import run_server
from src.mcp.fundamentals.server import create_app

if __name__ == "__main__":
    run_server(create_app())
