"""python -m src.mcp.futures 入口。"""

from src.mcp.common.server import run_server
from src.mcp.futures.server import create_app

if __name__ == "__main__":
    run_server(create_app())
