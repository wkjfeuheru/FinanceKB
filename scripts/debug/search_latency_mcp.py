"""Time the original queries through Caddy and MCP, without logging credentials."""

import argparse
import asyncio
import json
import os
import ssl
import time

import httpx
from mcp.client.streamable_http import streamablehttp_client

from mcp import ClientSession


async def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="https://localhost/mcp/knowledge")
    parser.add_argument("--ca-file")
    parser.add_argument("--budget", type=float, default=10)
    parser.add_argument("--runs", type=int, default=2)
    parser.add_argument("--concurrent", action="store_true")
    args = parser.parse_args()
    token = (os.environ.get("MCP_API_TOKENS") or os.environ["DATA_API_TOKENS"]).split(",")[0].strip()
    context = ssl.create_default_context(cafile=args.ca_file)

    def client_factory(**kwargs):
        return httpx.AsyncClient(verify=context, **kwargs)

    passed = True
    async with streamablehttp_client(
        args.url, headers={"Authorization": "Bearer " + token}, timeout=120,
        httpx_client_factory=client_factory,
    ) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()

            async def measure(run, tool, query):
                started = time.perf_counter()
                result = await session.call_tool(tool, {"query": query, "page": 1, "page_size": 3})
                elapsed = time.perf_counter() - started
                data = result.structuredContent or {}
                expected = 3 if tool == "search_research" else 0
                ok = not result.isError and data.get("returned") == expected and elapsed < args.budget
                print(json.dumps({
                    "run": run + 1, "tool": tool, "query": query,
                    "elapsed_s": round(elapsed, 3), "returned": data.get("returned"), "pass": ok,
                    "hit_ids": [item["id"] for item in data.get("items", [])],
                    "rerank_scores": [item["rerank_score"] for item in data.get("items", [])],
                }, ensure_ascii=False), flush=True)
                return ok

            for run in range(args.runs):
                queries = [
                    ("search_research", "半导体行业景气度"),
                    ("search_research", "PCB 行业景气度"),
                    ("search_research", "光伏组件价格"),
                    ("search_policy", "利率"),
                ]
                if args.concurrent:
                    verdicts = await asyncio.gather(*(measure(run, tool, query) for tool, query in queries))
                else:
                    verdicts = [await measure(run, tool, query) for tool, query in queries]
                passed = passed and all(verdicts)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
