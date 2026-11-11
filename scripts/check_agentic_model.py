"""不访问知识库的模型协议验收；默认离线，--live 才调用已配置端点。"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from time import perf_counter

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


async def check_model(model) -> dict:
    from src.agent.planning import Subproblem
    from src.agent.nodes.decision import request_tool_calls
    from src.answer.citations import validate_claims
    started = perf_counter()
    if model is None:
        return {"ready": False, "tool_protocol": "unavailable", "claim_protocol": "unavailable"}
    tool = await asyncio.to_thread(request_tool_calls,
        [Subproblem(id="q1", query="测试公司营业收入")], ["测试公司营业收入"], model)
    compatible = not tool["error"] and not tool["policy_forced"]
    if not compatible:
        return {"ready": False, "tool_protocol": "failed", "claim_protocol": "not_run",
                "latency_seconds": round(perf_counter() - started, 3)}
    validation = await validate_claims("测试公司营业收入10亿元[1]",
        [{"content":"测试公司营业收入10亿元", "security_name":"测试公司"}], model=model, enabled=True)
    return {"ready": validation["status"] == "verified", "tool_protocol": "passed",
            "claim_protocol": validation["status"], "latency_seconds": round(perf_counter() - started, 3),
            "scope": "synthetic_protocol_smoke_only; not a quality evaluation"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true", help="使用已配置模型做两项合成协议测试（产生API费用）")
    args = parser.parse_args()
    if not args.live:
        print(json.dumps({"ready": False, "status": "not_run", "reason": "未请求真实端点测试"}, ensure_ascii=False))
        return 0
    from src.core.config import get_model
    try:
        result = asyncio.run(check_model(get_model()))
    except Exception as exc:
        result = {"ready": False, "status": "error", "error_type": type(exc).__name__}
    print(json.dumps(result, ensure_ascii=False))
    return 0 if result.get("ready") else 1


if __name__ == "__main__":
    raise SystemExit(main())
