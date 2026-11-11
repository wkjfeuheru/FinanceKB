"""Read-only latency gate; pipe into the app container with docker exec -i.

Reads DATA_API_TOKENS from the container environment; never prints credentials.
The default 10 second budget is a diagnostic threshold, not a product SLO.
"""

import argparse
import json
import os
import time
import urllib.request


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query", action="append", help="Repeat to benchmark several queries")
    parser.add_argument("--page-size", type=int, default=3)
    parser.add_argument("--runs", type=int, default=2)
    parser.add_argument("--budget", type=float, default=10)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    args = parser.parse_args()
    token = os.environ.get("DATA_API_TOKENS", "").split(",")[0].strip()
    if not token:
        raise SystemExit("DATA_API_TOKENS is required")
    passed = True
    for run, query in enumerate((args.query or ["新能源汽车"]) * args.runs):
        payload = json.dumps(
            {"query": query, "page": 1, "page_size": args.page_size}, ensure_ascii=False
        ).encode()
        request = urllib.request.Request(
            args.base_url.rstrip("/") + "/api/v1/search/research",
            data=payload,
            headers={"Authorization": "Bearer " + token, "Content-Type": "application/json"},
        )
        started = time.perf_counter()
        with urllib.request.urlopen(request, timeout=120) as response:
            data = json.load(response)
        elapsed = time.perf_counter() - started
        ok = elapsed < args.budget and data["returned"] > 0
        passed = passed and ok
        print(json.dumps({
            "run": run + 1, "query": query, "elapsed_s": round(elapsed, 3), "returned": data["returned"],
            "budget_s": args.budget, "pass": ok,
            "hit_ids": [item["id"] for item in data["items"]],
            "rerank_scores": [item["rerank_score"] for item in data["items"]],
        }, ensure_ascii=False), flush=True)
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
