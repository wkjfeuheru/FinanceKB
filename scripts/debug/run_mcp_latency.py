"""Run the Caddy/MCP latency gate using Docker; no host MCP installation needed.

Example: .venv/Scripts/python.exe scripts/debug/run_mcp_latency.py --runs 2
Append --concurrent to submit all original queries together.
Tokens stay in the child environment; the local CA is copied to a temporary dir.
"""

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path


def main():
    info = json.loads(subprocess.check_output(["docker", "inspect", "financekb-mcp-knowledge-1"]))[0]
    runtime = dict(item.split("=", 1) for item in info["Config"]["Env"])
    env = os.environ.copy()
    env["DATA_API_TOKENS"] = runtime.get("MCP_API_TOKENS") or runtime["DATA_API_TOKENS"]
    with tempfile.TemporaryDirectory(prefix="financekb-latency-") as directory:
        subprocess.run([
            "docker", "cp", "financekb-caddy-1:/data/caddy/pki/authorities/local/root.crt",
            str(Path(directory) / "root.crt"),
        ], check=True)
        return subprocess.call([
            "docker", "run", "--rm", "--network", "container:financekb-caddy-1",
            "--env", "DATA_API_TOKENS",
            "--mount", f"type=bind,source={Path(__file__).resolve().parent},target=/diagnostics,readonly",
            "--mount", f"type=bind,source={directory},target=/local-ca,readonly",
            "financekb-app:latest", "python", "/diagnostics/search_latency_mcp.py",
            "--ca-file", "/local-ca/root.crt", *sys.argv[1:],
        ], env=env)


if __name__ == "__main__":
    raise SystemExit(main())
