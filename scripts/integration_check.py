"""End-to-end integration check.

Starts two mock vLLM workers with a real prefix cache, then drives the gateway
with AGENTSERVE_BACKEND=vllm against them. Exercises the code path that ships to
a cluster: adapter parsing, retry, failover, streaming, auth and Redis, with no
GPU needed.
"""
import os
import subprocess
import sys
import time
from pathlib import Path

import httpx

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS.parent))

REDIS_URL = os.environ.get("INTEGRATION_REDIS_URL", "redis://localhost:6399/1")
PORTS = [8201, 8202]


def wait_for(url: str, timeout: float = 25.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            if httpx.get(url, timeout=1.0).status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(0.3)
    return False


def main() -> int:
    workers = [
        subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "mock_vllm:app", "--app-dir", str(SCRIPTS),
             "--port", str(p), "--log-level", "error"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        for p in PORTS
    ]
    try:
        for p in PORTS:
            if not wait_for(f"http://localhost:{p}/health"):
                print(f"FAIL: mock worker on {p} never became healthy")
                return 1
        print(f"mock vLLM workers up on {PORTS}")

        os.environ.update({
            "AGENTSERVE_BACKEND": "vllm",
            "AGENTSERVE_VLLM_ENDPOINTS": ",".join(f"http://localhost:{p}" for p in PORTS),
            "AGENTSERVE_REDIS_URL": REDIS_URL,
            "AGENTSERVE_API_KEY": "topsecret",
        })

        from fastapi.testclient import TestClient

        from agentserve.gateway import create_app
        from agentserve.settings import load_settings

        settings = load_settings()
        print(f"settings: backend={settings.backend} replicas={settings.cluster.replicas} "
              f"multi_pod_safe={settings.multi_pod_safe}")

        failures = []
        with TestClient(create_app(settings)) as c:
            auth = {"Authorization": "Bearer topsecret"}

            h = c.get("/health").json()
            print(f"\n/health   backend={h['backend']} store={h['session_store']} "
                  f"replicas={h['replicas']} multi_pod_safe={h['multi_pod_safe']}")
            if h["backend"] != "vllm":
                failures.append("gateway did not select the vllm backend")
            if h["session_store"] != "redis":
                failures.append("gateway did not select the redis store")

            r = c.get("/ready")
            print(f"/ready    {r.status_code} {r.json()}")
            if r.status_code != 200:
                failures.append("readiness failed against live workers")

            print("\nauth:")
            for label, hdr in [("no token", {}), ("bad token", {"Authorization": "Bearer x"})]:
                code = c.post("/v1/chat/completions", headers={"X-Session-Id": "a", **hdr},
                              json={"messages": [{"role": "user", "content": "hi"}]}).status_code
                print(f"  {label:10} -> {code}")
                if code != 401:
                    failures.append(f"auth not enforced for {label}")

            print("\nagent session against real workers (prompt grows each turn):")
            convo = "SYSTEM " + "z" * 8000
            hdr = {"X-Session-Id": "real-1", "X-Tool-Name": "run_tests", **auth}
            replicas, hits = set(), 0
            for turn in range(1, 6):
                d = c.post("/v1/chat/completions", headers=hdr,
                           json={"messages": [{"role": "user", "content": convo}],
                                 "max_tokens": 20}).json()
                a, u = d["agentserve"], d["usage"]
                cached = u["prompt_tokens_details"]["cached_tokens"]
                replicas.add(a["replica"])
                hits += a["cache_tier"] != "miss"
                print(f"  turn {turn}: replica={a['replica']} tier={a['cache_tier']:<5} "
                      f"cached={cached:>6} prompt={u['prompt_tokens']:>6} "
                      f"text={d['choices'][0]['message']['content']!r}")
                convo += " TOOL OUTPUT " + "q" * 2000

            if len(replicas) != 1:
                failures.append(f"session was scattered across replicas {replicas}")
            if hits < 3:
                failures.append(f"expected repeat cache hits, saw {hits}")

            print("\nstreaming through the real backend:")
            with c.stream("POST", "/v1/chat/completions",
                          headers={"X-Session-Id": "real-stream", **auth},
                          json={"messages": [{"role": "user", "content": "w" * 9000}],
                                "max_tokens": 20, "stream": True}) as s:
                print(f"  status={s.status_code} ct={s.headers.get('content-type')} "
                      f"replica={s.headers.get('x-agentserve-replica')}")
                lines = [ln for ln in s.iter_lines() if ln.strip()]
            for ln in lines:
                print("   ", ln[:120])
            if not lines or not lines[-1].endswith("[DONE]"):
                failures.append("stream did not terminate with [DONE]")
            if not any("fix" in ln for ln in lines):
                failures.append("stream did not carry worker tokens through")

            print("\nmetrics:")
            for line in c.get("/metrics").text.splitlines():
                if line.startswith(("agentserve_requests_total{",
                                    "agentserve_tokens_reused_total",
                                    "agentserve_affinity_decisions_total{")):
                    print("   ", line)

        print()
        if failures:
            for f in failures:
                print(f"FAIL: {f}")
            return 1
        print("INTEGRATION OK: every check passed against the real vLLM code path")
        return 0
    finally:
        for w in workers:
            w.terminate()
        for w in workers:
            w.wait(timeout=5)


if __name__ == "__main__":
    sys.exit(main())
