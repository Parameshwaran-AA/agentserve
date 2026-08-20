"""Run the A/B on real hardware.

Launches N vLLM workers with prefix caching on, then runs the same agent workload
twice through the same gateway binary, once as round-robin + LRU (the control)
and once as session-affinity + adaptive TTL, and prints the difference.

Sized for free hardware. The idea is about routing and eviction, not model scale,
and a 0.5B model exercises vLLM's prefix cache with the same cached_tokens
accounting a 70B does. One T4 is enough to test whether the mechanism works. It
is not enough to claim datacenter throughput, and this prints no such number.

    Kaggle (2x T4)      python scripts/gpu_validate.py --replicas 2
    Colab  (1x T4)      python scripts/gpu_validate.py --replicas 2 --share-gpu
    HPC    (4x A100)    python scripts/gpu_validate.py --replicas 4 \
                            --model Qwen/Qwen2.5-1.5B-Instruct

Add --dry-run to check the plan without starting anything.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bench.live_replay import compare_table, replay  # noqa: E402
from bench.traces import generate_trace  # noqa: E402


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def detect_gpus() -> list[str]:
    if not shutil.which("nvidia-smi"):
        return []
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            text=True, timeout=15,
        )
    except Exception:
        return []
    return [line.strip() for line in out.splitlines() if line.strip()]


def wait_http(url: str, timeout: float, what: str) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            if httpx.get(url, timeout=2.0).status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(2.0)
        if int(time.time()) % 30 == 0:
            log(f"  still waiting for {what} ...")
    return False


def start_vllm(args, index: int, gpu: int, port: int) -> subprocess.Popen:
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", args.model,
        "--port", str(port),
        "--enable-prefix-caching",
        "--gpu-memory-utilization", str(args.gpu_memory_utilization),
        "--max-model-len", str(args.max_model_len),
        "--swap-space", str(args.swap_space_gb),
        "--disable-log-requests",
    ]
    log(f"  worker {index} -> GPU {gpu}, port {port}")
    return subprocess.Popen(cmd, env=env, stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL)


def start_gateway(args, endpoints: list[str], router: str, policy: str,
                  port: int) -> subprocess.Popen:
    env = dict(
        os.environ,
        AGENTSERVE_BACKEND="vllm",
        AGENTSERVE_VLLM_ENDPOINTS=",".join(endpoints),
        AGENTSERVE_VLLM_MODEL=args.model,
        AGENTSERVE_ROUTER=router,
        AGENTSERVE_POLICY=policy,
        AGENTSERVE_GPU_CACHE_TOKENS=str(args.kv_budget_tokens),
        AGENTSERVE_PIN_THRESHOLD_MS=str(args.pin_threshold_ms),
        AGENTSERVE_OFFLOAD_THRESHOLD_MS=str(args.offload_threshold_ms),
        PYTHONPATH=str(ROOT),
    )
    env.pop("AGENTSERVE_API_KEY", None)
    cmd = [sys.executable, "-m", "uvicorn", "agentserve.gateway:app",
           "--host", "127.0.0.1", "--port", str(port), "--log-level", "warning"]
    return subprocess.Popen(cmd, env=env, cwd=str(ROOT),
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def stop(procs) -> None:
    for p in procs:
        if p and p.poll() is None:
            p.send_signal(signal.SIGINT)
    time.sleep(3)
    for p in procs:
        if p and p.poll() is None:
            p.kill()


def main() -> int:
    ap = argparse.ArgumentParser(description="Validate AgentServe on real GPUs")
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--replicas", type=int, default=2)
    ap.add_argument("--share-gpu", action="store_true",
                    help="pack all replicas onto GPU 0 (single-GPU machines)")
    ap.add_argument("--gpu-memory-utilization", type=float, default=None)
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--swap-space-gb", type=int, default=4)
    ap.add_argument("--kv-budget-tokens", type=int, default=40_000)
    ap.add_argument("--pin-threshold-ms", type=float, default=2_000)
    ap.add_argument("--offload-threshold-ms", type=float, default=300_000)
    ap.add_argument("--sessions", type=int, default=24)
    ap.add_argument("--turns", type=int, default=10)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--prompt-tokens", type=int, default=1200)
    ap.add_argument("--speedup", type=float, default=20.0)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--startup-timeout", type=float, default=900.0)
    ap.add_argument("--out", default="gpu_validation.json")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    gpus = detect_gpus()
    log(f"GPUs detected: {gpus or 'none'}")
    if not gpus and not args.dry_run:
        log("No NVIDIA GPU found. This script needs one; use `python -m bench.run` "
            "for the simulated benchmark instead.")
        return 2

    n_gpus = max(1, len(gpus))
    share = args.share_gpu or args.replicas > n_gpus
    if args.gpu_memory_utilization is None:
        # Packing k replicas onto one card means each may claim at most ~1/k of
        # it, minus headroom for the CUDA context and activations.
        per_gpu = args.replicas / n_gpus if share else 1
        args.gpu_memory_utilization = round(min(0.85, 0.80 / per_gpu), 3)

    placement = [(i, 0 if share else i % n_gpus, 8300 + i) for i in range(args.replicas)]
    log(f"model={args.model} replicas={args.replicas} share_gpu={share} "
        f"gpu_mem_util={args.gpu_memory_utilization} kv_budget={args.kv_budget_tokens:,}")
    log(f"workload: {args.sessions} sessions x ~{args.turns} turns, "
        f"concurrency {args.concurrency}, prompts ~{args.prompt_tokens} tokens")

    if args.dry_run:
        for i, gpu, port in placement:
            log(f"  would start worker {i} on GPU {gpu} port {port}")
        log("dry run complete")
        return 0

    workers, gateway = [], None
    try:
        log("starting vLLM workers (first run downloads the model, be patient)")
        for i, gpu, port in placement:
            workers.append(start_vllm(args, i, gpu, port))
        endpoints = [f"http://127.0.0.1:{port}" for _, _, port in placement]
        for ep in endpoints:
            if not wait_http(f"{ep}/health", args.startup_timeout, ep):
                log(f"FAILED: {ep} never became healthy")
                return 1
        log("all workers healthy")

        traces = generate_trace(
            sessions=args.sessions, seed=args.seed, mean_turns=args.turns,
            base_prompt_tokens=args.prompt_tokens, arrival_window_ms=1.0,
        )
        total_calls = sum(len(t.turns) for t in traces)
        log(f"trace: {len(traces)} sessions, {total_calls} calls per arm")

        import asyncio
        runs = {}
        for label, router, policy in [
            ("round-robin + LRU", "round-robin", "lru"),
            ("AgentServe", "affinity", "adaptive"),
        ]:
            log(f"--- arm: {label} ---")
            gateway = start_gateway(args, endpoints, router, policy, 8299)
            if not wait_http("http://127.0.0.1:8299/health", 90, "gateway"):
                log("FAILED: gateway did not start")
                return 1
            runs[label] = asyncio.run(replay(
                "http://127.0.0.1:8299/v1/chat/completions", traces,
                args.concurrency, args.speedup, None, label, 120.0,
            ))
            r = runs[label].report()
            log(f"  hit rate {r['cache_hit_rate']:.1%} | reuse {r['token_reuse_rate']:.1%} "
                f"| p95 {r['p95_latency_ms']:,.0f} ms | failed {r['failed']}")
            stop([gateway])
            gateway = None
            # Let the workers' prefix caches drain so the second arm does not
            # inherit warm cache from the first. Without this the control arm
            # would be measured cold and the treatment warm.
            log("  cooling down between arms")
            time.sleep(20)

        baseline = runs["round-robin + LRU"]
        treatment = runs["AgentServe"]
        table = compare_table(baseline, treatment)
        print(table)

        payload = {
            "hardware": gpus,
            "model": args.model,
            "replicas": args.replicas,
            "share_gpu": share,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "kv_budget_tokens": args.kv_budget_tokens,
            "sessions": args.sessions,
            "concurrency": args.concurrency,
            "calls_per_arm": total_calls,
            "baseline": baseline.report(),
            "agentserve": treatment.report(),
        }
        Path(args.out).write_text(json.dumps(payload, indent=2))
        log(f"wrote {args.out}")

        b, t = baseline.report(), treatment.report()
        if t["failed"] or b["failed"]:
            log("NOTE: some calls failed; treat the numbers as indicative only")
        if t["cache_hit_rate"] > b["cache_hit_rate"]:
            log("RESULT: AgentServe improved cache hit rate on real hardware")
        else:
            log("RESULT: no improvement measured. Likely the KV budget is too "
                "large for the workload, lower --kv-budget-tokens or raise "
                "--sessions so the cache actually comes under pressure.")
        return 0
    finally:
        stop(workers + ([gateway] if gateway else []))
        log("stopped all processes")


if __name__ == "__main__":
    sys.exit(main())
