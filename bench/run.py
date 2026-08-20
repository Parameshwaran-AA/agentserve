"""Benchmark entry point.

    python -m bench.run --sessions 300 --seeds 8

Runs both arms over identical traces and prints a comparison table. Reports the
spread across seeds rather than a single number, because a scheduling result
from one trace is an anecdote.
"""
from __future__ import annotations

import argparse
import json
from statistics import mean, pstdev

from agentserve.config import ClusterConfig, HardwareProfile, PolicyConfig
from bench.replay import compare
from bench.traces import generate_trace, trace_summary

METRICS = [
    ("gpu_hit_rate", "GPU cache hit rate", "rate"),
    ("any_hit_rate", "Hit rate incl. DRAM", "rate"),
    ("token_reuse_rate", "Prompt tokens reused", "rate"),
    ("tokens_recomputed", "Tokens prefilled cold", "int"),
    ("gpu_busy_ms", "GPU compute time (ms)", "int"),
    ("p95_ttft_ms", "p95 time-to-first-token (ms)", "int"),
    ("p95_jct_ms", "p95 job completion (ms)", "int"),
]


def fmt(value: float, kind: str) -> str:
    if kind == "rate":
        return f"{value:.1%}"
    return f"{value:,.0f}"


def main() -> None:
    ap = argparse.ArgumentParser(description="AgentServe trace replay benchmark")
    ap.add_argument("--sessions", type=int, default=300)
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--replicas", type=int, default=4)
    ap.add_argument("--window-ms", type=float, default=2_000_000.0)
    ap.add_argument("--gpu-cache-tokens", type=int, default=150_000)
    ap.add_argument("--json", action="store_true", help="emit machine-readable output")
    args = ap.parse_args()

    config = ClusterConfig(
        replicas=args.replicas,
        hardware=HardwareProfile(gpu_cache_tokens=args.gpu_cache_tokens),
        policy=PolicyConfig(),
    )

    runs = []
    for seed in range(1, args.seeds + 1):
        traces = generate_trace(
            sessions=args.sessions, seed=seed, arrival_window_ms=args.window_ms
        )
        baseline, treatment = compare(traces, config)
        runs.append((baseline.report(), treatment.report(), treatment))

    if args.json:
        print(json.dumps({
            "config": {
                "sessions": args.sessions, "seeds": args.seeds,
                "replicas": args.replicas, "window_ms": args.window_ms,
            },
            "baseline": {k: mean(r[0][k] for r in runs) for k, _, _ in METRICS},
            "agentserve": {k: mean(r[1][k] for r in runs) for k, _, _ in METRICS},
        }, indent=2))
        return

    summary = trace_summary(
        generate_trace(sessions=args.sessions, seed=1, arrival_window_ms=args.window_ms)
    )
    util = mean(r[0]["gpu_busy_ms"] for r in runs) / (args.replicas * args.window_ms)

    print(f"\nWorkload: {summary['sessions']:.0f} agent sessions, "
          f"{summary['total_calls']:.0f} LLM calls, "
          f"{summary['mean_prompt_tokens']:,.0f} mean prompt tokens")
    print(f"Tool gaps: median {summary['median_tool_gap_ms']:,.0f} ms, "
          f"p90 {summary['p90_tool_gap_ms']:,.0f} ms")
    print(f"Cluster: {args.replicas} replicas, "
          f"{args.gpu_cache_tokens:,} token KV budget each, "
          f"~{util:.0%} baseline utilization")
    print(f"Averaged over {args.seeds} seeds\n")

    head = f"{'metric':<30}{'stock vLLM':>16}{'AgentServe':>16}{'delta':>12}"
    print(head)
    print("-" * len(head))
    for key, label, kind in METRICS:
        base_vals = [r[0][key] for r in runs]
        agent_vals = [r[1][key] for r in runs]
        b, a = mean(base_vals), mean(agent_vals)
        if kind == "rate":
            delta = f"{(a - b) * 100:+.1f} pp"
        else:
            delta = f"{(a / b - 1) * 100:+.1f}%" if b else "n/a"
        print(f"{label:<30}{fmt(b, kind):>16}{fmt(a, kind):>16}{delta:>12}")

    sd = pstdev([r[1]["gpu_hit_rate"] for r in runs])
    breaks = mean(t.affinity_breaks for _, _, t in runs)
    hits = mean(t.affinity_hits for _, _, t in runs)
    print(f"\nHit-rate stability across seeds: sd {sd:.4f}")
    print(f"Affinity honored {hits:,.0f}, rebalanced for load {breaks:,.0f}")
    print("Baseline = round-robin routing with LRU eviction, GPU tier only.\n")


if __name__ == "__main__":
    main()
