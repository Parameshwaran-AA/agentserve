"""Replay agent traces against a live gateway over HTTP.

The measurement path for real hardware. Nothing here is modelled: prompts are
real text, requests go over the network, latency is wall clock, and reuse is
whatever vLLM reports in prompt_tokens_details.cached_tokens.

Sessions run concurrently, each sleeping for its tool gap between calls. That
concurrency is the point, since the problem only exists when sessions compete for
cache capacity.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time
from dataclasses import dataclass, field

import httpx

from bench.traces import generate_trace


@dataclass
class CallRecord:
    session_id: str
    turn: int
    prompt_tokens: int
    cached_tokens: int
    latency_ms: float
    replica: int | None
    ok: bool = True


@dataclass
class ArmRun:
    label: str
    calls: list[CallRecord] = field(default_factory=list)
    session_ms: dict[str, float] = field(default_factory=dict)
    wall_ms: float = 0.0

    @property
    def ok_calls(self) -> list[CallRecord]:
        return [c for c in self.calls if c.ok]

    def pct(self, values: list[float], q: float) -> float:
        if not values:
            return 0.0
        ordered = sorted(values)
        return ordered[min(len(ordered) - 1, int(round(q * (len(ordered) - 1))))]

    def report(self) -> dict[str, float]:
        calls = self.ok_calls
        reused = sum(c.cached_tokens for c in calls)
        prompt = sum(c.prompt_tokens for c in calls)
        lat = [c.latency_ms for c in calls]
        jct = list(self.session_ms.values())
        return {
            "calls": len(calls),
            "failed": len(self.calls) - len(calls),
            "cache_hit_rate": sum(1 for c in calls if c.cached_tokens > 0) / max(1, len(calls)),
            "token_reuse_rate": reused / max(1, prompt),
            "tokens_recomputed": prompt - reused,
            "replicas_touched": len({c.replica for c in calls if c.replica is not None}),
            "p50_latency_ms": self.pct(lat, 0.50),
            "p95_latency_ms": self.pct(lat, 0.95),
            "mean_latency_ms": statistics.mean(lat) if lat else 0.0,
            "p95_jct_ms": self.pct(jct, 0.95),
            "wall_clock_s": self.wall_ms / 1000.0,
        }


def build_prompt(tokens: int, session_id: str, turn: int) -> str:
    """Text that grows only by appending, like a real agent transcript.

    Turn N's prompt has to be an exact prefix of turn N+1's. Anything appended
    after the growing body (a turn counter, a timestamp) breaks the prefix and
    silently destroys the cache hits being measured, so everything variable goes
    in the header.

    The session id leads so two sessions never share a prefix, otherwise vLLM
    reports hits that have nothing to do with our routing.
    """
    header = (f"SESSION {session_id}\n"
              f"You are a coding agent working on a repository.\n")
    body_words = max(1, tokens - 40)
    return header + " ".join(f"tok{i}" for i in range(body_words))


async def run_session(client, url, trace, headers, record, sem, speedup: float):
    started = time.perf_counter()
    async with sem:
        for turn_index, turn in enumerate(trace.turns):
            prompt = build_prompt(turn.prompt_tokens, trace.session_id, turn_index)
            assert prompt.startswith(f"SESSION {trace.session_id}")
            payload = {
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": min(turn.output_tokens, 64),
            }
            call_headers = {
                **headers,
                "X-Session-Id": trace.session_id,
                **({"X-Tool-Name": turn.tool_name} if turn.tool_name else {}),
            }
            t0 = time.perf_counter()
            try:
                r = await client.post(url, json=payload, headers=call_headers)
                r.raise_for_status()
                body = r.json()
                usage = body.get("usage", {})
                details = usage.get("prompt_tokens_details") or {}
                record(CallRecord(
                    session_id=trace.session_id,
                    turn=turn_index,
                    prompt_tokens=int(usage.get("prompt_tokens", 0)),
                    cached_tokens=int(details.get("cached_tokens", 0) or 0),
                    latency_ms=(time.perf_counter() - t0) * 1000.0,
                    replica=(body.get("agentserve") or {}).get("replica"),
                ))
            except Exception:
                record(CallRecord(trace.session_id, turn_index, 0, 0,
                                  (time.perf_counter() - t0) * 1000.0, None, ok=False))

            if turn.tool_name:
                # Compress the wall clock but keep the shape of the gaps. Short
                # tool calls stay short relative to long ones, and that ratio is
                # what the TTL policy keys on.
                await asyncio.sleep(turn.tool_duration_ms / 1000.0 / speedup)

    return trace.session_id, (time.perf_counter() - started) * 1000.0


async def replay(url: str, traces, concurrency: int, speedup: float,
                 api_key: str | None, label: str, timeout_s: float) -> ArmRun:
    run = ArmRun(label=label)
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    sem = asyncio.Semaphore(concurrency)
    started = time.perf_counter()

    async with httpx.AsyncClient(
        timeout=httpx.Timeout(timeout_s, connect=10.0),
        limits=httpx.Limits(max_connections=concurrency * 2),
    ) as client:
        results = await asyncio.gather(*[
            run_session(client, url, t, headers, run.calls.append, sem, speedup)
            for t in traces
        ])

    run.wall_ms = (time.perf_counter() - started) * 1000.0
    run.session_ms = dict(results)
    return run


def compare_table(baseline: ArmRun, treatment: ArmRun) -> str:
    rows = [
        ("Cache hit rate", "cache_hit_rate", "rate"),
        ("Prompt tokens reused", "token_reuse_rate", "rate"),
        ("Tokens prefilled cold", "tokens_recomputed", "int"),
        ("Replicas touched", "replicas_touched", "int"),
        ("Mean latency (ms)", "mean_latency_ms", "int"),
        ("p50 latency (ms)", "p50_latency_ms", "int"),
        ("p95 latency (ms)", "p95_latency_ms", "int"),
        ("p95 job completion (ms)", "p95_jct_ms", "int"),
        ("Failed calls", "failed", "int"),
    ]
    b, t = baseline.report(), treatment.report()
    head = f"{'metric':<28}{'round-robin + LRU':>20}{'AgentServe':>16}{'delta':>12}"
    lines = ["", head, "-" * len(head)]
    for label, key, kind in rows:
        bv, tv = b[key], t[key]
        if kind == "rate":
            cell_b, cell_t = f"{bv:.1%}", f"{tv:.1%}"
            delta = f"{(tv - bv) * 100:+.1f} pp"
        else:
            cell_b, cell_t = f"{bv:,.0f}", f"{tv:,.0f}"
            delta = f"{(tv / bv - 1) * 100:+.1f}%" if bv else "n/a"
        lines.append(f"{label:<28}{cell_b:>20}{cell_t:>16}{delta:>12}")
    lines.append("")
    lines.append(f"Calls: {b['calls']:,} per arm.  "
                 f"Wall clock: {b['wall_clock_s']:.0f}s vs {t['wall_clock_s']:.0f}s.")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description="Replay agent traces against a live gateway")
    ap.add_argument("--url", default="http://localhost:8000/v1/chat/completions")
    ap.add_argument("--sessions", type=int, default=24)
    ap.add_argument("--turns", type=int, default=10)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--prompt-tokens", type=int, default=1200)
    ap.add_argument("--speedup", type=float, default=20.0,
                    help="divide tool gaps by this to keep the run short")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--api-key", default=None)
    ap.add_argument("--timeout", type=float, default=120.0)
    ap.add_argument("--label", default="AgentServe")
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    traces = generate_trace(
        sessions=args.sessions, seed=args.seed, mean_turns=args.turns,
        base_prompt_tokens=args.prompt_tokens, arrival_window_ms=1.0,
    )
    run = asyncio.run(replay(args.url, traces, args.concurrency, args.speedup,
                             args.api_key, args.label, args.timeout))
    report = run.report()
    print(f"\n{args.label}")
    for k, v in report.items():
        print(f"  {k:22} {v:,.4f}" if isinstance(v, float) else f"  {k:22} {v:,}")
    if args.json_out:
        with open(args.json_out, "w") as fh:
            json.dump(report, fh, indent=2)
        print(f"\nwrote {args.json_out}")


if __name__ == "__main__":
    main()
