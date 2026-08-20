"""Trace replay harness.

Discrete-event simulation. Both arms get the same trace, hardware profile and
arrival times; only the router and cache policy differ, so the delta is
attributable to the scheduler.
"""
from __future__ import annotations

import heapq
from dataclasses import dataclass

from agentserve.backends.simulated import SimulatedBackend
from agentserve.config import ClusterConfig
from agentserve.models import CacheTier, RunStats, SessionTrace
from agentserve.policy import AdaptiveTtlPolicy, LruPolicy
from agentserve.predictor import ToolDurationPredictor
from agentserve.replica import Replica
from agentserve.router import RoundRobinRouter, SessionAffinityRouter


@dataclass
class ArmResult:
    label: str
    stats: RunStats
    evictions: int
    offloads: int
    affinity_hits: int = 0
    affinity_breaks: int = 0

    def percentile(self, values: list[float], q: float) -> float:
        if not values:
            return 0.0
        ordered = sorted(values)
        idx = min(len(ordered) - 1, int(round(q * (len(ordered) - 1))))
        return ordered[idx]

    def report(self) -> dict[str, float]:
        s = self.stats
        return {
            "requests": s.requests,
            "gpu_hit_rate": s.gpu_hit_rate,
            "any_hit_rate": s.any_hit_rate,
            "token_reuse_rate": s.reuse_rate,
            "tokens_recomputed": s.tokens_computed,
            "p50_ttft_ms": self.percentile(s.ttfts, 0.50),
            "p95_ttft_ms": self.percentile(s.ttfts, 0.95),
            "p50_jct_ms": self.percentile(s.job_completion_ms, 0.50),
            "p95_jct_ms": self.percentile(s.job_completion_ms, 0.95),
            "gpu_busy_ms": s.gpu_busy_ms,
            "evictions": self.evictions,
            "offloads": self.offloads,
        }


def _build(config: ClusterConfig, adaptive: bool):
    predictor = ToolDurationPredictor(config.predictor)
    policy = AdaptiveTtlPolicy(predictor, config.policy) if adaptive else LruPolicy()
    replicas = [Replica(i, config.hardware, policy) for i in range(config.replicas)]
    router = (
        SessionAffinityRouter(replicas, config.policy) if adaptive
        else RoundRobinRouter(replicas)
    )
    return predictor, replicas, router


def run_arm(
    traces: list[SessionTrace], config: ClusterConfig, adaptive: bool, label: str
) -> ArmResult:
    predictor, replicas, router = _build(config, adaptive)
    backend = SimulatedBackend(config.hardware)
    stats = RunStats()

    # (time_ms, seq, session_index, turn_index)
    queue: list[tuple[float, int, int, int]] = []
    seq = 0
    for i, trace in enumerate(traces):
        heapq.heappush(queue, (trace.arrival_ms, seq, i, 0))
        seq += 1

    session_start: dict[str, float] = {}
    session_end: dict[str, float] = {}

    while queue:
        now_ms, _, s_idx, t_idx = heapq.heappop(queue)
        trace = traces[s_idx]
        turn = trace.turns[t_idx]
        session_start.setdefault(trace.session_id, now_ms)

        for r in replicas:
            r.prune(now_ms)

        # On resume we learn how long the previous tool call actually took.
        if t_idx > 0:
            prev = trace.turns[t_idx - 1]
            if prev.tool_name:
                predictor.observe(trace.session_id, prev.tool_name, prev.tool_duration_ms)

        replica = router.select(trace.session_id, now_ms)
        gap_ms = (
            predictor.predict_ms(trace.session_id, turn.tool_name)
            if turn.tool_name else config.policy.dead_session_ms * 2
        )

        # Predicted wall-clock time this session will next need its cache.
        provisional_done = max(replica.free_at_ms, now_ms)
        result = backend.execute(
            replica=replica,
            session_id=trace.session_id,
            prompt_tokens=turn.prompt_tokens,
            output_tokens=turn.output_tokens,
            now_ms=now_ms,
            predicted_return_ms=provisional_done + gap_ms,
        )

        completed_ms = now_ms + result.latency_ms
        replica.pending.append(completed_ms)

        stats.requests += 1
        stats.tokens_reused += result.reused_tokens
        stats.tokens_computed += result.computed_tokens
        stats.ttfts.append(result.ttft_ms)
        stats.gpu_busy_ms += result.restore_ms + result.prefill_ms + result.decode_ms
        if result.tier is CacheTier.GPU:
            stats.gpu_hits += 1
        elif result.tier is CacheTier.DRAM:
            stats.dram_hits += 1
        else:
            stats.misses += 1

        session_end[trace.session_id] = completed_ms

        if t_idx + 1 < len(trace.turns):
            next_at = completed_ms + turn.tool_duration_ms
            heapq.heappush(queue, (next_at, seq, s_idx, t_idx + 1))
            seq += 1

    for sid, end in session_end.items():
        stats.job_completion_ms.append(end - session_start[sid])

    return ArmResult(
        label=label,
        stats=stats,
        evictions=sum(r.evictions for r in replicas),
        offloads=sum(r.offloads for r in replicas),
        affinity_hits=getattr(router, "affinity_hits", 0),
        affinity_breaks=getattr(router, "affinity_breaks", 0),
    )


def compare(traces: list[SessionTrace], config: ClusterConfig) -> tuple[ArmResult, ArmResult]:
    baseline = run_arm(traces, config, adaptive=False, label="stock vLLM (round-robin + LRU)")
    treatment = run_arm(traces, config, adaptive=True, label="AgentServe (affinity + adaptive TTL)")
    return baseline, treatment
