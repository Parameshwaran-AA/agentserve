"""Prometheus instrumentation.

The metric set answers the three questions an operator has: is the cache
working, is the router respecting affinity, and is any replica starving.
"""
from __future__ import annotations

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram, generate_latest

REGISTRY = CollectorRegistry()

REQUESTS = Counter(
    "agentserve_requests_total", "Requests served", ["tier"], registry=REGISTRY
)
TOKENS_REUSED = Counter(
    "agentserve_tokens_reused_total", "Prompt tokens served from cache",
    registry=REGISTRY,
)
TOKENS_COMPUTED = Counter(
    "agentserve_tokens_computed_total", "Prompt tokens prefilled from scratch",
    registry=REGISTRY,
)
AFFINITY = Counter(
    "agentserve_affinity_decisions_total", "Routing decisions", ["outcome"],
    registry=REGISTRY,
)
CACHE_ACTIONS = Counter(
    "agentserve_cache_actions_total", "Cache placement decisions", ["action"],
    registry=REGISTRY,
)
TTFT = Histogram(
    "agentserve_ttft_seconds", "Time to first token",
    buckets=(0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 30, 60),
    registry=REGISTRY,
)
GPU_CACHE_UTIL = Gauge(
    "agentserve_gpu_cache_utilization", "Fraction of KV cache capacity in use",
    ["replica"], registry=REGISTRY,
)
DRAM_CACHE_TOKENS = Gauge(
    "agentserve_dram_cache_tokens", "Tokens parked in host DRAM", ["replica"],
    registry=REGISTRY,
)
PREDICTED_GAP = Gauge(
    "agentserve_predicted_tool_gap_seconds", "EWMA estimate of tool duration",
    ["tool"], registry=REGISTRY,
)


def observe_result(result) -> None:
    REQUESTS.labels(tier=result.tier.value).inc()
    TOKENS_REUSED.inc(result.reused_tokens)
    TOKENS_COMPUTED.inc(result.computed_tokens)
    TTFT.observe(result.ttft_ms / 1000.0)


def observe_cluster(replicas, predictor) -> None:
    for r in replicas:
        GPU_CACHE_UTIL.labels(replica=str(r.id)).set(r.gpu_utilization)
        DRAM_CACHE_TOKENS.labels(replica=str(r.id)).set(r.dram_tokens)
    for tool, ms in predictor.snapshot().items():
        PREDICTED_GAP.labels(tool=tool).set(ms / 1000.0)


def render() -> bytes:
    return generate_latest(REGISTRY)
