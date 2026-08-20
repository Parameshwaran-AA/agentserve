"""Core value types shared by the router, policy, and backends."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class CacheTier(str, Enum):
    GPU = "gpu"
    DRAM = "dram"
    MISS = "miss"


@dataclass
class Turn:
    """One LLM call inside an agent session, plus the tool call that follows."""
    prompt_tokens: int
    output_tokens: int
    tool_name: str | None = None
    tool_duration_ms: float = 0.0


@dataclass
class SessionTrace:
    session_id: str
    arrival_ms: float
    turns: list[Turn]

    @property
    def total_prompt_tokens(self) -> int:
        return sum(t.prompt_tokens for t in self.turns)


@dataclass
class CacheEntry:
    """A session's KV cache resident on one replica."""
    session_id: str
    tokens: int
    tier: CacheTier
    last_touch_ms: float
    predicted_return_ms: float = 0.0

    @property
    def predicted_idle_ms(self) -> float:
        return max(0.0, self.predicted_return_ms - self.last_touch_ms)


@dataclass
class RequestResult:
    session_id: str
    replica_id: int
    tier: CacheTier
    reused_tokens: int
    computed_tokens: int
    prefill_ms: float
    decode_ms: float
    queue_ms: float
    restore_ms: float = 0.0
    # Populated by real backends; the simulator has no text to return.
    text: str = ""
    completion_tokens: int = 0

    @property
    def latency_ms(self) -> float:
        return self.queue_ms + self.restore_ms + self.prefill_ms + self.decode_ms

    @property
    def ttft_ms(self) -> float:
        return self.queue_ms + self.restore_ms + self.prefill_ms


@dataclass
class RunStats:
    """Aggregate counters produced by a benchmark run."""
    requests: int = 0
    gpu_hits: int = 0
    dram_hits: int = 0
    misses: int = 0
    tokens_reused: int = 0
    tokens_computed: int = 0
    ttfts: list[float] = field(default_factory=list)
    job_completion_ms: list[float] = field(default_factory=list)
    gpu_busy_ms: float = 0.0

    @property
    def gpu_hit_rate(self) -> float:
        return self.gpu_hits / self.requests if self.requests else 0.0

    @property
    def any_hit_rate(self) -> float:
        return (self.gpu_hits + self.dram_hits) / self.requests if self.requests else 0.0

    @property
    def reuse_rate(self) -> float:
        total = self.tokens_reused + self.tokens_computed
        return self.tokens_reused / total if total else 0.0
