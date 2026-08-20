"""One vLLM replica: its two-tier KV cache and request queue.

Capacity is counted in tokens, not bytes. Token count is proportional to KV
footprint for a fixed model and dtype, and it keeps the simulated backend and the
real adapter in the same units.
"""
from __future__ import annotations

from contextlib import contextmanager

from .config import HardwareProfile
from .models import CacheEntry, CacheTier
from .policy import CachePolicy, Placement


class Replica:
    def __init__(self, replica_id: int, hardware: HardwareProfile, policy: CachePolicy) -> None:
        self.id = replica_id
        self.hw = hardware
        self.policy = policy
        self.gpu: dict[str, CacheEntry] = {}
        self.dram: dict[str, CacheEntry] = {}
        self.free_at_ms: float = 0.0
        # pending holds simulated completion timestamps for the benchmark;
        # active counts live requests the gateway has dispatched. Both feed
        # inflight. If only the simulator populated it, the router's release
        # valve would be dead code in production.
        self.pending: list[float] = []
        self.active: int = 0
        self.peak_active: int = 0
        self.evictions: int = 0
        self.offloads: int = 0

    @property
    def inflight(self) -> int:
        """Requests dispatched here that have not finished."""
        return len(self.pending) + self.active

    def prune(self, now_ms: float) -> None:
        self.pending = [t for t in self.pending if t > now_ms]

    @contextmanager
    def slot(self):
        """Hold an in-flight slot for the life of a request.

        A context manager, not manual inc/dec: requests fail, time out and get
        cancelled mid-stream, and a leaked counter makes this replica look busy
        forever so it never receives another affinity-routed call.
        """
        self.active += 1
        self.peak_active = max(self.peak_active, self.active)
        try:
            yield self
        finally:
            self.active = max(0, self.active - 1)

    @property
    def gpu_tokens(self) -> int:
        return sum(e.tokens for e in self.gpu.values())

    @property
    def dram_tokens(self) -> int:
        return sum(e.tokens for e in self.dram.values())

    @property
    def gpu_utilization(self) -> float:
        return self.gpu_tokens / self.hw.gpu_cache_tokens

    def lookup(self, session_id: str) -> tuple[CacheTier, int]:
        """Return the best tier holding this session and how many tokens it covers."""
        if session_id in self.gpu:
            return CacheTier.GPU, self.gpu[session_id].tokens
        if session_id in self.dram:
            return CacheTier.DRAM, self.dram[session_id].tokens
        return CacheTier.MISS, 0

    def promote_from_dram(self, session_id: str) -> None:
        entry = self.dram.pop(session_id, None)
        if entry is not None:
            entry.tier = CacheTier.GPU
            self.gpu[session_id] = entry

    def store(
        self, session_id: str, tokens: int, now_ms: float, predicted_return_ms: float
    ) -> None:
        """Install or refresh a session's cache after a request completes."""
        entry = CacheEntry(
            session_id=session_id,
            tokens=tokens,
            tier=CacheTier.GPU,
            last_touch_ms=now_ms,
            predicted_return_ms=predicted_return_ms,
        )
        self.dram.pop(session_id, None)
        self.gpu[session_id] = entry

        placement = self.policy.on_release(entry, now_ms)
        if placement == Placement.DROP:
            del self.gpu[session_id]
            self.evictions += 1
        elif placement == Placement.OFFLOAD_DRAM:
            self._demote(entry)

        self._enforce_capacity(now_ms)

    def _demote(self, entry: CacheEntry) -> None:
        """Move an entry to host DRAM if there is room."""
        self.gpu.pop(entry.session_id, None)
        if self.dram_tokens + entry.tokens <= self.hw.dram_cache_tokens:
            entry.tier = CacheTier.DRAM
            self.dram[entry.session_id] = entry
            self.offloads += 1
        else:
            self.evictions += 1

    def _enforce_capacity(self, now_ms: float) -> None:
        """Shed GPU entries until we are back under capacity."""
        if self.gpu_tokens <= self.hw.gpu_cache_tokens:
            return
        for victim in self.policy.victim_order(list(self.gpu.values()), now_ms):
            if self.gpu_tokens <= self.hw.gpu_cache_tokens:
                break
            if self.policy.on_release(victim, now_ms) == Placement.OFFLOAD_DRAM:
                self._demote(victim)
            else:
                self.gpu.pop(victim.session_id, None)
                self.evictions += 1
        self._trim_dram()

    def _trim_dram(self) -> None:
        while self.dram_tokens > self.hw.dram_cache_tokens and self.dram:
            oldest = min(self.dram.values(), key=lambda e: e.last_touch_ms)
            del self.dram[oldest.session_id]
            self.evictions += 1
