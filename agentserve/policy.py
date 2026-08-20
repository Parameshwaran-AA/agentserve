"""Cache retention policies, sharing one interface so the benchmark can swap them.

LruPolicy is what a stock serving engine does: evict whatever was touched longest
ago, with no way to tell a paused session from a finished one.

AdaptiveTtlPolicy consults the predictor. Sessions due back within
pin_threshold_ms stay in HBM, longer gaps go to host DRAM instead of being
discarded, dead sessions are dropped.
"""
from __future__ import annotations

from typing import Protocol

from .config import PolicyConfig
from .models import CacheEntry
from .predictor import ToolDurationPredictor


class Placement:
    """Where a just-completed session's cache should live."""
    PIN_GPU = "pin_gpu"
    OFFLOAD_DRAM = "offload_dram"
    DROP = "drop"


class CachePolicy(Protocol):
    name: str

    def on_release(self, entry: CacheEntry, now_ms: float) -> str: ...

    def victim_order(self, entries: list[CacheEntry], now_ms: float) -> list[CacheEntry]: ...


class LruPolicy:
    """Baseline. Keep everything on GPU, evict oldest touch first, no DRAM tier."""
    name = "lru"

    def on_release(self, entry: CacheEntry, now_ms: float) -> str:
        return Placement.PIN_GPU

    def victim_order(self, entries: list[CacheEntry], now_ms: float) -> list[CacheEntry]:
        return sorted(entries, key=lambda e: e.last_touch_ms)


class AdaptiveTtlPolicy:
    """Predictive retention with a DRAM second tier."""
    name = "adaptive-ttl"

    def __init__(
        self, predictor: ToolDurationPredictor, config: PolicyConfig | None = None
    ) -> None:
        self.predictor = predictor
        self.config = config or PolicyConfig()

    def on_release(self, entry: CacheEntry, now_ms: float) -> str:
        idle = entry.predicted_idle_ms
        if idle <= self.config.pin_threshold_ms:
            return Placement.PIN_GPU
        if idle <= self.config.offload_threshold_ms:
            return Placement.OFFLOAD_DRAM
        return Placement.DROP

    def victim_order(self, entries: list[CacheEntry], now_ms: float) -> list[CacheEntry]:
        """Evict the session least likely to need its cache soon.

        Sorting by predicted return time means a session due back in 40s is given
        up before one due back in 300ms, even if the latter was touched longer
        ago. Recency is a poor proxy for imminence in agent traffic.
        """
        def score(e: CacheEntry) -> tuple[float, float]:
            if now_ms - e.last_touch_ms > self.config.dead_session_ms:
                return (float("inf"), e.last_touch_ms)
            return (e.predicted_return_ms - now_ms, e.last_touch_ms)

        return sorted(entries, key=score, reverse=True)
