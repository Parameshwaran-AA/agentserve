"""Replica selection.

RoundRobinRouter is the baseline: a rotation with no memory, which is what a
Kubernetes Service gives you today.

SessionAffinityRouter sends follow-up calls back to the replica already holding
the session's prefix. Affinity is a preference, not a pin. Past
affinity_queue_limit in-flight requests the call goes to the least loaded replica
instead, otherwise a hot session serializes behind itself while the cluster idles.
"""
from __future__ import annotations

from .config import PolicyConfig
from .models import CacheTier
from .replica import Replica
from .state import InMemorySessionStore, SessionStore


class RoundRobinRouter:
    name = "round-robin"

    def __init__(self, replicas: list[Replica]) -> None:
        self.replicas = replicas
        self._cursor = 0

    def select(self, session_id: str, now_ms: float) -> Replica:
        replica = self.replicas[self._cursor % len(self.replicas)]
        self._cursor += 1
        return replica


class SessionAffinityRouter:
    name = "session-affinity"

    def __init__(
        self,
        replicas: list[Replica],
        config: PolicyConfig | None = None,
        store: SessionStore | None = None,
    ) -> None:
        self.replicas = replicas
        self.config = config or PolicyConfig()
        self.store = store or InMemorySessionStore()
        self.affinity_hits = 0
        self.affinity_breaks = 0

    def select(self, session_id: str, now_ms: float) -> Replica:
        home_id = self.store.get_home(session_id)
        # A shared store outlives a scale-down, so a stale id must not index off
        # the end of the list.
        if home_id is not None and 0 <= home_id < len(self.replicas):
            home = self.replicas[home_id]
            tier, _ = home.lookup(session_id)
            if tier is not CacheTier.MISS:
                if home.inflight <= self.config.affinity_queue_limit:
                    self.affinity_hits += 1
                    return home
                # Warm but congested. Serve this one call elsewhere and leave the
                # home pointer alone. Re-homing on a momentary queue spike turns
                # a one-request detour into a discarded 40k-token prefix.
                self.affinity_breaks += 1
                return self._least_loaded(now_ms, avoid=home.id)

        replica = self._least_loaded(now_ms)
        self.store.set_home(session_id, replica.id)
        return replica

    @property
    def tracked_sessions(self) -> int:
        return self.store.size()

    def _least_loaded(self, now_ms: float, avoid: int | None = None) -> Replica:
        """Work-stealing placement: prefer the replica that frees up soonest."""
        candidates = [r for r in self.replicas if r.id != avoid] or self.replicas
        return min(
            candidates,
            key=lambda r: (r.inflight, max(r.free_at_ms, now_ms), r.gpu_utilization),
        )
