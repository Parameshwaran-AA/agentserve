"""Session -> replica map.

Two implementations. InMemorySessionStore is correct for a single gateway pod;
it is bounded with a TTL so a long-running process doesn't leak. RedisSessionStore
is for multiple pods, which need to agree on where each session lives.

Entries are allowed to expire. A stale or missing mapping costs one cache miss
and the router re-homes the session on the next call, so this is a cache and not
a consensus problem.
"""
from __future__ import annotations

import time
from collections import OrderedDict
from typing import Protocol


class SessionStore(Protocol):
    def get_home(self, session_id: str) -> int | None: ...

    def set_home(self, session_id: str, replica_id: int) -> None: ...

    def get_last_seen(self, session_id: str) -> float | None: ...

    def set_last_seen(self, session_id: str, when_ms: float) -> None: ...

    def size(self) -> int: ...

    def close(self) -> None: ...


class InMemorySessionStore:
    """Bounded LRU with per-entry TTL. Correct for exactly one gateway pod."""

    name = "memory"

    def __init__(self, max_sessions: int = 100_000, ttl_s: float = 3_600.0) -> None:
        self.max_sessions = max_sessions
        self.ttl_s = ttl_s
        self._home: OrderedDict[str, tuple[int, float]] = OrderedDict()
        self._seen: OrderedDict[str, tuple[float, float]] = OrderedDict()

    def _get(self, table: OrderedDict, key: str):
        entry = table.get(key)
        if entry is None:
            return None
        value, expires_at = entry
        if time.time() > expires_at:
            table.pop(key, None)
            return None
        table.move_to_end(key)
        return value

    def _put(self, table: OrderedDict, key: str, value) -> None:
        table[key] = (value, time.time() + self.ttl_s)
        table.move_to_end(key)
        while len(table) > self.max_sessions:
            table.popitem(last=False)

    def get_home(self, session_id: str) -> int | None:
        return self._get(self._home, session_id)

    def set_home(self, session_id: str, replica_id: int) -> None:
        self._put(self._home, session_id, replica_id)

    def get_last_seen(self, session_id: str) -> float | None:
        return self._get(self._seen, session_id)

    def set_last_seen(self, session_id: str, when_ms: float) -> None:
        self._put(self._seen, session_id, when_ms)

    def size(self) -> int:
        return len(self._home)

    def close(self) -> None:
        self._home.clear()
        self._seen.clear()


class RedisSessionStore:
    """Shared map for multi-pod gateways.

    If Redis is unreachable we degrade to a cache miss rather than raising.
    Losing affinity costs one redundant prefill; refusing to serve costs the
    request.
    """

    name = "redis"

    def __init__(self, client, ttl_s: float = 3_600.0, prefix: str = "agentserve") -> None:
        self.client = client
        self.ttl_s = int(ttl_s)
        self.prefix = prefix
        self.errors = 0

    def _key(self, kind: str, session_id: str) -> str:
        return f"{self.prefix}:{kind}:{session_id}"

    def get_home(self, session_id: str) -> int | None:
        try:
            raw = self.client.get(self._key("home", session_id))
        except Exception:
            self.errors += 1
            return None
        return int(raw) if raw is not None else None

    def set_home(self, session_id: str, replica_id: int) -> None:
        try:
            self.client.set(self._key("home", session_id), replica_id, ex=self.ttl_s)
        except Exception:
            self.errors += 1

    def get_last_seen(self, session_id: str) -> float | None:
        try:
            raw = self.client.get(self._key("seen", session_id))
        except Exception:
            self.errors += 1
            return None
        return float(raw) if raw is not None else None

    def set_last_seen(self, session_id: str, when_ms: float) -> None:
        try:
            self.client.set(self._key("seen", session_id), when_ms, ex=self.ttl_s)
        except Exception:
            self.errors += 1

    def size(self) -> int:
        try:
            return len(list(self.client.scan_iter(match=f"{self.prefix}:home:*", count=1000)))
        except Exception:
            self.errors += 1
            return -1

    def close(self) -> None:
        try:
            self.client.close()
        except Exception:
            pass


def build_session_store(url: str | None, ttl_s: float, max_sessions: int) -> SessionStore:
    """Pick a store from config."""
    if not url:
        return InMemorySessionStore(max_sessions=max_sessions, ttl_s=ttl_s)

    import redis  # lazy: single-pod deployments need no redis client

    client = redis.Redis.from_url(url, decode_responses=True, socket_timeout=0.25)
    client.ping()  # fail at startup, not on the first agent request
    return RedisSessionStore(client, ttl_s=ttl_s)
