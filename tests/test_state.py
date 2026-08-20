import time

import fakeredis
import pytest

from agentserve.state import InMemorySessionStore, RedisSessionStore, build_session_store


@pytest.fixture(params=["memory", "redis"])
def store(request):
    if request.param == "memory":
        return InMemorySessionStore(max_sessions=50, ttl_s=60)
    return RedisSessionStore(fakeredis.FakeRedis(decode_responses=True), ttl_s=60)


def test_roundtrip_home(store):
    assert store.get_home("s1") is None
    store.set_home("s1", 3)
    assert store.get_home("s1") == 3


def test_roundtrip_last_seen(store):
    store.set_last_seen("s1", 1234.5)
    assert store.get_last_seen("s1") == pytest.approx(1234.5)


def test_sessions_are_isolated(store):
    store.set_home("a", 0)
    store.set_home("b", 1)
    assert store.get_home("a") == 0
    assert store.get_home("b") == 1


def test_memory_store_is_bounded():
    """An unbounded map is a memory leak on a long-running gateway."""
    s = InMemorySessionStore(max_sessions=10, ttl_s=60)
    for i in range(100):
        s.set_home(f"s{i}", i % 4)
    assert s.size() == 10
    assert s.get_home("s99") is not None   # newest survives
    assert s.get_home("s0") is None        # oldest evicted


def test_memory_store_expires():
    s = InMemorySessionStore(ttl_s=0.05)
    s.set_home("s1", 2)
    time.sleep(0.08)
    assert s.get_home("s1") is None


def test_redis_failure_degrades_to_miss_not_error():
    """The gateway is on the hot path; a Redis blip must not fail inference."""
    class Broken:
        def get(self, *a, **k): raise ConnectionError("redis down")
        def set(self, *a, **k): raise ConnectionError("redis down")
        def scan_iter(self, *a, **k): raise ConnectionError("redis down")
        def close(self): pass

    s = RedisSessionStore(Broken())
    assert s.get_home("s1") is None      # degrades, does not raise
    s.set_home("s1", 1)
    assert s.errors >= 2


def test_build_session_store_defaults_to_memory():
    assert build_session_store(None, 60, 100).name == "memory"


def test_two_gateways_sharing_redis_agree():
    """The whole point of the shared store: pod A and pod B route identically."""
    client = fakeredis.FakeRedis(decode_responses=True)
    pod_a = RedisSessionStore(client)
    pod_b = RedisSessionStore(client)
    pod_a.set_home("task-42", 3)
    assert pod_b.get_home("task-42") == 3


def test_two_in_memory_gateways_do_not_agree():
    """Documents the failure this design exists to prevent."""
    pod_a, pod_b = InMemorySessionStore(), InMemorySessionStore()
    pod_a.set_home("task-42", 3)
    assert pod_b.get_home("task-42") is None
