"""In-flight accounting and concurrent behaviour.

The bug these guard against: replica.pending was only populated by the benchmark,
so inflight was permanently 0 in the live gateway. That made the affinity release
valve dead code, since 0 <= queue_limit is always true, and left _least_loaded
blind to load. Every test passed, because every test was single-request.
"""
import asyncio

import pytest
from fastapi.testclient import TestClient

from agentserve.config import HardwareProfile, PolicyConfig
from agentserve.gateway import create_app
from agentserve.policy import LruPolicy
from agentserve.replica import Replica
from agentserve.router import SessionAffinityRouter
from agentserve.settings import Settings

PROMPT = [{"role": "user", "content": "x" * 30_000}]


def replica(i=0):
    return Replica(i, HardwareProfile(), LruPolicy())


# ---- slot accounting -------------------------------------------------------

def test_slot_raises_and_lowers_inflight():
    r = replica()
    assert r.inflight == 0
    with r.slot():
        assert r.inflight == 1
    assert r.inflight == 0


def test_nested_slots_count_up():
    r = replica()
    with r.slot(), r.slot(), r.slot():
        assert r.inflight == 3
    assert r.inflight == 0


def test_slot_is_released_on_exception():
    """A leaked counter would make this replica look busy forever and never
    receive another affinity-routed request."""
    r = replica()
    with pytest.raises(ValueError):
        with r.slot():
            raise ValueError("upstream blew up")
    assert r.inflight == 0


def test_peak_inflight_is_recorded():
    r = replica()
    with r.slot(), r.slot():
        pass
    assert r.peak_active == 2
    assert r.inflight == 0


def test_inflight_never_goes_negative():
    r = replica()
    r.active = 0
    with r.slot():
        pass
    assert r.inflight >= 0


# ---- the valve is now reachable -------------------------------------------

def test_release_valve_fires_from_live_slots():
    """Previously impossible: only the simulator could raise inflight."""
    reps = [replica(i) for i in range(3)]
    router = SessionAffinityRouter(reps, PolicyConfig(affinity_queue_limit=2))
    home = router.select("s1", 0.0)
    home.store("s1", 5_000, now_ms=0.0, predicted_return_ms=100.0)

    assert router.select("s1", 1.0).id == home.id          # quiet: affinity holds

    with home.slot(), home.slot(), home.slot():            # now genuinely busy
        assert router.select("s1", 2.0).id != home.id
    assert router.affinity_breaks == 1

    assert router.select("s1", 3.0).id == home.id          # drains, affinity returns


def test_least_loaded_sees_live_load():
    reps = [replica(i) for i in range(3)]
    router = SessionAffinityRouter(reps, PolicyConfig())
    with reps[0].slot(), reps[1].slot():
        assert router.select("fresh", 0.0).id == 2


# ---- concurrent requests through the gateway -------------------------------

@pytest.mark.anyio
async def test_concurrent_requests_leave_no_leaked_slots():
    app = create_app(Settings())
    with TestClient(app) as c:
        async def call(i):
            return await asyncio.to_thread(
                c.post, "/v1/chat/completions",
                json={"messages": PROMPT, "max_tokens": 50},
                headers={"X-Session-Id": f"s{i}", "X-Tool-Name": "grep"},
            )

        responses = await asyncio.gather(*(call(i) for i in range(24)))
        assert all(r.status_code == 200 for r in responses)
        for r in app.state.replicas:
            assert r.inflight == 0, f"replica {r.id} leaked {r.inflight} slots"


@pytest.mark.anyio
async def test_concurrent_same_session_stays_on_one_replica():
    app = create_app(Settings())
    with TestClient(app) as c:
        async def call():
            return await asyncio.to_thread(
                c.post, "/v1/chat/completions",
                json={"messages": PROMPT, "max_tokens": 50},
                headers={"X-Session-Id": "hot", "X-Tool-Name": "read_file"},
            )

        results = await asyncio.gather(*(call() for _ in range(12)))
        replicas = {r.json()["agentserve"]["replica"] for r in results}
        assert len(replicas) == 1


@pytest.mark.anyio
async def test_cache_capacity_holds_under_concurrency():
    """Two in-flight requests can both read pre-store capacity; the replica
    must still not end up over its token budget."""
    app = create_app(Settings())
    with TestClient(app) as c:
        async def call(i):
            return await asyncio.to_thread(
                c.post, "/v1/chat/completions",
                json={"messages": PROMPT, "max_tokens": 50},
                headers={"X-Session-Id": f"cap{i}", "X-Tool-Name": "read_file"},
            )

        await asyncio.gather(*(call(i) for i in range(40)))
        for r in app.state.replicas:
            assert r.gpu_tokens <= r.hw.gpu_cache_tokens


def test_debug_endpoint_exposes_peak_inflight():
    with TestClient(create_app(Settings())) as c:
        c.post("/v1/chat/completions", json={"messages": PROMPT, "max_tokens": 20},
               headers={"X-Session-Id": "s1"})
        body = c.get("/debug/sessions").json()
        assert "peak_inflight" in body["replicas"][0]


# ---- baseline mode ---------------------------------------------------------

def test_gateway_can_run_as_the_baseline_arm():
    """Same binary, same network path, only routing and eviction differ."""
    with TestClient(create_app(Settings(router="round-robin", policy="lru"))) as c:
        h = c.get("/health").json()
        assert h["router"] == "round-robin"
        assert h["policy"] == "lru"
        replicas = {
            c.post("/v1/chat/completions", json={"messages": PROMPT, "max_tokens": 20},
                   headers={"X-Session-Id": "same"}).json()["agentserve"]["replica"]
            for _ in range(4)
        }
        assert len(replicas) > 1, "round-robin must scatter, that is the control"
