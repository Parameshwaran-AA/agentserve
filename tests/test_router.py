from agentserve.config import HardwareProfile, PolicyConfig
from agentserve.policy import LruPolicy
from agentserve.replica import Replica
from agentserve.router import RoundRobinRouter, SessionAffinityRouter

HW = HardwareProfile(gpu_cache_tokens=100_000)


def cluster(n=4):
    return [Replica(i, HW, LruPolicy()) for i in range(n)]


def test_round_robin_scatters_a_single_session():
    reps = cluster()
    r = RoundRobinRouter(reps)
    chosen = {r.select("s1", 0.0).id for _ in range(4)}
    assert chosen == {0, 1, 2, 3}, "baseline must scatter, that is the problem"


def test_affinity_returns_session_to_its_cache():
    reps = cluster()
    router = SessionAffinityRouter(reps, PolicyConfig())
    first = router.select("s1", 0.0)
    first.store("s1", 5_000, now_ms=0.0, predicted_return_ms=100.0)
    for _ in range(5):
        assert router.select("s1", 1.0).id == first.id
    assert router.affinity_hits == 5


def test_affinity_breaks_when_home_replica_is_saturated():
    reps = cluster()
    router = SessionAffinityRouter(reps, PolicyConfig(affinity_queue_limit=2))
    home = router.select("s1", 0.0)
    home.store("s1", 5_000, now_ms=0.0, predicted_return_ms=100.0)
    home.pending = [999.0] * 5  # deeply queued
    assert router.select("s1", 1.0).id != home.id
    assert router.affinity_breaks == 1


def test_cold_session_goes_to_least_loaded():
    reps = cluster()
    router = SessionAffinityRouter(reps, PolicyConfig())
    reps[0].pending = [999.0] * 3
    reps[1].pending = [999.0] * 3
    reps[2].pending = [999.0] * 3
    assert router.select("fresh", 1.0).id == 3


def test_load_break_does_not_discard_the_home_pointer():
    """A momentary queue spike must not cost the session its warm cache.

    Re-homing on transient load turns a one-request detour into a permanently
    discarded prefix, which is the opposite of what the whole system is for.
    """
    reps = cluster(3)
    router = SessionAffinityRouter(reps, PolicyConfig(affinity_queue_limit=2))
    home = router.select("s1", 0.0)
    home.store("s1", 5_000, now_ms=0.0, predicted_return_ms=100.0)

    with home.slot(), home.slot(), home.slot():
        detour = router.select("s1", 1.0)
        assert detour.id != home.id          # valve fires

    assert router.select("s1", 2.0).id == home.id  # and the session comes back
    assert router.store.get_home("s1") == home.id


def test_cold_session_does_set_a_home():
    reps = cluster(3)
    router = SessionAffinityRouter(reps, PolicyConfig())
    chosen = router.select("cold", 0.0)
    assert router.store.get_home("cold") == chosen.id
