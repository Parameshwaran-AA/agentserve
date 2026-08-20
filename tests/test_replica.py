from agentserve.config import HardwareProfile, PolicyConfig, PredictorConfig
from agentserve.models import CacheTier
from agentserve.policy import AdaptiveTtlPolicy, LruPolicy
from agentserve.predictor import ToolDurationPredictor
from agentserve.replica import Replica

HW = HardwareProfile(gpu_cache_tokens=10_000, dram_cache_tokens=50_000)


def adaptive():
    return AdaptiveTtlPolicy(ToolDurationPredictor(PredictorConfig()),
                             PolicyConfig(pin_threshold_ms=2_000, offload_threshold_ms=300_000))


def test_lookup_miss_on_empty():
    r = Replica(0, HW, LruPolicy())
    assert r.lookup("nope") == (CacheTier.MISS, 0)


def test_store_then_gpu_hit():
    r = Replica(0, HW, LruPolicy())
    r.store("s1", 4_000, now_ms=0, predicted_return_ms=100)
    assert r.lookup("s1") == (CacheTier.GPU, 4_000)


def test_gpu_capacity_is_enforced():
    r = Replica(0, HW, LruPolicy())
    for i in range(6):
        r.store(f"s{i}", 3_000, now_ms=float(i), predicted_return_ms=float(i) + 10)
    assert r.gpu_tokens <= HW.gpu_cache_tokens
    assert r.evictions > 0


def test_adaptive_policy_offloads_instead_of_dropping():
    r = Replica(0, HW, adaptive())
    r.store("slow", 4_000, now_ms=0, predicted_return_ms=45_000)
    assert r.lookup("slow") == (CacheTier.DRAM, 4_000)
    assert r.offloads == 1


def test_lru_has_no_dram_tier():
    r = Replica(0, HW, LruPolicy())
    for i in range(6):
        r.store(f"s{i}", 3_000, now_ms=float(i), predicted_return_ms=float(i))
    assert r.dram_tokens == 0


def test_promote_from_dram_moves_tier():
    r = Replica(0, HW, adaptive())
    r.store("s1", 4_000, now_ms=0, predicted_return_ms=45_000)
    r.promote_from_dram("s1")
    assert r.lookup("s1") == (CacheTier.GPU, 4_000)


def test_dram_capacity_is_bounded():
    hw = HardwareProfile(gpu_cache_tokens=10_000, dram_cache_tokens=9_000)
    r = Replica(0, hw, adaptive())
    for i in range(8):
        r.store(f"s{i}", 3_000, now_ms=float(i), predicted_return_ms=float(i) + 45_000)
    assert r.dram_tokens <= hw.dram_cache_tokens
