from agentserve.config import PolicyConfig, PredictorConfig
from agentserve.models import CacheEntry, CacheTier
from agentserve.policy import AdaptiveTtlPolicy, LruPolicy, Placement
from agentserve.predictor import ToolDurationPredictor


def entry(sid, tokens=1000, touch=0.0, ret=0.0):
    return CacheEntry(sid, tokens, CacheTier.GPU, touch, ret)


def make_policy():
    return AdaptiveTtlPolicy(
        ToolDurationPredictor(PredictorConfig()),
        PolicyConfig(pin_threshold_ms=2_000, offload_threshold_ms=300_000),
    )


def test_short_gap_pins_to_gpu():
    p = make_policy()
    assert p.on_release(entry("s", touch=0, ret=500), 0) == Placement.PIN_GPU


def test_long_gap_offloads_to_dram():
    p = make_policy()
    assert p.on_release(entry("s", touch=0, ret=45_000), 0) == Placement.OFFLOAD_DRAM


def test_very_long_gap_drops():
    p = make_policy()
    assert p.on_release(entry("s", touch=0, ret=900_000), 0) == Placement.DROP


def test_victim_order_prefers_evicting_late_returners():
    """The core inversion: recency loses to imminence."""
    p = make_policy()
    soon = entry("soon", touch=0, ret=1_000)      # older touch, back shortly
    later = entry("later", touch=900, ret=60_000)  # newer touch, back much later
    order = p.victim_order([soon, later], now_ms=1_000)
    assert order[0].session_id == "later"


def test_lru_ignores_prediction():
    p = LruPolicy()
    soon = entry("soon", touch=0, ret=1_000)
    later = entry("later", touch=900, ret=60_000)
    order = p.victim_order([soon, later], now_ms=1_000)
    assert order[0].session_id == "soon"
    assert p.on_release(soon, 0) == Placement.PIN_GPU
