"""Tunable constants for the serving model.

Order-of-magnitude figures for a 7B model on a datacenter accelerator. Decode is
amortized per request under continuous batching, which is why it looks so cheap
next to a batch-of-one measurement. Prefill dominates, and that asymmetry is why
cache reuse matters.

Every hardware assumption lives here and nowhere else.
"""
from dataclasses import dataclass, field


@dataclass(frozen=True)
class HardwareProfile:
    prefill_ms_per_token: float = 0.065
    decode_ms_per_token: float = 0.35
    dram_restore_ms_per_token: float = 0.004
    dram_offload_ms_per_token: float = 0.003
    gpu_cache_tokens: int = 150_000
    dram_cache_tokens: int = 1_400_000

    @property
    def dram_speedup(self) -> float:
        """How much cheaper a DRAM restore is than a cold prefill."""
        return self.prefill_ms_per_token / self.dram_restore_ms_per_token


@dataclass(frozen=True)
class PolicyConfig:
    """Thresholds for the adaptive TTL decision.

    Under pin_threshold_ms the cache stays in HBM, up to offload_threshold_ms it
    goes to host DRAM, beyond that it is dropped.
    """
    pin_threshold_ms: float = 2_000.0
    offload_threshold_ms: float = 300_000.0
    dead_session_ms: float = 600_000.0
    affinity_queue_limit: int = 6


@dataclass(frozen=True)
class PredictorConfig:
    ewma_alpha: float = 0.3
    prior_ms: float = 5_000.0
    min_observations: int = 2


@dataclass(frozen=True)
class ClusterConfig:
    replicas: int = 4
    hardware: HardwareProfile = field(default_factory=HardwareProfile)
    policy: PolicyConfig = field(default_factory=PolicyConfig)
    predictor: PredictorConfig = field(default_factory=PredictorConfig)
