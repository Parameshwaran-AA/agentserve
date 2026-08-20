"""Environment config.

Everything the Helm chart and compose file set is read here. Two rules: every
value has a working default so the gateway runs with no environment set, and bad
values fail at startup naming the variable rather than on the thousandth request.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

from .config import ClusterConfig, HardwareProfile, PolicyConfig, PredictorConfig


class ConfigError(RuntimeError):
    pass


def _num(name: str, default: float, cast=float):
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return cast(default)
    try:
        value = cast(raw)
    except ValueError as exc:
        raise ConfigError(f"{name}={raw!r} is not a valid {cast.__name__}") from exc
    if value <= 0:
        raise ConfigError(f"{name} must be positive, got {value}")
    return value


def _flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _csv(name: str) -> list[str]:
    raw = os.environ.get(name, "")
    return [part.strip().rstrip(",") for part in raw.split(",") if part.strip().rstrip(",")]


@dataclass
class Settings:
    backend: str = "simulated"
    # Lets the same binary serve as the control arm of an A/B. Same gateway,
    # tokenizer and network path, so a measured difference comes from routing
    # and eviction rather than from two different pieces of software.
    router: str = "affinity"
    policy: str = "adaptive"
    vllm_endpoints: list[str] = field(default_factory=list)
    vllm_model: str = "mistralai/Mistral-7B-Instruct-v0.3"
    vllm_timeout_s: float = 300.0
    vllm_max_retries: int = 2
    redis_url: str | None = None
    session_ttl_s: float = 3_600.0
    max_sessions: int = 100_000
    api_key: str | None = None
    tokenizer: str = "cl100k_base"
    cluster: ClusterConfig = field(default_factory=ClusterConfig)

    @property
    def multi_pod_safe(self) -> bool:
        """True when the routing map is shared, so >1 gateway replica is correct."""
        return self.redis_url is not None


def load_settings() -> Settings:
    backend = os.environ.get("AGENTSERVE_BACKEND", "simulated").strip().lower()
    router = os.environ.get("AGENTSERVE_ROUTER", "affinity").strip().lower()
    policy = os.environ.get("AGENTSERVE_POLICY", "adaptive").strip().lower()
    if router not in {"affinity", "round-robin"}:
        raise ConfigError(f"AGENTSERVE_ROUTER={router!r} must be 'affinity' or 'round-robin'")
    if policy not in {"adaptive", "lru"}:
        raise ConfigError(f"AGENTSERVE_POLICY={policy!r} must be 'adaptive' or 'lru'")
    if backend not in {"simulated", "vllm"}:
        raise ConfigError(
            f"AGENTSERVE_BACKEND={backend!r} must be 'simulated' or 'vllm'"
        )

    endpoints = _csv("AGENTSERVE_VLLM_ENDPOINTS")
    if backend == "vllm" and not endpoints:
        raise ConfigError(
            "AGENTSERVE_BACKEND=vllm requires AGENTSERVE_VLLM_ENDPOINTS "
            "(comma-separated worker URLs)"
        )

    # Replica count follows the endpoint list. Letting them drift indexes off
    # the end of the list on the first request to the highest replica.
    replicas = len(endpoints) if backend == "vllm" else int(
        _num("AGENTSERVE_REPLICAS", 4, int)
    )

    cluster = ClusterConfig(
        replicas=replicas,
        hardware=HardwareProfile(
            gpu_cache_tokens=int(_num("AGENTSERVE_GPU_CACHE_TOKENS", 150_000, int)),
            dram_cache_tokens=int(_num("AGENTSERVE_DRAM_CACHE_TOKENS", 1_400_000, int)),
        ),
        policy=PolicyConfig(
            pin_threshold_ms=_num("AGENTSERVE_PIN_THRESHOLD_MS", 2_000.0),
            offload_threshold_ms=_num("AGENTSERVE_OFFLOAD_THRESHOLD_MS", 300_000.0),
            dead_session_ms=_num("AGENTSERVE_DEAD_SESSION_MS", 600_000.0),
            affinity_queue_limit=int(_num("AGENTSERVE_AFFINITY_QUEUE_LIMIT", 6, int)),
        ),
        predictor=PredictorConfig(
            ewma_alpha=_num("AGENTSERVE_EWMA_ALPHA", 0.3),
            prior_ms=_num("AGENTSERVE_PREDICTOR_PRIOR_MS", 5_000.0),
            min_observations=int(_num("AGENTSERVE_PREDICTOR_MIN_OBS", 2, int)),
        ),
    )

    if not 0 < cluster.predictor.ewma_alpha <= 1:
        raise ConfigError("AGENTSERVE_EWMA_ALPHA must be in (0, 1]")
    if cluster.policy.pin_threshold_ms >= cluster.policy.offload_threshold_ms:
        raise ConfigError(
            "AGENTSERVE_PIN_THRESHOLD_MS must be below AGENTSERVE_OFFLOAD_THRESHOLD_MS"
        )

    return Settings(
        backend=backend,
        router=router,
        policy=policy,
        vllm_endpoints=endpoints,
        vllm_model=os.environ.get("AGENTSERVE_VLLM_MODEL", Settings.vllm_model),
        vllm_timeout_s=_num("AGENTSERVE_VLLM_TIMEOUT_S", 300.0),
        vllm_max_retries=int(_num("AGENTSERVE_VLLM_MAX_RETRIES", 2, int)),
        redis_url=os.environ.get("AGENTSERVE_REDIS_URL") or None,
        session_ttl_s=_num("AGENTSERVE_SESSION_TTL_S", 3_600.0),
        max_sessions=int(_num("AGENTSERVE_MAX_SESSIONS", 100_000, int)),
        api_key=os.environ.get("AGENTSERVE_API_KEY") or None,
        tokenizer=os.environ.get("AGENTSERVE_TOKENIZER", "cl100k_base"),
        cluster=cluster,
    )
