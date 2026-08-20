import pytest

from agentserve.settings import ConfigError, load_settings


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for key in list(__import__("os").environ):
        if key.startswith("AGENTSERVE_"):
            monkeypatch.delenv(key, raising=False)


def test_defaults_work_with_no_environment():
    s = load_settings()
    assert s.backend == "simulated"
    assert s.cluster.replicas == 4
    assert s.multi_pod_safe is False


def test_env_actually_reaches_the_policy(monkeypatch):
    """Guards the bug where the Helm ConfigMap was decorative."""
    monkeypatch.setenv("AGENTSERVE_PIN_THRESHOLD_MS", "500")
    monkeypatch.setenv("AGENTSERVE_OFFLOAD_THRESHOLD_MS", "60000")
    monkeypatch.setenv("AGENTSERVE_AFFINITY_QUEUE_LIMIT", "12")
    monkeypatch.setenv("AGENTSERVE_REPLICAS", "9")
    s = load_settings()
    assert s.cluster.policy.pin_threshold_ms == 500
    assert s.cluster.policy.offload_threshold_ms == 60_000
    assert s.cluster.policy.affinity_queue_limit == 12
    assert s.cluster.replicas == 9


def test_vllm_backend_requires_endpoints(monkeypatch):
    monkeypatch.setenv("AGENTSERVE_BACKEND", "vllm")
    with pytest.raises(ConfigError, match="VLLM_ENDPOINTS"):
        load_settings()


def test_replica_count_follows_endpoint_list(monkeypatch):
    """Otherwise a request to the highest replica indexes off the endpoint list."""
    monkeypatch.setenv("AGENTSERVE_BACKEND", "vllm")
    monkeypatch.setenv("AGENTSERVE_REPLICAS", "99")
    monkeypatch.setenv("AGENTSERVE_VLLM_ENDPOINTS", "http://a:8000,http://b:8000")
    s = load_settings()
    assert s.cluster.replicas == 2
    assert s.vllm_endpoints == ["http://a:8000", "http://b:8000"]


def test_trailing_commas_from_helm_templates_are_tolerated(monkeypatch):
    monkeypatch.setenv("AGENTSERVE_BACKEND", "vllm")
    monkeypatch.setenv("AGENTSERVE_VLLM_ENDPOINTS", " http://a:8000, http://b:8000, ")
    assert load_settings().vllm_endpoints == ["http://a:8000", "http://b:8000"]


def test_unknown_backend_rejected(monkeypatch):
    monkeypatch.setenv("AGENTSERVE_BACKEND", "tensorrt")
    with pytest.raises(ConfigError):
        load_settings()


def test_bad_number_fails_at_startup_naming_the_variable(monkeypatch):
    monkeypatch.setenv("AGENTSERVE_PIN_THRESHOLD_MS", "soon")
    with pytest.raises(ConfigError, match="AGENTSERVE_PIN_THRESHOLD_MS"):
        load_settings()


def test_inverted_thresholds_rejected(monkeypatch):
    monkeypatch.setenv("AGENTSERVE_PIN_THRESHOLD_MS", "90000")
    monkeypatch.setenv("AGENTSERVE_OFFLOAD_THRESHOLD_MS", "1000")
    with pytest.raises(ConfigError, match="must be below"):
        load_settings()


def test_redis_url_marks_deployment_multi_pod_safe(monkeypatch):
    monkeypatch.setenv("AGENTSERVE_REDIS_URL", "redis://localhost:6379/0")
    assert load_settings().multi_pod_safe is True
