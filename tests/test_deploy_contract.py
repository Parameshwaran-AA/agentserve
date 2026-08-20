"""Keeps the deployment manifests honest about the code.

An earlier version of the chart shipped a ConfigMap whose keys nothing read. It
rendered fine, mounted fine and changed nothing, which is worse than having no
ConfigMap because the cluster looks configured. These fail the build if the
manifests and settings.py drift apart again.
"""
import ast
import re
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
CHART = ROOT / "deploy" / "helm"
SETTINGS = (ROOT / "agentserve" / "settings.py").read_text()
CONFIGMAP = (CHART / "templates" / "configmap.yaml").read_text()
VALUES = yaml.safe_load((CHART / "values.yaml").read_text())

ENV_IN_SETTINGS = set(re.findall(r'"(AGENTSERVE_[A-Z0-9_]+)"', SETTINGS))
ENV_IN_CONFIGMAP = set(re.findall(r"^\s{2}(AGENTSERVE_[A-Z0-9_]+):", CONFIGMAP, re.M))


def test_settings_reads_some_env():
    assert len(ENV_IN_SETTINGS) > 10


@pytest.mark.parametrize("key", sorted(ENV_IN_CONFIGMAP))
def test_every_configmap_key_is_actually_read(key):
    assert key in ENV_IN_SETTINGS, (
        f"{key} is set by the Helm ConfigMap but never read in settings.py. "
        "A key nothing reads makes the chart look configurable when it is not."
    )


def test_policy_knobs_are_exposed_by_the_chart():
    for key in [
        "AGENTSERVE_PIN_THRESHOLD_MS",
        "AGENTSERVE_OFFLOAD_THRESHOLD_MS",
        "AGENTSERVE_AFFINITY_QUEUE_LIMIT",
        "AGENTSERVE_VLLM_ENDPOINTS",
        "AGENTSERVE_BACKEND",
    ]:
        assert key in ENV_IN_CONFIGMAP, f"{key} should be tunable via the chart"


def test_chart_defaults_to_a_single_gateway_pod():
    """Multi-pod without Redis is the silent-failure case; the default must be safe."""
    assert VALUES["gateway"]["replicaCount"] == 1
    assert VALUES["redis"]["enabled"] is False


def test_chart_guards_multi_pod_without_redis():
    helpers = (CHART / "templates" / "_helpers.tpl").read_text()
    assert "fail" in helpers
    assert "redis.enabled" in helpers
    for template in ("configmap.yaml", "deployment.yaml"):
        body = (CHART / "templates" / template).read_text()
        assert "agentserve.validate" in body, f"{template} must run the guard"


def test_vllm_runs_as_statefulset_for_stable_dns():
    body = (CHART / "templates" / "vllm-statefulset.yaml").read_text()
    assert "kind: StatefulSet" in body
    assert "clusterIP: None" in body, "headless Service is what gives per-pod DNS"


def test_prefix_caching_and_swap_space_are_enabled_by_default():
    """These are the two vLLM features the whole design depends on."""
    assert VALUES["vllm"]["enablePrefixCaching"] is True
    assert VALUES["vllm"]["swapSpaceGb"] > 0


def test_deployment_rolls_pods_when_config_changes():
    body = (CHART / "templates" / "deployment.yaml").read_text()
    assert "checksum/config" in body


def test_liveness_and_readiness_probe_different_endpoints():
    """/health must not touch dependencies; /ready must."""
    body = (CHART / "templates" / "deployment.yaml").read_text()
    assert "path: /health" in body and "path: /ready" in body


def test_compose_wires_a_shared_session_store():
    compose = yaml.safe_load((ROOT / "docker-compose.yml").read_text())
    env = compose["services"]["gateway"]["environment"]
    assert "redis" in env["AGENTSERVE_REDIS_URL"]
    assert "redis" in compose["services"]


def test_dockerfile_pins_one_worker():
    """Replica cache state is per-process; multiple uvicorn workers diverge."""
    body = (ROOT / "Dockerfile").read_text()
    assert '"--workers", "1"' in body


# ---- packaging -------------------------------------------------------------

PYPROJECT = (ROOT / "pyproject.toml").read_text()

LOCAL = {"agentserve", "bench", "scripts", "tests"}


def _third_party_imports(*dirs: str) -> set[str]:
    """Top-level imports, via ast so prose inside docstrings is not mistaken
    for an import statement."""
    found: set[str] = set()
    for d in dirs:
        for path in (ROOT / d).rglob("*.py"):
            for node in ast.walk(ast.parse(path.read_text())):
                if isinstance(node, ast.Import):
                    found |= {a.name.split(".")[0] for a in node.names}
                elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                    found.add(node.module.split(".")[0])
    return {
        m for m in found - LOCAL
        if m not in sys.stdlib_module_names and m != "__future__"
    }


def test_setuptools_packages_are_explicit():
    """Auto-discovery fails on this layout: bench/, deploy/ and notebooks/ sit
    beside agentserve/ and setuptools refuses to guess. Without this, a clean
    `pip install -e .` errors out and every CI job fails at the install step."""
    assert "[tool.setuptools]" in PYPROJECT
    assert 'packages = ["agentserve", "agentserve.backends", "bench"]' in PYPROJECT


def test_build_system_is_declared():
    assert "[build-system]" in PYPROJECT


def test_runtime_imports_are_declared_dependencies():
    for module in _third_party_imports("agentserve", "bench"):
        assert module.replace("_", "-") in PYPROJECT.lower() or module in PYPROJECT, (
            f"{module} is imported at runtime but not declared in pyproject.toml"
        )


def test_test_only_imports_are_declared_dev_dependencies():
    """PyYAML was imported here and declared nowhere, which broke a clean install."""
    for module in _third_party_imports("tests", "scripts"):
        assert module.replace("_", "-") in PYPROJECT.lower() or module in PYPROJECT, (
            f"{module} is imported by tests/scripts but not declared in pyproject.toml"
        )


def test_dockerfile_installs_from_pyproject():
    """A hand-written package list in the Dockerfile drifts from pyproject silently."""
    body = (ROOT / "Dockerfile").read_text()
    assert "pip install --no-cache-dir ." in body
    assert "COPY pyproject.toml" in body
