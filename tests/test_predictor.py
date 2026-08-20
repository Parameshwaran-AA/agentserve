from agentserve.config import PredictorConfig
from agentserve.predictor import ToolDurationPredictor


def test_falls_back_to_prior_when_unseen():
    p = ToolDurationPredictor(PredictorConfig(prior_ms=5000.0))
    assert p.predict_ms("s1", "run_tests") == 5000.0


def test_learns_per_tool_after_min_observations():
    p = ToolDurationPredictor(PredictorConfig(min_observations=2, ewma_alpha=0.5))
    p.observe("s1", "read_file", 300.0)
    p.observe("s1", "read_file", 300.0)
    assert p.predict_ms("s1", "read_file") == 300.0


def test_separates_fast_and_slow_tools():
    p = ToolDurationPredictor(PredictorConfig(min_observations=2))
    for _ in range(5):
        p.observe("s1", "read_file", 300.0)
        p.observe("s1", "run_tests", 45_000.0)
    fast = p.predict_ms("s1", "read_file")
    slow = p.predict_ms("s1", "run_tests")
    assert slow > fast * 50, "predictor must distinguish bimodal tool latencies"


def test_ewma_tracks_drift():
    p = ToolDurationPredictor(PredictorConfig(min_observations=1, ewma_alpha=0.5))
    p.observe("s1", "build", 1_000.0)
    early = p.predict_ms("s1", "build")
    for _ in range(10):
        p.observe("s1", "build", 90_000.0)
    assert p.predict_ms("s1", "build") > early * 10


def test_rejects_negative_duration():
    p = ToolDurationPredictor()
    try:
        p.observe("s1", "grep", -1.0)
    except ValueError:
        return
    raise AssertionError("expected ValueError on negative duration")
