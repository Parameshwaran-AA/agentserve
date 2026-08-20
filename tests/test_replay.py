"""End-to-end checks on the benchmark itself.

These guard the claims made about the system, so a regression in the policy
shows up as a failing assertion rather than a quietly worse number.
"""
from agentserve.config import ClusterConfig
from bench.replay import compare, run_arm
from bench.traces import generate_trace, trace_summary


def small_trace(seed=1):
    return generate_trace(sessions=60, seed=seed, arrival_window_ms=500_000)


def test_trace_prompts_grow_monotonically():
    for t in small_trace():
        sizes = [turn.prompt_tokens for turn in t.turns]
        assert sizes == sorted(sizes), "agent transcripts only ever append"


def test_trace_tool_gaps_are_bimodal():
    s = trace_summary(small_trace())
    assert s["p90_tool_gap_ms"] > 20 * s["median_tool_gap_ms"]


def test_both_arms_serve_identical_request_counts():
    traces = small_trace()
    base, agent = compare(traces, ClusterConfig())
    assert base.stats.requests == agent.stats.requests
    assert base.stats.requests == sum(len(t.turns) for t in traces)


def test_agentserve_improves_cache_hit_rate():
    base, agent = compare(small_trace(), ClusterConfig())
    assert agent.stats.gpu_hit_rate > base.stats.gpu_hit_rate


def test_agentserve_recomputes_fewer_tokens():
    base, agent = compare(small_trace(), ClusterConfig())
    assert agent.stats.tokens_computed < base.stats.tokens_computed


def test_agentserve_reduces_gpu_compute_time():
    base, agent = compare(small_trace(), ClusterConfig())
    assert agent.stats.gpu_busy_ms < base.stats.gpu_busy_ms


def test_result_is_deterministic():
    a = run_arm(small_trace(), ClusterConfig(), adaptive=True, label="a")
    b = run_arm(small_trace(), ClusterConfig(), adaptive=True, label="b")
    assert a.report() == b.report()


def test_improvement_holds_across_seeds():
    for seed in (1, 2, 3, 4):
        base, agent = compare(small_trace(seed), ClusterConfig())
        assert agent.stats.gpu_hit_rate > base.stats.gpu_hit_rate, f"seed {seed}"
