"""Guards on the live measurement harness.

The prompt builder is the easiest place to silently invalidate a GPU run. If turn
N's prompt isn't an exact prefix of turn N+1's, vLLM reports near-zero hits and
the experiment measures nothing. The failure is invisible in the output: it just
looks like the idea doesn't work.
"""
from bench.live_replay import ArmRun, CallRecord, build_prompt, compare_table


def test_prompt_grows_strictly_by_appending():
    a = build_prompt(400, "s1", 0)
    b = build_prompt(700, "s1", 1)
    c = build_prompt(1100, "s1", 2)
    assert b.startswith(a), "turn 1 must contain turn 0 verbatim"
    assert c.startswith(b), "turn 2 must contain turn 1 verbatim"


def test_nothing_variable_is_appended_after_the_body():
    """A trailing turn counter or timestamp would break the prefix."""
    a = build_prompt(500, "s1", 0)
    b = build_prompt(500, "s1", 9)
    assert a == b, "turn index must not alter a prompt of the same length"


def test_sessions_never_share_a_prefix():
    a = build_prompt(600, "sess-a", 0)
    b = build_prompt(600, "sess-b", 0)
    assert not b.startswith(a) and not a.startswith(b)


def test_prompt_length_tracks_requested_tokens():
    assert len(build_prompt(2000, "s", 0)) > len(build_prompt(500, "s", 0))


def _run(label, hits, cached=800, prompt=1000):
    run = ArmRun(label=label)
    for i in range(10):
        run.calls.append(CallRecord("s", i, prompt, cached if i < hits else 0, 10.0, 0))
    run.session_ms = {"s": 100.0}
    return run


def test_report_computes_hit_and_reuse_rates():
    r = _run("x", hits=7).report()
    assert r["cache_hit_rate"] == 0.7
    assert 0 < r["token_reuse_rate"] < 1
    assert r["calls"] == 10


def test_failed_calls_are_excluded_from_rates_but_counted():
    run = _run("x", hits=10)
    run.calls.append(CallRecord("s", 99, 0, 0, 5.0, None, ok=False))
    r = run.report()
    assert r["failed"] == 1
    assert r["cache_hit_rate"] == 1.0


def test_compare_table_renders_both_arms():
    table = compare_table(_run("base", hits=2), _run("agentserve", hits=9))
    assert "Cache hit rate" in table
    assert "pp" in table
