import json

import fakeredis
import pytest
from fastapi.testclient import TestClient

from agentserve.gateway import create_app
from agentserve.settings import Settings
from agentserve.state import RedisSessionStore

PROMPT = [{"role": "user", "content": "x" * 40_000}]


def make_client(**kw):
    return TestClient(create_app(Settings(**kw)))


@pytest.fixture()
def client():
    with make_client() as c:
        yield c


def post(client, session="s1", tool="run_tests", stream=False, **headers):
    body = {"messages": PROMPT, "max_tokens": 200}
    if stream:
        body["stream"] = True
    return client.post(
        "/v1/chat/completions", json=body,
        headers={"X-Session-Id": session, "X-Tool-Name": tool, **headers},
    )


# ---- basics ----------------------------------------------------------------

def test_health_reports_wiring(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["policy"] == "adaptive-ttl"
    assert body["backend"] == "simulated"
    assert body["multi_pod_safe"] is False


def test_ready_endpoint(client):
    r = client.get("/ready")
    assert r.status_code == 200
    assert r.json()["ready"] is True


def test_session_header_required(client):
    assert client.post("/v1/chat/completions", json={"messages": PROMPT}).status_code == 400


def test_absurd_session_id_rejected(client):
    assert post(client, session="s" * 300).status_code == 400


def test_empty_messages_rejected(client):
    r = client.post("/v1/chat/completions", json={"messages": []},
                    headers={"X-Session-Id": "s1"})
    assert r.status_code == 400


def test_malformed_body_rejected(client):
    r = client.post("/v1/chat/completions", content=b"{not json",
                    headers={"X-Session-Id": "s1", "Content-Type": "application/json"})
    assert r.status_code == 400


# ---- the actual feature ----------------------------------------------------

def test_first_call_misses_then_hits(client):
    assert post(client).json()["agentserve"]["cache_tier"] == "miss"
    tiers = [post(client).json()["agentserve"]["cache_tier"] for _ in range(3)]
    assert all(t in ("gpu", "dram") for t in tiers)


def test_session_stays_on_one_replica(client):
    replicas = {post(client).json()["agentserve"]["replica"] for _ in range(6)}
    assert len(replicas) == 1


def test_different_sessions_spread_across_replicas(client):
    replicas = {post(client, session=f"s{i}").json()["agentserve"]["replica"]
                for i in range(8)}
    assert len(replicas) > 1


def test_cached_tokens_reported_openai_style(client):
    post(client)
    assert post(client).json()["usage"]["prompt_tokens_details"]["cached_tokens"] > 0


def test_token_count_uses_real_tokenizer_not_char_div_4(client):
    """40k characters is not 10k tokens; the heuristic was wrong by ~25%."""
    usage = post(client).json()["usage"]
    assert usage["prompt_tokens"] != len(PROMPT[0]["content"]) // 4


# ---- streaming -------------------------------------------------------------

def test_streaming_returns_sse(client):
    r = post(client, stream=True)
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/event-stream")
    assert r.headers["x-accel-buffering"] == "no"


def test_stream_terminates_with_done(client):
    body = post(client, stream=True).text
    assert body.rstrip().endswith("data: [DONE]")


def test_stream_chunks_are_valid_openai_format(client):
    lines = [ln for ln in post(client, stream=True).text.splitlines()
             if ln.startswith("data: ") and "[DONE]" not in ln]
    assert lines
    first = json.loads(lines[0][6:])
    assert first["object"] == "chat.completion.chunk"
    assert "delta" in first["choices"][0]


def test_streaming_still_updates_the_cache(client):
    post(client, stream=True)
    assert post(client).json()["agentserve"]["cache_tier"] in ("gpu", "dram")


# ---- auth ------------------------------------------------------------------

def test_no_auth_required_by_default(client):
    assert post(client).status_code == 200


def test_api_key_enforced_when_configured():
    with make_client(api_key="secret") as c:
        assert post(c).status_code == 401
        assert post(c, Authorization="Bearer wrong").status_code == 401
        assert post(c, Authorization="Bearer secret").status_code == 200


# ---- multi-pod -------------------------------------------------------------

def test_two_gateways_on_shared_redis_route_identically():
    """The gap this release closes: without a shared store these disagree."""
    shared = fakeredis.FakeRedis(decode_responses=True)
    store = RedisSessionStore(shared)
    apps = [create_app(Settings(), backend=None) for _ in range(2)]
    for app in apps:
        app.state.router.store = store

    with TestClient(apps[0]) as a, TestClient(apps[1]) as b:
        first = post(a, session="shared").json()["agentserve"]["replica"]
        second = post(b, session="shared").json()["agentserve"]["replica"]
        assert first == second


def test_debug_sessions_exposes_store(client):
    post(client)
    body = client.get("/debug/sessions").json()
    assert body["session_store"] == "memory"
    assert body["tracked_sessions"] >= 1
    assert len(body["replicas"]) == 4
