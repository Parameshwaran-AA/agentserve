"""Contract tests for the vLLM adapter.

No GPU needed. These run the adapter against a mock server speaking vLLM's
OpenAI-compatible wire format, including the prefix-cache usage fields and SSE
streaming, which covers the parsing, retry and failover logic where adapter bugs
actually live. Throughput claims need real hardware.
"""
import json

import httpx
import pytest

from agentserve.backends.vllm import VllmBackend
from agentserve.config import HardwareProfile
from agentserve.policy import LruPolicy
from agentserve.replica import Replica


def replica(i=0):
    return Replica(i, HardwareProfile(), LruPolicy())


def completion_body(cached=0, prompt=1000, completion=50, text="hello"):
    return {
        "choices": [{"message": {"role": "assistant", "content": text}}],
        "usage": {
            "prompt_tokens": prompt,
            "completion_tokens": completion,
            "prompt_tokens_details": {"cached_tokens": cached},
        },
    }


def backend_with(handler, endpoints=None, max_retries=2):
    b = VllmBackend(endpoints or ["http://w0:8000"], model="m", max_retries=max_retries)
    b._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return b


@pytest.mark.anyio
async def test_reports_cache_miss_when_no_cached_tokens():
    b = backend_with(lambda r: httpx.Response(200, json=completion_body(cached=0)))
    res = await b.aexecute(replica(), "s1", 1000, 50, 0.0, 100.0, messages=[])
    assert res.tier.value == "miss"
    assert res.computed_tokens == 1000
    await b.aclose()


@pytest.mark.anyio
async def test_reports_cache_hit_from_vllm_usage_fields():
    b = backend_with(lambda r: httpx.Response(200, json=completion_body(cached=950)))
    res = await b.aexecute(replica(), "s1", 1000, 50, 0.0, 100.0, messages=[])
    assert res.tier.value == "gpu"
    assert res.reused_tokens == 950
    assert res.computed_tokens == 50
    await b.aclose()


@pytest.mark.anyio
async def test_response_text_is_returned():
    b = backend_with(lambda r: httpx.Response(200, json=completion_body(text="fixed it")))
    res = await b.aexecute(replica(), "s1", 10, 5, 0.0, 100.0, messages=[])
    assert res.text == "fixed it"
    await b.aclose()


@pytest.mark.anyio
async def test_fails_over_to_another_worker():
    seen = []

    def handler(request):
        seen.append(str(request.url))
        if "w0" in str(request.url):
            return httpx.Response(503)
        return httpx.Response(200, json=completion_body(cached=0))

    b = backend_with(handler, endpoints=["http://w0:8000", "http://w1:8000"])
    res = await b.aexecute(replica(0), "s1", 1000, 50, 0.0, 100.0, messages=[])
    assert res is not None
    assert b.failovers == 1
    assert any("w1" in u for u in seen)
    await b.aclose()


@pytest.mark.anyio
async def test_raises_only_when_every_worker_fails():
    b = backend_with(lambda r: httpx.Response(503),
                     endpoints=["http://w0:8000", "http://w1:8000"])
    with pytest.raises(RuntimeError, match="all vLLM workers failed"):
        await b.aexecute(replica(0), "s1", 1000, 50, 0.0, 100.0, messages=[])
    await b.aclose()


@pytest.mark.anyio
async def test_streaming_parses_sse_and_returns_final_result():
    chunks = [
        {"choices": [{"delta": {"content": "he"}}]},
        {"choices": [{"delta": {"content": "llo"}}]},
        {"choices": [], "usage": {"prompt_tokens": 900, "completion_tokens": 2,
                                  "prompt_tokens_details": {"cached_tokens": 880}}},
    ]
    payload = "".join(f"data: {json.dumps(c)}\n\n" for c in chunks) + "data: [DONE]\n\n"
    b = backend_with(lambda r: httpx.Response(200, text=payload,
                                              headers={"content-type": "text/event-stream"}))

    texts, result = [], None
    async for text, maybe in b.astream(
        replica=replica(), session_id="s1", messages=[], prompt_tokens=900,
        output_tokens=2, now_ms=0.0, predicted_return_ms=100.0,
    ):
        if maybe is not None:
            result = maybe
        elif text:
            texts.append(text)

    assert "".join(texts) == "hello"
    assert result.reused_tokens == 880
    assert result.tier.value == "gpu"
    await b.aclose()


@pytest.mark.anyio
async def test_probe_true_if_any_worker_alive():
    def handler(request):
        return httpx.Response(200 if "w1" in str(request.url) else 503)

    b = backend_with(handler, endpoints=["http://w0:8000", "http://w1:8000"])
    assert await b.aprobe() is True
    await b.aclose()


@pytest.mark.anyio
async def test_probe_false_if_all_workers_down():
    b = backend_with(lambda r: httpx.Response(503), endpoints=["http://w0:8000"])
    assert await b.aprobe() is False
    await b.aclose()


def test_requires_at_least_one_endpoint():
    with pytest.raises(ValueError):
        VllmBackend([], model="m")
