"""OpenAI-compatible gateway.

Clients point their existing SDK here instead of at vLLM and add one header,
X-Session-Id. Routing, cache retention and DRAM offload happen behind the API.

X-Tool-Name is optional. Passing it lets the predictor key its estimate on the
tool rather than a session-level average.

Error policy throughout: this sits on the hot path for every agent call, so a
failure in our own machinery degrades to a cache miss, never to a failed request.
"""
from __future__ import annotations

import json
import logging
import time
from contextlib import asynccontextmanager
from typing import Any

from fastapi import Depends, FastAPI, Header, HTTPException, Request, Response
from fastapi.responses import StreamingResponse

from .backends.simulated import SimulatedBackend
from .backends.vllm import VllmBackend
from .metrics import AFFINITY, CACHE_ACTIONS, observe_cluster, observe_result, render
from .policy import AdaptiveTtlPolicy, LruPolicy
from .predictor import ToolDurationPredictor
from .replica import Replica
from .router import RoundRobinRouter, SessionAffinityRouter
from .settings import Settings, load_settings
from .state import build_session_store
from .tokenizer import Tokenizer

log = logging.getLogger("agentserve")


def build_backend(settings: Settings):
    """Without this the vLLM adapter is dead code and every deployment silently
    serves the simulator."""
    if settings.backend == "vllm":
        return VllmBackend(
            endpoints=settings.vllm_endpoints,
            model=settings.vllm_model,
            timeout_s=settings.vllm_timeout_s,
            max_retries=settings.vllm_max_retries,
        )
    return SimulatedBackend(settings.cluster.hardware)


def create_app(settings: Settings | None = None, backend: Any = None) -> FastAPI:
    settings = settings or load_settings()
    config = settings.cluster

    predictor = ToolDurationPredictor(config.predictor)
    policy = (
        AdaptiveTtlPolicy(predictor, config.policy)
        if settings.policy == "adaptive" else LruPolicy()
    )
    replicas = [Replica(i, config.hardware, policy) for i in range(config.replicas)]
    store = build_session_store(
        settings.redis_url, settings.session_ttl_s, settings.max_sessions
    )
    router = (
        SessionAffinityRouter(replicas, config.policy, store=store)
        if settings.router == "affinity" else RoundRobinRouter(replicas)
    )
    backend = backend if backend is not None else build_backend(settings)
    tokenizer = Tokenizer(settings.tokenizer)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if not settings.multi_pod_safe:
            log.warning(
                "Session map is process-local (no AGENTSERVE_REDIS_URL). Run exactly "
                "one gateway replica, or affinity degrades silently to round-robin."
            )
        log.info(
            "AgentServe up: backend=%s replicas=%d store=%s tokenizer=%s",
            backend.name, len(replicas), store.name, tokenizer.name,
        )
        yield
        closer = getattr(backend, "aclose", None)
        if closer is not None:
            await closer()
        store.close()

    app = FastAPI(title="AgentServe", version="1.0.0", lifespan=lifespan)
    app.state.settings = settings
    app.state.replicas = replicas
    app.state.router = router
    app.state.predictor = predictor
    app.state.store = store

    def require_api_key(authorization: str | None = Header(default=None)) -> None:
        """No-op unless AGENTSERVE_API_KEY is set."""
        if settings.api_key is None:
            return
        if authorization != f"Bearer {settings.api_key}":
            raise HTTPException(401, "invalid or missing bearer token")

    # ---- operational endpoints ------------------------------------------------

    @app.get("/health")
    def health() -> dict[str, Any]:
        """Liveness. Touches no dependency: a probe that fails on a Redis blip
        gets a healthy pod restarted."""
        return {
            "status": "ok",
            "version": "1.0.0",
            "backend": backend.name,
            "replicas": len(replicas),
            "policy": policy.name,
            "router": router.name,
            "session_store": store.name,
            "multi_pod_safe": settings.multi_pod_safe,
        }

    @app.get("/ready")
    async def ready() -> Response:
        """Readiness. Checks dependencies, because a pod that can't reach its
        workers should leave the Service."""
        checks: dict[str, Any] = {"session_store": store.size() >= 0}
        probe = getattr(backend, "aprobe", None)
        checks["workers"] = await probe() if probe is not None else True
        healthy = all(bool(v) for v in checks.values())
        return Response(
            content=json.dumps({"ready": healthy, "checks": checks}),
            status_code=200 if healthy else 503,
            media_type="application/json",
        )

    @app.get("/metrics")
    def metrics() -> Response:
        observe_cluster(replicas, predictor)
        return Response(content=render(), media_type="text/plain; version=0.0.4")

    @app.get("/debug/sessions")
    def debug_sessions() -> dict[str, Any]:
        return {
            "tracked_sessions": getattr(router, "tracked_sessions", 0),
            "router": router.name,
            "session_store": store.name,
            "affinity_hits": getattr(router, "affinity_hits", 0),
            "affinity_breaks": getattr(router, "affinity_breaks", 0),
            "replicas": [
                {
                    "id": r.id,
                    "gpu_sessions": len(r.gpu),
                    "dram_sessions": len(r.dram),
                    "gpu_utilization": round(r.gpu_utilization, 4),
                    "inflight": r.inflight,
                    "peak_inflight": r.peak_active,
                    "evictions": r.evictions,
                    "offloads": r.offloads,
                }
                for r in replicas
            ],
            "predicted_tool_gaps_ms": predictor.snapshot(),
        }

    # ---- inference ------------------------------------------------------------

    def schedule(session_id: str, tool_name: str | None, now_ms: float):
        """Routing and prediction, shared by both response paths."""
        for r in replicas:
            r.prune(now_ms)

        last = store.get_last_seen(session_id)
        if last is not None:
            # Clamped at zero: the simulated backend reports a logical
            # completion ahead of wall clock, and real clusters see clock skew.
            predictor.observe(session_id, tool_name, max(0.0, now_ms - last))

        before = getattr(router, "affinity_hits", 0)
        replica = router.select(session_id, now_ms)
        AFFINITY.labels(
            outcome="affinity"
            if getattr(router, "affinity_hits", 0) > before else "rebalanced"
        ).inc()
        return replica, predictor.predict_ms(session_id, tool_name)

    def record(session_id: str, result, now_ms: float) -> None:
        observe_result(result)
        CACHE_ACTIONS.labels(action=result.tier.value).inc()
        store.set_last_seen(session_id, now_ms + result.latency_ms)

    @app.post("/v1/chat/completions", dependencies=[Depends(require_api_key)])
    async def chat_completions(
        request: Request,
        x_session_id: str | None = Header(default=None),
        x_tool_name: str | None = Header(default=None),
    ):
        if not x_session_id:
            raise HTTPException(400, "X-Session-Id header is required")
        if len(x_session_id) > 256:
            raise HTTPException(400, "X-Session-Id must be at most 256 characters")

        try:
            body = await request.json()
        except Exception as exc:
            raise HTTPException(400, "request body must be valid JSON") from exc

        messages = body.get("messages")
        if not isinstance(messages, list) or not messages:
            raise HTTPException(400, "messages must be a non-empty array")
        max_tokens = int(body.get("max_tokens", 256))
        wants_stream = bool(body.get("stream", False))

        now_ms = time.time() * 1000.0
        prompt_tokens = tokenizer.count_messages(messages)
        replica, gap_ms = schedule(x_session_id, x_tool_name, now_ms)

        if wants_stream:
            return StreamingResponse(
                stream_response(
                    backend, replica, x_session_id, messages, prompt_tokens,
                    max_tokens, now_ms, gap_ms, body, record,
                ),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "X-Accel-Buffering": "no",  # nginx must not buffer SSE
                    "X-AgentServe-Replica": str(replica.id),
                },
            )

        try:
            # Holding the slot for the upstream call is what makes
            # replica.inflight real and the release valve live.
            with replica.slot():
                result = await execute(
                    backend, replica, x_session_id, messages, prompt_tokens,
                    max_tokens, now_ms, gap_ms,
                )
        except Exception as exc:
            log.exception("upstream failure for session %s", x_session_id)
            raise HTTPException(502, f"upstream worker error: {exc}") from exc

        record(x_session_id, result, now_ms)
        completion = int(getattr(result, "completion_tokens", max_tokens))

        return {
            "id": f"agentserve-{x_session_id}-{int(now_ms)}",
            "object": "chat.completion",
            "created": int(now_ms / 1000),
            "model": body.get("model", settings.vllm_model),
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": getattr(result, "text", "")},
                "finish_reason": "stop",
            }],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion,
                "total_tokens": prompt_tokens + completion,
                "prompt_tokens_details": {"cached_tokens": result.reused_tokens},
            },
            "agentserve": {
                "replica": result.replica_id,
                "cache_tier": result.tier.value,
                "reused_tokens": result.reused_tokens,
                "computed_tokens": result.computed_tokens,
                "ttft_ms": round(result.ttft_ms, 2),
                "predicted_gap_ms": round(gap_ms, 1),
            },
        }

    return app


async def execute(backend, replica, session_id, messages, prompt_tokens,
                  max_tokens, now_ms, gap_ms):
    """Call the backend, handling both sync and async signatures."""
    kwargs = dict(
        replica=replica,
        session_id=session_id,
        prompt_tokens=prompt_tokens,
        output_tokens=max_tokens,
        now_ms=now_ms,
        predicted_return_ms=now_ms + gap_ms,
    )
    runner = getattr(backend, "aexecute", None)
    if runner is not None:
        return await runner(messages=messages, **kwargs)
    return backend.execute(**kwargs)


async def stream_response(backend, replica, session_id, messages, prompt_tokens,
                          max_tokens, now_ms, gap_ms, body, record):
    """SSE in OpenAI's chat.completion.chunk format.

    Not cosmetic: TTFT is what this project optimizes and a client can't observe
    it through a non-streaming response.
    """
    stream_id = f"agentserve-{session_id}-{int(now_ms)}"
    model = body.get("model", "agentserve")

    def envelope(delta: dict[str, Any], finish: str | None = None) -> str:
        payload = {
            "id": stream_id,
            "object": "chat.completion.chunk",
            "created": int(now_ms / 1000),
            "model": model,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
        }
        return f"data: {json.dumps(payload)}\n\n"

    astream = getattr(backend, "astream", None)
    result = None
    try:
        # Held for the whole stream, not just the first byte. A session
        # streaming 2,000 tokens occupies the replica the entire time.
        with replica.slot():
            if astream is not None:
                first = True
                async for text, maybe_result in astream(
                    replica=replica, session_id=session_id, messages=messages,
                    prompt_tokens=prompt_tokens, output_tokens=max_tokens,
                    now_ms=now_ms, predicted_return_ms=now_ms + gap_ms,
                ):
                    if maybe_result is not None:
                        result = maybe_result
                        break
                    yield envelope({"role": "assistant", "content": text} if first
                                   else {"content": text})
                    first = False
            else:
                result = backend.execute(
                    replica=replica, session_id=session_id,
                    prompt_tokens=prompt_tokens, output_tokens=max_tokens,
                    now_ms=now_ms, predicted_return_ms=now_ms + gap_ms,
                )
                yield envelope(
                    {"role": "assistant", "content": getattr(result, "text", "")}
                )

        if result is not None:
            record(session_id, result, now_ms)
            yield "data: " + json.dumps({
                "id": stream_id,
                "object": "chat.completion.chunk",
                "agentserve": {
                    "replica": result.replica_id,
                    "cache_tier": result.tier.value,
                    "reused_tokens": result.reused_tokens,
                    "ttft_ms": round(result.ttft_ms, 2),
                },
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            }) + "\n\n"
        yield "data: [DONE]\n\n"
    except Exception as exc:
        # Headers are already sent, so an upstream failure mid-stream can't
        # become a status code. Report it inside the stream instead.
        log.exception("stream failed for session %s", session_id)
        yield "data: " + json.dumps(
            {"error": {"message": str(exc), "type": "upstream_error"}}
        ) + "\n\n"
        yield "data: [DONE]\n\n"


app = create_app()
