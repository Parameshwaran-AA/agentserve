"""Adapter for real vLLM workers. Selected with AGENTSERVE_BACKEND=vllm.

Depends on two vLLM flags. --enable-prefix-caching provides the cache the router
steers toward; vLLM can't help if the request lands on a worker that never saw
the conversation. --swap-space backs the DRAM tier; vLLM will swap but won't
decide when on agent-aware grounds, which is what AdaptiveTtlPolicy is for.

Reuse comes from prompt_tokens_details.cached_tokens on the response.
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import AsyncIterator
from typing import Any

import httpx

from ..models import CacheTier, RequestResult
from ..replica import Replica

log = logging.getLogger("agentserve.vllm")


class VllmBackend:
    name = "vllm"

    def __init__(
        self,
        endpoints: list[str],
        model: str,
        timeout_s: float = 300.0,
        max_retries: int = 2,
    ) -> None:
        if not endpoints:
            raise ValueError("VllmBackend requires at least one endpoint")
        self.endpoints = [e.rstrip("/") for e in endpoints]
        self.model = model
        self.max_retries = max_retries
        self.failovers = 0
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(timeout_s, connect=5.0),
            limits=httpx.Limits(max_connections=256, max_keepalive_connections=64),
        )

    #, helpers ---------------------------------------------------------------

    def _url(self, replica_id: int) -> str:
        return f"{self.endpoints[replica_id % len(self.endpoints)]}/v1/chat/completions"

    def _fallbacks(self, replica: Replica) -> list[int]:
        """Scheduled worker first, then the rest.

        Failover costs the cache hit, since another worker doesn't have this
        prefix. Worth it: a redundant prefill beats a failed agent task, and a
        worker that just refused a connection won't serve the cached copy anyway.
        """
        order = [replica.id]
        order += [i for i in range(len(self.endpoints)) if i != replica.id]
        return order[: 1 + self.max_retries]

    def _payload(self, messages, max_tokens: int, stream: bool) -> dict[str, Any]:
        return {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "stream": stream,
            **({"stream_options": {"include_usage": True}} if stream else {}),
        }

    def _result(self, replica: Replica, session_id: str, usage: dict[str, Any],
                fallback_prompt: int, fallback_completion: int,
                elapsed_ms: float, ttft_ms: float,
                now_ms: float, predicted_return_ms: float, text: str) -> RequestResult:
        details = usage.get("prompt_tokens_details") or {}
        cached = int(details.get("cached_tokens", 0) or 0)
        prompt = int(usage.get("prompt_tokens", fallback_prompt) or fallback_prompt)
        completion = int(
            usage.get("completion_tokens", fallback_completion) or fallback_completion
        )

        tier = CacheTier.GPU if cached > 0 else CacheTier.MISS
        replica.store(
            session_id=session_id,
            tokens=prompt + completion,
            now_ms=now_ms + elapsed_ms,
            predicted_return_ms=predicted_return_ms,
        )

        result = RequestResult(
            session_id=session_id,
            replica_id=replica.id,
            tier=tier,
            reused_tokens=cached,
            computed_tokens=max(0, prompt - cached),
            prefill_ms=ttft_ms,
            decode_ms=max(0.0, elapsed_ms - ttft_ms),
            queue_ms=0.0,
        )
        result.text = text
        result.completion_tokens = completion
        return result

    #, non-streaming ---------------------------------------------------------

    async def aexecute(
        self,
        replica: Replica,
        session_id: str,
        prompt_tokens: int,
        output_tokens: int,
        now_ms: float,
        predicted_return_ms: float,
        messages: list[dict[str, Any]] | None = None,
    ) -> RequestResult:
        payload = self._payload(messages or [], output_tokens, stream=False)
        last_error: Exception | None = None

        for attempt, target in enumerate(self._fallbacks(replica)):
            started = time.perf_counter()
            try:
                response = await self._client.post(self._url(target), json=payload)
                response.raise_for_status()
            except Exception as exc:
                last_error = exc
                log.warning("worker %s failed (%s); trying next", target, exc)
                await asyncio.sleep(min(0.25 * (2 ** attempt), 2.0))
                continue

            # Counted at success, not failure. The number that matters is
            # requests served by a worker without the prefix, each of which
            # costs a redundant prefill.
            if attempt:
                self.failovers += 1
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            body = response.json()
            text = (body.get("choices") or [{}])[0].get("message", {}).get("content", "")
            return self._result(
                replica, session_id, body.get("usage", {}) or {},
                prompt_tokens, output_tokens, elapsed_ms, elapsed_ms,
                now_ms, predicted_return_ms, text,
            )

        raise RuntimeError(f"all vLLM workers failed: {last_error}")

    #, streaming -------------------------------------------------------------

    async def astream(
        self,
        replica: Replica,
        session_id: str,
        messages: list[dict[str, Any]],
        prompt_tokens: int,
        output_tokens: int,
        now_ms: float,
        predicted_return_ms: float,
    ) -> AsyncIterator[tuple[str, RequestResult | None]]:
        """Yield (chunk, None) per token, then ("", RequestResult) at the end.

        No mid-stream failover: once bytes are on the wire, retrying would replay
        tokens the client already saw. Failover covers connection setup only.
        """
        payload = self._payload(messages, output_tokens, stream=True)
        started = time.perf_counter()
        ttft_ms = 0.0
        usage: dict[str, Any] = {}
        collected: list[str] = []

        async with self._client.stream("POST", self._url(replica.id), json=payload) as r:
            r.raise_for_status()
            async for line in r.aiter_lines():
                if not line.startswith("data: "):
                    continue
                blob = line[6:].strip()
                if blob == "[DONE]":
                    break
                try:
                    chunk = json.loads(blob)
                except json.JSONDecodeError:
                    continue

                if chunk.get("usage"):
                    usage = chunk["usage"]
                choices = chunk.get("choices") or []
                if not choices:
                    continue
                delta = (choices[0].get("delta") or {}).get("content")
                if not delta:
                    continue
                if not ttft_ms:
                    ttft_ms = (time.perf_counter() - started) * 1000.0
                collected.append(delta)
                yield delta, None

        elapsed_ms = (time.perf_counter() - started) * 1000.0
        yield "", self._result(
            replica, session_id, usage, prompt_tokens, output_tokens,
            elapsed_ms, ttft_ms or elapsed_ms, now_ms, predicted_return_ms,
            "".join(collected),
        )

    #, lifecycle -------------------------------------------------------------

    async def aprobe(self) -> bool:
        """At least one worker must answer. Losing some workers is degraded,
        not down."""
        async def alive(endpoint: str) -> bool:
            try:
                r = await self._client.get(f"{endpoint}/health", timeout=2.0)
                return r.status_code == 200
            except Exception:
                return False

        results = await asyncio.gather(*(alive(e) for e in self.endpoints))
        return any(results)

    async def aclose(self) -> None:
        await self._client.aclose()
