"""Deterministic cost model of a vLLM worker.

No randomness, so the same trace and policy give the same numbers every run.

Three cases: a GPU hit prefills only the appended tokens, a DRAM hit pays a PCIe
restore first (about 16x cheaper per token than recomputing), and a miss prefills
the whole prompt.
"""
from __future__ import annotations

from ..config import HardwareProfile
from ..models import CacheTier, RequestResult
from ..replica import Replica


class SimulatedBackend:
    name = "simulated"

    def __init__(self, hardware: HardwareProfile | None = None) -> None:
        self.hw = hardware or HardwareProfile()

    def execute(
        self,
        replica: Replica,
        session_id: str,
        prompt_tokens: int,
        output_tokens: int,
        now_ms: float,
        predicted_return_ms: float,
    ) -> RequestResult:
        tier, cached_tokens = replica.lookup(session_id)

        # A cached prefix only helps up to the current prompt length.
        reusable = min(cached_tokens, prompt_tokens) if tier is not CacheTier.MISS else 0
        computed = prompt_tokens - reusable

        restore_ms = 0.0
        if tier is CacheTier.DRAM:
            restore_ms = reusable * self.hw.dram_restore_ms_per_token
            replica.promote_from_dram(session_id)

        prefill_ms = computed * self.hw.prefill_ms_per_token
        decode_ms = output_tokens * self.hw.decode_ms_per_token

        queue_ms = max(0.0, replica.free_at_ms - now_ms)
        service_ms = restore_ms + prefill_ms + decode_ms
        replica.free_at_ms = max(replica.free_at_ms, now_ms) + service_ms

        replica.store(
            session_id=session_id,
            tokens=prompt_tokens + output_tokens,
            now_ms=now_ms + queue_ms + service_ms,
            predicted_return_ms=predicted_return_ms,
        )

        return RequestResult(
            session_id=session_id,
            replica_id=replica.id,
            tier=tier,
            reused_tokens=reusable,
            computed_tokens=computed,
            prefill_ms=prefill_ms,
            decode_ms=decode_ms,
            queue_ms=queue_ms,
            restore_ms=restore_ms,
        )
