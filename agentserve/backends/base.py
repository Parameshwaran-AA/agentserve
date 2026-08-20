"""Backend interface.

Router, policy, predictor and metrics are all backend agnostic, so the same
scheduler runs against the simulator on a laptop and real vLLM workers on a
cluster without a line of policy code changing.
"""
from __future__ import annotations

from typing import Protocol

from ..models import RequestResult
from ..replica import Replica


class Backend(Protocol):
    name: str

    def execute(
        self,
        replica: Replica,
        session_id: str,
        prompt_tokens: int,
        output_tokens: int,
        now_ms: float,
        predicted_return_ms: float,
    ) -> RequestResult:
        """Serve one call and update the replica's cache state."""
        ...
