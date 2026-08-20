"""Estimates how long an agent will be away between calls.

Nothing in the request says whether the session returns in 300ms or 40s, so we
learn it from history keyed on tool name, falling back to a per-session average
and then a global prior.

EWMA rather than a plain mean because tool latencies drift within a session as
test suites grow and repos warm up.
"""
from __future__ import annotations

from collections import defaultdict

from .config import PredictorConfig


class ToolDurationPredictor:
    def __init__(self, config: PredictorConfig | None = None) -> None:
        self.config = config or PredictorConfig()
        self._tool_ewma: dict[str, float] = {}
        self._tool_counts: dict[str, int] = defaultdict(int)
        self._session_ewma: dict[str, float] = {}
        self._global_ewma: float = self.config.prior_ms
        self._observations: int = 0

    def observe(self, session_id: str, tool_name: str | None, duration_ms: float) -> None:
        """Record an actual measured gap. Called when a session resumes."""
        if duration_ms < 0:
            raise ValueError("duration_ms must be non-negative")
        a = self.config.ewma_alpha
        if tool_name:
            prev = self._tool_ewma.get(tool_name, duration_ms)
            self._tool_ewma[tool_name] = a * duration_ms + (1 - a) * prev
            self._tool_counts[tool_name] += 1
        prev_s = self._session_ewma.get(session_id, duration_ms)
        self._session_ewma[session_id] = a * duration_ms + (1 - a) * prev_s
        self._global_ewma = a * duration_ms + (1 - a) * self._global_ewma
        self._observations += 1

    def predict_ms(self, session_id: str, tool_name: str | None) -> float:
        """Estimated idle gap before this session issues its next call."""
        if tool_name and self._tool_counts[tool_name] >= self.config.min_observations:
            return self._tool_ewma[tool_name]
        if session_id in self._session_ewma:
            return self._session_ewma[session_id]
        if self._observations:
            return self._global_ewma
        return self.config.prior_ms

    def snapshot(self) -> dict[str, float]:
        """Exposed on /debug/predictor so the estimates are inspectable."""
        return dict(sorted(self._tool_ewma.items()))
