"""Prompt length measurement.

Cache accounting is in tokens, so a bad count corrupts every capacity decision.
The chars/4 heuristic underestimates code and badly misjudges non-Latin scripts,
which makes a replica think it has headroom it doesn't and over-admit sessions.

Uses tiktoken when available, heuristic as fallback.
"""
from __future__ import annotations

from typing import Any

try:
    import tiktoken
except ImportError:  # pragma: no cover
    tiktoken = None


class Tokenizer:
    def __init__(self, encoding: str = "cl100k_base") -> None:
        self.name = "heuristic"
        self._encoder = None
        if tiktoken is not None:
            try:
                self._encoder = tiktoken.get_encoding(encoding)
                self.name = encoding
            except Exception:  # pragma: no cover - unknown encoding name
                self._encoder = None

    def count_text(self, text: str) -> int:
        if self._encoder is not None:
            return len(self._encoder.encode(text, disallowed_special=()))
        return max(1, len(text) // 4)

    def count_messages(self, messages: list[dict[str, Any]]) -> int:
        """Token count for a chat prompt, including per-message framing overhead.

        Chat templates wrap each message in role markers; four tokens per
        message is the standard approximation and keeps the estimate on the
        conservative side rather than under-counting.
        """
        total = 0
        for message in messages:
            total += 4
            for value in message.values():
                if isinstance(value, str):
                    total += self.count_text(value)
                elif isinstance(value, list):
                    for part in value:
                        if isinstance(part, dict) and isinstance(part.get("text"), str):
                            total += self.count_text(part["text"])
        return max(1, total)
