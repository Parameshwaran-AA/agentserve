"""Synthetic agent traces.

Three properties are reproduced because the result depends on them. Prompts grow
monotonically, so turn N+1 contains turn N verbatim. Tool latency is bimodal and
heavy tailed, with greps in the hundreds of milliseconds and test suites in the
tens of seconds; a single mean would erase the effect entirely. And sessions are
long, around 25 calls per task.

Seeded, so runs are reproducible.
"""
from __future__ import annotations

import random

from agentserve.models import SessionTrace, Turn

# name -> (mean_ms, sigma for lognormal, selection weight)
TOOL_PROFILE: dict[str, tuple[float, float, float]] = {
    "read_file": (280.0, 0.45, 0.32),
    "grep": (420.0, 0.50, 0.18),
    "edit_file": (650.0, 0.40, 0.16),
    "git_diff": (900.0, 0.55, 0.08),
    "run_linter": (6_500.0, 0.60, 0.10),
    "run_tests": (44_000.0, 0.70, 0.13),
    "build": (95_000.0, 0.65, 0.03),
}


def _sample_tool(rng: random.Random) -> tuple[str, float]:
    names = list(TOOL_PROFILE)
    weights = [TOOL_PROFILE[n][2] for n in names]
    name = rng.choices(names, weights=weights, k=1)[0]
    mean, sigma, _ = TOOL_PROFILE[name]
    mu = math_log(mean) - (sigma ** 2) / 2
    return name, max(50.0, rng.lognormvariate(mu, sigma))


def math_log(x: float) -> float:
    import math

    return math.log(x)


def generate_trace(
    sessions: int = 240,
    seed: int = 7,
    mean_turns: int = 25,
    base_prompt_tokens: int = 4_200,
    arrival_window_ms: float = 240_000.0,
) -> list[SessionTrace]:
    rng = random.Random(seed)
    traces: list[SessionTrace] = []

    for i in range(sessions):
        turns_count = max(4, int(rng.gauss(mean_turns, 6)))
        prompt = base_prompt_tokens + rng.randint(-900, 2_600)
        turns: list[Turn] = []

        for t in range(turns_count):
            output = rng.randint(90, 420)
            last = t == turns_count - 1
            if last:
                tool_name, tool_ms = None, 0.0
            else:
                tool_name, tool_ms = _sample_tool(rng)
            turns.append(
                Turn(
                    prompt_tokens=prompt,
                    output_tokens=output,
                    tool_name=tool_name,
                    tool_duration_ms=tool_ms,
                )
            )
            # Next prompt carries this prompt plus the model's output plus the
            # tool result appended to the transcript.
            tool_result_tokens = rng.randint(120, 1_400) if tool_name else 0
            prompt = prompt + output + tool_result_tokens

        traces.append(
            SessionTrace(
                session_id=f"sess-{i:04d}",
                arrival_ms=rng.uniform(0.0, arrival_window_ms),
                turns=turns,
            )
        )

    traces.sort(key=lambda s: s.arrival_ms)
    return traces


def trace_summary(traces: list[SessionTrace]) -> dict[str, float]:
    turns = [len(t.turns) for t in traces]
    prompts = [turn.prompt_tokens for t in traces for turn in t.turns]
    gaps = [turn.tool_duration_ms for t in traces for turn in t.turns if turn.tool_name]
    gaps.sort()
    return {
        "sessions": len(traces),
        "total_calls": sum(turns),
        "mean_turns": sum(turns) / len(turns),
        "max_prompt_tokens": max(prompts),
        "mean_prompt_tokens": sum(prompts) / len(prompts),
        "median_tool_gap_ms": gaps[len(gaps) // 2],
        "p90_tool_gap_ms": gaps[int(len(gaps) * 0.9)],
    }
