"""Shared workload estimates for tournament-backed experiments."""

from __future__ import annotations


def tournament_pairings(strategy_count: int, *, include_self_play: bool = False) -> int:
    """Return the number of unique pairings in one round-robin repetition."""
    count = int(strategy_count)
    if count < 2:
        raise ValueError("strategy_count must be at least 2")
    return count * (count + 1) // 2 if include_self_play else count * (count - 1) // 2


def tournament_workload(
    strategy_count: int,
    rounds_per_match: int,
    repetitions: int,
    *,
    include_self_play: bool = False,
) -> int:
    """Return the exact number of simulated rounds for a tournament."""
    rounds = int(rounds_per_match)
    repeats = int(repetitions)
    if rounds < 1:
        raise ValueError("rounds_per_match must be positive")
    if repeats < 1:
        raise ValueError("repetitions must be positive")
    return tournament_pairings(strategy_count, include_self_play=include_self_play) * rounds * repeats
