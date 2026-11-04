"""Deterministic, factual summaries for completed experiment artifacts."""

from __future__ import annotations

from collections.abc import Mapping, Sequence


def _strategy_name(row: Mapping[str, object]) -> str:
    return str(row.get("strategy", "Unknown strategy"))


def robustness_insights(result: object) -> list[str]:
    """Summarize a validated seed sweep without making causal claims."""
    if not isinstance(result, Mapping):
        return []
    summary = result.get("summary")
    if not isinstance(summary, Sequence) or isinstance(summary, (str, bytes)) or not summary:
        return []
    rows = [row for row in summary if isinstance(row, Mapping)]
    if not rows:
        return []

    highest = max(rows, key=lambda row: float(row.get("mean_payoff", 0.0)))
    stable = min(rows, key=lambda row: float(row.get("payoff_std", 0.0)))
    best_rank = min(rows, key=lambda row: (float(row.get("mean_rank", float("inf"))), -float(row.get("mean_payoff", 0.0))))
    widest = max(rows, key=lambda row: int(row.get("worst_rank", 0)) - int(row.get("best_rank", 0)))
    seed_count = max(int(row.get("seeds", 0)) for row in rows)

    return [
        f"{_strategy_name(highest)} had the highest mean payoff at {float(highest.get('mean_payoff', 0.0)):.3f} points per round across {seed_count} seeds.",
        f"{_strategy_name(stable)} had the lowest seed-to-seed payoff variation with a standard deviation of {float(stable.get('payoff_std', 0.0)):.3f}.",
        f"{_strategy_name(best_rank)} had the best average rank at {float(best_rank.get('mean_rank', 0.0)):.2f}.",
        f"{_strategy_name(widest)} had the widest observed rank range, from {int(widest.get('best_rank', 0))} to {int(widest.get('worst_rank', 0))}.",
    ]


def evolution_insights(result: object) -> list[str]:
    """Summarize a validated evolution history using descriptive statements."""
    if not isinstance(result, Mapping):
        return []
    history = result.get("history")
    if not isinstance(history, Sequence) or isinstance(history, (str, bytes)) or not history:
        return []
    rows = [row for row in history if isinstance(row, Mapping)]
    if not rows:
        return []

    first = rows[0]
    final = rows[-1]
    first_shares = first.get("shares")
    final_shares = final.get("shares")
    if not isinstance(first_shares, Mapping) or not isinstance(final_shares, Mapping) or not final_shares:
        return []

    initial_leader = max(first_shares, key=lambda name: float(first_shares[name]))
    final_leader = max(final_shares, key=lambda name: float(final_shares[name]))
    changes = {
        str(name): float(final_shares[name]) - float(first_shares.get(name, 0.0))
        for name in final_shares
    }
    largest_gain = max(changes, key=changes.get)
    largest_loss = min(changes, key=changes.get)
    generation = int(final.get("generation", len(rows) - 1))

    return [
        f"{initial_leader} was the largest population at generation 0; {final_leader} was largest at generation {generation} with {float(final_shares[final_leader]):.1%} of the population.",
        f"{largest_gain} had the largest population-share increase at {changes[largest_gain]:+.1%}.",
        f"{largest_loss} had the largest population-share decrease at {changes[largest_loss]:+.1%}.",
        f"The final population had an average payoff of {float(final.get('average_payoff', 0.0)):.3f}, cooperation of {float(final.get('cooperation_rate', 0.0)):.1%}, and diversity of {float(final.get('diversity', 0.0)):.3f}.",
    ]
