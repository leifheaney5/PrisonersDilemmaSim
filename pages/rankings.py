"""Deterministic rankings and interactive figures for experiment results."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math

import plotly.graph_objects as go


ROBUSTNESS_METRICS = {
    "mean_payoff": {"label": "Mean payoff", "descending": True, "format": ".3f", "kind": "performance"},
    "payoff_std": {"label": "Payoff variation", "descending": False, "format": ".3f", "kind": "stability"},
    "mean_rank": {"label": "Average rank", "descending": False, "format": ".2f", "kind": "performance"},
    "win_rate": {"label": "First-place rate", "descending": True, "format": ".1%", "kind": "performance"},
    "mean_cooperation": {"label": "Cooperation rate", "descending": True, "format": ".1%", "kind": "behavior"},
}

RANK_TOLERANCE = 1e-12


def _competition_ranks(values: Sequence[float], *, tolerance: float = RANK_TOLERANCE) -> list[int]:
    """Assign 1, 1, 3 competition ranks to values already sorted best first."""
    ranks: list[int] = []
    previous: float | None = None
    current_rank = 0
    for index, value in enumerate(values, start=1):
        if previous is None or not math.isclose(value, previous, rel_tol=tolerance, abs_tol=tolerance):
            current_rank = index
        ranks.append(current_rank)
        previous = value
    return ranks


def _summary_rows(result: object) -> list[Mapping[str, object]]:
    if not isinstance(result, Mapping):
        return []
    summary = result.get("summary")
    if not isinstance(summary, Sequence) or isinstance(summary, (str, bytes)):
        return []
    return [row for row in summary if isinstance(row, Mapping) and row.get("strategy")]


def _selected_strategy(click_data: object) -> str | None:
    if not isinstance(click_data, Mapping):
        return None
    points = click_data.get("points")
    if not isinstance(points, Sequence) or not points or not isinstance(points[0], Mapping):
        return None
    custom = points[0].get("customdata")
    if isinstance(custom, Sequence) and not isinstance(custom, (str, bytes)):
        return str(custom[0]) if custom else None
    return str(custom) if custom is not None else None


def robustness_ranking_view(
    result: object,
    metric: str = "mean_payoff",
    click_data: object = None,
) -> tuple[go.Figure, str]:
    """Build a sortable robustness ranking and its deterministic narrative."""
    definition = ROBUSTNESS_METRICS.get(metric, ROBUSTNESS_METRICS["mean_payoff"])
    metric = metric if metric in ROBUSTNESS_METRICS else "mean_payoff"
    rows = _summary_rows(result)
    if not rows:
        return go.Figure().update_layout(title="Run or open a robustness result to rank strategies"), ""

    ranked = sorted(
        rows,
        key=lambda row: (
            -float(row.get(metric, 0.0)) if definition["descending"] else float(row.get(metric, 0.0)),
            str(row["strategy"]),
        ),
    )
    names = [str(row["strategy"]) for row in ranked]
    values = [float(row.get(metric, 0.0)) for row in ranked]
    ranks = _competition_ranks(values)
    selected = _selected_strategy(click_data)
    hover = [
        (
            f"Mean payoff: {float(row.get('mean_payoff', 0.0)):.3f}<br>"
            f"Average rank: {float(row.get('mean_rank', 0.0)):.2f}<br>"
            f"Payoff variation: {float(row.get('payoff_std', 0.0)):.3f}<br>"
            f"First-place rate: {float(row.get('win_rate', 0.0)):.1%}"
        )
        for row in ranked
    ]
    figure = go.Figure(
        go.Bar(
            x=values,
            y=names,
            orientation="h",
            customdata=[[name] for name in names],
            text=[f"#{rank}" for rank in ranks],
            textposition="outside",
            hovertext=hover,
            hovertemplate="<b>%{y}</b><br>%{hovertext}<extra></extra>",
            marker={
                "color": values,
                "colorscale": "Viridis",
                "showscale": False,
                "line": {
                    "color": ["#0f172a" if name == selected else "rgba(15, 23, 42, 0.2)" for name in names],
                    "width": [4 if name == selected else 1 for name in names],
                },
            },
        )
    )
    figure.update_yaxes(autorange="reversed", title=None)
    figure.update_xaxes(title=str(definition["label"]))
    figure.update_layout(
        title=f"Strategy ranking by {str(definition['label']).lower()}",
        height=max(360, 46 * len(names) + 120),
        margin=dict(l=10, r=55, t=60, b=45),
        clickmode="event+select",
    )

    formatter = str(definition["format"])
    ranking = "; ".join(
        f"{rank}. {name} ({format(value, formatter)})"
        for rank, name, value in zip(ranks, names, values, strict=True)
    )
    direction = "highest to lowest" if definition["descending"] else "lowest to highest"
    statement = f"Deterministic {definition['kind']} ranking by {str(definition['label']).lower()} ({direction}): {ranking}."

    if selected in names:
        index = names.index(selected)
        row = ranked[index]
        statement += (
            f" Selected: {selected} ranks #{ranks[index]}; mean payoff {float(row.get('mean_payoff', 0.0)):.3f}, "
            f"average rank {float(row.get('mean_rank', 0.0)):.2f}, and first-place rate {float(row.get('win_rate', 0.0)):.1%}."
        )
    return figure, statement


def evolution_generation_ranking_view(result: object, generation: int | None) -> tuple[go.Figure, str]:
    """Rank population shares at one stored generation without rerunning evolution."""
    history = result.get("history") if isinstance(result, Mapping) else None
    if not isinstance(history, Sequence) or isinstance(history, (str, bytes)) or not history:
        return go.Figure().update_layout(title="Run or open an evolution result to rank populations"), ""
    index = max(0, min(int(generation or 0), len(history) - 1))
    row = history[index]
    shares = row.get("shares") if isinstance(row, Mapping) else None
    if not isinstance(shares, Mapping) or not shares:
        return go.Figure().update_layout(title="No population shares are available"), ""

    ranked = sorted(((str(name), float(share)) for name, share in shares.items()), key=lambda item: (-item[1], item[0]))
    names = [name for name, _share in ranked]
    values = [share for _name, share in ranked]
    ranks = _competition_ranks(values)
    figure = go.Figure(
        go.Bar(
            x=values,
            y=names,
            orientation="h",
            customdata=[[name] for name in names],
            text=[f"#{rank} · {share:.1%}" for rank, share in zip(ranks, values, strict=True)],
            textposition="outside",
            marker={"color": values, "colorscale": "Tealgrn", "showscale": False},
            hovertemplate="<b>%{y}</b><br>Population share: %{x:.1%}<extra></extra>",
        )
    )
    figure.update_yaxes(autorange="reversed", title=None)
    figure.update_xaxes(range=[0, 1], tickformat=".0%", title="Population share")
    figure.update_layout(
        title=f"Population ranking at generation {int(row.get('generation', index))}",
        height=max(340, 46 * len(names) + 110),
        margin=dict(l=10, r=75, t=60, b=45),
    )
    ranking = "; ".join(
        f"{rank}. {name} ({share:.1%})"
        for rank, (name, share) in zip(ranks, ranked, strict=True)
    )
    return figure, f"Deterministic population ranking at generation {int(row.get('generation', index))}: {ranking}."


def tournament_ranking_statement(state: object, metric: str = "points_per_round") -> str:
    """Describe the live tournament order using the same normalized metrics as its chart."""
    if not isinstance(state, Mapping):
        return ""
    names = [str(name) for name in state.get("strategy_names", [])]
    rounds = state.get("rounds_played", {})
    totals = state.get("totals", {})
    wins = state.get("match_wins", {})
    losses = state.get("match_losses", {})
    ties = state.get("match_ties", {})
    cooperations = state.get("cooperate", {})
    if not names or not isinstance(rounds, Mapping) or sum(int(rounds.get(name, 0)) for name in names) == 0:
        return ""

    definitions = {
        "points_per_round": ("points per round", ".3f"),
        "total_points": ("total points", ".0f"),
        "win_rate": ("win rate", ".1%"),
        "cooperation_rate": ("cooperation rate", ".1%"),
    }
    metric = metric if metric in definitions else "points_per_round"

    def value(name: str) -> float:
        played = int(rounds.get(name, 0))
        if metric == "total_points":
            return float(totals.get(name, 0))
        if metric == "points_per_round":
            return float(totals.get(name, 0)) / played if played else 0.0
        if metric == "cooperation_rate":
            return float(cooperations.get(name, 0)) / played if played else 0.0
        matches = int(wins.get(name, 0)) + int(losses.get(name, 0)) + int(ties.get(name, 0))
        return float(wins.get(name, 0)) / matches if matches else 0.0

    ranked = sorted(((name, value(name)) for name in names), key=lambda item: (-item[1], item[0]))
    ranks = _competition_ranks([score for _name, score in ranked])
    label, formatter = definitions[metric]
    order = "; ".join(
        f"{rank}. {name} ({format(score, formatter)})"
        for rank, (name, score) in zip(ranks, ranked, strict=True)
    )
    status = "Final" if bool(state.get("done")) else "Provisional"
    matches_done = int(state.get("matches_done", 0))
    total_matches = int(state.get("total_matches", 0))
    progress = f" after {matches_done} of {total_matches} matches" if total_matches else ""
    return f"{status} deterministic ranking by {label}{progress}: {order}."
