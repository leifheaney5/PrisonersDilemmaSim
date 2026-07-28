"""Dash-independent presentation helpers for experiment result artifacts."""

from __future__ import annotations

from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def empty_experiment_figure(title: str) -> go.Figure:
    """Return a consistent empty-state figure for an experiment panel."""
    return px.scatter(title=title)


def robustness_result_view(result: dict[str, Any] | None) -> tuple[go.Figure, go.Figure, list[dict], list[dict]]:
    """Build aggregate robustness figures and table data from a result artifact."""
    if not isinstance(result, dict) or not result.get("seed_results"):
        empty = empty_experiment_figure("Run or open a robustness result")
        return empty, empty, [], []

    seed_frame = pd.DataFrame(result["seed_results"])
    summary_frame = pd.DataFrame(result["summary"])
    payoff_figure = px.box(
        seed_frame,
        x="strategy",
        y="points_per_round",
        color="strategy",
        points="all",
        title="Payoff distribution across independent seeds",
    )
    payoff_figure.update_yaxes(range=[0, 5], title="Points per round")
    payoff_figure.update_layout(showlegend=False, height=430, margin=dict(l=10, r=10, t=55, b=80))

    rank_matrix = seed_frame.pivot(index="seed", columns="strategy", values="rank")
    rank_figure = px.imshow(
        rank_matrix,
        text_auto=True,
        color_continuous_scale="RdYlGn_r",
        title="Rank by seed",
        labels={"x": "Strategy", "y": "Seed", "color": "Rank"},
        aspect="auto",
    )
    rank_figure.update_layout(height=430, margin=dict(l=10, r=10, t=55, b=80))

    table_rows = summary_frame.round(4).to_dict("records")
    columns = [{"name": column.replace("_", " ").title(), "id": column} for column in summary_frame.columns]
    return payoff_figure, rank_figure, columns, table_rows


def robustness_seed_view(result: dict[str, Any] | None, seed: int | None) -> tuple[go.Figure, go.Figure, list[dict], list[dict]]:
    """Build the charts and table for one seed without rerunning its tournament."""
    rows = [
        row
        for row in list((result or {}).get("seed_results", []))
        if int(row["seed"]) == int(seed or 0)
    ]
    if not rows:
        empty = empty_experiment_figure("Select a seed to inspect")
        return empty, empty, [], []

    frame = pd.DataFrame(rows).sort_values(["rank", "strategy"])
    payoff = px.bar(
        frame,
        x="points_per_round",
        y="strategy",
        color="rank",
        orientation="h",
        title=f"Points per round, seed {int(seed)}",
        range_x=[0, 5],
    )
    cooperation = px.bar(
        frame,
        x="cooperation_rate",
        y="strategy",
        color="cooperation_rate",
        orientation="h",
        title=f"Cooperation rate, seed {int(seed)}",
        range_x=[0, 1],
        color_continuous_scale="RdYlGn",
    )
    cooperation.update_xaxes(tickformat=".0%")
    table_rows = frame.round(4).to_dict("records")
    columns = [{"name": column.replace("_", " ").title(), "id": column} for column in frame.columns]
    return payoff, cooperation, columns, table_rows


def evolution_result_view(history: list[dict[str, Any]], strategy_names: list[str]) -> tuple[go.Figure, go.Figure, list[dict], list[dict]]:
    """Build evolution trajectory figures and the final population table."""
    if not history:
        return go.Figure(), go.Figure(), [], []

    share_frame = pd.DataFrame(
        [
            {"generation": row["generation"], "strategy": name, "share": row["shares"][name]}
            for row in history
            for name in strategy_names
        ]
    )
    population_figure = px.area(
        share_frame,
        x="generation",
        y="share",
        color="strategy",
        groupnorm=None,
        title="Population share by generation",
    )
    population_figure.update_yaxes(range=[0, 1], tickformat=".0%", title="Population share")
    population_figure.update_layout(height=470, margin=dict(l=10, r=10, t=55, b=35), hovermode="x unified")

    outcome_frame = pd.DataFrame(
        {
            "generation": [row["generation"] for row in history],
            "average_payoff": [row["average_payoff"] for row in history],
            "cooperation_rate": [row["cooperation_rate"] for row in history],
            "diversity": [row["diversity"] for row in history],
        }
    )
    outcomes_figure = make_subplots(specs=[[{"secondary_y": True}]])
    outcomes_figure.add_trace(
        go.Scatter(x=outcome_frame["generation"], y=outcome_frame["average_payoff"], name="Average payoff"),
        secondary_y=False,
    )
    outcomes_figure.add_trace(
        go.Scatter(x=outcome_frame["generation"], y=outcome_frame["cooperation_rate"], name="Cooperation rate"),
        secondary_y=True,
    )
    outcomes_figure.add_trace(
        go.Scatter(x=outcome_frame["generation"], y=outcome_frame["diversity"], name="Diversity"),
        secondary_y=True,
    )
    outcomes_figure.update_yaxes(title_text="Average payoff", secondary_y=False, range=[0, 5])
    outcomes_figure.update_yaxes(title_text="Rate", secondary_y=True, range=[0, 1], tickformat=".0%")
    outcomes_figure.update_layout(
        title="Population outcomes",
        height=420,
        margin=dict(l=10, r=10, t=55, b=35),
        hovermode="x unified",
    )

    final = history[-1]
    table_rows = [
        {
            "strategy": name,
            "final_share": round(float(final["shares"][name]), 6),
            "fitness": round(float(final["fitness"][name]), 4),
        }
        for name in strategy_names
    ]
    table_rows.sort(key=lambda row: row["final_share"], reverse=True)
    columns = [
        {"name": "Strategy", "id": "strategy"},
        {"name": "Final share", "id": "final_share", "type": "numeric", "format": {"specifier": ".1%"}},
        {"name": "Fitness", "id": "fitness", "type": "numeric"},
    ]
    return population_figure, outcomes_figure, columns, table_rows
