"""Dash-independent tournament analytics and matchup transformations."""

from __future__ import annotations

import pandas as pd


def perspective_rows(results: pd.DataFrame) -> pd.DataFrame:
    """Convert round rows into one row from each player's perspective."""
    player_one = results.rename(
        columns={
            "strategy_1": "strategy",
            "strategy_2": "opponent",
            "move_1": "move",
            "move_2": "opp_move",
            "points_1": "points",
            "points_2": "opp_points",
        }
    )[["repetition", "round", "strategy", "opponent", "move", "opp_move", "points", "opp_points"]]
    player_two = results.rename(
        columns={
            "strategy_2": "strategy",
            "strategy_1": "opponent",
            "move_2": "move",
            "move_1": "opp_move",
            "points_2": "points",
            "points_1": "opp_points",
        }
    )[["repetition", "round", "strategy", "opponent", "move", "opp_move", "points", "opp_points"]]
    rows = pd.concat([player_one, player_two], ignore_index=True)
    for column in ("strategy", "opponent", "move", "opp_move"):
        rows[column] = rows[column].astype("string")
    return rows


def match_level(perspectives: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-round perspectives into match outcomes."""
    matches = (
        perspectives.groupby(["repetition", "strategy", "opponent"], as_index=False)
        .agg(points=("points", "sum"), opp_points=("opp_points", "sum"), rounds=("points", "size"))
    )
    matches["outcome"] = "tie"
    matches.loc[matches["points"] > matches["opp_points"], "outcome"] = "win"
    matches.loc[matches["points"] < matches["opp_points"], "outcome"] = "loss"
    return matches


def tournament_metrics_frame(state: dict) -> pd.DataFrame:
    """Return comparable live metrics without favoring strategies that played first."""
    rows = []
    for strategy in state.get("strategy_names", []) or []:
        rounds = int((state.get("rounds_played", {}) or {}).get(strategy, 0))
        points = int((state.get("totals", {}) or {}).get(strategy, 0))
        cooperations = int((state.get("cooperate", {}) or {}).get(strategy, 0))
        wins = int((state.get("match_wins", {}) or {}).get(strategy, 0))
        losses = int((state.get("match_losses", {}) or {}).get(strategy, 0))
        ties = int((state.get("match_ties", {}) or {}).get(strategy, 0))
        matches = wins + losses + ties
        rows.append(
            {
                "strategy": strategy,
                "total_points": points,
                "points_per_round": points / rounds if rounds else 0.0,
                "cooperation_rate": cooperations / rounds if rounds else 0.0,
                "defection_rate": (rounds - cooperations) / rounds if rounds else 0.0,
                "win_rate": wins / matches if matches else 0.0,
                "rounds_played": rounds,
                "matches_played": matches,
            }
        )
    return pd.DataFrame(rows)


def strategy_landscape_frame(results: pd.DataFrame) -> pd.DataFrame:
    """Summarize strategy behavior for the interactive strategy landscape.

    The horizontal axis measures how much more often a strategy cooperates after
    cooperation than after defection. The vertical axis measures how consistently
    it keeps the same action from one round to the next.
    """
    perspectives = perspective_rows(results)
    columns = [
        "strategy",
        "cooperation_rate",
        "cooperate_after_cooperation",
        "cooperate_after_defection",
        "response_gap",
        "stability",
        "points_per_round",
    ]
    if perspectives.empty:
        return pd.DataFrame(columns=columns)

    summary = perspectives.groupby("strategy", as_index=False).agg(
        cooperation_rate=("move", lambda moves: float((moves == "cooperate").mean())),
        points_per_round=("points", "mean"),
    )
    ordered = perspectives.sort_values(["repetition", "strategy", "opponent", "round"]).copy()
    match_groups = ordered.groupby(["repetition", "strategy", "opponent"], sort=False)
    ordered["previous_opp_move"] = match_groups["opp_move"].shift()
    conditional = (
        ordered[ordered["previous_opp_move"].notna()]
        .assign(cooperated=lambda frame: (frame["move"] == "cooperate").astype(float))
        .pivot_table(
            index="strategy",
            columns="previous_opp_move",
            values="cooperated",
            aggfunc="mean",
        )
        .rename(
            columns={
                "cooperate": "cooperate_after_cooperation",
                "defect": "cooperate_after_defection",
            }
        )
        .reset_index()
    )
    summary = summary.merge(conditional, on="strategy", how="left")
    for column in ("cooperate_after_cooperation", "cooperate_after_defection"):
        summary[column] = summary[column].fillna(summary["cooperation_rate"])

    ordered["previous_move"] = match_groups["move"].shift()
    observed = ordered[ordered["previous_move"].notna()].copy()
    observed["kept_action"] = (observed["move"] == observed["previous_move"]).astype(float)
    stability = observed.groupby("strategy", as_index=False).agg(stability=("kept_action", "mean"))
    summary = summary.merge(stability, on="strategy", how="left")
    summary["stability"] = summary["stability"].fillna(1.0)
    summary["response_gap"] = (
        summary["cooperate_after_cooperation"] - summary["cooperate_after_defection"]
    )
    return summary[columns].sort_values("strategy").reset_index(drop=True)


def pairwise_metric_frame(results: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Build a strategy-by-opponent matrix for a selected matchup metric."""
    perspectives = perspective_rows(results)
    if perspectives.empty:
        return pd.DataFrame()
    perspectives = perspectives.assign(
        combined_payoff=perspectives["points"] + perspectives["opp_points"],
        payoff_inequality=(perspectives["points"] - perspectives["opp_points"]).abs(),
        mutual_cooperation=((perspectives["move"] == "cooperate") & (perspectives["opp_move"] == "cooperate")).astype(float),
        mutual_defection=((perspectives["move"] == "defect") & (perspectives["opp_move"] == "defect")).astype(float),
        exploitation=(perspectives["move"] != perspectives["opp_move"]).astype(float),
    )
    grouped = perspectives.groupby(["strategy", "opponent"], as_index=False).agg(
        points_per_round=("points", "mean"),
        opponent_points_per_round=("opp_points", "mean"),
        cooperation_rate=("move", lambda moves: float((moves == "cooperate").mean())),
        combined_payoff=("combined_payoff", "mean"),
        payoff_inequality=("payoff_inequality", "mean"),
        mutual_cooperation_rate=("mutual_cooperation", "mean"),
        mutual_defection_rate=("mutual_defection", "mean"),
        exploitation_rate=("exploitation", "mean"),
    )
    grouped["score_margin"] = grouped["points_per_round"] - grouped["opponent_points_per_round"]
    if metric == "win_rate":
        matches = match_level(perspectives)
        win_rates = matches.groupby(["strategy", "opponent"], as_index=False).agg(
            win_rate=("outcome", lambda outcomes: float((outcomes == "win").mean()))
        )
        grouped = grouped.merge(win_rates, on=["strategy", "opponent"], how="left")
    supported = {
        "points_per_round",
        "cooperation_rate",
        "score_margin",
        "win_rate",
        "combined_payoff",
        "payoff_inequality",
        "mutual_cooperation_rate",
        "mutual_defection_rate",
        "exploitation_rate",
    }
    value = metric if metric in supported else "points_per_round"
    return grouped.pivot(index="strategy", columns="opponent", values=value)


def matchup_replay_frame(results: pd.DataFrame, strategy: str, opponent: str, repetition: int = 0) -> pd.DataFrame:
    """Return one matchup from the selected strategy's perspective."""
    if results.empty or not strategy or not opponent:
        return pd.DataFrame()
    forward = results[(results["strategy_1"] == strategy) & (results["strategy_2"] == opponent)]
    reverse = results[(results["strategy_2"] == strategy) & (results["strategy_1"] == opponent)]
    source = forward if not forward.empty else reverse
    source = source[source["repetition"] == int(repetition)].sort_values("round")
    if source.empty:
        return pd.DataFrame()
    is_forward = not forward.empty
    player = "1" if is_forward else "2"
    other = "2" if is_forward else "1"
    replay = pd.DataFrame(
        {
            "round": source["round"].astype(int) + 1,
            "strategy_move": source[f"move_{player}"].astype(str),
            "opponent_move": source[f"move_{other}"].astype(str),
            "strategy_intended": source.get(f"intended_move_{player}", source[f"move_{player}"]).astype(str),
            "opponent_intended": source.get(f"intended_move_{other}", source[f"move_{other}"]).astype(str),
            "strategy_points": source[f"points_{player}"].astype(int),
            "opponent_points": source[f"points_{other}"].astype(int),
        }
    )
    replay["combined_payoff"] = replay["strategy_points"] + replay["opponent_points"]
    replay["score_margin"] = replay["strategy_points"] - replay["opponent_points"]
    replay["cumulative_strategy"] = replay["strategy_points"].cumsum()
    replay["cumulative_opponent"] = replay["opponent_points"].cumsum()
    replay["outcome"] = "mixed"
    replay.loc[(replay["strategy_move"] == "cooperate") & (replay["opponent_move"] == "cooperate"), "outcome"] = "mutual cooperation"
    replay.loc[(replay["strategy_move"] == "defect") & (replay["opponent_move"] == "defect"), "outcome"] = "mutual defection"
    return replay
