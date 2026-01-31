"""
Prisoner's Dilemma tournament simulation utilities.

This module is intentionally side-effect free on import:
- no automatic simulations
- no automatic CSV writes

The Dash app can call `simulate_tournament()` and build analytics on top.
"""

from __future__ import annotations

from typing import Callable, Literal, Optional

import pandas as pd
import random

Move = Literal["cooperate", "defect"]


def payoff(move_1: Move, move_2: Move) -> tuple[int, int]:
    """Return (points_1, points_2) for a single round."""

    if move_1 == "cooperate" and move_2 == "cooperate":
        return 3, 3
    if move_1 == "defect" and move_2 == "defect":
        return 1, 1
    if move_1 == "cooperate" and move_2 == "defect":
        return 0, 5
    return 5, 0


class Strategy:
    def __init__(self, name: str):
        self.name = name

    def play(self, opponent_history: list[Move]) -> Move:
        raise NotImplementedError


class MrNiceGuy(Strategy):
    def play(self, opponent_history: list[Move]) -> Move:
        return "cooperate"


class BadCop(Strategy):
    def play(self, opponent_history: list[Move]) -> Move:
        return "defect"


class TitForTat(Strategy):
    def play(self, opponent_history: list[Move]) -> Move:
        return "cooperate" if not opponent_history else opponent_history[-1]


class ImSoRandom(Strategy):
    def play(self, opponent_history: list[Move]) -> Move:
        return "cooperate" if random.random() < 0.5 else "defect"


class CalculatedDefector(Strategy):
    def play(self, opponent_history: list[Move]) -> Move:
        return "defect" if opponent_history.count("defect") > len(opponent_history) * 0.25 else "cooperate"


class HoldingAGrudge(Strategy):
    def play(self, opponent_history: list[Move]) -> Move:
        return "defect" if "defect" in opponent_history else "cooperate"


class ForgiveButDontForget(Strategy):
    def play(self, opponent_history: list[Move]) -> Move:
        return "defect" if opponent_history.count("defect") > len(opponent_history) * 0.5 else "cooperate"


class BadAlternator(Strategy):
    def __init__(self, name: str):
        super().__init__(name)
        self.turn = 0

    def play(self, opponent_history: list[Move]) -> Move:
        self.turn += 1
        return "cooperate" if self.turn % 2 == 1 else "defect"


class RitualDefection(Strategy):
    def __init__(self, name: str):
        super().__init__(name)
        self.turn = 0

    def play(self, opponent_history: list[Move]) -> Move:
        self.turn += 1
        return "defect" if self.turn % 5 == 0 else "cooperate"


class TripleThreat(Strategy):
    def __init__(self, name: str):
        super().__init__(name)
        self.turn = 0

    def play(self, opponent_history: list[Move]) -> Move:
        self.turn += 1
        cycle_position = self.turn % 6
        return "defect" if 3 <= cycle_position < 6 else "cooperate"


def make_strategy_factories() -> list[Callable[[], Strategy]]:
    return [
        lambda: MrNiceGuy("MrNiceGuy"),
        lambda: BadCop("BadCop"),
        lambda: TitForTat("TitForTat"),
        lambda: ImSoRandom("ImSoRandom"),
        lambda: CalculatedDefector("CalculatedDefector"),
        lambda: HoldingAGrudge("HoldingAGrudge"),
        lambda: ForgiveButDontForget("ForgiveButDontForget"),
        lambda: BadAlternator("BadAlternator"),
        lambda: RitualDefection("RitualDefection"),
        lambda: TripleThreat("TripleThreat"),
    ]


def list_strategy_names() -> list[str]:
    return [factory().name for factory in make_strategy_factories()]


def simulate_tournament(
    *,
    rounds_per_match: int = 10,
    repetitions: int = 30,
    seed: Optional[int] = 0,
) -> pd.DataFrame:
    """
    Round-level tournament simulation.

    One row per round played with:
    - repetition, round
    - strategy_1, strategy_2
    - move_1, move_2
    - points_1, points_2
    """

    if rounds_per_match <= 0:
        raise ValueError("rounds_per_match must be > 0")
    if repetitions <= 0:
        raise ValueError("repetitions must be > 0")

    if seed is not None:
        random.seed(seed)

    factories = make_strategy_factories()
    rows: list[dict] = []

    for rep in range(repetitions):
        for i in range(len(factories)):
            for j in range(i + 1, len(factories)):
                s1 = factories[i]()
                s2 = factories[j]()

                history1: list[Move] = []
                history2: list[Move] = []

                for r in range(rounds_per_match):
                    m1 = s1.play(history2)
                    m2 = s2.play(history1)
                    history1.append(m1)
                    history2.append(m2)

                    p1, p2 = payoff(m1, m2)

                    rows.append(
                        {
                            "repetition": rep,
                            "round": r,
                            "strategy_1": s1.name,
                            "strategy_2": s2.name,
                            "move_1": m1,
                            "move_2": m2,
                            "points_1": p1,
                            "points_2": p2,
                        }
                    )

    df = pd.DataFrame(rows)
    for col in ("strategy_1", "strategy_2", "move_1", "move_2"):
        if col in df.columns:
            df[col] = df[col].astype("string")
    return df


def strategy_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Per-strategy totals aggregated across both player roles."""

    if df.empty:
        return pd.DataFrame(columns=["strategy", "total_points", "avg_points_per_round", "cooperate_rate"])

    s1 = (
        df.groupby("strategy_1")
        .agg(
            total_points=("points_1", "sum"),
            rounds=("points_1", "size"),
            coop=("move_1", lambda s: (s == "cooperate").mean()),
        )
        .rename_axis("strategy")
    )
    s2 = (
        df.groupby("strategy_2")
        .agg(
            total_points=("points_2", "sum"),
            rounds=("points_2", "size"),
            coop=("move_2", lambda s: (s == "cooperate").mean()),
        )
        .rename_axis("strategy")
    )

    merged = s1.add(s2, fill_value=0)
    merged["avg_points_per_round"] = merged["total_points"] / merged["rounds"].where(merged["rounds"] != 0, 1)
    merged["cooperate_rate"] = merged["coop"] / 2.0

    out = merged.reset_index()[["strategy", "total_points", "avg_points_per_round", "cooperate_rate"]]
    return out.sort_values("total_points", ascending=False, ignore_index=True)