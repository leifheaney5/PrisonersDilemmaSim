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


class ThePushover(Strategy):
    """
    Starts reasonably responsive, then "gives in" later by cooperating regardless.

    Implementation: tit-for-tat for a few turns, then always cooperate.
    """

    def __init__(self, name: str):
        super().__init__(name)
        self.turn = 0

    def play(self, opponent_history: list[Move]) -> Move:
        self.turn += 1
        if self.turn <= 4:
            return "cooperate" if not opponent_history else opponent_history[-1]
        return "cooperate"


class TheThief(Strategy):
    """
    Tries to "steal" late by shifting behavior over time.

    Implementation:
    - Early: cooperate
    - Mid: tit-for-tat
    - Late: defect-heavy pattern (D, D, C repeating)
    """

    def __init__(self, name: str):
        super().__init__(name)
        self.turn = 0

    def play(self, opponent_history: list[Move]) -> Move:
        self.turn += 1
        if self.turn <= 3:
            return "cooperate"
        if self.turn <= 6:
            return "cooperate" if not opponent_history else opponent_history[-1]
        # Late-game theft: mostly defect with occasional cooperate
        return "cooperate" if (self.turn % 3 == 0) else "defect"


class Pattern(Strategy):
    """
    Fixed repeating pattern: 3 of one move, then 3 of the other, repeat (DDD CCC ...).

    Note: This is intentionally different from `TripleThreat`, which is (CCC DDD ...).
    """

    def __init__(self, name: str):
        super().__init__(name)
        self.turn = 0

    def play(self, opponent_history: list[Move]) -> Move:
        self.turn += 1
        idx = (self.turn - 1) % 6
        return "defect" if idx < 3 else "cooperate"


class NeverSwitchUp(Strategy):
    """
    Chooses cooperate/defect once at random, then never changes for the whole match.
    """

    def __init__(self, name: str):
        super().__init__(name)
        self.choice: Optional[Move] = None

    def play(self, opponent_history: list[Move]) -> Move:
        if self.choice is None:
            self.choice = "cooperate" if random.random() < 0.5 else "defect"
        return self.choice


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
        lambda: ThePushover("ThePushover"),
        lambda: TheThief("TheThief"),
        lambda: Pattern("Pattern"),
        lambda: NeverSwitchUp("NeverSwitchUp"),
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


# ----------------------------
# Real-time / incremental experiment runner
# ----------------------------


def _lcg_next(state: int) -> int:
    """
    Tiny deterministic RNG for UI stepping.

    We use an LCG so the entire experiment state is JSON-serializable (single int),
    which plays nicely with `dcc.Store`.
    """

    # Numerical Recipes LCG parameters
    a = 1664525
    c = 1013904223
    m = 2**32
    return (a * (state & 0xFFFFFFFF) + c) % m


def _lcg_float01(state: int) -> tuple[float, int]:
    state = _lcg_next(state)
    return (state / float(2**32)), state


def _init_strategy_state(name: str, custom_config: Optional[dict] = None) -> dict:
    """
    Per-match state for the incremental runner.

    If `custom_config` is provided, it will be embedded into the state so
    `play_strategy()` can execute the custom policy.
    """

    if custom_config is not None:
        return {"turn": 0, "custom": custom_config}
    if name in {"BadAlternator", "RitualDefection", "TripleThreat", "ThePushover", "TheThief", "Pattern"}:
        return {"turn": 0}
    if name == "NeverSwitchUp":
        return {"choice": None}
    return {}


def play_strategy(name: str, opponent_history: list[Move], state: dict, rng_state: int) -> tuple[Move, dict, int]:
    """
    Stateless-ish strategy execution used by the incremental runner.

    Returns: (move, new_state, new_rng_state)
    """

    # Custom strategy (from UI builder)
    custom = state.get("custom") if isinstance(state, dict) else None
    if isinstance(custom, dict):
        turn = int(state.get("turn", 0)) + 1
        state = {**state, "turn": turn}

        base_move: Move = "cooperate" if custom.get("start_move") == "cooperate" else "defect"
        use_tft = bool(custom.get("use_tft", False))
        use_grudge = bool(custom.get("use_grudge", False))
        min_history = int(custom.get("min_history", 0) or 0)
        threshold = float(custom.get("defect_rate_threshold", 1.0))
        endgame_after = int(custom.get("endgame_after_turn", 0) or 0)
        noise = float(custom.get("noise", 0.0) or 0.0)

        move: Move
        if use_grudge and ("defect" in opponent_history):
            move = "defect"
        elif use_tft:
            move = "cooperate" if not opponent_history else opponent_history[-1]
        else:
            move = base_move

        if len(opponent_history) >= min_history and min_history > 0:
            defect_rate = opponent_history.count("defect") / float(len(opponent_history) or 1)
            if defect_rate > threshold:
                move = "defect"

        if endgame_after > 0 and turn >= endgame_after:
            move = "defect"

        # Noise: occasionally flip the decision
        if noise > 0:
            x, rng_state = _lcg_float01(rng_state)
            if x < noise:
                move = "defect" if move == "cooperate" else "cooperate"

        return move, state, rng_state

    if name == "MrNiceGuy":
        return "cooperate", state, rng_state
    if name == "BadCop":
        return "defect", state, rng_state
    if name == "TitForTat":
        return ("cooperate" if not opponent_history else opponent_history[-1]), state, rng_state
    if name == "ImSoRandom":
        x, rng_state = _lcg_float01(rng_state)
        return ("cooperate" if x < 0.5 else "defect"), state, rng_state
    if name == "CalculatedDefector":
        return ("defect" if opponent_history.count("defect") > len(opponent_history) * 0.25 else "cooperate"), state, rng_state
    if name == "HoldingAGrudge":
        return ("defect" if "defect" in opponent_history else "cooperate"), state, rng_state
    if name == "ForgiveButDontForget":
        return ("defect" if opponent_history.count("defect") > len(opponent_history) * 0.5 else "cooperate"), state, rng_state
    if name == "BadAlternator":
        turn = int(state.get("turn", 0)) + 1
        state = {**state, "turn": turn}
        return ("cooperate" if (turn % 2 == 1) else "defect"), state, rng_state
    if name == "RitualDefection":
        turn = int(state.get("turn", 0)) + 1
        state = {**state, "turn": turn}
        return ("defect" if (turn % 5 == 0) else "cooperate"), state, rng_state
    if name == "TripleThreat":
        turn = int(state.get("turn", 0)) + 1
        state = {**state, "turn": turn}
        cycle_position = turn % 6
        return ("defect" if 3 <= cycle_position < 6 else "cooperate"), state, rng_state

    if name == "ThePushover":
        turn = int(state.get("turn", 0)) + 1
        state = {**state, "turn": turn}
        if turn <= 4:
            return ("cooperate" if not opponent_history else opponent_history[-1]), state, rng_state
        return "cooperate", state, rng_state

    if name == "TheThief":
        turn = int(state.get("turn", 0)) + 1
        state = {**state, "turn": turn}
        if turn <= 3:
            return "cooperate", state, rng_state
        if turn <= 6:
            return ("cooperate" if not opponent_history else opponent_history[-1]), state, rng_state
        return ("cooperate" if (turn % 3 == 0) else "defect"), state, rng_state

    if name == "Pattern":
        turn = int(state.get("turn", 0)) + 1
        state = {**state, "turn": turn}
        idx = (turn - 1) % 6
        return ("defect" if idx < 3 else "cooperate"), state, rng_state

    if name == "NeverSwitchUp":
        choice = state.get("choice")
        if choice in ("cooperate", "defect"):
            return choice, state, rng_state
        x, rng_state = _lcg_float01(rng_state)
        choice = "cooperate" if x < 0.5 else "defect"
        return choice, {**state, "choice": choice}, rng_state

    # Fallback: treat unknown as always cooperate (safe default)
    return "cooperate", state, rng_state


def init_tournament_state(
    *,
    strategy_names: list[str],
    rounds_per_match: int,
    repetitions: int,
    seed: int = 0,
    recent_limit: int = 200,
    timeline_limit: int = 400,
    custom_strategies: Optional[dict[str, dict]] = None,
) -> dict:
    """
    Create a JSON-serializable state object for incremental tournament stepping.
    """

    names = [str(s) for s in (strategy_names or [])]
    if len(names) < 2:
        raise ValueError("Select at least 2 strategies to run a tournament.")

    rounds_per_match = int(rounds_per_match)
    repetitions = int(repetitions)
    seed = int(seed)
    if rounds_per_match <= 0:
        raise ValueError("rounds_per_match must be > 0")
    if repetitions <= 0:
        raise ValueError("repetitions must be > 0")

    n = len(names)
    total_matches = repetitions * (n * (n - 1) // 2)

    custom_strategies = dict(custom_strategies or {})

    # Initialize first pair (0,1)
    s1 = names[0]
    s2 = names[1]

    return {
        "strategy_names": names,
        "rounds_per_match": rounds_per_match,
        "repetitions": repetitions,
        "seed": seed,
        "rng_state": seed & 0xFFFFFFFF,
        "recent_limit": int(recent_limit),
        # iteration state
        "rep": 0,
        "i": 0,
        "j": 1,
        "round": 0,
        "s1": s1,
        "s2": s2,
        "custom_strategies": custom_strategies,
        "s1_state": _init_strategy_state(s1, custom_strategies.get(s1)),
        "s2_state": _init_strategy_state(s2, custom_strategies.get(s2)),
        "history1": [],
        "history2": [],
        "match_points1": 0,
        "match_points2": 0,
        # aggregates
        "totals": {name: 0 for name in names},
        "rounds_played": {name: 0 for name in names},
        "cooperate": {name: 0 for name in names},
        "match_wins": {name: 0 for name in names},
        "match_losses": {name: 0 for name in names},
        "match_ties": {name: 0 for name in names},
        "matches_done": 0,
        "total_matches": total_matches,
        "recent": [],
        # match-level timeline snapshots (for live charts)
        "timeline_limit": int(timeline_limit),
        "timeline": [],
        # UI helpers
        "summary_shown": False,
        "done": False,
    }


def _advance_pair(state: dict) -> dict:
    names: list[str] = state["strategy_names"]
    n = len(names)
    i = int(state["i"])
    j = int(state["j"])
    rep = int(state["rep"])

    # next pair
    j += 1
    if j >= n:
        i += 1
        j = i + 1

    # end of rep
    if i >= n - 1:
        rep += 1
        i = 0
        j = 1

    state["rep"] = rep
    state["i"] = i
    state["j"] = j

    if rep >= int(state["repetitions"]):
        state["done"] = True
        return state

    s1 = names[i]
    s2 = names[j]
    state["s1"] = s1
    state["s2"] = s2
    custom = state.get("custom_strategies", {}) or {}
    state["s1_state"] = _init_strategy_state(s1, custom.get(s1))
    state["s2_state"] = _init_strategy_state(s2, custom.get(s2))
    state["history1"] = []
    state["history2"] = []
    state["round"] = 0
    state["match_points1"] = 0
    state["match_points2"] = 0
    return state


def step_tournament(state: dict, *, max_rounds: int = 500) -> dict:
    """
    Advance the tournament by up to `max_rounds` rounds and return the updated state.
    """

    if not state or state.get("done"):
        return state

    max_rounds = int(max_rounds)
    if max_rounds <= 0:
        return state

    rounds_per_match = int(state["rounds_per_match"])
    recent_limit = int(state.get("recent_limit", 200))
    timeline_limit = int(state.get("timeline_limit", 400))

    for _ in range(max_rounds):
        if state.get("done"):
            break

        s1 = state["s1"]
        s2 = state["s2"]
        history1: list[Move] = state["history1"]
        history2: list[Move] = state["history2"]

        m1, s1_state, rng_state = play_strategy(s1, history2, state["s1_state"], int(state["rng_state"]))
        m2, s2_state, rng_state = play_strategy(s2, history1, state["s2_state"], rng_state)

        state["s1_state"] = s1_state
        state["s2_state"] = s2_state
        state["rng_state"] = rng_state

        history1.append(m1)
        history2.append(m2)

        p1, p2 = payoff(m1, m2)
        state["match_points1"] += int(p1)
        state["match_points2"] += int(p2)

        state["totals"][s1] += int(p1)
        state["totals"][s2] += int(p2)
        state["rounds_played"][s1] += 1
        state["rounds_played"][s2] += 1
        state["cooperate"][s1] += 1 if m1 == "cooperate" else 0
        state["cooperate"][s2] += 1 if m2 == "cooperate" else 0

        state["recent"].append(
            {
                "rep": int(state["rep"]),
                "strategy_1": s1,
                "strategy_2": s2,
                "round": int(state["round"]) + 1,
                "move_1": m1,
                "move_2": m2,
                "points_1": int(p1),
                "points_2": int(p2),
            }
        )
        if len(state["recent"]) > recent_limit:
            state["recent"] = state["recent"][-recent_limit:]

        state["round"] = int(state["round"]) + 1

        if int(state["round"]) >= rounds_per_match:
            # finalize match outcome
            mp1 = int(state["match_points1"])
            mp2 = int(state["match_points2"])
            if mp1 > mp2:
                state["match_wins"][s1] += 1
                state["match_losses"][s2] += 1
            elif mp2 > mp1:
                state["match_wins"][s2] += 1
                state["match_losses"][s1] += 1
            else:
                state["match_ties"][s1] += 1
                state["match_ties"][s2] += 1

            state["matches_done"] = int(state["matches_done"]) + 1

            # snapshot match-level aggregates for live charts
            state["timeline"].append(
                {
                    "matches_done": int(state["matches_done"]),
                    "totals": dict(state["totals"]),
                    "match_wins": dict(state["match_wins"]),
                    "match_losses": dict(state["match_losses"]),
                    "match_ties": dict(state["match_ties"]),
                }
            )
            if len(state["timeline"]) > timeline_limit:
                state["timeline"] = state["timeline"][-timeline_limit:]

            state = _advance_pair(state)

    return state


def init_human_match_state(*, opponent: str, rounds: int = 10, seed: int = 0, custom_strategies: Optional[dict[str, dict]] = None) -> dict:
    """
    Initialize a human-vs-strategy match state for interactive play.
    """

    opponent = str(opponent)
    rounds = int(rounds)
    seed = int(seed)
    if rounds <= 0:
        raise ValueError("rounds must be > 0")

    custom_strategies = dict(custom_strategies or {})
    return {
        "opponent": opponent,
        "rounds": rounds,
        "seed": seed,
        "rng_state": seed & 0xFFFFFFFF,
        "custom_strategies": custom_strategies,
        "opponent_state": _init_strategy_state(opponent, custom_strategies.get(opponent)),
        "human_history": [],
        "opponent_history": [],
        "round": 0,
        "human_points": 0,
        "opponent_points": 0,
        "events": [],
        "done": False,
    }


def step_human_match(state: dict, *, human_move: Move) -> dict:
    """
    Apply one human move, compute opponent response, and advance match by one round.
    """

    if not state or state.get("done"):
        return state

    human_move = "cooperate" if human_move == "cooperate" else "defect"

    opponent = state["opponent"]
    rng_state = int(state["rng_state"])
    opp_move, opp_state, rng_state = play_strategy(opponent, state["human_history"], state["opponent_state"], rng_state)

    state["rng_state"] = rng_state
    state["opponent_state"] = opp_state

    # update histories
    state["human_history"].append(human_move)
    state["opponent_history"].append(opp_move)

    p_h, p_o = payoff(human_move, opp_move)
    state["human_points"] += int(p_h)
    state["opponent_points"] += int(p_o)

    state["round"] = int(state["round"]) + 1
    state["events"].append(
        {
            "round": int(state["round"]),
            "human_move": human_move,
            "opponent_move": opp_move,
            "human_points": int(p_h),
            "opponent_points": int(p_o),
        }
    )

    if int(state["round"]) >= int(state["rounds"]):
        state["done"] = True

    return state