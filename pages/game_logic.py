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
import math

Move = Literal["cooperate", "defect"]

_NAME_ALIASES: dict[str, str] = {
    # Legacy names (kept for backwards compatibility)
    "ThePushover": "Pushover",
    "TheThief": "Thief",
    "ParrotPicker": "Parrot",
    "KeepingThePeace": "KeepingPeace",
}


def _canonical_strategy_name(name: str) -> str:
    nm = str(name or "")
    return _NAME_ALIASES.get(nm, nm)


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


# ----------------------------
# Number helpers (for novelty strategies)
# ----------------------------


def _is_prime(n: int) -> bool:
    n = int(n)
    if n < 2:
        return False
    if n in (2, 3):
        return True
    if (n % 2) == 0:
        return False
    r = int(math.isqrt(n))
    f = 3
    while f <= r:
        if (n % f) == 0:
            return False
        f += 2
    return True


def _is_fibonacci(n: int) -> bool:
    """
    True if n is a Fibonacci number (1-indexed notion still works).
    Test: 5n^2 ± 4 is a perfect square.
    """

    n = int(n)
    if n < 0:
        return False
    x1 = 5 * n * n + 4
    x2 = 5 * n * n - 4
    r1 = int(math.isqrt(x1))
    r2 = int(math.isqrt(x2))
    return (r1 * r1 == x1) or (r2 * r2 == x2)


def _is_proth(n: int) -> bool:
    """
    Proth number test: n = k*2^m + 1 with k odd, m>0 and 2^m > k.

    Reference definition: https://www.numbersaplenty.com/set/Proth_number/
    """

    n = int(n)
    if n <= 2:
        return False
    x = n - 1
    # Factor out powers of 2: x = k * 2^m
    m = 0
    while x % 2 == 0:
        x //= 2
        m += 1
        k = x
        if m > 0 and (k % 2 == 1) and (2**m > k):
            return True
    return False


# A small, explicit set of Friedman numbers (as listed on numbersaplenty).
# In typical app usage, "round numbers" are small, so this is enough and avoids
# a heavy expression-search algorithm.
_FRIEDMAN_NUMBERS: set[int] = {
    25,
    121,
    125,
    126,
    127,
    128,
    153,
    216,
    289,
    343,
    347,
    625,
    688,
    736,
    1022,
    1024,
    1206,
    1255,
    1260,
    1285,
    1296,
    1395,
    1435,
    1503,
    1530,
    1792,
    1827,
    2048,
    2187,
    2349,
    2500,
    2501,
}


def _is_friedman(n: int) -> bool:
    """
    Friedman-number membership using a curated set.

    Reference / examples: https://www.numbersaplenty.com/set/Friedman_number/
    """

    return int(n) in _FRIEDMAN_NUMBERS


def _is_square(n: int) -> bool:
    n = int(n)
    if n < 0:
        return False
    r = int(math.isqrt(n))
    return r * r == n


class MrNiceGuy(Strategy):
    def play(self, opponent_history: list[Move]) -> Move:
        return "cooperate"


class BadCop(Strategy):
    def play(self, opponent_history: list[Move]) -> Move:
        return "defect"


class TitForTat(Strategy):
    def play(self, opponent_history: list[Move]) -> Move:
        return "cooperate" if not opponent_history else opponent_history[-1]


class WinStayLoseShift(Strategy):
    """
    Win-Stay, Lose-Shift (WSLS / Pavlov).

    Repeats the previous move if the previous outcome was "good" (R=3 or T=5),
    otherwise switches (P=1 or S=0). Starts with cooperate.
    """

    def __init__(self, name: str):
        super().__init__(name)
        self.last_move: Move = "cooperate"

    def play(self, opponent_history: list[Move]) -> Move:
        if not opponent_history:
            self.last_move = "cooperate"
            return self.last_move

        prev_payoff, _ = payoff(self.last_move, opponent_history[-1])
        if prev_payoff in (3, 5):
            move: Move = self.last_move
        else:
            move = "defect" if self.last_move == "cooperate" else "cooperate"
        self.last_move = move
        return move


class TitForTwoTats(Strategy):
    """
    Tit-for-Two-Tats (TF2T).

    Cooperates by default; defects only after two consecutive opponent defections.
    """

    def play(self, opponent_history: list[Move]) -> Move:
        if len(opponent_history) >= 2 and opponent_history[-1] == "defect" and opponent_history[-2] == "defect":
            return "defect"
        return "cooperate"


class SuspiciousTitForTat(Strategy):
    """
    Suspicious Tit-for-Tat (STFT): starts with defect, then mirrors the opponent.
    """

    def play(self, opponent_history: list[Move]) -> Move:
        if not opponent_history:
            return "defect"
        return opponent_history[-1]


class GenerousTitForTat(Strategy):
    """
    Generous Tit-for-Tat (GTFT): like TFT, but sometimes forgives defections.
    """

    def __init__(self, name: str, forgive_prob: float = 0.1):
        super().__init__(name)
        self.forgive_prob = float(forgive_prob)

    def play(self, opponent_history: list[Move]) -> Move:
        if not opponent_history:
            return "cooperate"
        if opponent_history[-1] == "defect":
            return "cooperate" if random.random() < self.forgive_prob else "defect"
        return "cooperate"


class Joss(Strategy):
    """
    Joss: TFT with occasional random defection ("spite") even after opponent cooperates.
    """

    def __init__(self, name: str, defect_prob: float = 0.1):
        super().__init__(name)
        self.defect_prob = float(defect_prob)

    def play(self, opponent_history: list[Move]) -> Move:
        if not opponent_history:
            return "cooperate"
        if opponent_history[-1] == "defect":
            return "defect"
        # opponent cooperated; sometimes defect anyway
        return "defect" if random.random() < self.defect_prob else "cooperate"


class Prober(Strategy):
    """
    Prober / Tester.

    Probes early with a fixed sequence, then:
    - if the opponent retaliates at least once, switch to TFT
    - otherwise exploit by always defecting
    """

    def __init__(self, name: str):
        super().__init__(name)
        self.turn = 0
        self.mode: str | None = None  # "tft" or "exploit"

    def play(self, opponent_history: list[Move]) -> Move:
        self.turn += 1
        if self.turn == 1:
            return "defect"
        if self.turn in (2, 3):
            return "cooperate"

        if self.mode is None:
            self.mode = "tft" if ("defect" in opponent_history) else "exploit"

        if self.mode == "tft":
            return "cooperate" if not opponent_history else opponent_history[-1]
        return "defect"


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
        # Intended: CCC DDD repeating
        idx = (self.turn - 1) % 6
        return "cooperate" if idx < 3 else "defect"


class Pushover(Strategy):
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


class Thief(Strategy):
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


class RandomPrime(Strategy):
    """
    Defects every turn, except on prime-numbered turns it chooses randomly.

    Note: "turn" here is the round number within the current match (1-indexed).
    """

    def __init__(self, name: str):
        super().__init__(name)
        self.turn = 0

    def play(self, opponent_history: list[Move]) -> Move:
        self.turn += 1
        if _is_prime(self.turn):
            return "cooperate" if random.random() < 0.5 else "defect"
        return "defect"


class Fibonacci(Strategy):
    """
    Starts with a random base choice, then:
    - plays that base choice on Fibonacci-numbered turns
    - otherwise plays the opposite choice

    Note: "turn" here is the round number within the current match (1-indexed).
    """

    def __init__(self, name: str):
        super().__init__(name)
        self.turn = 0
        self.base: Optional[Move] = None

    def play(self, opponent_history: list[Move]) -> Move:
        self.turn += 1
        if self.base is None:
            self.base = "cooperate" if random.random() < 0.5 else "defect"
        if _is_fibonacci(self.turn):
            return self.base
        return "defect" if self.base == "cooperate" else "cooperate"


class DefectiveFriedman(Strategy):
    """
    Defects on turns whose (1-indexed) round number is a Friedman number,
    otherwise cooperates.

    Reference: https://www.numbersaplenty.com/set/Friedman_number/
    """

    def __init__(self, name: str):
        super().__init__(name)
        self.turn = 0

    def play(self, opponent_history: list[Move]) -> Move:
        self.turn += 1
        return "defect" if _is_friedman(self.turn) else "cooperate"


class CooperativeProth(Strategy):
    """
    Cooperates on turns whose (1-indexed) round number is a Proth number,
    otherwise defects.

    Reference: https://www.numbersaplenty.com/set/Proth_number/
    """

    def __init__(self, name: str):
        super().__init__(name)
        self.turn = 0

    def play(self, opponent_history: list[Move]) -> Move:
        self.turn += 1
        return "cooperate" if _is_proth(self.turn) else "defect"


class LongTermRelationship(Strategy):
    """
    A "relationship health" strategy based on overall cooperation rate.

    Rules (based on cooperation across BOTH players so far):
    - If combined cooperation rate >= 67%: cooperate ("hopeful future")
    - If combined cooperation rate is between 33% and 67%: random
    - If combined cooperation rate < 33%: defect

    Starts with cooperate on the first move.
    """

    def __init__(self, name: str):
        super().__init__(name)
        self.turn = 0
        self.self_coop = 0

    def play(self, opponent_history: list[Move]) -> Move:
        # Turn number within this match (1-indexed)
        self.turn += 1

        # First move is always cooperate
        if self.turn == 1:
            self.self_coop += 1
            return "cooperate"

        rounds_played = len(opponent_history)
        opp_coop = opponent_history.count("cooperate")
        denom = 2 * rounds_played if rounds_played > 0 else 1
        combined_rate = (self.self_coop + opp_coop) / float(denom)

        if combined_rate >= (2.0 / 3.0):
            move: Move = "cooperate"
        elif combined_rate < (1.0 / 3.0):
            move = "defect"
        else:
            move = "cooperate" if random.random() < 0.5 else "defect"

        self.self_coop += 1 if move == "cooperate" else 0
        return move


class Parrot(Strategy):
    """
    Starts with random, then copies the opponent's last move for 5 turns,
    then goes back to random for 1 turn, repeating.

    Cycle after the first move: 5x copy, 1x random.
    """

    def __init__(self, name: str):
        super().__init__(name)
        self.turn = 0

    def play(self, opponent_history: list[Move]) -> Move:
        self.turn += 1

        # First move: random
        if self.turn == 1:
            return "cooperate" if random.random() < 0.5 else "defect"

        # Then repeat: 5 turns copy, 1 turn random
        pos = (self.turn - 2) % 6  # 0..5
        if pos <= 4:
            return "cooperate" if not opponent_history else opponent_history[-1]
        return "cooperate" if random.random() < 0.5 else "defect"


class OneStepBehind(Strategy):
    """
    Starts with random, then always plays the opposite of the opponent's last move
    (i.e., what would have beaten it in the previous round).
    """

    def __init__(self, name: str):
        super().__init__(name)
        self.turn = 0

    def play(self, opponent_history: list[Move]) -> Move:
        self.turn += 1
        if self.turn == 1:
            return "cooperate" if random.random() < 0.5 else "defect"
        if not opponent_history:
            return "cooperate"
        return "defect" if opponent_history[-1] == "cooperate" else "cooperate"


class FriendlySquare(Strategy):
    """
    Cooperates on perfect-square turns (1, 4, 9, 16, ...), otherwise random.
    """

    def __init__(self, name: str):
        super().__init__(name)
        self.turn = 0

    def play(self, opponent_history: list[Move]) -> Move:
        self.turn += 1
        if _is_square(self.turn):
            return "cooperate"
        return "cooperate" if random.random() < 0.5 else "defect"


class LosingMyMind(Strategy):
    """
    Starts fully cooperative, but becomes increasingly random over time.

    This implementation does NOT assume the match length is known.
    Randomness ramps up with turn count and eventually becomes fully random.
    """

    def __init__(self, name: str):
        super().__init__(name)
        self.turn = 0

    def play(self, opponent_history: list[Move]) -> Move:
        self.turn += 1
        # Linear ramp: by turn 21 it's fully random (p = 1.0).
        p = min(1.0, max(0.0, (self.turn - 1) * 0.05))
        if p == 0.0:
            return "cooperate"
        if random.random() < p:
            return "cooperate" if random.random() < 0.5 else "defect"
        return "cooperate"


class BadJudgeOfCharacter(Strategy):
    """
    Starts with defect.

    After observing the opponent's first 3 moves:
    - If opponent defects more than cooperates: always defect forever
    - Otherwise: choose randomly forever
    """

    def __init__(self, name: str):
        super().__init__(name)
        self.turn = 0
        self.mode: str | None = None  # "always_defect" | "random"

    def play(self, opponent_history: list[Move]) -> Move:
        self.turn += 1
        if self.turn <= 3:
            return "defect"

        if self.mode not in ("always_defect", "random"):
            first3 = opponent_history[:3]
            d = first3.count("defect")
            c = first3.count("cooperate")
            self.mode = "always_defect" if d > c else "random"

        if self.mode == "always_defect":
            return "defect"
        return "cooperate" if random.random() < 0.5 else "defect"


class DefectiveDeputy(Strategy):
    """
    Defect-leaning strategy that becomes more and more likely to defect each turn.
    """

    def __init__(self, name: str):
        super().__init__(name)
        self.turn = 0

    def play(self, opponent_history: list[Move]) -> Move:
        self.turn += 1
        # Start defect-heavy and ramp to always defect.
        p_defect = min(1.0, 0.65 + 0.02 * (self.turn - 1))
        return "defect" if random.random() < p_defect else "cooperate"


class PastTrauma(Strategy):
    """
    Cooperates until the opponent defects at least 3 times (not necessarily consecutively),
    then defects forever.
    """

    def play(self, opponent_history: list[Move]) -> Move:
        return "defect" if opponent_history.count("defect") >= 3 else "cooperate"


class Lottery(Strategy):
    """
    Defects every turn, but on the last turn chooses randomly.

    If match length is unknown, it will defect every turn.
    """

    def __init__(self, name: str):
        super().__init__(name)
        self.turn = 0
        self.total_rounds: int | None = None

    def set_total_rounds(self, total_rounds: int) -> None:
        self.total_rounds = int(total_rounds)

    def play(self, opponent_history: list[Move]) -> Move:
        self.turn += 1
        if self.total_rounds and self.turn >= int(self.total_rounds):
            return "cooperate" if random.random() < 0.5 else "defect"
        return "defect"


class KeepingPeace(Strategy):
    """
    Starts with cooperate and tries to keep the match as even as possible in points.

    Heuristic:
    - Track cumulative points for self and opponent.
    - If ahead: cooperate (reduce risk of widening lead).
    - If behind: defect (attempt to catch up).
    - If tied: cooperate.

    Note: without knowing the opponent's next move, "perfect tying" isn't possible;
    this is a simple, stable approximation.
    """

    def __init__(self, name: str):
        super().__init__(name)
        self.turn = 0
        self.self_points = 0
        self.opp_points = 0
        self.last_move: Move | None = None

    def play(self, opponent_history: list[Move]) -> Move:
        self.turn += 1

        # Update points from the previous round (if any)
        if self.turn > 1 and self.last_move is not None and opponent_history:
            p_self, p_opp = payoff(self.last_move, opponent_history[-1])
            self.self_points += int(p_self)
            self.opp_points += int(p_opp)

        if self.turn == 1:
            move: Move = "cooperate"
        else:
            if self.self_points > self.opp_points:
                move = "cooperate"
            elif self.self_points < self.opp_points:
                move = "defect"
            else:
                move = "cooperate"

        self.last_move = move
        return move


class BadDivorce(Strategy):
    """
    Starts defecting, then defects every turn except ONE random cooperation.

    If there are N rounds, it defects N-1 times and cooperates on one random round
    (never the first round).
    """

    def __init__(self, name: str):
        super().__init__(name)
        self.turn = 0
        self.total_rounds: int | None = None
        self.coop_turn: int | None = None

    def set_total_rounds(self, total_rounds: int) -> None:
        self.total_rounds = int(total_rounds)
        n = int(self.total_rounds or 0)
        if n <= 1:
            self.coop_turn = None
        else:
            self.coop_turn = random.randint(2, n)

    def play(self, opponent_history: list[Move]) -> Move:
        self.turn += 1
        if self.coop_turn is None and self.total_rounds is None:
            # Horizon unknown: pick a default one-time cooperate moment.
            self.coop_turn = random.randint(2, 10)
        return "cooperate" if (self.coop_turn is not None and self.turn == self.coop_turn) else "defect"


class RandomStranger(Strategy):
    """
    Chooses randomly for all but the last turn of the match, then defects on the last turn.
    """

    def __init__(self, name: str):
        super().__init__(name)
        self.turn = 0
        self.total_rounds: int | None = None

    def set_total_rounds(self, total_rounds: int) -> None:
        self.total_rounds = int(total_rounds)

    def play(self, opponent_history: list[Move]) -> Move:
        self.turn += 1
        if self.total_rounds and self.turn >= int(self.total_rounds):
            return "defect"
        return "cooperate" if random.random() < 0.5 else "defect"


class MarkedMan(Strategy):
    """
    Defects 90% of the time, cooperates 10% of the time (stochastic).
    """

    def play(self, opponent_history: list[Move]) -> Move:
        return "defect" if random.random() < 0.9 else "cooperate"


class Shootout(Strategy):
    """
    Cooperates on the first turn, then defects every other turn (even turns).
    """

    def __init__(self, name: str):
        super().__init__(name)
        self.turn = 0

    def play(self, opponent_history: list[Move]) -> Move:
        self.turn += 1
        if self.turn == 1:
            return "cooperate"
        return "defect" if (self.turn % 2 == 0) else "cooperate"


class ParkBus(Strategy):
    """
    "Park the bus": defects until it gets ahead on cumulative points, then cooperates forever.
    """

    def __init__(self, name: str):
        super().__init__(name)
        self.turn = 0
        self.self_points = 0
        self.opp_points = 0
        self.last_move: Move | None = None
        self.parked = False

    def play(self, opponent_history: list[Move]) -> Move:
        self.turn += 1
        if self.turn > 1 and self.last_move in ("cooperate", "defect") and opponent_history:
            p_self, p_opp = payoff(self.last_move, opponent_history[-1])
            self.self_points += int(p_self)
            self.opp_points += int(p_opp)

        if not self.parked and self.self_points > self.opp_points:
            self.parked = True

        move: Move = "cooperate" if self.parked else "defect"
        self.last_move = move
        return move


class Illuminati(Strategy):
    """
    Intentionally a "black box" strategy (not explained to end users).

    Goal: be robust vs defectors, stable vs reciprocators, and opportunistic vs naive cooperators.
    """

    def __init__(self, name: str):
        super().__init__(name)
        self.turn = 0
        self.self_points = 0
        self.opp_points = 0
        self.last_move: Move | None = None
        self.retaliate_left = 0
        self.mode: str | None = None  # "exploit" | "shield" | None

    def play(self, opponent_history: list[Move]) -> Move:
        self.turn += 1

        if self.turn > 1 and self.last_move in ("cooperate", "defect") and opponent_history:
            p_self, p_opp = payoff(self.last_move, opponent_history[-1])
            self.self_points += int(p_self)
            self.opp_points += int(p_opp)

        if self.turn == 1:
            self.last_move = "cooperate"
            return "cooperate"

        opp_defects = opponent_history.count("defect")
        opp_rate = opp_defects / float(len(opponent_history) or 1)

        if self.mode is None and len(opponent_history) >= 6:
            if opp_defects == 0:
                self.mode = "exploit"
            elif opp_rate >= 0.6:
                self.mode = "shield"

        if self.mode in {"exploit", "shield"}:
            self.last_move = "defect"
            return "defect"

        if (self.opp_points - self.self_points) >= 6:
            self.retaliate_left = max(self.retaliate_left, 2)

        if opponent_history and opponent_history[-1] == "defect":
            self.retaliate_left = max(self.retaliate_left, 2)

        if self.retaliate_left > 0:
            self.retaliate_left -= 1
            self.last_move = "defect"
            return "defect"

        move: Move = "cooperate" if (opponent_history and opponent_history[-1] == "cooperate") else "defect"
        self.last_move = move
        return move


def make_strategy_factories() -> list[Callable[[], Strategy]]:
    return [
        lambda: MrNiceGuy("MrNiceGuy"),
        lambda: BadCop("BadCop"),
        lambda: TitForTat("TitForTat"),
        lambda: WinStayLoseShift("WinStayLoseShift"),
        lambda: TitForTwoTats("TitForTwoTats"),
        lambda: SuspiciousTitForTat("SuspiciousTitForTat"),
        lambda: GenerousTitForTat("GenerousTitForTat"),
        lambda: Joss("Joss"),
        lambda: Prober("Prober"),
        lambda: ImSoRandom("ImSoRandom"),
        lambda: CalculatedDefector("CalculatedDefector"),
        lambda: HoldingAGrudge("HoldingAGrudge"),
        lambda: ForgiveButDontForget("ForgiveButDontForget"),
        lambda: BadAlternator("BadAlternator"),
        lambda: RitualDefection("RitualDefection"),
        lambda: TripleThreat("TripleThreat"),
        lambda: Pushover("Pushover"),
        lambda: Thief("Thief"),
        lambda: Pattern("Pattern"),
        lambda: NeverSwitchUp("NeverSwitchUp"),
        lambda: RandomPrime("RandomPrime"),
        lambda: Fibonacci("Fibonacci"),
        lambda: DefectiveFriedman("DefectiveFriedman"),
        lambda: CooperativeProth("CooperativeProth"),
        lambda: LongTermRelationship("LongTermRelationship"),
        lambda: Parrot("Parrot"),
        lambda: OneStepBehind("OneStepBehind"),
        lambda: FriendlySquare("FriendlySquare"),
        lambda: LosingMyMind("LosingMyMind"),
        lambda: KeepingPeace("KeepingPeace"),
        lambda: BadJudgeOfCharacter("BadJudgeOfCharacter"),
        lambda: DefectiveDeputy("DefectiveDeputy"),
        lambda: PastTrauma("PastTrauma"),
        lambda: BadDivorce("BadDivorce"),
        lambda: RandomStranger("RandomStranger"),
        lambda: MarkedMan("MarkedMan"),
        lambda: Lottery("Lottery"),
        lambda: Shootout("Shootout"),
        lambda: ParkBus("ParkBus"),
        lambda: Illuminati("Illuminati"),
    ]


def list_strategy_names() -> list[str]:
    return [factory().name for factory in make_strategy_factories()]


def simulate_tournament(
    *,
    rounds_per_match: int = 10,
    repetitions: int = 30,
    seed: Optional[int] = 0,
    horizon_known: bool = True,
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

    def _maybe_set_total_rounds(s: Strategy) -> None:
        if not bool(horizon_known):
            return
        setter = getattr(s, "set_total_rounds", None)
        if callable(setter):
            try:
                setter(rounds_per_match)
            except Exception:
                # Never let optional metadata crash simulation.
                pass

    for rep in range(repetitions):
        for i in range(len(factories)):
            for j in range(i + 1, len(factories)):
                s1 = factories[i]()
                s2 = factories[j]()
                _maybe_set_total_rounds(s1)
                _maybe_set_total_rounds(s2)

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
    name = _canonical_strategy_name(name)
    if name == "WinStayLoseShift":
        return {"last_move": "cooperate"}
    if name == "Prober":
        return {"turn": 0, "mode": None}
    if name in {"BadAlternator", "RitualDefection", "TripleThreat", "Pushover", "Thief", "Pattern"}:
        return {"turn": 0}
    if name == "NeverSwitchUp":
        return {"choice": None}
    if name in {"RandomPrime", "DefectiveFriedman", "CooperativeProth"}:
        return {"turn": 0}
    if name == "Fibonacci":
        return {"turn": 0, "base": None}
    if name == "LongTermRelationship":
        return {"turn": 0, "self_coop": 0}
    if name in {"Parrot", "OneStepBehind", "FriendlySquare", "Shootout"}:
        return {"turn": 0}
    if name == "LosingMyMind":
        return {"turn": 0}
    if name == "KeepingPeace":
        return {"turn": 0, "self_points": 0, "opp_points": 0, "last_move": None}
    if name == "BadJudgeOfCharacter":
        return {"turn": 0, "mode": None}
    if name == "DefectiveDeputy":
        return {"turn": 0}
    if name == "PastTrauma":
        return {"turn": 0, "opp_defects": 0, "seen": 0}
    if name == "BadDivorce":
        return {"turn": 0, "match_total_rounds": None, "coop_turn": None}
    if name == "RandomStranger":
        return {"turn": 0, "match_total_rounds": None}
    if name == "MarkedMan":
        return {"turn": 0}
    if name == "Lottery":
        return {"turn": 0, "match_total_rounds": None}
    if name == "ParkBus":
        return {"turn": 0, "self_points": 0, "opp_points": 0, "last_move": None, "parked": False}
    if name == "Illuminati":
        return {"turn": 0, "self_points": 0, "opp_points": 0, "last_move": None, "retaliate_left": 0, "mode": None}
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

    name = _canonical_strategy_name(name)

    if name == "MrNiceGuy":
        return "cooperate", state, rng_state
    if name == "BadCop":
        return "defect", state, rng_state
    if name == "TitForTat":
        return ("cooperate" if not opponent_history else opponent_history[-1]), state, rng_state
    if name == "SuspiciousTitForTat":
        return ("defect" if not opponent_history else opponent_history[-1]), state, rng_state
    if name == "TitForTwoTats":
        if len(opponent_history) >= 2 and opponent_history[-1] == "defect" and opponent_history[-2] == "defect":
            return "defect", state, rng_state
        return "cooperate", state, rng_state
    if name == "WinStayLoseShift":
        last_move = state.get("last_move", "cooperate")
        if last_move not in ("cooperate", "defect"):
            last_move = "cooperate"
        if not opponent_history:
            return "cooperate", {**state, "last_move": "cooperate"}, rng_state
        prev_payoff, _ = payoff(last_move, opponent_history[-1])
        if prev_payoff in (3, 5):
            move = last_move
        else:
            move = "defect" if last_move == "cooperate" else "cooperate"
        return move, {**state, "last_move": move}, rng_state
    if name == "GenerousTitForTat":
        forgive_prob = 0.1
        if not opponent_history:
            return "cooperate", state, rng_state
        if opponent_history[-1] == "defect":
            x, rng_state = _lcg_float01(rng_state)
            return ("cooperate" if x < forgive_prob else "defect"), state, rng_state
        return "cooperate", state, rng_state
    if name == "Joss":
        defect_prob = 0.1
        if not opponent_history:
            return "cooperate", state, rng_state
        if opponent_history[-1] == "defect":
            return "defect", state, rng_state
        x, rng_state = _lcg_float01(rng_state)
        return ("defect" if x < defect_prob else "cooperate"), state, rng_state
    if name == "Prober":
        turn = int(state.get("turn", 0)) + 1
        mode = state.get("mode")
        state = {**state, "turn": turn, "mode": mode}
        if turn == 1:
            return "defect", state, rng_state
        if turn in (2, 3):
            return "cooperate", state, rng_state
        if mode not in ("tft", "exploit"):
            mode = "tft" if ("defect" in opponent_history) else "exploit"
            state = {**state, "mode": mode}
        if mode == "tft":
            return ("cooperate" if not opponent_history else opponent_history[-1]), state, rng_state
        return "defect", state, rng_state
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

    if name == "Pushover":
        turn = int(state.get("turn", 0)) + 1
        state = {**state, "turn": turn}
        if turn <= 4:
            return ("cooperate" if not opponent_history else opponent_history[-1]), state, rng_state
        return "cooperate", state, rng_state

    if name == "Thief":
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

    if name == "RandomPrime":
        turn = int(state.get("turn", 0)) + 1
        state = {**state, "turn": turn}
        if _is_prime(turn):
            x, rng_state = _lcg_float01(rng_state)
            return ("cooperate" if x < 0.5 else "defect"), state, rng_state
        return "defect", state, rng_state

    if name == "Fibonacci":
        turn = int(state.get("turn", 0)) + 1
        base = state.get("base")
        if base not in ("cooperate", "defect"):
            x, rng_state = _lcg_float01(rng_state)
            base = "cooperate" if x < 0.5 else "defect"
        state = {**state, "turn": turn, "base": base}
        if _is_fibonacci(turn):
            return base, state, rng_state
        return ("defect" if base == "cooperate" else "cooperate"), state, rng_state

    if name == "DefectiveFriedman":
        turn = int(state.get("turn", 0)) + 1
        state = {**state, "turn": turn}
        return ("defect" if _is_friedman(turn) else "cooperate"), state, rng_state

    if name == "CooperativeProth":
        turn = int(state.get("turn", 0)) + 1
        state = {**state, "turn": turn}
        return ("cooperate" if _is_proth(turn) else "defect"), state, rng_state

    if name == "LongTermRelationship":
        turn = int(state.get("turn", 0)) + 1
        self_coop = int(state.get("self_coop", 0))

        # First move: cooperate
        if turn == 1:
            return "cooperate", {**state, "turn": 1, "self_coop": self_coop + 1}, rng_state

        rounds_played = len(opponent_history)
        opp_coop = opponent_history.count("cooperate")
        denom = 2 * rounds_played if rounds_played > 0 else 1
        combined_rate = (self_coop + opp_coop) / float(denom)

        if combined_rate >= (2.0 / 3.0):
            move: Move = "cooperate"
        elif combined_rate < (1.0 / 3.0):
            move = "defect"
        else:
            x, rng_state = _lcg_float01(rng_state)
            move = "cooperate" if x < 0.5 else "defect"

        self_coop += 1 if move == "cooperate" else 0
        return move, {**state, "turn": turn, "self_coop": self_coop}, rng_state

    if name == "Parrot":
        turn = int(state.get("turn", 0)) + 1
        state = {**state, "turn": turn}
        if turn == 1:
            x, rng_state = _lcg_float01(rng_state)
            return ("cooperate" if x < 0.5 else "defect"), state, rng_state
        pos = (turn - 2) % 6  # 0..5
        if pos <= 4:
            return ("cooperate" if not opponent_history else opponent_history[-1]), state, rng_state
        x, rng_state = _lcg_float01(rng_state)
        return ("cooperate" if x < 0.5 else "defect"), state, rng_state

    if name == "OneStepBehind":
        turn = int(state.get("turn", 0)) + 1
        state = {**state, "turn": turn}
        if turn == 1:
            x, rng_state = _lcg_float01(rng_state)
            return ("cooperate" if x < 0.5 else "defect"), state, rng_state
        if not opponent_history:
            return "cooperate", state, rng_state
        return ("defect" if opponent_history[-1] == "cooperate" else "cooperate"), state, rng_state

    if name == "FriendlySquare":
        turn = int(state.get("turn", 0)) + 1
        state = {**state, "turn": turn}
        if _is_square(turn):
            return "cooperate", state, rng_state
        x, rng_state = _lcg_float01(rng_state)
        return ("cooperate" if x < 0.5 else "defect"), state, rng_state

    if name == "LosingMyMind":
        turn = int(state.get("turn", 0)) + 1
        # Linear ramp: by turn 21 it's fully random (p = 1.0).
        p = min(1.0, max(0.0, (turn - 1) * 0.05))
        state = {**state, "turn": turn}
        if p <= 0:
            return "cooperate", state, rng_state
        x, rng_state = _lcg_float01(rng_state)
        if x < p:
            y, rng_state = _lcg_float01(rng_state)
            return ("cooperate" if y < 0.5 else "defect"), state, rng_state
        return "cooperate", state, rng_state

    if name == "KeepingPeace":
        turn = int(state.get("turn", 0)) + 1
        self_points = int(state.get("self_points", 0))
        opp_points = int(state.get("opp_points", 0))
        last_move = state.get("last_move")

        # Update points from previous round (if any)
        if turn > 1 and last_move in ("cooperate", "defect") and opponent_history:
            p_self, p_opp = payoff(last_move, opponent_history[-1])
            self_points += int(p_self)
            opp_points += int(p_opp)

        if turn == 1:
            move: Move = "cooperate"
        else:
            if self_points > opp_points:
                move = "cooperate"
            elif self_points < opp_points:
                move = "defect"
            else:
                move = "cooperate"

        return (
            move,
            {
                **state,
                "turn": turn,
                "self_points": self_points,
                "opp_points": opp_points,
                "last_move": move,
            },
            rng_state,
        )

    if name == "BadJudgeOfCharacter":
        turn = int(state.get("turn", 0)) + 1
        mode = state.get("mode")
        state = {**state, "turn": turn, "mode": mode}
        if turn <= 3:
            return "defect", state, rng_state
        if mode not in ("always_defect", "random"):
            first3 = opponent_history[:3]
            d = first3.count("defect")
            c = first3.count("cooperate")
            mode = "always_defect" if d > c else "random"
            state = {**state, "mode": mode}
        if mode == "always_defect":
            return "defect", state, rng_state
        x, rng_state = _lcg_float01(rng_state)
        return ("cooperate" if x < 0.5 else "defect"), state, rng_state

    if name == "DefectiveDeputy":
        turn = int(state.get("turn", 0)) + 1
        state = {**state, "turn": turn}
        p_defect = min(1.0, 0.65 + 0.02 * (turn - 1))
        x, rng_state = _lcg_float01(rng_state)
        return ("defect" if x < p_defect else "cooperate"), state, rng_state

    if name == "PastTrauma":
        turn = int(state.get("turn", 0)) + 1
        seen = int(state.get("seen", 0))
        opp_defects = int(state.get("opp_defects", 0))
        # Update defect count incrementally
        if len(opponent_history) > seen:
            for mv in opponent_history[seen:]:
                if mv == "defect":
                    opp_defects += 1
            seen = len(opponent_history)
        state = {**state, "turn": turn, "seen": seen, "opp_defects": opp_defects}
        return ("defect" if opp_defects >= 3 else "cooperate"), state, rng_state

    if name == "BadDivorce":
        turn = int(state.get("turn", 0)) + 1
        mt = state.get("match_total_rounds")
        n = int(mt) if isinstance(mt, int) else 0
        coop_turn = state.get("coop_turn")
        if coop_turn is None:
            # If horizon known: choose from [2..n]. Otherwise: default window [2..10].
            if n >= 2:
                x, rng_state = _lcg_float01(rng_state)
                coop_turn = 2 + int(x * (n - 1))
            else:
                x, rng_state = _lcg_float01(rng_state)
                coop_turn = 2 + int(x * 9)  # 2..10
        state = {**state, "turn": turn, "coop_turn": int(coop_turn), "match_total_rounds": (n if n > 0 else None)}
        return ("cooperate" if turn == int(coop_turn) else "defect"), state, rng_state

    if name == "RandomStranger":
        turn = int(state.get("turn", 0)) + 1
        mt = state.get("match_total_rounds")
        n = int(mt) if isinstance(mt, int) else 0
        state = {**state, "turn": turn, "match_total_rounds": (n if n > 0 else None)}
        if n > 0 and turn >= n:
            return "defect", state, rng_state
        x, rng_state = _lcg_float01(rng_state)
        return ("cooperate" if x < 0.5 else "defect"), state, rng_state

    if name == "MarkedMan":
        turn = int(state.get("turn", 0)) + 1
        state = {**state, "turn": turn}
        x, rng_state = _lcg_float01(rng_state)
        return ("defect" if x < 0.9 else "cooperate"), state, rng_state

    if name == "Lottery":
        turn = int(state.get("turn", 0)) + 1
        mt = state.get("match_total_rounds")
        n = int(mt) if isinstance(mt, int) else 0
        state = {**state, "turn": turn, "match_total_rounds": (n if n > 0 else None)}
        if n > 0 and turn >= n:
            x, rng_state = _lcg_float01(rng_state)
            return ("cooperate" if x < 0.5 else "defect"), state, rng_state
        return "defect", state, rng_state

    if name == "Shootout":
        turn = int(state.get("turn", 0)) + 1
        state = {**state, "turn": turn}
        if turn == 1:
            return "cooperate", state, rng_state
        return ("defect" if (turn % 2 == 0) else "cooperate"), state, rng_state

    if name == "ParkBus":
        turn = int(state.get("turn", 0)) + 1
        self_points = int(state.get("self_points", 0))
        opp_points = int(state.get("opp_points", 0))
        last_move = state.get("last_move")
        parked = bool(state.get("parked", False))
        # Update points from previous round (if any)
        if turn > 1 and last_move in ("cooperate", "defect") and opponent_history:
            p_self, p_opp = payoff(last_move, opponent_history[-1])
            self_points += int(p_self)
            opp_points += int(p_opp)
        if not parked and self_points > opp_points:
            parked = True
        move: Move = "cooperate" if parked else "defect"
        return (
            move,
            {**state, "turn": turn, "self_points": self_points, "opp_points": opp_points, "last_move": move, "parked": parked},
            rng_state,
        )

    if name == "Illuminati":
        turn = int(state.get("turn", 0)) + 1
        self_points = int(state.get("self_points", 0))
        opp_points = int(state.get("opp_points", 0))
        last_move = state.get("last_move")
        retaliate_left = int(state.get("retaliate_left", 0))
        mode = state.get("mode")

        if turn > 1 and last_move in ("cooperate", "defect") and opponent_history:
            p_self, p_opp = payoff(last_move, opponent_history[-1])
            self_points += int(p_self)
            opp_points += int(p_opp)

        if turn == 1:
            return "cooperate", {**state, "turn": 1, "self_points": 0, "opp_points": 0, "last_move": "cooperate", "retaliate_left": 0, "mode": None}, rng_state

        opp_defects = opponent_history.count("defect")
        opp_rate = opp_defects / float(len(opponent_history) or 1)

        if mode not in ("exploit", "shield") and len(opponent_history) >= 6:
            if opp_defects == 0:
                mode = "exploit"
            elif opp_rate >= 0.6:
                mode = "shield"

        if mode in ("exploit", "shield"):
            return (
                "defect",
                {**state, "turn": turn, "self_points": self_points, "opp_points": opp_points, "last_move": "defect", "retaliate_left": retaliate_left, "mode": mode},
                rng_state,
            )

        if (opp_points - self_points) >= 6:
            retaliate_left = max(retaliate_left, 2)
        if opponent_history and opponent_history[-1] == "defect":
            retaliate_left = max(retaliate_left, 2)

        if retaliate_left > 0:
            retaliate_left -= 1
            return (
                "defect",
                {**state, "turn": turn, "self_points": self_points, "opp_points": opp_points, "last_move": "defect", "retaliate_left": retaliate_left, "mode": mode},
                rng_state,
            )

        move: Move = "cooperate" if (opponent_history and opponent_history[-1] == "cooperate") else "defect"
        return (
            move,
            {**state, "turn": turn, "self_points": self_points, "opp_points": opp_points, "last_move": move, "retaliate_left": retaliate_left, "mode": mode},
            rng_state,
        )

    # Fallback: treat unknown as always cooperate (safe default)
    return "cooperate", state, rng_state


def init_tournament_state(
    *,
    strategy_names: list[str],
    rounds_per_match: int,
    repetitions: int,
    seed: int = 0,
    horizon_known: bool = True,
    recent_limit: int = 200,
    timeline_limit: int = 400,
    timeline_stride: int = 1,
    custom_strategies: Optional[dict[str, dict]] = None,
) -> dict:
    """
    Create a JSON-serializable state object for incremental tournament stepping.
    """

    names = [_canonical_strategy_name(str(s)) for s in (strategy_names or [])]
    if len(names) < 2:
        raise ValueError("Select at least 2 strategies to run a tournament.")

    rounds_per_match = int(rounds_per_match)
    repetitions = int(repetitions)
    seed = int(seed)
    horizon_known = bool(horizon_known)
    timeline_stride = max(1, int(timeline_stride))
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

    s1_state = _init_strategy_state(s1, custom_strategies.get(s1))
    s2_state = _init_strategy_state(s2, custom_strategies.get(s2))
    # Provide match length only when explicitly enabled (horizon known).
    # Used by endgame-based strategies like BadDivorce / RandomStranger.
    if horizon_known and s1 in {"BadDivorce", "RandomStranger", "Lottery"}:
        s1_state["match_total_rounds"] = rounds_per_match
    if horizon_known and s2 in {"BadDivorce", "RandomStranger", "Lottery"}:
        s2_state["match_total_rounds"] = rounds_per_match

    return {
        "strategy_names": names,
        "rounds_per_match": rounds_per_match,
        "repetitions": repetitions,
        "seed": seed,
        "horizon_known": horizon_known,
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
        "s1_state": s1_state,
        "s2_state": s2_state,
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
        # Capture a snapshot every N matches to keep state compact.
        # (Dash stores pass state back/forth; large timelines hurt performance.)
        "timeline_stride": int(timeline_stride),
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
    # Provide match length to strategies that use it (endgame-based),
    # but only if the horizon is considered "known".
    if bool(state.get("horizon_known", True)):
        rpm = int(state.get("rounds_per_match", 0) or 0)
        if s1 in {"BadDivorce", "RandomStranger", "Lottery"}:
            state["s1_state"]["match_total_rounds"] = rpm
        if s2 in {"BadDivorce", "RandomStranger", "Lottery"}:
            state["s2_state"]["match_total_rounds"] = rpm
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
    timeline_stride = max(1, int(state.get("timeline_stride", 1)))

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

            # Snapshot match-level aggregates for live charts.
            #
            # IMPORTANT: keep snapshots compact. Repeated dict copies (with strategy names as keys)
            # balloon the JSON state and slow Dash callbacks dramatically as strategy count grows.
            md = int(state["matches_done"])
            should_snapshot = (md % timeline_stride == 0)
            if should_snapshot or bool(state.get("done")):
                names: list[str] = state.get("strategy_names", [])
                state["timeline"].append(
                    {
                        "matches_done": md,
                        # Compact: align arrays to `strategy_names` order (names stored once).
                        "totals": [int(state["totals"].get(s, 0)) for s in names],
                        "match_wins": [int(state["match_wins"].get(s, 0)) for s in names],
                        "match_losses": [int(state["match_losses"].get(s, 0)) for s in names],
                        "match_ties": [int(state["match_ties"].get(s, 0)) for s in names],
                    }
                )
                if len(state["timeline"]) > timeline_limit:
                    state["timeline"] = state["timeline"][-timeline_limit:]

            state = _advance_pair(state)

    return state


def init_human_match_state(
    *,
    opponent: str,
    rounds: int = 10,
    seed: int = 0,
    horizon_known: bool = True,
    custom_strategies: Optional[dict[str, dict]] = None,
) -> dict:
    """
    Initialize a human-vs-strategy match state for interactive play.
    """

    opponent = _canonical_strategy_name(str(opponent))
    rounds = int(rounds)
    seed = int(seed)
    horizon_known = bool(horizon_known)
    if rounds <= 0:
        raise ValueError("rounds must be > 0")

    custom_strategies = dict(custom_strategies or {})
    opp_state = _init_strategy_state(opponent, custom_strategies.get(opponent))
    if horizon_known and opponent in {"BadDivorce", "RandomStranger", "Lottery"}:
        opp_state["match_total_rounds"] = rounds
    return {
        "opponent": opponent,
        "rounds": rounds,
        "seed": seed,
        "horizon_known": horizon_known,
        "rng_state": seed & 0xFFFFFFFF,
        "custom_strategies": custom_strategies,
        "opponent_state": opp_state,
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