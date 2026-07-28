"""Canonical presentation metadata and traits for built-in strategies.

This module has no Dash dependency, allowing simulation code and tests to use
strategy metadata without importing the web application.
"""

from __future__ import annotations

STRATEGY_ALIASES: dict[str, str] = {
    "ThePushover": "Pushover",
    "TheThief": "Thief",
    "ParrotPicker": "Parrot",
    "KeepingThePeace": "KeepingPeace",
}

HORIZON_AWARE_STRATEGIES = frozenset({"BadDivorce", "RandomStranger", "Lottery"})


def canonical_strategy_name(name: object) -> str:
    """Return the canonical built-in name for a current or legacy name."""
    normalized = str(name or "")
    return STRATEGY_ALIASES.get(normalized, normalized)


STRATEGY_PROFILES: dict[str, dict[str, str]] = {
    "MrNiceGuy": {
        "description": "Always cooperates. Baseline for measuring how strategies exploit unconditional cooperation.",
        "origin": "Project baseline strategy (unconditional cooperator).",
        "notes": "Scores well against other cooperators, but is heavily exploited by defect-heavy strategies.",
    },
    "BadCop": {
        "description": "Always defects. Baseline for measuring robustness against exploitation.",
        "origin": "Project baseline strategy (unconditional defector).",
        "notes": "Often wins short-term vs cooperators; performs poorly in mutual-defection matchups.",
    },
    "TitForTat": {
        "description": "Cooperates first, then copies the opponent’s previous move.",
        "origin": "Classic strategy popularized by the Axelrod tournaments (submitted by Anatol Rapoport).",
        "notes": "Typically strong in repeated games: nice, retaliatory, forgiving, and clear.",
    },
    "ImSoRandom": {
        "description": "Randomly cooperates or defects each turn (50/50).",
        "origin": "Project baseline strategy (stochastic behavior).",
        "notes": "Useful to test whether strategies handle noise/unpredictability.",
    },
    "CalculatedDefector": {
        "description": "Cooperates unless the opponent defects “too often” (threshold-based).",
        "origin": "Project-defined heuristic.",
        "notes": "Tolerates occasional defection but punishes sustained defection.",
    },
    "HoldingAGrudge": {
        "description": "Cooperates until the opponent defects once, then defects forever.",
        "origin": "Project-defined grudge strategy (grim-trigger style).",
        "notes": "Very punishing; can do well against defectors but can lock into mutual defection after a single defection.",
    },
    "ForgiveButDontForget": {
        "description": "Defects if the opponent’s historical defection rate is high; otherwise cooperates.",
        "origin": "Project-defined forgiveness heuristic.",
        "notes": "More forgiving than a pure grudge; still punishes frequent defectors.",
    },
    "BadAlternator": {
        "description": "Alternates cooperate/defect each turn (C, D, C, D...).",
        "origin": "Project-defined deterministic cycle strategy.",
        "notes": "Can confuse reactive opponents; performance depends strongly on opponent’s response to alternation.",
    },
    "RitualDefection": {
        "description": "Mostly cooperates, but defects on a fixed schedule (every 5th move).",
        "origin": "Project-defined periodic strategy.",
        "notes": "Introduces predictable “ritual” defection; tests opponent retaliation/forgiveness.",
    },
    "TripleThreat": {
        "description": "Defects for a 3-turn block in a 6-turn cycle (CCC DDD repeating).",
        "origin": "Project-defined cyclic strategy.",
        "notes": "Creates sustained defection bursts; can trigger long retaliation cycles in grudge-like opponents.",
    },
    "Pushover": {
        "description": "Starts responsive, then eventually gives in and cooperates regardless of the opponent.",
        "origin": "Project-defined 'softening' strategy.",
        "notes": "Can reduce long retaliation cycles, but risks being exploited late in the match.",
    },
    "Thief": {
        "description": "Builds cooperation early, then shifts behavior later to try to take advantage.",
        "origin": "Project-defined 'phase shift' strategy.",
        "notes": "Useful for studying end-game betrayal and how retaliation-based opponents react.",
    },
    "Pattern": {
        "description": "Repeats a fixed pattern: 3 defects, then 3 cooperates, then repeat (DDD CCC ...).",
        "origin": "Project-defined deterministic pattern strategy.",
        "notes": "Predictable by design; tests whether opponents adapt to periodic behavior. (Intentionally distinct from TripleThreat.)",
    },
    "NeverSwitchUp": {
        "description": "Randomly chooses cooperate or defect once, then sticks with it for the entire match.",
        "origin": "Project-defined commitment strategy (stochastic initialization).",
        "notes": "A controlled way to test 'committed' behavior vs reactive opponents.",
    },
    "WinStayLoseShift": {
        "description": "Repeats its last move if it was rewarded; otherwise switches (Pavlov / WSLS).",
        "origin": "Classic IPD baseline (Pavlov / Win‑Stay, Lose‑Shift).",
        "notes": "Often strong in noisy settings; can quickly return to cooperation after mutual cooperation.",
    },
    "TitForTwoTats": {
        "description": "Cooperates by default; defects only after two consecutive opponent defections.",
        "origin": "Classic forgiving TFT variant (TF2T).",
        "notes": "More forgiving than TFT; less likely to spiral into retaliation after a single defection.",
    },
    "TwoTitsForTat": {
        "description": "Cooperates first, then answers each opponent defection with two rounds of defection.",
        "origin": "Classic retaliatory Tit-for-Tat variant (2TFT).",
        "notes": "More punitive than TFT; a single defection can provoke a longer retaliation cycle.",
    },
    "HardTitForTat": {
        "description": "Defects if the opponent defected at least once during the previous three rounds.",
        "origin": "Classic finite-memory Tit-for-Tat variant (Hard TFT).",
        "notes": "Retains a short memory of recent defections without holding a permanent grudge.",
    },
    "SoftMajority": {
        "description": "Cooperates when the opponent has cooperated at least as often as it has defected.",
        "origin": "Classic majority-based Iterated Prisoner’s Dilemma strategy.",
        "notes": "Cooperative on opening and tied histories; sustained defection eventually changes its response.",
    },
    "HardMajority": {
        "description": "Cooperates only when the opponent has cooperated more often than it has defected.",
        "origin": "Classic majority-based Iterated Prisoner’s Dilemma strategy.",
        "notes": "Defects on opening and tied histories, making it the less trusting majority variant.",
    },
    "Gradual": {
        "description": "Answers repeated defections with increasingly long punishments, followed by two cooperative moves.",
        "origin": "Classic Gradual strategy associated with IPD tournament research.",
        "notes": "Combines proportional punishment with an explicit attempt to restore cooperation.",
    },
    "DebtCollector": {
        "description": "Maintains a trust ledger: defections add two debt tokens, cooperation repays one, and outstanding debt triggers retaliation.",
        "origin": "Project-defined restorative-accounting strategy.",
        "notes": "Unlike a permanent grudge, good behavior can repay the debt; repeated exploitation creates proportionally longer consequences.",
    },
    "PatternHunter": {
        "description": "Learns which move usually follows the opponent’s current move and plays against the predicted next action.",
        "origin": "Project-defined one-step sequence learner.",
        "notes": "Falls back to Tit-for-Tat while learning, then exploits recurring transitions without knowing the opponent’s identity.",
    },
    "EntropyBroker": {
        "description": "Measures recent switching: it cooperates with stable majorities but defects when behavior becomes maximally erratic.",
        "origin": "Project-defined uncertainty-sensitive strategy.",
        "notes": "Treats chaos as risk. Its eight-move window lets an opponent rebuild trust after an unstable phase.",
    },
    "SuspiciousTitForTat": {
        "description": "Defects on the first move, then mirrors the opponent’s previous move.",
        "origin": "Classic TFT variant (STFT).",
        "notes": "A 'hostile start' version of TFT; useful for testing strategies against early aggression.",
    },
    "GenerousTitForTat": {
        "description": "Like TFT, but sometimes forgives defections and cooperates anyway (stochastic).",
        "origin": "Classic TFT variant (GTFT).",
        "notes": "Uses controlled forgiveness to recover cooperation after occasional defections.",
    },
    "Joss": {
        "description": "TFT with occasional random defection ('spite') even after opponent cooperation.",
        "origin": "Classic stochastic variant of TFT (Joss).",
        "notes": "Injects unpredictability; can exploit overly trusting opponents but may reduce cooperation stability.",
    },
    "Prober": {
        "description": "Probes early (D, C, C), then exploits if the opponent never retaliates; otherwise switches to TFT.",
        "origin": "Classic 'tester' strategy (Prober).",
        "notes": "Aims to detect unconditional cooperators; otherwise behaves similarly to TFT.",
    },
    "RandomPrime": {
        "description": "Defects by default; on prime-numbered turns it plays randomly.",
        "origin": "Project-defined novelty strategy (number-based turn schedule).",
        "notes": "A mostly-defect strategy with periodic randomness tied to primes.",
    },
    "Fibonacci": {
        "description": "Starts with a random base choice; plays it on Fibonacci-numbered turns, otherwise plays the opposite.",
        "origin": "Project-defined novelty strategy (Fibonacci turn schedule).",
        "notes": "Creates structured alternation driven by the Fibonacci sequence.",
    },
    "DefectiveFriedman": {
        "description": "Defects on turns whose round number is a Friedman number; otherwise cooperates.",
        "origin": "Project-defined novelty strategy inspired by Friedman numbers.",
        "notes": "Rare, irregular defections determined by a curated Friedman set.",
    },
    "CooperativeProth": {
        "description": "Cooperates on Proth-numbered turns; otherwise defects.",
        "origin": "Project-defined novelty strategy inspired by Proth numbers.",
        "notes": "Structured cooperation tied to a number-theory predicate.",
    },
    "LongTermRelationship": {
        "description": "Cooperates when overall cooperation is high, defects when it’s low, and randomizes in the middle.",
        "origin": "Project-defined relationship-health heuristic.",
        "notes": "Uses overall cooperation rate as a proxy for 'trust' and adapts accordingly.",
    },
    "Parrot": {
        "description": "Starts random, then copies the opponent for 5 turns, then goes random for 1 turn, repeating.",
        "origin": "Project-defined periodic mimic strategy.",
        "notes": "Mostly reactive (copying) with occasional 'reset' randomness.",
    },
    "OneStepBehind": {
        "description": "Starts random, then always plays the opposite of the opponent’s previous move.",
        "origin": "Project-defined anti-mirroring strategy.",
        "notes": "Tries to 'beat' what the opponent did last round; can destabilize cooperation loops.",
    },
    "FriendlySquare": {
        "description": "Cooperates on perfect-square turns (1, 4, 9, 16, …); otherwise plays randomly.",
        "origin": "Project-defined novelty strategy (square-number schedule).",
        "notes": "Mostly random with occasional deterministic cooperation markers.",
    },
    "LosingMyMind": {
        "description": "Starts fully cooperative and becomes increasingly random each turn.",
        "origin": "Project-defined gradual-noise strategy.",
        "notes": "Models 'deteriorating consistency' over time without relying on match-length knowledge.",
    },
    "KeepingPeace": {
        "description": "Starts cooperative and tries to keep the match as close to a tie as possible in points.",
        "origin": "Project-defined 'tie-seeking' heuristic.",
        "notes": "Tracks its own and the opponent’s points and adjusts to reduce score imbalance.",
    },
    "BadJudgeOfCharacter": {
        "description": "Starts defecting; after 3 rounds it either defects forever or randomizes based on early opponent behavior.",
        "origin": "Project-defined early-judgment strategy.",
        "notes": "If the opponent defects more than cooperates in the first 3 rounds, it commits to defecting forever.",
    },
    "DefectiveDeputy": {
        "description": "Defect-leaning strategy that becomes more likely to defect each turn.",
        "origin": "Project-defined ramping defector.",
        "notes": "A gradually-hardening policy that trends toward always defecting.",
    },
    "BadDivorce": {
        "description": "Defects almost every turn, with one surprise cooperation on a random round.",
        "origin": "Project-defined endgame-flavored strategy.",
        "notes": "Defects N−1 times and cooperates once (never on the first move).",
    },
    "RandomStranger": {
        "description": "Mostly random, but defects at the end to try to 'get one over' on the opponent.",
        "origin": "Project-defined endgame betrayal strategy.",
        "notes": "Random for most of the match; defects on the final turn when the horizon is known.",
    },
    "PastTrauma": {
        "description": "Cooperates until the opponent defects 3 total times, then defects forever.",
        "origin": "Project-defined threshold grudge strategy.",
        "notes": "The three defections do not need to be consecutive.",
    },
    "MarkedMan": {
        "description": "Defects about 90% of the time, cooperates about 10% (paranoia breaks occasionally).",
        "origin": "Project-defined stochastic paranoia strategy.",
        "notes": "A fixed-probability mixture policy (mostly defect).",
    },
    "Lottery": {
        "description": "Defects throughout, then plays randomly on the final turn (if the horizon is known).",
        "origin": "Project-defined endgame gamble strategy.",
        "notes": "If match length is unknown, it behaves as an always-defect policy.",
    },
    "Shootout": {
        "description": "Cooperates on the first move, then defects every other turn.",
        "origin": "Project-defined alternating duel strategy.",
        "notes": "Creates a predictable C/D rhythm after the opening cooperate.",
    },
    "ParkBus": {
        "description": "Defects until it gets ahead on points, then cooperates forever.",
        "origin": "Project-defined 'lock-in lead' strategy.",
        "notes": "Attempts to secure an early advantage and then play defensively (cooperate) to preserve it.",
    },
    "Illuminati": {
        "description": "Classified (black box).",
        "origin": "Project-defined hidden strategy.",
        "notes": "This strategy’s logic is intentionally not disclosed in the UI.",
    },
    "ForgetfulGrudger": {
        "description": "Punishes an opponent defection for ten decisions, then returns to cooperation.",
        "origin": "Finite-memory variant of the common grudger strategy.",
        "notes": "Unlike Grim Trigger, an old defection eventually leaves its active memory.",
    },
    "StochasticPavlov": {
        "description": "Uses Win-Stay, Lose-Shift, then flips five percent of its decisions.",
        "origin": "Stochastic variant of Pavlov / Win-Stay, Lose-Shift.",
        "notes": "Tests whether outcome-based adaptation remains stable with decision noise.",
    },
    "Appeaser": {
        "description": "Repeats its own action after cooperation and switches after opponent defection.",
        "origin": "Common stateful IPD response strategy.",
        "notes": "Its attempt to respond to aggression can produce unusual cycles.",
    },
    "Forgiver": {
        "description": "Retaliates once after a defection, then makes a cooperative reconciliation move.",
        "origin": "Deterministic forgiveness strategy.",
        "notes": "Provides a simple comparison with probabilistic and escalating forgiveness rules.",
    },
    "ReactivePlayer": {
        "description": "Cooperates with probability 90% after cooperation and 20% after defection.",
        "origin": "Parameterized reactive R(p,q) strategy family.",
        "notes": "A fixed research preset for comparing general reactive policies.",
    },
    "MemoryOnePlayer": {
        "description": "Conditions cooperation probability on the previous CC, CD, DC, or DD outcome.",
        "origin": "General memory-one strategy family.",
        "notes": "Uses the probability vector (0.9, 0.2, 0.8, 0.1).",
    },
    "ZDExtort2": {
        "description": "Uses an extortionate zero-determinant memory-one probability vector.",
        "origin": "Advanced research strategy based on zero-determinant IPD policies.",
        "notes": "Its theoretical payoff relationship is a long-run property and may be less visible in short or noisy matches.",
    },
    "ZDGenerous2": {
        "description": "Uses a generous zero-determinant memory-one probability vector.",
        "origin": "Advanced research strategy based on generous zero-determinant policies.",
        "notes": "Designed for research comparisons with extortionate and ordinary memory-one strategies.",
    },
    "ZDEqualizer": {
        "description": "Uses an equalizer-style memory-one probability preset.",
        "origin": "Advanced research strategy in the zero-determinant family.",
        "notes": "The profile exposes a fixed preset for finite and noisy tournament experiments.",
    },
    "Handshake": {
        "description": "Emits C-D-C-C, cooperates with recognized copies, and defects otherwise.",
        "origin": "Recognition strategy using only legal IPD actions.",
        "notes": "Most informative when self-play or population evolution is enabled.",
    },
    "AdaptiveBestResponse": {
        "description": "Estimates conditional cooperation and selects the action with greater expected payoff.",
        "origin": "Project-defined interpretable opponent-modeling strategy.",
        "notes": "Uses cooperative priors so its earliest observations do not dominate permanently.",
    },
    "HedgeMetaStrategy": {
        "description": "Combines four simple policies using multiplicative expert weights.",
        "origin": "Project-defined online expert-selection strategy.",
        "notes": "Adapts the influence of cooperation, defection, Tit-for-Tat, and Pavlov recommendations.",
    },
}


def strategy_scorecard(name: str) -> dict[str, object]:
    """
    Static-ish strategy classification for UI badges/scorecards.

    This is intentionally simple (human-readable, not 'perfect' taxonomy).
    """

    nm = canonical_strategy_name(name)

    deterministic = nm in {
        "MrNiceGuy",
        "BadCop",
        "TitForTat",
        "WinStayLoseShift",
        "TitForTwoTats",
        "TwoTitsForTat",
        "HardTitForTat",
        "SoftMajority",
        "HardMajority",
        "Gradual",
        "DebtCollector",
        "PatternHunter",
        "EntropyBroker",
        "SuspiciousTitForTat",
        "Prober",
        "CalculatedDefector",
        "HoldingAGrudge",
        "ForgiveButDontForget",
        "BadAlternator",
        "RitualDefection",
        "TripleThreat",
        "Pushover",
        "Thief",
        "Pattern",
        "DefectiveFriedman",
        "CooperativeProth",
        "KeepingPeace",
        "ParkBus",
        "Shootout",
        "Illuminati",
        "PastTrauma",
        "ForgetfulGrudger",
        "Appeaser",
        "Forgiver",
        "Handshake",
        "AdaptiveBestResponse",
        "HedgeMetaStrategy",
    }

    stochastic = nm in {
        "ImSoRandom",
        "NeverSwitchUp",
        "GenerousTitForTat",
        "Joss",
        "OneStepBehind",
        "RandomPrime",
        "Fibonacci",
        "LongTermRelationship",
        "Parrot",
        "FriendlySquare",
        "LosingMyMind",
        "BadJudgeOfCharacter",
        "DefectiveDeputy",
        "BadDivorce",
        "RandomStranger",
        "MarkedMan",
        "Lottery",
        "StochasticPavlov",
        "ReactivePlayer",
        "MemoryOnePlayer",
        "ZDExtort2",
        "ZDGenerous2",
        "ZDEqualizer",
    }

    # Human-readable memory requirement for profile scorecards.
    if nm in {
        "MrNiceGuy",
        "BadCop",
        "ImSoRandom",
        "BadAlternator",
        "RitualDefection",
        "TripleThreat",
        "Pattern",
        "NeverSwitchUp",
        "RandomPrime",
        "Fibonacci",
        "DefectiveFriedman",
        "CooperativeProth",
        "FriendlySquare",
        "LosingMyMind",
        "DefectiveDeputy",
        "BadDivorce",
        "RandomStranger",
        "MarkedMan",
        "Lottery",
        "Shootout",
    }:
        memory: object = 0
    elif nm in {
        "TitForTat",
        "SuspiciousTitForTat",
        "WinStayLoseShift",
        "GenerousTitForTat",
        "Joss",
        "Pushover",
        "Thief",
        "OneStepBehind",
        "Parrot",
        "StochasticPavlov",
        "Appeaser",
        "Forgiver",
        "ReactivePlayer",
        "MemoryOnePlayer",
        "ZDExtort2",
        "ZDGenerous2",
        "ZDEqualizer",
    }:
        memory = 1
    elif nm in {"TitForTwoTats", "TwoTitsForTat"}:
        memory = 2
    elif nm == "HardTitForTat":
        memory = 3
    elif nm in {
        "HoldingAGrudge",
        "CalculatedDefector",
        "ForgiveButDontForget",
        "SoftMajority",
        "HardMajority",
        "Prober",
        "LongTermRelationship",
        "BadJudgeOfCharacter",
        "PastTrauma",
        "PatternHunter",
    }:
        memory = "full history"
    elif nm == "Handshake":
        memory = 4
    elif nm in {"Gradual", "DebtCollector", "EntropyBroker", "KeepingPeace", "ParkBus", "Illuminati", "ForgetfulGrudger", "AdaptiveBestResponse", "HedgeMetaStrategy"}:
        memory = "stateful"
    else:
        memory = "unknown"

    # Primary tendency (very rough)
    primary_coop = nm in {"MrNiceGuy", "TitForTat", "WinStayLoseShift", "TitForTwoTats", "TwoTitsForTat", "HardTitForTat", "SoftMajority", "Gradual", "GenerousTitForTat", "Pushover", "KeepingPeace", "ForgetfulGrudger", "Forgiver", "ZDGenerous2"}
    primary_defect = nm in {"BadCop", "HardMajority", "CalculatedDefector", "HoldingAGrudge", "Thief", "OneStepBehind", "DefectiveDeputy", "BadDivorce", "ParkBus"}

    # Uses turn counter / schedule
    time_based = nm in {
        "BadAlternator",
        "RitualDefection",
        "TripleThreat",
        "Pattern",
        "NeverSwitchUp",
        "RandomPrime",
        "Fibonacci",
        "DefectiveFriedman",
        "CooperativeProth",
        "Parrot",
        "FriendlySquare",
        "LosingMyMind",
        "Shootout",
        "BadDivorce",
        "RandomStranger",
        "Lottery",
    }

    reactive = nm in {
        "TitForTat",
        "SuspiciousTitForTat",
        "TitForTwoTats",
        "TwoTitsForTat",
        "HardTitForTat",
        "SoftMajority",
        "HardMajority",
        "Gradual",
        "WinStayLoseShift",
        "Joss",
        "Prober",
        "CalculatedDefector",
        "HoldingAGrudge",
        "ForgiveButDontForget",
        "Pushover",
        "Parrot",
        "OneStepBehind",
        "LongTermRelationship",
        "KeepingPeace",
        "ParkBus",
        "Illuminati",
        "PastTrauma",
        "DebtCollector",
        "PatternHunter",
        "EntropyBroker",
        "ForgetfulGrudger",
        "StochasticPavlov",
        "Appeaser",
        "Forgiver",
        "ReactivePlayer",
        "MemoryOnePlayer",
        "ZDExtort2",
        "ZDGenerous2",
        "ZDEqualizer",
        "Handshake",
        "AdaptiveBestResponse",
        "HedgeMetaStrategy",
    }

    return {
        "deterministic": bool(deterministic and not stochastic),
        "stochastic": bool(stochastic),
        "memory": memory,
        "primarily_cooperative": bool(primary_coop),
        "primarily_defective": bool(primary_defect),
        "reactive": bool(reactive),
        "time_based": bool(time_based),
        "horizon_aware": nm in HORIZON_AWARE_STRATEGIES,
    }
