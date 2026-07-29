"""Population evolution for built-in Prisoner's Dilemma strategies."""

from __future__ import annotations

import math
import random
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass

import pandas as pd

try:
    from .workloads import tournament_workload
except ImportError:
    from workloads import tournament_workload  # type: ignore


MAX_EVOLUTION_STRATEGIES = 8
MAX_EVOLUTION_GENERATIONS = 500
MAX_EVOLUTION_WORKLOAD = 250_000
MAX_MORAN_EVENTS = 500_000
EVOLUTION_RESULT_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class EvolutionConfig:
    strategy_names: tuple[str, ...]
    initial_shares: dict[str, float]
    generations: int = 100
    rounds_per_match: int = 20
    repetitions: int = 3
    seed: int = 0
    selection_strength: float = 1.0
    mutation_rate: float = 0.0
    execution_error_rate: float = 0.0
    horizon_known: bool = False
    model: str = "replicator"
    population_size: int = 100

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def normalize_evolution_config(config: EvolutionConfig) -> EvolutionConfig:
    """Validate a configuration and return normalized population shares."""
    names = tuple(str(name) for name in config.strategy_names)
    shares = normalize_population(names, config.initial_shares)
    generations = int(config.generations)
    rounds = int(config.rounds_per_match)
    repetitions = int(config.repetitions)
    selection = float(config.selection_strength)
    mutation = float(config.mutation_rate)
    error_rate = float(config.execution_error_rate)
    model = str(config.model)
    population_size = int(config.population_size)
    if not 1 <= generations <= MAX_EVOLUTION_GENERATIONS:
        raise ValueError(f"generations must be between 1 and {MAX_EVOLUTION_GENERATIONS}")
    if not 1 <= rounds <= 100:
        raise ValueError("rounds_per_match must be between 1 and 100")
    if not 1 <= repetitions <= 30:
        raise ValueError("repetitions must be between 1 and 30")
    if not math.isfinite(selection) or not 0 <= selection <= 1:
        raise ValueError("selection_strength must be between 0 and 1")
    if not math.isfinite(mutation) or not 0 <= mutation <= 0.25:
        raise ValueError("mutation_rate must be between 0 and 0.25")
    if not math.isfinite(error_rate) or not 0 <= error_rate <= 1:
        raise ValueError("execution_error_rate must be between 0 and 1")
    if model not in {"replicator", "moran"}:
        raise ValueError("model must be replicator or moran")
    if not 10 <= population_size <= 10_000:
        raise ValueError("population_size must be between 10 and 10000")
    if model == "moran" and population_size * generations > MAX_MORAN_EVENTS:
        raise ValueError(f"Moran process must not exceed {MAX_MORAN_EVENTS} birth-death events")
    workload = tournament_workload(len(names), rounds, repetitions, include_self_play=True)
    if workload > MAX_EVOLUTION_WORKLOAD:
        raise ValueError(f"evolution matchup workload must not exceed {MAX_EVOLUTION_WORKLOAD} rounds")
    return EvolutionConfig(
        strategy_names=names,
        initial_shares=shares,
        generations=generations,
        rounds_per_match=rounds,
        repetitions=repetitions,
        seed=int(config.seed),
        selection_strength=selection,
        mutation_rate=mutation,
        execution_error_rate=error_rate,
        horizon_known=bool(config.horizon_known),
        model=model,
        population_size=population_size,
    )


def normalize_population(strategy_names: Sequence[str], initial_shares: Mapping[str, float] | None = None) -> dict[str, float]:
    """Validate and normalize population shares."""
    names = [str(name) for name in strategy_names]
    if len(names) < 2:
        raise ValueError("Select at least 2 strategies for evolution.")
    if len(names) > MAX_EVOLUTION_STRATEGIES:
        raise ValueError(f"Evolution supports at most {MAX_EVOLUTION_STRATEGIES} strategies.")
    if len(names) != len(set(names)):
        raise ValueError("Evolution strategy names must be unique.")
    if initial_shares is None:
        return {name: 1.0 / len(names) for name in names}
    unknown = set(initial_shares) - set(names)
    if unknown:
        raise ValueError(f"Initial shares contain unknown strategies: {', '.join(sorted(unknown))}")
    shares = {}
    for name in names:
        value = float(initial_shares.get(name, 0.0))
        if not math.isfinite(value) or value < 0:
            raise ValueError("Initial shares must be finite and non-negative.")
        shares[name] = value
    total = sum(shares.values())
    if total <= 0:
        raise ValueError("Initial shares must have a positive total.")
    return {name: value / total for name, value in shares.items()}


def matchup_statistics(results: pd.DataFrame, strategy_names: Sequence[str]) -> dict[tuple[str, str], dict[str, float]]:
    """Calculate directional payoff and cooperation rates for selected matchups."""
    names = set(strategy_names)
    stats: dict[tuple[str, str], dict[str, float]] = {}
    if results.empty:
        return stats
    selected = results[results["strategy_1"].isin(names) & results["strategy_2"].isin(names)]
    perspectives = pd.concat(
        [
            selected.rename(columns={"strategy_1": "strategy", "strategy_2": "opponent", "points_1": "payoff", "move_1": "move"})[
                ["strategy", "opponent", "payoff", "move"]
            ],
            selected.rename(columns={"strategy_2": "strategy", "strategy_1": "opponent", "points_2": "payoff", "move_2": "move"})[
                ["strategy", "opponent", "payoff", "move"]
            ],
        ],
        ignore_index=True,
    )
    grouped = perspectives.groupby(["strategy", "opponent"], as_index=False).agg(
        payoff=("payoff", "mean"),
        cooperation=("move", lambda moves: float((moves == "cooperate").mean())),
    )
    for row in grouped.itertuples(index=False):
        stats[(str(row.strategy), str(row.opponent))] = {
            "payoff": float(row.payoff),
            "cooperation": float(row.cooperation),
        }
    missing = [(a, b) for a in names for b in names if (a, b) not in stats]
    if missing:
        raise ValueError(f"Missing matchup statistics for {len(missing)} ordered pair(s).")
    return stats


def evolve_population(
    strategy_names: Sequence[str],
    matchup_stats: Mapping[tuple[str, str], Mapping[str, float]],
    *,
    generations: int,
    initial_shares: Mapping[str, float] | None = None,
    selection_strength: float = 1.0,
    mutation_rate: float = 0.0,
) -> list[dict[str, object]]:
    """Run deterministic discrete replicator dynamics with optional mutation."""
    generations = int(generations)
    selection_strength = float(selection_strength)
    mutation_rate = float(mutation_rate)
    if not 1 <= generations <= MAX_EVOLUTION_GENERATIONS:
        raise ValueError(f"generations must be between 1 and {MAX_EVOLUTION_GENERATIONS}")
    if not math.isfinite(selection_strength) or not 0.0 <= selection_strength <= 1.0:
        raise ValueError("selection_strength must be between 0 and 1")
    if not math.isfinite(mutation_rate) or not 0.0 <= mutation_rate <= 0.25:
        raise ValueError("mutation_rate must be between 0 and 0.25")

    names = [str(name) for name in strategy_names]
    shares = normalize_population(names, initial_shares)
    history: list[dict[str, object]] = []

    def snapshot(generation: int, fitness: Mapping[str, float]) -> dict[str, object]:
        average_payoff = sum(shares[name] * fitness[name] for name in names)
        cooperation_rate = sum(
            shares[a] * shares[b] * float(matchup_stats[(a, b)]["cooperation"])
            for a in names
            for b in names
        )
        diversity = -sum(value * math.log(value) for value in shares.values() if value > 0)
        maximum_diversity = math.log(len(names))
        return {
            "generation": generation,
            "shares": dict(shares),
            "fitness": dict(fitness),
            "average_payoff": average_payoff,
            "cooperation_rate": cooperation_rate,
            "diversity": diversity / maximum_diversity if maximum_diversity else 0.0,
        }

    for generation in range(generations + 1):
        fitness = {
            name: sum(shares[opponent] * float(matchup_stats[(name, opponent)]["payoff"]) for opponent in names)
            for name in names
        }
        history.append(snapshot(generation, fitness))
        if generation == generations:
            break
        average = sum(shares[name] * fitness[name] for name in names)
        if average > 0:
            updated = {
                name: shares[name] * ((1.0 - selection_strength) + selection_strength * fitness[name] / average)
                for name in names
            }
        else:
            updated = dict(shares)
        total = sum(updated.values())
        updated = {name: value / total for name, value in updated.items()}
        if mutation_rate > 0:
            uniform = 1.0 / len(names)
            updated = {name: (1.0 - mutation_rate) * value + mutation_rate * uniform for name, value in updated.items()}
        shares = updated
    return history


def evolve_moran_population(
    strategy_names: Sequence[str],
    matchup_stats: Mapping[tuple[str, str], Mapping[str, float]],
    *,
    generations: int,
    population_size: int,
    initial_shares: Mapping[str, float] | None = None,
    selection_strength: float = 1.0,
    mutation_rate: float = 0.0,
    seed: int = 0,
) -> list[dict[str, object]]:
    """Run a seeded finite-population Moran birth-death process."""
    generations = int(generations)
    population_size = int(population_size)
    selection_strength = float(selection_strength)
    mutation_rate = float(mutation_rate)
    if not 1 <= generations <= MAX_EVOLUTION_GENERATIONS:
        raise ValueError(f"generations must be between 1 and {MAX_EVOLUTION_GENERATIONS}")
    if not 10 <= population_size <= 10_000:
        raise ValueError("population_size must be between 10 and 10000")
    if population_size * generations > MAX_MORAN_EVENTS:
        raise ValueError(f"Moran process must not exceed {MAX_MORAN_EVENTS} birth-death events")
    if not math.isfinite(selection_strength) or not 0 <= selection_strength <= 1:
        raise ValueError("selection_strength must be between 0 and 1")
    if not math.isfinite(mutation_rate) or not 0 <= mutation_rate <= 0.25:
        raise ValueError("mutation_rate must be between 0 and 0.25")

    names = [str(name) for name in strategy_names]
    shares = normalize_population(names, initial_shares)
    raw_counts = {name: shares[name] * population_size for name in names}
    counts = {name: int(math.floor(raw_counts[name])) for name in names}
    remainder = population_size - sum(counts.values())
    order = sorted(names, key=lambda name: (raw_counts[name] - counts[name], name), reverse=True)
    for name in order[:remainder]:
        counts[name] += 1
    rng = random.Random(int(seed))
    history: list[dict[str, object]] = []

    def weighted_choice(weights: Mapping[str, float]) -> str:
        total = sum(weights.values())
        target = rng.random() * total
        running = 0.0
        for name in names:
            running += weights[name]
            if target <= running:
                return name
        return names[-1]

    def snapshot(generation: int) -> dict[str, object]:
        current_shares = {name: counts[name] / population_size for name in names}
        fitness = {
            name: sum(current_shares[opponent] * float(matchup_stats[(name, opponent)]["payoff"]) for opponent in names)
            for name in names
        }
        average_payoff = sum(current_shares[name] * fitness[name] for name in names)
        cooperation_rate = sum(
            current_shares[a] * current_shares[b] * float(matchup_stats[(a, b)]["cooperation"])
            for a in names
            for b in names
        )
        diversity = -sum(value * math.log(value) for value in current_shares.values() if value > 0)
        return {
            "generation": generation,
            "shares": current_shares,
            "counts": dict(counts),
            "fitness": fitness,
            "average_payoff": average_payoff,
            "cooperation_rate": cooperation_rate,
            "diversity": diversity / math.log(len(names)),
        }

    for generation in range(generations + 1):
        history.append(snapshot(generation))
        if generation == generations:
            break
        for _event in range(population_size):
            current_shares = {name: counts[name] / population_size for name in names}
            fitness = {
                name: sum(current_shares[opponent] * float(matchup_stats[(name, opponent)]["payoff"]) for opponent in names)
                for name in names
            }
            birth_weights = {
                name: counts[name] * ((1.0 - selection_strength) + selection_strength * float(fitness[name]))
                for name in names
            }
            if sum(birth_weights.values()) <= 0:
                birth_weights = {name: float(counts[name]) for name in names}
            parent = weighted_choice(birth_weights)
            offspring = rng.choice(names) if rng.random() < mutation_rate else parent
            death = weighted_choice({name: float(counts[name]) for name in names})
            counts[death] -= 1
            counts[offspring] += 1
    return history


def evolution_events(history: Sequence[Mapping[str, object]], low_share_threshold: float = 0.01) -> list[dict[str, object]]:
    """Derive deterministic population events from generation history."""
    if not history:
        return []
    events: list[dict[str, object]] = []
    previous = history[0]
    previous_shares = dict(previous.get("shares", {}))
    previous_leader = max(previous_shares, key=previous_shares.get) if previous_shares else None
    for current in history[1:]:
        generation = int(current.get("generation", 0))
        shares = dict(current.get("shares", {}))
        if not shares:
            previous = current
            continue
        leader = max(shares, key=shares.get)
        if leader != previous_leader:
            events.append({"generation": generation, "type": "leader", "message": f"{leader} became the largest population."})
        for name, share in shares.items():
            old_share = float(previous_shares.get(name, 0.0))
            share = float(share)
            if old_share >= low_share_threshold and share < low_share_threshold:
                events.append({"generation": generation, "type": "low_share", "message": f"{name} fell below {low_share_threshold:.0%}."})
            elif old_share < low_share_threshold and share >= low_share_threshold:
                events.append({"generation": generation, "type": "recovery", "message": f"{name} returned to at least {low_share_threshold:.0%}."})
            if old_share < 0.5 <= share:
                events.append({"generation": generation, "type": "majority", "message": f"{name} passed 50% of the population."})
        previous_cooperation = float(previous.get("cooperation_rate", 0.0))
        cooperation = float(current.get("cooperation_rate", 0.0))
        for threshold in (0.25, 0.5, 0.75):
            if previous_cooperation < threshold <= cooperation:
                events.append({"generation": generation, "type": "cooperation", "message": f"Population cooperation passed {threshold:.0%}."})
            elif previous_cooperation >= threshold > cooperation:
                events.append({"generation": generation, "type": "cooperation", "message": f"Population cooperation fell below {threshold:.0%}."})
        previous = current
        previous_shares = shares
        previous_leader = leader
    return events


def build_evolution_result(
    config: EvolutionConfig,
    history: Sequence[Mapping[str, object]],
    matchup_stats: Mapping[tuple[str, str], Mapping[str, float]],
) -> dict[str, object]:
    """Build a portable, versioned evolution result artifact."""
    normalized = normalize_evolution_config(config)
    history_rows = [dict(row) for row in history]
    if not history_rows:
        raise ValueError("evolution history must not be empty")
    if int(history_rows[0].get("generation", -1)) != 0:
        raise ValueError("evolution history must begin at generation 0")
    if int(history_rows[-1].get("generation", -1)) != normalized.generations:
        raise ValueError("evolution history does not match the configured generation count")
    statistics = [
        {
            "strategy": strategy,
            "opponent": opponent,
            "payoff": float(values["payoff"]),
            "cooperation": float(values["cooperation"]),
        }
        for (strategy, opponent), values in sorted(matchup_stats.items())
    ]
    return {
        "schema_version": EVOLUTION_RESULT_SCHEMA_VERSION,
        "experiment_type": "evolution_result",
        "game": "prisoners_dilemma",
        "game_version": 1,
        "format_notice": "Experimental population model. Individual matches use the classic Prisoner's Dilemma actions and payoff matrix.",
        "config": normalized.to_dict(),
        "history": history_rows,
        "events": evolution_events(history_rows),
        "matchup_statistics": statistics,
    }


def validate_evolution_result(payload: object, known_strategies: Sequence[str] | None = None) -> dict[str, object]:
    """Validate a saved evolution result without rerunning its matches."""
    if not isinstance(payload, Mapping):
        raise ValueError("evolution result must be an object")
    if payload.get("schema_version") != EVOLUTION_RESULT_SCHEMA_VERSION:
        raise ValueError("unsupported or missing evolution result schema_version")
    if payload.get("experiment_type") != "evolution_result":
        raise ValueError("experiment_type must be evolution_result")
    if payload.get("game") != "prisoners_dilemma" or payload.get("game_version") != 1:
        raise ValueError("unsupported game or game_version")
    raw_config = payload.get("config")
    if not isinstance(raw_config, Mapping):
        raise ValueError("evolution result config must be an object")
    config = normalize_evolution_config(
        EvolutionConfig(
            strategy_names=tuple(raw_config.get("strategy_names", [])),
            initial_shares=dict(raw_config.get("initial_shares", {})),
            generations=raw_config.get("generations", 100),
            rounds_per_match=raw_config.get("rounds_per_match", 20),
            repetitions=raw_config.get("repetitions", 3),
            seed=raw_config.get("seed", 0),
            selection_strength=raw_config.get("selection_strength", 1.0),
            mutation_rate=raw_config.get("mutation_rate", 0.0),
            execution_error_rate=raw_config.get("execution_error_rate", 0.0),
            horizon_known=raw_config.get("horizon_known", False),
            model=raw_config.get("model", "replicator"),
            population_size=raw_config.get("population_size", 100),
        )
    )
    if known_strategies is not None:
        unknown = sorted(set(config.strategy_names) - set(known_strategies))
        if unknown:
            raise ValueError(f"unknown evolution strategies: {', '.join(unknown)}")
    history = payload.get("history")
    if not isinstance(history, list) or not history:
        raise ValueError("evolution result history must be a non-empty list")
    if not all(isinstance(row, Mapping) for row in history):
        raise ValueError("each evolution history row must be an object")
    try:
        generations = [int(row.get("generation", -1)) for row in history]
    except (TypeError, ValueError) as exc:
        raise ValueError("evolution result history has invalid generations") from exc
    if generations != list(range(config.generations + 1)):
        raise ValueError("evolution result history has invalid generations")
    for row in history:
        shares = row.get("shares")
        fitness = row.get("fitness")
        if not isinstance(shares, Mapping) or set(shares) != set(config.strategy_names):
            raise ValueError("evolution result contains invalid population shares")
        if not isinstance(fitness, Mapping) or set(fitness) != set(config.strategy_names):
            raise ValueError("evolution result contains invalid fitness values")
        normalized_shares = normalize_population(config.strategy_names, shares)
        if any(abs(float(shares[name]) - normalized_shares[name]) > 1e-8 for name in config.strategy_names):
            raise ValueError("evolution result population shares must sum to 1")
        fitness_values = [float(fitness[name]) for name in config.strategy_names]
        if any(not math.isfinite(value) or not 0 <= value <= 5 for value in fitness_values):
            raise ValueError("evolution result fitness values must be finite and between 0 and 5")
        for field, lower, upper in (
            ("average_payoff", 0.0, 5.0),
            ("cooperation_rate", 0.0, 1.0),
            ("diversity", 0.0, 1.0),
        ):
            value = float(row.get(field, float("nan")))
            if not math.isfinite(value) or not lower <= value <= upper:
                raise ValueError(f"evolution result {field} must be finite and between {lower:g} and {upper:g}")
        if config.model == "moran":
            counts = row.get("counts")
            if not isinstance(counts, Mapping) or set(counts) != set(config.strategy_names):
                raise ValueError("Moran evolution history must contain counts for every strategy")
            normalized_counts: dict[str, int] = {}
            for name in config.strategy_names:
                count = counts[name]
                if isinstance(count, bool) or not isinstance(count, int) or count < 0:
                    raise ValueError("Moran population counts must be non-negative integers")
                normalized_counts[name] = count
            if sum(normalized_counts.values()) != config.population_size:
                raise ValueError("Moran population counts must sum to population_size")
            if any(
                abs(float(shares[name]) - normalized_counts[name] / config.population_size) > 1e-8
                for name in config.strategy_names
            ):
                raise ValueError("Moran population shares must match population counts")

    raw_statistics = payload.get("matchup_statistics")
    if not isinstance(raw_statistics, list):
        raise ValueError("evolution result matchup_statistics must be a list")
    expected_pairs = {(strategy, opponent) for strategy in config.strategy_names for opponent in config.strategy_names}
    seen_pairs: set[tuple[str, str]] = set()
    normalized_statistics: list[dict[str, object]] = []
    for raw_statistic in raw_statistics:
        if not isinstance(raw_statistic, Mapping):
            raise ValueError("each evolution matchup statistic must be an object")
        strategy = str(raw_statistic.get("strategy", ""))
        opponent = str(raw_statistic.get("opponent", ""))
        pair = (strategy, opponent)
        if pair not in expected_pairs:
            raise ValueError("evolution matchup statistic contains an unknown strategy pair")
        if pair in seen_pairs:
            raise ValueError("evolution matchup_statistics contains duplicate strategy pairs")
        payoff = float(raw_statistic.get("payoff", float("nan")))
        cooperation = float(raw_statistic.get("cooperation", float("nan")))
        if not math.isfinite(payoff) or not 0 <= payoff <= 5:
            raise ValueError("evolution matchup payoff must be finite and between 0 and 5")
        if not math.isfinite(cooperation) or not 0 <= cooperation <= 1:
            raise ValueError("evolution matchup cooperation must be finite and between 0 and 1")
        seen_pairs.add(pair)
        normalized_statistics.append(
            {"strategy": strategy, "opponent": opponent, "payoff": payoff, "cooperation": cooperation}
        )
    if seen_pairs != expected_pairs:
        raise ValueError("evolution matchup_statistics is missing configured strategy pairs")
    result = dict(payload)
    result["config"] = config.to_dict()
    result["history"] = [dict(row) for row in history]
    result["events"] = evolution_events(result["history"])
    result["matchup_statistics"] = sorted(
        normalized_statistics,
        key=lambda row: (str(row["strategy"]), str(row["opponent"])),
    )
    return result
