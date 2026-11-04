"""Seed-sweep analysis for reproducible tournament robustness studies."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from importlib import import_module
from typing import Callable

import pandas as pd

MAX_SWEEP_SEEDS = 100
MAX_SWEEP_WORKLOAD = 2_000_000
SEED_SWEEP_RESULT_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class SeedSweepConfig:
    strategy_names: tuple[str, ...]
    start_seed: int = 0
    seed_count: int = 20
    seed_step: int = 1
    rounds_per_match: int = 100
    repetitions: int = 5
    horizon_known: bool = False
    include_self_play: bool = False
    execution_error_rate: float = 0.0

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def normalize_seed_sweep_config(config: SeedSweepConfig) -> SeedSweepConfig:
    """Validate a seed sweep and reject excessive workloads before execution."""
    names = tuple(str(name) for name in config.strategy_names)
    if len(names) < 2:
        raise ValueError("Select at least 2 strategies for a seed sweep.")
    if len(names) != len(set(names)):
        raise ValueError("Seed sweep strategy names must be unique.")
    seed_count = int(config.seed_count)
    seed_step = int(config.seed_step)
    rounds = int(config.rounds_per_match)
    repetitions = int(config.repetitions)
    error_rate = float(config.execution_error_rate)
    if not 1 <= seed_count <= MAX_SWEEP_SEEDS:
        raise ValueError(f"seed_count must be between 1 and {MAX_SWEEP_SEEDS}")
    if seed_step == 0:
        raise ValueError("seed_step must not be 0")
    if not 1 <= rounds <= 1_000:
        raise ValueError("rounds_per_match must be between 1 and 1000")
    if not 1 <= repetitions <= 100:
        raise ValueError("repetitions must be between 1 and 100")
    if not math.isfinite(error_rate) or not 0 <= error_rate <= 1:
        raise ValueError("execution_error_rate must be between 0 and 1")
    strategy_count = len(names)
    pairings = strategy_count * (strategy_count + (1 if config.include_self_play else -1)) // 2
    workload = seed_count * pairings * rounds * repetitions
    if workload > MAX_SWEEP_WORKLOAD:
        raise ValueError(f"seed sweep workload must not exceed {MAX_SWEEP_WORKLOAD} rounds")
    return SeedSweepConfig(
        strategy_names=names,
        start_seed=int(config.start_seed),
        seed_count=seed_count,
        seed_step=seed_step,
        rounds_per_match=rounds,
        repetitions=repetitions,
        horizon_known=bool(config.horizon_known),
        include_self_play=bool(config.include_self_play),
        execution_error_rate=error_rate,
    )


def _seed_strategy_metrics(results: pd.DataFrame, seed: int) -> list[dict[str, object]]:
    perspectives = pd.concat(
        [
            results.rename(columns={"strategy_1": "strategy", "points_1": "points", "move_1": "move"})[
                ["strategy", "points", "move"]
            ],
            results.rename(columns={"strategy_2": "strategy", "points_2": "points", "move_2": "move"})[
                ["strategy", "points", "move"]
            ],
        ],
        ignore_index=True,
    )
    metrics = perspectives.groupby("strategy", as_index=False).agg(
        points_per_round=("points", "mean"),
        cooperation_rate=("move", lambda moves: float((moves == "cooperate").mean())),
        rounds=("points", "size"),
    )
    metrics["seed"] = int(seed)
    metrics["rank"] = metrics["points_per_round"].rank(method="min", ascending=False).astype(int)
    return metrics[["seed", "strategy", "points_per_round", "cooperation_rate", "rounds", "rank"]].to_dict("records")


def summarize_seed_sweep(seed_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    """Aggregate independent seed-level measurements into descriptive statistics."""
    if not seed_rows:
        return []
    frame = pd.DataFrame(seed_rows)
    strategy_count = int(frame["strategy"].nunique())
    frame["won"] = frame["rank"] == 1
    frame["top_three"] = frame["rank"] <= min(3, strategy_count)
    summary = frame.groupby("strategy", as_index=False).agg(
        seeds=("seed", "nunique"),
        mean_payoff=("points_per_round", "mean"),
        median_payoff=("points_per_round", "median"),
        payoff_std=("points_per_round", lambda values: float(values.std(ddof=0))),
        payoff_q1=("points_per_round", lambda values: float(values.quantile(0.25))),
        payoff_q3=("points_per_round", lambda values: float(values.quantile(0.75))),
        min_payoff=("points_per_round", "min"),
        max_payoff=("points_per_round", "max"),
        mean_cooperation=("cooperation_rate", "mean"),
        mean_rank=("rank", "mean"),
        best_rank=("rank", "min"),
        worst_rank=("rank", "max"),
        win_rate=("won", "mean"),
        top_three_rate=("top_three", "mean"),
    )
    summary = summary.sort_values(["mean_rank", "mean_payoff"], ascending=[True, False])
    return summary.to_dict("records")


def run_seed_sweep(
    config: SeedSweepConfig,
    simulator: Callable[..., pd.DataFrame] | None = None,
) -> dict[str, object]:
    """Run independent tournament seeds and return portable raw and summary rows."""
    normalized = normalize_seed_sweep_config(config)
    if simulator is None:
        module_name = f"{__package__}.game_logic" if __package__ else "game_logic"
        simulator = import_module(module_name).simulate_tournament
    seeds = [normalized.start_seed + index * normalized.seed_step for index in range(normalized.seed_count)]
    rows: list[dict[str, object]] = []
    for seed in seeds:
        results = simulator(
            strategy_names=list(normalized.strategy_names),
            rounds_per_match=normalized.rounds_per_match,
            repetitions=normalized.repetitions,
            seed=seed,
            horizon_known=normalized.horizon_known,
            include_self_play=normalized.include_self_play,
            execution_error_rate=normalized.execution_error_rate,
        )
        rows.extend(_seed_strategy_metrics(results, seed))
    return {
        "schema_version": SEED_SWEEP_RESULT_SCHEMA_VERSION,
        "experiment_type": "seed_sweep_result",
        "game": "prisoners_dilemma",
        "game_version": 1,
        "format_notice": "Experimental robustness analysis. Each seed runs the configured Prisoner's Dilemma tournament independently.",
        "config": normalized.to_dict(),
        "seed_results": rows,
        "summary": summarize_seed_sweep(rows),
    }


def validate_seed_sweep_result(payload: object, known_strategies: Sequence[str] | None = None) -> dict[str, object]:
    """Validate a saved seed sweep and rebuild its descriptive summary."""
    if not isinstance(payload, Mapping):
        raise ValueError("seed sweep result must be an object")
    if payload.get("schema_version") != SEED_SWEEP_RESULT_SCHEMA_VERSION:
        raise ValueError("unsupported or missing seed sweep schema_version")
    if payload.get("experiment_type") != "seed_sweep_result":
        raise ValueError("experiment_type must be seed_sweep_result")
    if payload.get("game") != "prisoners_dilemma" or payload.get("game_version") != 1:
        raise ValueError("unsupported game or game_version")
    raw_config = payload.get("config")
    if not isinstance(raw_config, Mapping):
        raise ValueError("seed sweep config must be an object")
    config = normalize_seed_sweep_config(
        SeedSweepConfig(
            strategy_names=tuple(raw_config.get("strategy_names", [])),
            start_seed=raw_config.get("start_seed", 0),
            seed_count=raw_config.get("seed_count", 20),
            seed_step=raw_config.get("seed_step", 1),
            rounds_per_match=raw_config.get("rounds_per_match", 100),
            repetitions=raw_config.get("repetitions", 5),
            horizon_known=raw_config.get("horizon_known", False),
            include_self_play=raw_config.get("include_self_play", False),
            execution_error_rate=raw_config.get("execution_error_rate", 0.0),
        )
    )
    if known_strategies is not None:
        unknown = sorted(set(config.strategy_names) - set(known_strategies))
        if unknown:
            raise ValueError(f"unknown seed sweep strategies: {', '.join(unknown)}")
    raw_rows = payload.get("seed_results")
    if not isinstance(raw_rows, list) or not raw_rows:
        raise ValueError("seed_results must be a non-empty list")
    expected_seeds = {
        config.start_seed + index * config.seed_step
        for index in range(config.seed_count)
    }
    expected_pairs = {(seed, strategy) for seed in expected_seeds for strategy in config.strategy_names}
    normalized_rows: list[dict[str, object]] = []
    seen_pairs: set[tuple[int, str]] = set()
    for raw_row in raw_rows:
        if not isinstance(raw_row, Mapping):
            raise ValueError("each seed result must be an object")
        seed = int(raw_row.get("seed", 0))
        strategy = str(raw_row.get("strategy", ""))
        payoff = float(raw_row.get("points_per_round", float("nan")))
        cooperation = float(raw_row.get("cooperation_rate", float("nan")))
        rounds = int(raw_row.get("rounds", 0))
        rank = int(raw_row.get("rank", 0))
        if seed not in expected_seeds or strategy not in config.strategy_names:
            raise ValueError("seed result does not match the configured seeds and strategies")
        if not math.isfinite(payoff) or not 0 <= payoff <= 5:
            raise ValueError("points_per_round must be finite and between 0 and 5")
        if not math.isfinite(cooperation) or not 0 <= cooperation <= 1:
            raise ValueError("cooperation_rate must be finite and between 0 and 1")
        if rounds <= 0:
            raise ValueError("seed result rounds must be positive")
        if not 1 <= rank <= len(config.strategy_names):
            raise ValueError("seed result rank is outside the strategy range")
        pair = (seed, strategy)
        if pair in seen_pairs:
            raise ValueError("seed_results contains duplicate seed and strategy rows")
        seen_pairs.add(pair)
        normalized_rows.append(
            {
                "seed": seed,
                "strategy": strategy,
                "points_per_round": payoff,
                "cooperation_rate": cooperation,
                "rounds": rounds,
                "rank": rank,
            }
        )
    if seen_pairs != expected_pairs:
        raise ValueError("seed_results is missing configured seed and strategy rows")
    rank_frame = pd.DataFrame(normalized_rows)
    rank_frame["expected_rank"] = rank_frame.groupby("seed")["points_per_round"].rank(method="min", ascending=False).astype(int)
    if bool((rank_frame["rank"] != rank_frame["expected_rank"]).any()):
        raise ValueError("seed result rank does not match points_per_round")
    result = dict(payload)
    result["config"] = config.to_dict()
    result["seed_results"] = sorted(normalized_rows, key=lambda row: (int(row["seed"]), str(row["strategy"])))
    result["summary"] = summarize_seed_sweep(result["seed_results"])
    result["format_notice"] = "Experimental robustness analysis. Each seed runs the configured Prisoner's Dilemma tournament independently."
    return result
