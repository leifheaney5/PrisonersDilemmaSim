import json
import unittest

from pages.robustness import SeedSweepConfig, normalize_seed_sweep_config, run_seed_sweep, summarize_seed_sweep, validate_seed_sweep_result
from pages.app import configure_robustness_seed_inspector, experiment_page, inspect_robustness_seed, render_robustness_result


def _component_ids(component):
    ids = []
    component_id = getattr(component, "id", None)
    if component_id:
        ids.append(component_id)
    children = getattr(component, "children", None)
    if isinstance(children, (list, tuple)):
        for child in children:
            ids.extend(_component_ids(child))
    elif children is not None and not isinstance(children, (str, int, float)):
        ids.extend(_component_ids(children))
    return ids


class SeedSweepTests(unittest.TestCase):
    def test_experiment_page_exposes_robustness_controls(self):
        ids = _component_ids(experiment_page())
        for component_id in (
            "robustness-strategies",
            "robustness-seed-count",
            "robustness-run",
            "robustness-payoff-chart",
            "robustness-rank-chart",
            "robustness-summary-table",
            "robustness-import",
            "robustness-seed-inspector",
            "robustness-seed-payoff-chart",
            "robustness-insights",
            "robustness-ranking-metric",
            "robustness-ranking-chart",
            "robustness-ranking-statement",
            "evolution-insights",
            "evolution-generation-ranking",
            "evolution-generation-ranking-statement",
            "tournament-ranking-statement",
        ):
            self.assertIn(component_id, ids)

    def test_configuration_rejects_invalid_ranges_and_workloads(self):
        base = {"strategy_names": ("TitForTat", "BadCop")}
        for field, value in (("seed_count", 0), ("seed_step", 0), ("rounds_per_match", 0), ("execution_error_rate", 2)):
            with self.subTest(field=field):
                with self.assertRaises(ValueError):
                    normalize_seed_sweep_config(SeedSweepConfig(**base, **{field: value}))
        with self.assertRaisesRegex(ValueError, "workload"):
            normalize_seed_sweep_config(
                SeedSweepConfig(strategy_names=tuple(f"S{i}" for i in range(20)), seed_count=100, rounds_per_match=1_000, repetitions=100)
            )

    def test_seed_sweep_is_reproducible_and_json_serializable(self):
        config = SeedSweepConfig(
            strategy_names=("TitForTat", "BadCop", "ImSoRandom"),
            start_seed=11,
            seed_count=3,
            rounds_per_match=8,
            repetitions=2,
            execution_error_rate=0.05,
        )
        first = run_seed_sweep(config)
        second = run_seed_sweep(config)
        self.assertEqual(first, second)
        self.assertEqual({row["seed"] for row in first["seed_results"]}, {11, 12, 13})
        self.assertEqual(len(first["summary"]), 3)
        json.dumps(first)

    def test_seed_sweep_result_round_trip_rebuilds_summary(self):
        result = run_seed_sweep(
            SeedSweepConfig(
                strategy_names=("TitForTat", "ImSoRandom"),
                start_seed=4,
                seed_count=2,
                rounds_per_match=5,
                repetitions=1,
            )
        )
        result["summary"] = []
        restored = validate_seed_sweep_result(json.loads(json.dumps(result)), ["TitForTat", "ImSoRandom"])
        self.assertEqual(len(restored["summary"]), 2)
        self.assertEqual({row["seed"] for row in restored["seed_results"]}, {4, 5})
        options, selected = configure_robustness_seed_inspector(restored)
        self.assertEqual([option["value"] for option in options], [4, 5])
        self.assertEqual(selected, 4)
        payoff, rank, columns, summary_rows = render_robustness_result(restored)
        self.assertTrue(payoff.data)
        self.assertTrue(rank.data)
        self.assertTrue(columns)
        self.assertEqual(len(summary_rows), 2)
        seed_payoff, cooperation, seed_columns, seed_rows = inspect_robustness_seed(restored, 5)
        self.assertTrue(seed_payoff.data)
        self.assertTrue(cooperation.data)
        self.assertTrue(seed_columns)
        self.assertEqual(len(seed_rows), 2)

    def test_seed_sweep_result_rejects_missing_and_invalid_rows(self):
        result = run_seed_sweep(
            SeedSweepConfig(
                strategy_names=("TitForTat", "BadCop"),
                seed_count=2,
                rounds_per_match=5,
                repetitions=1,
            )
        )
        missing = json.loads(json.dumps(result))
        missing["seed_results"].pop()
        with self.assertRaisesRegex(ValueError, "missing"):
            validate_seed_sweep_result(missing)
        invalid = json.loads(json.dumps(result))
        invalid["seed_results"][0]["cooperation_rate"] = 2
        with self.assertRaisesRegex(ValueError, "cooperation_rate"):
            validate_seed_sweep_result(invalid)
        invalid_rank = json.loads(json.dumps(result))
        invalid_rank["seed_results"][0]["rank"] = 2 if invalid_rank["seed_results"][0]["rank"] == 1 else 1
        with self.assertRaisesRegex(ValueError, "rank does not match"):
            validate_seed_sweep_result(invalid_rank)

    def test_summary_keeps_seed_level_rank_and_uncertainty(self):
        rows = [
            {"seed": 1, "strategy": "A", "points_per_round": 4.0, "cooperation_rate": 0.2, "rounds": 10, "rank": 1},
            {"seed": 1, "strategy": "B", "points_per_round": 2.0, "cooperation_rate": 0.8, "rounds": 10, "rank": 2},
            {"seed": 2, "strategy": "A", "points_per_round": 2.0, "cooperation_rate": 0.4, "rounds": 10, "rank": 2},
            {"seed": 2, "strategy": "B", "points_per_round": 3.0, "cooperation_rate": 0.6, "rounds": 10, "rank": 1},
        ]
        summary = {row["strategy"]: row for row in summarize_seed_sweep(rows)}
        self.assertEqual(summary["A"]["mean_payoff"], 3.0)
        self.assertEqual(summary["A"]["payoff_std"], 1.0)
        self.assertEqual(summary["A"]["win_rate"], 0.5)
        self.assertEqual(summary["A"]["best_rank"], 1)
        self.assertEqual(summary["A"]["worst_rank"], 2)


if __name__ == "__main__":
    unittest.main()
