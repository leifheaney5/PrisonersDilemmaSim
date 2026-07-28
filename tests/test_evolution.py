import json
import unittest

import pandas as pd

from pages.evolution import (
    EvolutionConfig,
    build_evolution_result,
    evolution_events,
    evolve_moran_population,
    evolve_population,
    matchup_statistics,
    normalize_evolution_config,
    normalize_population,
    validate_evolution_result,
)


class EvolutionTests(unittest.TestCase):
    def test_equal_fitness_preserves_population(self):
        stats = {
            (a, b): {"payoff": 3.0, "cooperation": 1.0}
            for a in ("A", "B")
            for b in ("A", "B")
        }
        history = evolve_population(["A", "B"], stats, generations=5, initial_shares={"A": 0.7, "B": 0.3})
        self.assertAlmostEqual(history[-1]["shares"]["A"], 0.7)
        self.assertAlmostEqual(history[-1]["shares"]["B"], 0.3)

    def test_higher_fitness_strategy_grows(self):
        stats = {
            ("A", "A"): {"payoff": 4.0, "cooperation": 0.5},
            ("A", "B"): {"payoff": 4.0, "cooperation": 0.5},
            ("B", "A"): {"payoff": 2.0, "cooperation": 0.5},
            ("B", "B"): {"payoff": 2.0, "cooperation": 0.5},
        }
        history = evolve_population(["A", "B"], stats, generations=3)
        self.assertGreater(history[-1]["shares"]["A"], history[0]["shares"]["A"])
        self.assertAlmostEqual(sum(history[-1]["shares"].values()), 1.0)

    def test_mutation_reintroduces_zero_share(self):
        stats = {
            (a, b): {"payoff": 3.0, "cooperation": 1.0}
            for a in ("A", "B")
            for b in ("A", "B")
        }
        history = evolve_population(
            ["A", "B"],
            stats,
            generations=1,
            initial_shares={"A": 1.0, "B": 0.0},
            mutation_rate=0.1,
        )
        self.assertGreater(history[-1]["shares"]["B"], 0.0)

    def test_moran_process_is_seeded_and_preserves_integer_population(self):
        stats = {
            ("A", "A"): {"payoff": 4.0, "cooperation": 0.5},
            ("A", "B"): {"payoff": 4.0, "cooperation": 0.5},
            ("B", "A"): {"payoff": 2.0, "cooperation": 0.5},
            ("B", "B"): {"payoff": 2.0, "cooperation": 0.5},
        }
        first = evolve_moran_population(["A", "B"], stats, generations=5, population_size=50, seed=9)
        second = evolve_moran_population(["A", "B"], stats, generations=5, population_size=50, seed=9)
        self.assertEqual(first, second)
        self.assertTrue(all(sum(row["counts"].values()) == 50 for row in first))
        self.assertTrue(all(abs(sum(row["shares"].values()) - 1.0) < 1e-9 for row in first))

    def test_moran_process_changes_with_seed(self):
        stats = {
            (a, b): {"payoff": 3.0, "cooperation": 1.0}
            for a in ("A", "B")
            for b in ("A", "B")
        }
        first = evolve_moran_population(["A", "B"], stats, generations=3, population_size=30, seed=1)
        second = evolve_moran_population(["A", "B"], stats, generations=3, population_size=30, seed=2)
        self.assertNotEqual(first, second)

    def test_history_is_json_serializable(self):
        stats = {
            (a, b): {"payoff": 3.0, "cooperation": 1.0}
            for a in ("A", "B")
            for b in ("A", "B")
        }
        json.dumps(evolve_population(["A", "B"], stats, generations=2))

    def test_matchup_statistics_are_directional(self):
        frame = pd.DataFrame(
            [
                {
                    "strategy_1": "A",
                    "strategy_2": "B",
                    "move_1": "cooperate",
                    "move_2": "defect",
                    "points_1": 0,
                    "points_2": 5,
                },
                {
                    "strategy_1": "A",
                    "strategy_2": "A",
                    "move_1": "cooperate",
                    "move_2": "cooperate",
                    "points_1": 3,
                    "points_2": 3,
                },
                {
                    "strategy_1": "B",
                    "strategy_2": "B",
                    "move_1": "defect",
                    "move_2": "defect",
                    "points_1": 1,
                    "points_2": 1,
                },
            ]
        )
        stats = matchup_statistics(frame, ["A", "B"])
        self.assertEqual(stats[("A", "B")]["payoff"], 0)
        self.assertEqual(stats[("B", "A")]["payoff"], 5)
        self.assertEqual(stats[("A", "A")]["cooperation"], 1)

    def test_population_validation(self):
        with self.assertRaisesRegex(ValueError, "at least 2"):
            normalize_population(["A"])
        with self.assertRaisesRegex(ValueError, "positive total"):
            normalize_population(["A", "B"], {"A": 0, "B": 0})
        with self.assertRaisesRegex(ValueError, "mutation_rate"):
            evolve_population(
                ["A", "B"],
                {(a, b): {"payoff": 3.0, "cooperation": 1.0} for a in ("A", "B") for b in ("A", "B")},
                generations=1,
                mutation_rate=0.3,
            )

    def test_configuration_normalizes_shares_and_estimates_bounds(self):
        config = normalize_evolution_config(
            EvolutionConfig(
                strategy_names=("A", "B"),
                initial_shares={"A": 3, "B": 1},
                generations=25,
                rounds_per_match=10,
                repetitions=2,
            )
        )
        self.assertEqual(config.initial_shares, {"A": 0.75, "B": 0.25})
        self.assertEqual(config.to_dict()["strategy_names"], ("A", "B"))

    def test_configuration_rejects_invalid_ranges(self):
        base = {"strategy_names": ("A", "B"), "initial_shares": {"A": 0.5, "B": 0.5}}
        for field, value in (
            ("rounds_per_match", 0),
            ("repetitions", 0),
            ("selection_strength", 2),
            ("execution_error_rate", -0.1),
            ("model", "unknown"),
            ("population_size", 1),
        ):
            with self.subTest(field=field):
                with self.assertRaises(ValueError):
                    normalize_evolution_config(EvolutionConfig(**base, **{field: value}))

    def test_event_log_reports_leader_majority_and_low_share_changes(self):
        history = [
            {"generation": 0, "shares": {"A": 0.49, "B": 0.51}, "cooperation_rate": 0.2},
            {"generation": 1, "shares": {"A": 0.6, "B": 0.4}, "cooperation_rate": 0.55},
            {"generation": 2, "shares": {"A": 0.995, "B": 0.005}, "cooperation_rate": 0.8},
        ]
        events = evolution_events(history)
        event_types = [event["type"] for event in events]
        self.assertIn("leader", event_types)
        self.assertIn("majority", event_types)
        self.assertIn("low_share", event_types)
        self.assertIn("cooperation", event_types)

    def test_event_log_is_empty_without_history(self):
        self.assertEqual(evolution_events([]), [])

    def test_evolution_result_round_trip_is_valid_and_portable(self):
        config = EvolutionConfig(
            strategy_names=("A", "B"),
            initial_shares={"A": 0.5, "B": 0.5},
            generations=2,
        )
        stats = {
            (a, b): {"payoff": 3.0, "cooperation": 1.0}
            for a in ("A", "B")
            for b in ("A", "B")
        }
        history = evolve_population(["A", "B"], stats, generations=2)
        artifact = build_evolution_result(config, history, stats)
        restored = validate_evolution_result(json.loads(json.dumps(artifact)), ["A", "B"])
        self.assertEqual(restored["history"], artifact["history"])
        self.assertIn("Experimental population model", restored["format_notice"])

    def test_evolution_result_rejects_tampered_history(self):
        config = EvolutionConfig(
            strategy_names=("A", "B"),
            initial_shares={"A": 0.5, "B": 0.5},
            generations=1,
        )
        stats = {
            (a, b): {"payoff": 3.0, "cooperation": 1.0}
            for a in ("A", "B")
            for b in ("A", "B")
        }
        artifact = build_evolution_result(config, evolve_population(["A", "B"], stats, generations=1), stats)
        artifact["history"][1]["shares"] = {"A": 0.9, "B": 0.9}
        with self.assertRaisesRegex(ValueError, "sum to 1"):
            validate_evolution_result(artifact)


if __name__ == "__main__":
    unittest.main()
