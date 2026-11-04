import unittest

from pages.experiment_views import (
    evolution_result_view,
    robustness_result_view,
    robustness_seed_view,
)


class ExperimentViewTests(unittest.TestCase):
    def setUp(self):
        self.robustness_result = {
            "seed_results": [
                {"seed": 4, "strategy": "A", "points_per_round": 3.2, "cooperation_rate": 0.8, "rank": 1, "rounds_played": 10},
                {"seed": 4, "strategy": "B", "points_per_round": 2.1, "cooperation_rate": 0.2, "rank": 2, "rounds_played": 10},
                {"seed": 5, "strategy": "A", "points_per_round": 2.8, "cooperation_rate": 0.7, "rank": 2, "rounds_played": 10},
                {"seed": 5, "strategy": "B", "points_per_round": 3.0, "cooperation_rate": 0.3, "rank": 1, "rounds_played": 10},
            ],
            "summary": [
                {"strategy": "A", "mean_points_per_round": 3.0, "mean_rank": 1.5},
                {"strategy": "B", "mean_points_per_round": 2.55, "mean_rank": 1.5},
            ],
        }

    def test_robustness_result_view_builds_figures_and_table(self):
        payoff, ranks, columns, rows = robustness_result_view(self.robustness_result)
        self.assertEqual(payoff.layout.title.text, "Payoff distribution across independent seeds")
        self.assertEqual(ranks.layout.title.text, "Rank by seed")
        self.assertEqual([column["id"] for column in columns], ["strategy", "mean_points_per_round", "mean_rank"])
        self.assertEqual(len(rows), 2)

    def test_robustness_views_have_safe_empty_states(self):
        aggregate = robustness_result_view(None)
        seed = robustness_seed_view(self.robustness_result, 999)
        self.assertEqual(aggregate[0].layout.title.text, "Run or open a robustness result")
        self.assertEqual(seed[0].layout.title.text, "Select a seed to inspect")
        self.assertEqual(aggregate[2:], ([], []))
        self.assertEqual(seed[2:], ([], []))

    def test_seed_view_selects_only_requested_seed(self):
        payoff, cooperation, columns, rows = robustness_seed_view(self.robustness_result, 4)
        self.assertEqual(payoff.layout.title.text, "Points per round, seed 4")
        self.assertEqual(cooperation.layout.title.text, "Cooperation rate, seed 4")
        self.assertEqual([row["strategy"] for row in rows], ["A", "B"])
        self.assertIn("rounds_played", [column["id"] for column in columns])

    def test_evolution_result_view_builds_trajectories_and_final_ranking(self):
        history = [
            {
                "generation": 0,
                "shares": {"A": 0.5, "B": 0.5},
                "fitness": {"A": 3.0, "B": 2.0},
                "average_payoff": 2.5,
                "cooperation_rate": 0.5,
                "diversity": 1.0,
            },
            {
                "generation": 1,
                "shares": {"A": 0.7, "B": 0.3},
                "fitness": {"A": 3.2, "B": 1.8},
                "average_payoff": 2.8,
                "cooperation_rate": 0.6,
                "diversity": 0.84,
            },
        ]
        population, outcomes, columns, rows = evolution_result_view(history, ["A", "B"])
        self.assertEqual(population.layout.title.text, "Population share by generation")
        self.assertEqual(outcomes.layout.title.text, "Population outcomes")
        self.assertEqual([trace.name for trace in outcomes.data], ["Average payoff", "Cooperation rate", "Diversity"])
        self.assertEqual(rows[0]["strategy"], "A")
        self.assertEqual(rows[0]["final_share"], 0.7)
        self.assertEqual([column["id"] for column in columns], ["strategy", "final_share", "fitness"])

    def test_evolution_result_view_handles_empty_history(self):
        population, outcomes, columns, rows = evolution_result_view([], [])
        self.assertEqual(len(population.data), 0)
        self.assertEqual(len(outcomes.data), 0)
        self.assertEqual(columns, [])
        self.assertEqual(rows, [])


if __name__ == "__main__":
    unittest.main()
