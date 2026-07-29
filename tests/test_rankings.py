import unittest

from pages.app import (
    render_evolution_generation_ranking,
    render_robustness_ranking,
    render_tournament_ranking,
)
from pages.rankings import (
    evolution_generation_ranking_view,
    robustness_ranking_view,
    tournament_ranking_statement,
)


class InteractiveRankingTests(unittest.TestCase):
    def setUp(self):
        self.robustness = {
            "summary": [
                {
                    "strategy": "Steady",
                    "mean_payoff": 3.0,
                    "payoff_std": 0.05,
                    "mean_rank": 1.5,
                    "win_rate": 0.4,
                    "mean_cooperation": 0.8,
                },
                {
                    "strategy": "Sprinter",
                    "mean_payoff": 3.2,
                    "payoff_std": 0.4,
                    "mean_rank": 1.8,
                    "win_rate": 0.6,
                    "mean_cooperation": 0.3,
                },
                {
                    "strategy": "Follower",
                    "mean_payoff": 2.5,
                    "payoff_std": 0.1,
                    "mean_rank": 2.7,
                    "win_rate": 0.0,
                    "mean_cooperation": 0.6,
                },
            ]
        }
        self.evolution = {
            "history": [
                {"generation": 0, "shares": {"Resident": 0.8, "Invader": 0.2}},
                {"generation": 1, "shares": {"Resident": 0.45, "Invader": 0.55}},
            ]
        }

    def test_robustness_ranking_is_deterministic_for_higher_metric(self):
        figure, statement = robustness_ranking_view(self.robustness, "mean_payoff")
        self.assertEqual(list(figure.data[0].y), ["Sprinter", "Steady", "Follower"])
        self.assertIn("1. Sprinter (3.200)", statement)
        self.assertIn("2. Steady (3.000)", statement)
        self.assertIn("performance ranking", statement)
        self.assertIn("highest to lowest", statement)

    def test_lower_metric_reverses_ranking_and_click_adds_detail(self):
        click = {"points": [{"customdata": ["Steady"]}]}
        figure, statement = render_robustness_ranking(self.robustness, "payoff_std", click)
        self.assertEqual(list(figure.data[0].y), ["Steady", "Follower", "Sprinter"])
        self.assertEqual(list(figure.data[0].marker.line.width), [4, 1, 1])
        self.assertIn("stability ranking", statement)
        self.assertIn("lowest to highest", statement)
        self.assertIn("Selected: Steady ranks #1", statement)

    def test_evolution_ranking_tracks_selected_generation(self):
        figure, statement = evolution_generation_ranking_view(self.evolution, 1)
        self.assertEqual(list(figure.data[0].y), ["Invader", "Resident"])
        self.assertIn("generation 1", statement)
        self.assertIn("1. Invader (55.0%)", statement)

        callback_figure, callback_statement = render_evolution_generation_ranking(self.evolution, 0)
        self.assertEqual(list(callback_figure.data[0].y), ["Resident", "Invader"])
        self.assertIn("1. Resident (80.0%)", callback_statement)

    def test_empty_and_unknown_inputs_have_safe_fallbacks(self):
        figure, statement = robustness_ranking_view(None, "unknown")
        self.assertEqual(len(figure.data), 0)
        self.assertEqual(statement, "")
        figure, statement = evolution_generation_ranking_view(None, 0)
        self.assertEqual(len(figure.data), 0)
        self.assertEqual(statement, "")

    def test_live_tournament_statement_uses_normalized_selected_metric(self):
        state = {
            "strategy_names": ["A", "B"],
            "totals": {"A": 30, "B": 20},
            "rounds_played": {"A": 10, "B": 5},
            "cooperate": {"A": 8, "B": 1},
            "match_wins": {"A": 1, "B": 2},
            "match_losses": {"A": 1, "B": 0},
            "match_ties": {"A": 0, "B": 0},
            "matches_done": 2,
            "total_matches": 3,
            "done": False,
        }
        statement = tournament_ranking_statement(state, "points_per_round")
        self.assertIn("Provisional", statement)
        self.assertIn("after 2 of 3 matches", statement)
        self.assertIn("1. B (4.000); 2. A (3.000)", statement)
        self.assertIn("1. B (100.0%); 2. A (50.0%)", render_tournament_ranking(state, "win_rate"))
        self.assertEqual(tournament_ranking_statement(None), "")

    def test_equal_values_share_competition_rank(self):
        tied = {
            "summary": [
                {"strategy": "B", "mean_payoff": 3.0},
                {"strategy": "A", "mean_payoff": 3.0},
                {"strategy": "C", "mean_payoff": 2.0},
            ]
        }
        figure, statement = robustness_ranking_view(tied, "mean_payoff")
        self.assertEqual(list(figure.data[0].y), ["A", "B", "C"])
        self.assertEqual(list(figure.data[0].text), ["#1", "#1", "#3"])
        self.assertIn("1. A (3.000); 1. B (3.000); 3. C (2.000)", statement)

        evolution = {"history": [{"generation": 0, "shares": {"B": 0.4, "A": 0.4, "C": 0.2}}]}
        figure, statement = evolution_generation_ranking_view(evolution, 0)
        self.assertEqual(list(figure.data[0].text), ["#1 · 40.0%", "#1 · 40.0%", "#3 · 20.0%"])
        self.assertIn("1. A (40.0%); 1. B (40.0%); 3. C (20.0%)", statement)


if __name__ == "__main__":
    unittest.main()
