import unittest

import pandas as pd

from pages.analytics import (
    match_level,
    matchup_replay_frame,
    pairwise_metric_frame,
    perspective_rows,
    strategy_landscape_frame,
    tournament_metrics_frame,
)


class AnalyticsModuleTests(unittest.TestCase):
    def setUp(self):
        self.results = pd.DataFrame(
            [
                {
                    "repetition": 0,
                    "round": 0,
                    "strategy_1": "A",
                    "strategy_2": "B",
                    "intended_move_1": "cooperate",
                    "intended_move_2": "defect",
                    "move_1": "cooperate",
                    "move_2": "defect",
                    "points_1": 0,
                    "points_2": 5,
                }
            ]
        )

    def test_perspective_and_match_helpers_are_dash_independent(self):
        perspectives = perspective_rows(self.results)
        self.assertEqual(len(perspectives), 2)
        matches = match_level(perspectives)
        outcomes = dict(zip(matches["strategy"], matches["outcome"]))
        self.assertEqual(outcomes, {"A": "loss", "B": "win"})

    def test_metric_and_replay_helpers_preserve_behavior(self):
        matrix = pairwise_metric_frame(self.results, "score_margin")
        self.assertEqual(matrix.loc["A", "B"], -5)
        replay = matchup_replay_frame(self.results, "B", "A")
        self.assertEqual(replay.iloc[0]["strategy_points"], 5)

    def test_live_metrics_normalize_by_rounds_played(self):
        frame = tournament_metrics_frame(
            {
                "strategy_names": ["A"],
                "totals": {"A": 9},
                "rounds_played": {"A": 3},
                "cooperate": {"A": 2},
                "match_wins": {"A": 1},
            }
        )
        self.assertEqual(frame.iloc[0]["points_per_round"], 3)
        self.assertAlmostEqual(frame.iloc[0]["cooperation_rate"], 2 / 3)

    def test_strategy_landscape_exposes_interpretable_behavior_axes(self):
        results = pd.DataFrame(
            [
                {
                    "repetition": 0,
                    "round": 0,
                    "strategy_1": "A",
                    "strategy_2": "B",
                    "move_1": "cooperate",
                    "move_2": "defect",
                    "points_1": 0,
                    "points_2": 5,
                },
                {
                    "repetition": 0,
                    "round": 1,
                    "strategy_1": "A",
                    "strategy_2": "B",
                    "move_1": "defect",
                    "move_2": "defect",
                    "points_1": 1,
                    "points_2": 1,
                },
            ]
        )
        landscape = strategy_landscape_frame(results).set_index("strategy")
        self.assertEqual(set(landscape.index), {"A", "B"})
        self.assertAlmostEqual(landscape.loc["A", "cooperation_rate"], 0.5)
        self.assertAlmostEqual(landscape.loc["A", "stability"], 0.0)
        self.assertAlmostEqual(landscape.loc["B", "stability"], 1.0)
        self.assertTrue(-1 <= landscape.loc["A", "response_gap"] <= 1)


if __name__ == "__main__":
    unittest.main()
