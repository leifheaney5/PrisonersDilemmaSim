import unittest

import pandas as pd

from pages.app import _custom_preview_rows, _pairwise_metric_frame, _tournament_metrics_frame


class TournamentVisualizationTests(unittest.TestCase):
    def test_metrics_are_normalized_for_unequal_live_progress(self):
        frame = _tournament_metrics_frame(
            {
                "strategy_names": ["Early", "Late"],
                "totals": {"Early": 30, "Late": 15},
                "rounds_played": {"Early": 10, "Late": 5},
                "cooperate": {"Early": 8, "Late": 1},
                "match_wins": {"Early": 2, "Late": 1},
                "match_losses": {"Early": 1, "Late": 0},
                "match_ties": {"Early": 1, "Late": 1},
            }
        ).set_index("strategy")

        self.assertEqual(frame.loc["Early", "points_per_round"], 3.0)
        self.assertEqual(frame.loc["Late", "points_per_round"], 3.0)
        self.assertEqual(frame.loc["Early", "cooperation_rate"], 0.8)
        self.assertEqual(frame.loc["Late", "defection_rate"], 0.8)
        self.assertEqual(frame.loc["Early", "win_rate"], 0.5)
        self.assertEqual(frame.loc["Late", "win_rate"], 0.5)

    def test_metrics_handle_strategies_that_have_not_played(self):
        frame = _tournament_metrics_frame({"strategy_names": ["Waiting"]})
        row = frame.iloc[0]
        for metric in ("points_per_round", "cooperation_rate", "defection_rate", "win_rate"):
            self.assertEqual(row[metric], 0.0)

    def test_pairwise_matrix_reports_payoff_and_margin(self):
        results = pd.DataFrame(
            [
                {
                    "repetition": 0,
                    "round": 0,
                    "strategy_1": "Cooperator",
                    "strategy_2": "Defector",
                    "move_1": "cooperate",
                    "move_2": "defect",
                    "points_1": 0,
                    "points_2": 5,
                }
            ]
        )
        payoff_matrix = _pairwise_metric_frame(results, "points_per_round")
        margin_matrix = _pairwise_metric_frame(results, "score_margin")
        self.assertEqual(payoff_matrix.loc["Cooperator", "Defector"], 0)
        self.assertEqual(payoff_matrix.loc["Defector", "Cooperator"], 5)
        self.assertEqual(margin_matrix.loc["Cooperator", "Defector"], -5)
        self.assertEqual(margin_matrix.loc["Defector", "Cooperator"], 5)

    def test_custom_preview_uses_unsaved_composed_policy(self):
        rows = _custom_preview_rows(
            {
                "start_move": "cooperate",
                "response_mode": "tft",
                "retaliation_window": 0,
                "threshold_enabled": False,
                "noise": 0.0,
            },
            ["cooperate", "defect", "cooperate"],
        )
        self.assertEqual([row["custom_move"] for row in rows], ["cooperate", "cooperate", "defect"])
        self.assertEqual(sum(row["custom_points"] for row in rows), 8)


if __name__ == "__main__":
    unittest.main()
