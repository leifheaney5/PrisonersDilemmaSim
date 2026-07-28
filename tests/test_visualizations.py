import unittest

import pandas as pd

from pages.app import (
    _custom_preview_rows,
    _game_format_notice,
    _matchup_replay_frame,
    _pairwise_metric_frame,
    _tournament_metrics_frame,
    profiles_page,
    select_landscape_strategy,
)


class TournamentVisualizationTests(unittest.TestCase):
    def test_profiles_page_includes_clickable_strategy_landscape(self):
        page = profiles_page()

        def component_ids(component):
            found = []
            component_id = getattr(component, "id", None)
            if component_id:
                found.append(component_id)
            children = getattr(component, "children", None)
            if isinstance(children, (list, tuple)):
                for child in children:
                    found.extend(component_ids(child))
            elif children is not None:
                found.extend(component_ids(children))
            return found

        self.assertIn("strategy-landscape", component_ids(page))
        self.assertIn("matchup-arena-round", component_ids(page))
        self.assertIn("matchup-arena-frame", component_ids(page))
        self.assertEqual(
            select_landscape_strategy({"points": [{"customdata": ["TitForTat"]}]}),
            "TitForTat",
        )

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

    def test_pairwise_matrix_reports_welfare_and_relationship_metrics(self):
        results = pd.DataFrame(
            [
                {"repetition": 0, "round": 0, "strategy_1": "A", "strategy_2": "B", "move_1": "cooperate", "move_2": "cooperate", "points_1": 3, "points_2": 3},
                {"repetition": 0, "round": 1, "strategy_1": "A", "strategy_2": "B", "move_1": "defect", "move_2": "cooperate", "points_1": 5, "points_2": 0},
            ]
        )
        combined = _pairwise_metric_frame(results, "combined_payoff")
        inequality = _pairwise_metric_frame(results, "payoff_inequality")
        mutual_cooperation = _pairwise_metric_frame(results, "mutual_cooperation_rate")
        exploitation = _pairwise_metric_frame(results, "exploitation_rate")
        self.assertEqual(combined.loc["A", "B"], 5.5)
        self.assertEqual(inequality.loc["A", "B"], 2.5)
        self.assertEqual(mutual_cooperation.loc["A", "B"], 0.5)
        self.assertEqual(exploitation.loc["A", "B"], 0.5)

    def test_matchup_replay_normalizes_player_order(self):
        results = pd.DataFrame(
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
        replay = _matchup_replay_frame(results, "B", "A")
        row = replay.iloc[0]
        self.assertEqual(row["strategy_move"], "defect")
        self.assertEqual(row["strategy_points"], 5)
        self.assertEqual(row["cumulative_strategy"], 5)
        self.assertEqual(row["combined_payoff"], 5)

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
        self.assertIn("Tit-for-Tat", rows[2]["base_rule"])
        self.assertEqual(rows[2]["noise_flip"], "No")

    def test_custom_preview_explains_rule_overrides(self):
        rows = _custom_preview_rows(
            {
                "start_move": "cooperate",
                "response_mode": "fixed",
                "retaliation_window": 2,
                "threshold_enabled": True,
                "defect_rate_threshold": 0.25,
                "min_history": 2,
                "endgame_after_turn": 4,
                "noise": 0.0,
            },
            ["cooperate", "defect", "cooperate", "cooperate"],
        )
        self.assertIn("Retaliation active", rows[2]["safety_rule"])
        self.assertIn("Active", rows[3]["threshold_rule"])
        self.assertIn("turn 4", rows[3]["endgame_rule"])
        self.assertEqual(rows[3]["custom_move"], "defect")

    def test_format_notice_distinguishes_classic_and_experimental_modes(self):
        classic_text, classic_color = _game_format_notice()
        variant_text, variant_color = _game_format_notice(True, 0.05)
        self.assertIn("Classic IPD format", classic_text)
        self.assertEqual(classic_color, "success")
        self.assertIn("Experimental variant", variant_text)
        self.assertIn("self-play", variant_text)
        self.assertIn("execution errors", variant_text)
        self.assertEqual(variant_color, "warning")


if __name__ == "__main__":
    unittest.main()
