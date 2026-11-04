import json
import unittest

from pages.app import configure_match_arena, render_match_arena
from pages.match_arena import build_arena_frames, select_arena_frame


class MatchArenaTests(unittest.TestCase):
    def setUp(self):
        self.rows = [
            {
                "round": 1,
                "strategy_move": "cooperate",
                "opponent_move": "cooperate",
                "strategy_intended": "cooperate",
                "opponent_intended": "cooperate",
                "strategy_points": 3,
                "opponent_points": 3,
                "cumulative_strategy": 3,
                "cumulative_opponent": 3,
            },
            {
                "round": 2,
                "strategy_move": "defect",
                "opponent_move": "cooperate",
                "strategy_intended": "cooperate",
                "opponent_intended": "cooperate",
                "strategy_points": 5,
                "opponent_points": 0,
                "cumulative_strategy": 8,
                "cumulative_opponent": 3,
            },
        ]

    def test_frames_preserve_moves_scores_and_execution_errors(self):
        frames = build_arena_frames(self.rows, "TitForTat", "Gradual")
        self.assertEqual(len(frames), 2)
        self.assertEqual(frames[0]["outcome"], "Mutual cooperation")
        self.assertEqual(frames[1]["outcome"], "The selected strategy exploited its opponent")
        self.assertTrue(frames[1]["strategy_error"])
        self.assertFalse(frames[1]["opponent_error"])
        self.assertEqual(frames[1]["strategy_total"], 8)
        self.assertIn("intended move was flipped", frames[1]["statement"])
        json.dumps(frames)

    def test_frame_selection_clamps_to_available_rounds(self):
        frames = build_arena_frames(self.rows, "A", "B")
        self.assertEqual(select_arena_frame(frames, 0)["round"], 1)
        self.assertEqual(select_arena_frame(frames, 99)["round"], 2)
        self.assertIsNone(select_arena_frame([], 1))

    def test_invalid_moves_are_rejected(self):
        invalid = [{**self.rows[0], "strategy_move": "wait"}]
        with self.assertRaisesRegex(ValueError, "cooperate or defect"):
            build_arena_frames(invalid, "A", "B")

    def test_slider_and_component_render_from_stored_frames(self):
        frames = build_arena_frames(self.rows, "TitForTat", "Gradual")
        maximum, marks, value = configure_match_arena(frames)
        self.assertEqual((maximum, value), (2, 1))
        self.assertEqual(marks, {1: "1", 2: "2"})

        component = render_match_arena(frames, 2)
        rendered = str(component)
        self.assertIn("TitForTat", rendered)
        self.assertIn("Gradual", rendered)
        self.assertIn("execution flipped", rendered)
        self.assertIn("8 total", rendered)

    def test_empty_arena_has_instructional_state(self):
        maximum, marks, value = configure_match_arena(None)
        self.assertEqual((maximum, marks, value), (1, {1: "1"}, 1))
        self.assertIn("Select a matchup matrix cell", str(render_match_arena(None, 1)))


if __name__ == "__main__":
    unittest.main()
