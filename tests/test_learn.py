import unittest

from pages.app import _payoff_lesson_result, learn_page


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


class LearnPageTests(unittest.TestCase):
    def test_payoff_lesson_covers_the_four_outcomes(self):
        expected = {
            ("cooperate", "cooperate"): (3, 3, 6),
            ("cooperate", "defect"): (0, 5, 5),
            ("defect", "cooperate"): (5, 0, 5),
            ("defect", "defect"): (1, 1, 2),
        }
        for moves, scores in expected.items():
            with self.subTest(moves=moves):
                result = _payoff_lesson_result(*moves)
                self.assertEqual(
                    (result["player_points"], result["opponent_points"], result["combined_points"]),
                    scores,
                )

    def test_learn_page_contains_interactive_controls(self):
        ids = _component_ids(learn_page())
        self.assertIn("learn-player-move", ids)
        self.assertIn("learn-opponent-move", ids)
        self.assertIn("learn-payoff-result", ids)
        self.assertEqual(len(ids), len(set(ids)))


if __name__ == "__main__":
    unittest.main()
