import unittest

from pages.app import get_results
from pages.workloads import tournament_pairings, tournament_workload


class WorkloadTests(unittest.TestCase):
    def test_pairing_counts_distinguish_classic_and_self_play(self):
        self.assertEqual(tournament_pairings(4), 6)
        self.assertEqual(tournament_pairings(4, include_self_play=True), 10)

    def test_tournament_workload_counts_all_rounds_and_repetitions(self):
        self.assertEqual(tournament_workload(4, 5, 3), 90)
        self.assertEqual(tournament_workload(4, 5, 3, include_self_play=True), 150)

    def test_invalid_workload_inputs_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "strategy_count"):
            tournament_workload(1, 5, 1)
        with self.assertRaisesRegex(ValueError, "rounds_per_match"):
            tournament_workload(2, 0, 1)
        with self.assertRaisesRegex(ValueError, "repetitions"):
            tournament_workload(2, 1, 0)

    def test_full_result_cache_is_bounded_to_one_configuration(self):
        self.assertEqual(get_results.cache_parameters()["maxsize"], 1)


if __name__ == "__main__":
    unittest.main()
