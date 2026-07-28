import unittest

from pages.insights import evolution_insights, robustness_insights
from pages.app import render_evolution_insights, render_robustness_insights


class ExperimentInsightTests(unittest.TestCase):
    def test_robustness_insights_report_performance_stability_and_rank(self):
        result = {
            "summary": [
                {
                    "strategy": "Reliable",
                    "seeds": 20,
                    "mean_payoff": 3.1,
                    "payoff_std": 0.05,
                    "mean_rank": 1.4,
                    "best_rank": 1,
                    "worst_rank": 3,
                },
                {
                    "strategy": "Volatile",
                    "seeds": 20,
                    "mean_payoff": 3.3,
                    "payoff_std": 0.4,
                    "mean_rank": 2.1,
                    "best_rank": 1,
                    "worst_rank": 5,
                },
            ]
        }
        insights = robustness_insights(result)
        self.assertEqual(len(insights), 4)
        self.assertIn("Volatile had the highest mean payoff", insights[0])
        self.assertIn("Reliable had the lowest", insights[1])
        self.assertIn("Reliable had the best average rank", insights[2])
        self.assertIn("from 1 to 5", insights[3])
        self.assertEqual(render_robustness_insights(result).children.children[1].children, "What happened across seeds")

    def test_evolution_insights_compare_initial_and_final_population(self):
        result = {
            "history": [
                {
                    "generation": 0,
                    "shares": {"Resident": 0.9, "Invader": 0.1},
                    "average_payoff": 1.2,
                    "cooperation_rate": 0.1,
                    "diversity": 0.2,
                },
                {
                    "generation": 40,
                    "shares": {"Resident": 0.25, "Invader": 0.75},
                    "average_payoff": 2.9,
                    "cooperation_rate": 0.7,
                    "diversity": 0.5,
                },
            ]
        }
        insights = evolution_insights(result)
        self.assertEqual(len(insights), 4)
        self.assertIn("Resident was the largest population at generation 0", insights[0])
        self.assertIn("Invader was largest at generation 40", insights[0])
        self.assertIn("Invader had the largest population-share increase", insights[1])
        self.assertIn("Resident had the largest population-share decrease", insights[2])
        self.assertIn("cooperation of 70.0%", insights[3])
        self.assertEqual(render_evolution_insights(result).children.children[1].children, "What changed in the population")

    def test_missing_results_return_no_claims(self):
        self.assertEqual(robustness_insights(None), [])
        self.assertEqual(robustness_insights({"summary": []}), [])
        self.assertEqual(evolution_insights(None), [])
        self.assertEqual(evolution_insights({"history": []}), [])


if __name__ == "__main__":
    unittest.main()
