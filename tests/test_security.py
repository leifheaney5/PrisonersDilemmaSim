import unittest

from pages.security import SlidingWindowLimiter, is_heavy_dash_submission


class SecurityControlTests(unittest.TestCase):
    def test_sliding_window_limiter_releases_expired_events(self):
        now = [0.0]
        limiter = SlidingWindowLimiter(2, 10, clock=lambda: now[0])
        self.assertTrue(limiter.check("client").allowed)
        self.assertTrue(limiter.check("client").allowed)
        blocked = limiter.check("client")
        self.assertFalse(blocked.allowed)
        self.assertEqual(blocked.retry_after, 10)
        now[0] = 11.0
        self.assertTrue(limiter.check("client").allowed)

    def test_heavy_callback_detection_is_narrow(self):
        self.assertTrue(
            is_heavy_dash_submission(
                "/_dash-update-component",
                "POST",
                {
                    "output": "..robustness-status.children...robustness-result.data..",
                    "changedPropIds": ["robustness-run.n_clicks"],
                },
            )
        )
        self.assertTrue(
            is_heavy_dash_submission(
                "/_dash-update-component",
                "POST",
                {
                    "output": "..evolution-status.children...evolution-result.data..",
                    "changedPropIds": ["evolution-run.n_clicks"],
                },
            )
        )
        self.assertFalse(is_heavy_dash_submission("/health", "GET", {}))
        self.assertFalse(
            is_heavy_dash_submission(
                "/_dash-update-component",
                "POST",
                {"output": "tournament-leaderboard.figure"},
            )
        )


if __name__ == "__main__":
    unittest.main()
