import unittest

from pages.security import RedisSlidingWindowLimiter, SlidingWindowLimiter, create_rate_limiter, is_heavy_dash_submission


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

    def test_process_limiter_bounds_and_expires_client_keys(self):
        now = [0.0]
        limiter = SlidingWindowLimiter(2, 10, clock=lambda: now[0], max_keys=2)
        limiter.check("oldest")
        now[0] = 1.0
        limiter.check("middle")
        now[0] = 2.0
        limiter.check("newest")
        self.assertEqual(set(limiter._events), {"middle", "newest"})
        now[0] = 20.0
        limiter.check("current")
        self.assertEqual(set(limiter._events), {"current"})

    def test_factory_uses_process_limiter_without_redis(self):
        limiter = create_rate_limiter("", 2, 10)
        self.assertEqual(limiter.backend, "process")

    def test_redis_limiter_translates_atomic_script_result(self):
        class FakeRedis:
            def __init__(self):
                self.results = ([1, 0], [0, 2_500])
                self.calls = []

            def eval(self, *args):
                self.calls.append(args)
                return self.results[len(self.calls) - 1]

        client = FakeRedis()
        limiter = RedisSlidingWindowLimiter(client, 1, 10, clock=lambda: 123.0, namespace="test")
        self.assertTrue(limiter.check("client").allowed)
        blocked = limiter.check("client")
        self.assertFalse(blocked.allowed)
        self.assertEqual(blocked.retry_after, 3)
        self.assertEqual(client.calls[0][2], "test:client")

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
