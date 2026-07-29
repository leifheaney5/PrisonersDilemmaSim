import io
import json
import unittest
from urllib.parse import urlparse

from scripts.verify_deployment import verify_deployment


class FakeResponse(io.BytesIO):
    def __init__(self, payload=b"", *, status=200, headers=None):
        super().__init__(payload)
        self.status = status
        self.headers = headers or {}

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.close()


class DeploymentSmokeTests(unittest.TestCase):
    @staticmethod
    def _opener(overrides=None):
        overrides = overrides or {}

        def open_request(request, timeout):
            del timeout
            path = urlparse(request.full_url).path
            if path in overrides:
                return overrides[path]
            if path == "/health/live":
                return FakeResponse(json.dumps({"status": "ok"}).encode())
            if path == "/health/ready":
                return FakeResponse(
                    json.dumps(
                        {
                            "status": "ready",
                            "strategy_count": 100,
                            "checks": {"strategy_catalog": True, "assets": True},
                        }
                    ).encode()
                )
            if path == "/version":
                return FakeResponse(
                    json.dumps(
                        {
                            "app_version": "2.0.0",
                            "game_version": 1,
                            "artifact_schema_versions": {},
                            "rate_limit_backend": "redis",
                        }
                    ).encode()
                )
            return FakeResponse(
                headers={
                    "X-Content-Type-Options": "nosniff",
                    "X-Frame-Options": "SAMEORIGIN",
                    "Content-Security-Policy": "object-src 'none'",
                    "X-Request-ID": "request-1",
                    "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
                }
            )

        return open_request

    def test_healthy_deployment_passes_every_check(self):
        results = verify_deployment("https://example.test", opener=self._opener())
        self.assertTrue(all(result.passed for result in results))

    def test_readiness_rejects_a_short_catalog(self):
        short = FakeResponse(
            json.dumps(
                {
                    "status": "ready",
                    "strategy_count": 99,
                    "checks": {"strategy_catalog": True},
                }
            ).encode()
        )
        results = verify_deployment(
            "https://example.test",
            opener=self._opener({"/health/ready": short}),
        )
        readiness = next(result for result in results if result.name == "readiness")
        self.assertFalse(readiness.passed)

    def test_security_check_rejects_missing_headers(self):
        results = verify_deployment(
            "https://example.test",
            opener=self._opener({"/": FakeResponse()}),
        )
        security = next(result for result in results if result.name == "security headers")
        self.assertFalse(security.passed)

    def test_public_deployment_requires_https(self):
        with self.assertRaisesRegex(ValueError, "must use HTTPS"):
            verify_deployment("http://example.test", opener=self._opener())

    def test_loopback_http_is_allowed_for_local_verification(self):
        results = verify_deployment("http://127.0.0.1:8050", opener=self._opener())
        self.assertTrue(all(result.passed for result in results))


if __name__ == "__main__":
    unittest.main()
