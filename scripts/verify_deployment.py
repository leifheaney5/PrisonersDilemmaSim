"""Verify the public contract of a deployed Strategy Lab instance."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from typing import Callable
from urllib.parse import urljoin, urlparse
from urllib.request import Request, urlopen


PUBLIC_ROUTES = ("/", "/learn", "/profiles", "/experiment")
REQUIRED_SECURITY_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "SAMEORIGIN",
}


@dataclass(frozen=True)
class CheckResult:
    name: str
    passed: bool
    detail: str


def _get(base_url: str, path: str, opener: Callable, timeout: float):
    url = urljoin(f"{base_url.rstrip('/')}/", path.lstrip("/"))
    request = Request(url, headers={"User-Agent": "prisoners-dilemma-deployment-check/1"})
    return opener(request, timeout=timeout)


def verify_deployment(
    base_url: str,
    *,
    opener: Callable = urlopen,
    timeout: float = 15.0,
    minimum_strategies: int = 100,
) -> list[CheckResult]:
    """Return deterministic checks for a running public deployment."""
    parsed_url = urlparse(base_url)
    loopback = parsed_url.hostname in {"127.0.0.1", "localhost", "::1"}
    if parsed_url.scheme not in {"http", "https"} or not parsed_url.netloc:
        raise ValueError("base_url must be an absolute HTTP(S) origin")
    if parsed_url.scheme != "https" and not loopback:
        raise ValueError("public deployments must use HTTPS")

    results: list[CheckResult] = []

    for route in PUBLIC_ROUTES:
        with _get(base_url, route, opener, timeout) as response:
            status = int(response.status)
            results.append(CheckResult(f"route {route}", status == 200, f"HTTP {status}"))

    with _get(base_url, "/health/live", opener, timeout) as response:
        payload = json.load(response)
        passed = int(response.status) == 200 and payload.get("status") == "ok"
        results.append(CheckResult("liveness", passed, f"HTTP {response.status}; status={payload.get('status')}"))

    with _get(base_url, "/health/ready", opener, timeout) as response:
        payload = json.load(response)
        checks = payload.get("checks", {})
        strategy_count = payload.get("strategy_count", 0)
        passed = (
            int(response.status) == 200
            and payload.get("status") == "ready"
            and isinstance(checks, dict)
            and bool(checks)
            and all(checks.values())
            and isinstance(strategy_count, int)
            and strategy_count >= minimum_strategies
        )
        results.append(
            CheckResult(
                "readiness",
                passed,
                f"HTTP {response.status}; strategies={strategy_count}; checks={checks}",
            )
        )

    with _get(base_url, "/version", opener, timeout) as response:
        payload = json.load(response)
        required = {"app_version", "game_version", "artifact_schema_versions", "rate_limit_backend"}
        missing = sorted(required - set(payload))
        results.append(
            CheckResult(
                "version metadata",
                int(response.status) == 200 and not missing,
                f"HTTP {response.status}; missing={missing}",
            )
        )

    with _get(base_url, "/", opener, timeout) as response:
        headers = response.headers
        mismatches = [
            name
            for name, expected in REQUIRED_SECURITY_HEADERS.items()
            if headers.get(name) != expected
        ]
        csp = headers.get("Content-Security-Policy", "")
        request_id = headers.get("X-Request-ID", "")
        hsts = headers.get("Strict-Transport-Security", "")
        hsts_valid = loopback or ("max-age=" in hsts and "includeSubDomains" in hsts)
        passed = not mismatches and "object-src 'none'" in csp and bool(request_id) and hsts_valid
        results.append(
            CheckResult(
                "security headers",
                passed,
                (
                    f"mismatches={mismatches}; csp={bool(csp)}; "
                    f"request_id={bool(request_id)}; hsts={hsts_valid}"
                ),
            )
        )

    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base_url", help="Deployment origin, for example https://example.com")
    parser.add_argument("--timeout", type=float, default=15.0)
    parser.add_argument("--minimum-strategies", type=int, default=100)
    args = parser.parse_args()

    try:
        results = verify_deployment(
            args.base_url,
            timeout=args.timeout,
            minimum_strategies=args.minimum_strategies,
        )
    except Exception as exc:
        print(f"FAIL deployment request: {type(exc).__name__}: {exc}")
        return 1

    for result in results:
        label = "PASS" if result.passed else "FAIL"
        print(f"{label} {result.name}: {result.detail}")
    return 0 if all(result.passed for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
