"""Environment-backed application configuration for deployment and diagnostics."""

from __future__ import annotations

import os


APP_ENV = os.getenv("APP_ENV", "development").strip() or "development"
COMMIT_SHA = (
    os.getenv("RENDER_GIT_COMMIT")
    or os.getenv("GIT_COMMIT")
    or os.getenv("COMMIT_SHA")
    or "unknown"
).strip()
DEPLOYED_AT = os.getenv("DEPLOYED_AT", "unknown").strip() or "unknown"
PAYPAL_CLIENT_ID = os.getenv(
    "PAYPAL_CLIENT_ID",
    "BAAl7kWTxi6DEkHN3OfgGG2D1JqpQdHd22tivmtDGJ574TMPPUoXoCqg0OlGQmeDM2aS4wbzBd0emGM7As",
).strip()
MAX_REQUEST_BYTES = int(os.getenv("MAX_REQUEST_BYTES", "3000000"))
HEAVY_REQUESTS_PER_MINUTE = int(os.getenv("HEAVY_REQUESTS_PER_MINUTE", "8"))


def public_runtime_metadata() -> dict[str, str]:
    """Return non-secret deployment metadata safe to expose publicly."""
    return {
        "environment": APP_ENV,
        "commit_sha": COMMIT_SHA,
        "deployed_at": DEPLOYED_AT,
    }
