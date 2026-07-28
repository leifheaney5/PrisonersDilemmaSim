"""Small dependency-free request controls for the public web process."""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from threading import Lock
import time


@dataclass(frozen=True)
class RateLimitDecision:
    allowed: bool
    retry_after: int = 0


class SlidingWindowLimiter:
    """Bound requests per key over a rolling window.

    This protects each web process from accidental repeated submissions. A shared
    Redis-backed limiter should replace it when background workers are introduced.
    """

    def __init__(self, limit: int, window_seconds: int = 60, *, clock=time.monotonic):
        if limit < 1 or window_seconds < 1:
            raise ValueError("rate limit and window must be positive")
        self.limit = int(limit)
        self.window_seconds = int(window_seconds)
        self._clock = clock
        self._events: dict[str, deque[float]] = defaultdict(deque)
        self._lock = Lock()

    def check(self, key: str) -> RateLimitDecision:
        now = float(self._clock())
        cutoff = now - self.window_seconds
        with self._lock:
            events = self._events[str(key)]
            while events and events[0] <= cutoff:
                events.popleft()
            if len(events) >= self.limit:
                retry_after = max(1, int(self.window_seconds - (now - events[0]) + 0.999))
                return RateLimitDecision(False, retry_after)
            events.append(now)
            return RateLimitDecision(True)


HEAVY_DASH_OUTPUT_MARKERS = (
    "robustness-result.data",
    "evolution-result.data",
)


def is_heavy_dash_submission(path: str, method: str, payload: object) -> bool:
    """Identify callbacks that can start expensive robustness or evolution work."""
    if method.upper() != "POST" or path != "/_dash-update-component" or not isinstance(payload, dict):
        return False
    output = str(payload.get("output", ""))
    changed = {str(value) for value in payload.get("changedPropIds", [])}
    run_triggered = bool(
        {"robustness-run.n_clicks", "evolution-run.n_clicks"} & changed
    )
    return run_triggered and any(marker in output for marker in HEAVY_DASH_OUTPUT_MARKERS)
