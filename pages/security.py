"""Small dependency-free request controls for the public web process."""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from threading import Lock
import time
import uuid


@dataclass(frozen=True)
class RateLimitDecision:
    allowed: bool
    retry_after: int = 0


class SlidingWindowLimiter:
    """Bound requests per key over a rolling window.

    This protects each web process from accidental repeated submissions. A shared
    Redis-backed limiter should replace it when background workers are introduced.
    """

    backend = "process"

    def __init__(self, limit: int, window_seconds: int = 60, *, clock=time.monotonic, max_keys: int = 10_000):
        if limit < 1 or window_seconds < 1 or max_keys < 1:
            raise ValueError("rate limit and window must be positive")
        self.limit = int(limit)
        self.window_seconds = int(window_seconds)
        self._clock = clock
        self._events: dict[str, deque[float]] = defaultdict(deque)
        self._last_seen: dict[str, float] = {}
        self.max_keys = int(max_keys)
        self._last_cleanup = float("-inf")
        self._lock = Lock()

    def _cleanup(self, now: float, cutoff: float) -> None:
        if now - self._last_cleanup < self.window_seconds:
            return
        expired = [key for key, seen in self._last_seen.items() if seen <= cutoff]
        for key in expired:
            self._events.pop(key, None)
            self._last_seen.pop(key, None)
        self._evict_overflow()
        self._last_cleanup = now

    def _evict_overflow(self) -> None:
        overflow = len(self._last_seen) - self.max_keys
        if overflow > 0:
            oldest = sorted(self._last_seen, key=self._last_seen.get)[:overflow]
            for key in oldest:
                self._events.pop(key, None)
                self._last_seen.pop(key, None)

    def check(self, key: str) -> RateLimitDecision:
        now = float(self._clock())
        cutoff = now - self.window_seconds
        with self._lock:
            normalized_key = str(key)
            self._cleanup(now, cutoff)
            events = self._events[normalized_key]
            while events and events[0] <= cutoff:
                events.popleft()
            if len(events) >= self.limit:
                self._last_seen[normalized_key] = now
                self._evict_overflow()
                retry_after = max(1, int(self.window_seconds - (now - events[0]) + 0.999))
                return RateLimitDecision(False, retry_after)
            events.append(now)
            self._last_seen[normalized_key] = now
            self._evict_overflow()
            return RateLimitDecision(True)


class RedisSlidingWindowLimiter:
    """Atomic Redis-backed sliding window shared by every web worker."""

    backend = "redis"
    _SCRIPT = """
local key = KEYS[1]
local now = tonumber(ARGV[1])
local window = tonumber(ARGV[2])
local limit = tonumber(ARGV[3])
local member = ARGV[4]
redis.call('ZREMRANGEBYSCORE', key, '-inf', now - window)
local count = redis.call('ZCARD', key)
if count >= limit then
  local oldest = redis.call('ZRANGE', key, 0, 0, 'WITHSCORES')
  redis.call('PEXPIRE', key, window)
  return {0, math.max(1, window - (now - tonumber(oldest[2])))}
end
redis.call('ZADD', key, now, member)
redis.call('PEXPIRE', key, window)
return {1, 0}
"""

    def __init__(self, client, limit: int, window_seconds: int = 60, *, clock=time.time, namespace: str = "pd:heavy"):
        if limit < 1 or window_seconds < 1:
            raise ValueError("rate limit and window must be positive")
        self.client = client
        self.limit = int(limit)
        self.window_seconds = int(window_seconds)
        self._clock = clock
        self.namespace = str(namespace).strip() or "pd:heavy"

    def check(self, key: str) -> RateLimitDecision:
        now_ms = int(float(self._clock()) * 1_000)
        window_ms = self.window_seconds * 1_000
        redis_key = f"{self.namespace}:{key}"
        member = f"{now_ms}:{uuid.uuid4().hex}"
        result = self.client.eval(self._SCRIPT, 1, redis_key, now_ms, window_ms, self.limit, member)
        allowed = bool(int(result[0]))
        retry_after = 0 if allowed else max(1, (int(result[1]) + 999) // 1_000)
        return RateLimitDecision(allowed, retry_after)


def create_rate_limiter(redis_url: str, limit: int, window_seconds: int = 60):
    """Create a shared Redis limiter when configured, otherwise a local limiter."""
    url = str(redis_url or "").strip()
    if not url:
        return SlidingWindowLimiter(limit, window_seconds)
    try:
        import redis
    except ImportError as exc:
        raise RuntimeError("Redis rate limiting is configured but redis-py is unavailable") from exc
    client = redis.Redis.from_url(url, socket_connect_timeout=2, socket_timeout=2, decode_responses=False)
    client.ping()
    return RedisSlidingWindowLimiter(client, limit, window_seconds)


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
