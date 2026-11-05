"""Rate-limiting, exponential backoff & circuit breaker (spec Section 4.4).

Wraps a callable with retries using exponential backoff (with jitter) and a
simple circuit breaker that opens after repeated consecutive failures so a run
is not dropped mid-blade by hammering a failing backend.
"""

from __future__ import annotations

import random
import time
from typing import Any, Callable, Optional


class CircuitOpenError(RuntimeError):
    pass


class RateLimiter:
    def __init__(
        self,
        max_retries: int = 5,
        base_delay_s: float = 1.0,
        max_delay_s: float = 60.0,
        circuit_breaker_threshold: int = 5,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self.max_retries = max(0, int(max_retries))
        self.base_delay_s = max(0.0, float(base_delay_s))
        self.max_delay_s = max(self.base_delay_s, float(max_delay_s))
        self.circuit_breaker_threshold = max(1, int(circuit_breaker_threshold))
        self._sleep = sleep
        self._consecutive_failures = 0
        self._circuit_open_until = 0.0

    def _backoff_delay(self, attempt: int) -> float:
        # attempt is 0-based; delay grows exponentially, capped, with jitter.
        delay = self.base_delay_s * (2 ** attempt)
        delay = min(delay, self.max_delay_s)
        return delay * (0.5 + random.random() * 0.5)  # jitter 50%-100%

    def call(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        attempts = self.max_retries + 1
        last_exc: Optional[BaseException] = None
        for attempt in range(attempts):
            if self._circuit_open_until and time.time() < self._circuit_open_until:
                raise CircuitOpenError(
                    "Circuit breaker is open; backend appears to be failing repeatedly."
                )
            try:
                result = fn(*args, **kwargs)
            except Exception as exc:  # noqa: BLE001 - retry boundary
                last_exc = exc
                self._consecutive_failures += 1
                if self._consecutive_failures >= self.circuit_breaker_threshold:
                    self._circuit_open_until = time.time() + self.max_delay_s
                if attempt < attempts - 1:
                    self._sleep(self._backoff_delay(attempt))
                    continue
                raise
            else:
                self._consecutive_failures = 0
                return result
        if last_exc:
            raise last_exc
        return fn(*args, **kwargs)

    def reset(self) -> None:
        self._consecutive_failures = 0
        self._circuit_open_until = 0.0
