"""Base class shared by all BladeRunner agents."""

from __future__ import annotations

from typing import Any, Optional

from ..config import Config
from ..events import EventBus
from ..logging_setup import BladeLogger
from ..rate_limit import RateLimiter


class BaseAgent:
    """Common plumbing: identity, config, logging, event bus, rate limiter."""

    name: str = "BaseAgent"

    def __init__(
        self,
        config: Config,
        event_bus: Optional[EventBus] = None,
        run_id: Optional[str] = None,
        logger: Optional[BladeLogger] = None,
        rate_limiter: Optional[RateLimiter] = None,
    ):
        self.config = config
        self.events = event_bus or EventBus()
        self.run_id = run_id
        self.log = logger or BladeLogger()
        rl = config.rate_limit
        self.rate = rate_limiter or RateLimiter(
            max_retries=rl.get("max_retries", 5),
            base_delay_s=rl.get("base_delay_s", 1.0),
            max_delay_s=rl.get("max_delay_s", 60.0),
            circuit_breaker_threshold=rl.get("circuit_breaker_threshold", 5),
        )

    def log_info(self, event: str, message: str, detail: Optional[dict] = None) -> None:
        self.log.info(self.name, event, message, detail=detail, run_id=self.run_id)

    def log_error(self, event: str, message: str, detail: Optional[dict] = None) -> None:
        self.log.error(self.name, event, message, detail=detail, run_id=self.run_id)
