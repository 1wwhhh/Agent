from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum

from app.llm.exceptions import CircuitBreakerOpenError


class CircuitBreakerStatus(str, Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@dataclass(frozen=True)
class CircuitBreakerConfig:
    failure_threshold: int = 3
    reset_timeout_seconds: int = 30


@dataclass
class CircuitBreaker:
    config: CircuitBreakerConfig = field(default_factory=CircuitBreakerConfig)
    state: CircuitBreakerStatus = CircuitBreakerStatus.CLOSED
    consecutive_failures: int = 0
    opened_at: datetime | None = None

    def before_request(self) -> CircuitBreakerStatus:
        if self.state == CircuitBreakerStatus.OPEN:
            if self._cooldown_elapsed():
                self.state = CircuitBreakerStatus.HALF_OPEN
                return self.state
            raise CircuitBreakerOpenError("llm circuit breaker is open")
        return self.state

    def record_success(self) -> None:
        self.state = CircuitBreakerStatus.CLOSED
        self.consecutive_failures = 0
        self.opened_at = None

    def record_failure(self) -> None:
        if self.state == CircuitBreakerStatus.HALF_OPEN:
            self._open()
            return
        self.consecutive_failures += 1
        if self.consecutive_failures >= self.config.failure_threshold:
            self._open()

    def snapshot(self) -> dict[str, object]:
        return {
            "state": self.state.value,
            "consecutive_failures": self.consecutive_failures,
            "opened_at": self.opened_at.isoformat() if self.opened_at is not None else None,
        }

    def _open(self) -> None:
        self.state = CircuitBreakerStatus.OPEN
        self.opened_at = datetime.now(timezone.utc)

    def _cooldown_elapsed(self) -> bool:
        if self.opened_at is None:
            return True
        reopen_at = self.opened_at + timedelta(seconds=self.config.reset_timeout_seconds)
        return datetime.now(timezone.utc) >= reopen_at
