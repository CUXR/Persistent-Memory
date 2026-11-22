"""Hard cap on what one command may spend on hosted APIs."""

from __future__ import annotations

import math
import threading


class SpendLimitExceeded(RuntimeError):
    """Hosted-API spend has passed the cap; no further paid call should be made."""


class SpendTracker:
    """Thread-safe running total of hosted-API cost, with a limit.

    One instance is shared by every hosted client in a command, so the cap
    covers LLM and Jev calls together. Clients call :meth:`check` before a
    paid request and :meth:`add` after it; responses served from the cache are
    never added.
    """

    def __init__(self, limit_usd: float) -> None:
        if not limit_usd >= 0:  # written this way to reject NaN too: it would switch the cap off
            raise ValueError(f"spend limit must be a non-negative dollar amount, got {limit_usd!r}")
        self.limit_usd = float(limit_usd)
        self._total = 0.0
        self._lock = threading.Lock()

    @property
    def total(self) -> float:
        """Dollars recorded so far."""

        with self._lock:
            return self._total

    def add(self, cost_usd: float) -> None:
        """Record ``cost_usd``; raise once the running total passes the limit.

        The cost is recorded even when this raises (the money is already
        spent), so every later ``add`` and ``check`` raises as well.
        """

        if not (math.isfinite(cost_usd) and cost_usd >= 0):
            raise ValueError(f"cost must be a finite, non-negative dollar amount, got {cost_usd!r}")
        with self._lock:
            self._total += cost_usd
            total = self._total
        self._raise_if_over(total)

    def check(self) -> None:
        """Raise if the limit has already been passed; call before a paid request."""

        self._raise_if_over(self.total)

    def _raise_if_over(self, total: float) -> None:
        if total > self.limit_usd:
            raise SpendLimitExceeded(
                f"hosted-API spend ${total:.4f} passed the ${self.limit_usd:.2f} cap; "
                "raise EVAL_MAX_SPEND_USD to continue (cached responses are not charged again)"
            )
