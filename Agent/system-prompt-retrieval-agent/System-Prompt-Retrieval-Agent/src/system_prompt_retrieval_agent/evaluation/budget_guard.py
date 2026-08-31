from __future__ import annotations


class CostExhausted(RuntimeError):
    """Raised when a charge would exceed daily or per-round budget cap."""


class BudgetGuard:
    """Tracks USD spending against configurable daily and per-round caps."""

    def __init__(self, daily_usd_cap: float, per_round_usd_cap: float) -> None:
        self.daily_cap = float(daily_usd_cap)
        self.round_cap = float(per_round_usd_cap)
        self.daily_spent = 0.0
        self.round_spent = 0.0

    def charge(self, usd: float) -> None:
        """
        Raise CostExhausted BEFORE charging if the charge would exceed either cap.
        Updates both daily_spent and round_spent on success.
        """
        usd = float(usd)
        if self.daily_spent + usd > self.daily_cap:
            raise CostExhausted(
                f"Daily budget cap ${self.daily_cap:.4f} would be exceeded "
                f"(spent ${self.daily_spent:.4f}, charging ${usd:.4f})"
            )
        if self.round_spent + usd > self.round_cap:
            raise CostExhausted(
                f"Per-round budget cap ${self.round_cap:.4f} would be exceeded "
                f"(spent ${self.round_spent:.4f}, charging ${usd:.4f})"
            )
        self.daily_spent += usd
        self.round_spent += usd

    def reset_round(self) -> None:
        """Reset per-round spending without touching daily_spent."""
        self.round_spent = 0.0
