"""First-pass cost model: commission + half-spread + square-root impact +
borrow. P3 replaces ``calibrated_cost`` with a fully calibrated version; the
interface is identical, so the P1 flagship plugs straight into it.

Per-unit-turnover cost (in dollars per dollar traded):
    cost = commission + half_spread
           + impact_k * sigma_daily * sqrt(participation)
           + (borrow_annual / 252) * short_weight_on_trade

``participation`` is share of the day's volume the trade consumes, capped at
``max_participation``. The capacity curve (P3) is a sweep of AUM through the
same function.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


class CostModel:
    def __init__(
        self,
        commission_bps: float = 0.5,
        half_spread_bps: float = 2.0,
        impact_k: float = 0.1,
        max_participation: float = 0.25,
        borrow_annual_pct: float = 2.0,
        sigma_daily: float = 0.01,
    ):
        self.commission_bps = commission_bps
        self.half_spread_bps = half_spread_bps
        self.impact_k = impact_k
        self.max_participation = max_participation
        self.borrow_annual_pct = borrow_annual_pct
        self.sigma_daily = sigma_daily

    @property
    def fixed_cost(self) -> float:
        return (self.commission_bps + self.half_spread_bps) / 10_000.0

    @property
    def borrow_daily(self) -> float:
        return self.borrow_annual_pct / 100.0 / 252.0

    def unit_cost(self, participation: float) -> float:
        """Per-dollar-traded cost at a given participation of ADV."""
        part = min(float(participation), self.max_participation)
        impact = self.impact_k * self.sigma_daily * np.sqrt(part)
        return self.fixed_cost + impact

    def shock(self, multiplier: float = 3.0) -> "CostModel":
        """Pessimistic variant: multiply the slippage-type costs only."""
        c = CostModel(
            commission_bps=self.commission_bps,
            half_spread_bps=self.half_spread_bps * multiplier,
            impact_k=self.impact_k * multiplier,
            max_participation=self.max_participation,
            borrow_annual_pct=self.borrow_annual_pct,
            sigma_daily=self.sigma_daily,
        )
        return c

    def __repr__(self) -> str:
        return (
            f"CostModel(commission={self.commission_bps}bp, half_spread={self.half_spread_bps}bp, "
            f"impact_k={self.impact_k}, max_part={self.max_participation}, borrow={self.borrow_annual_pct}%/yr)"
        )


def calibrated_cost(adv: pd.Series, dollar_trade: float, model: CostModel, adv_window: int = 5) -> float:
    """Cost in dollars for ``dollar_trade`` dollars executed against the
    trailing-average daily dollar volume in ``adv``."""
    adv_now = adv.rolling(adv_window, min_periods=1).mean().iloc[-1]
    if not np.isfinite(adv_now) or adv_now <= 0:
        return float("inf")
    participation = dollar_trade / adv_now
    return float(dollar_trade * model.unit_cost(participation))


def capacity_curve(
    model: CostModel,
    adv: pd.Series,
    gross_edge: float,
    ann: float = 252.0,
    n_points: int = 20,
    turnover_x: float = 3.5,
    n_names: int = 100,
) -> pd.DataFrame:
    """Net edge as a function of AUM, at a fixed monthly turnover of
    ``turnover_x``x AUM spread across a ``n_names``-name universe.

    ``gross_edge`` is gross alpha per dollar deployed per rebalance. As AUM
    rises, the share of each name's ADV the trade consumes (participation)
    rises, square-root impact rises with it, and net edge decays — the
    break-even AUM is where net edge hits zero.

    Two deliberate modeling choices:

    - Per-name participation, not aggregate: a cross-sectional strategy
      spreads its turnover across its holdings, so the monthly dollar trade
      per name is ``turnover_x * AUM / 21 / n_names``.
    - Participation is *uncapped*: ``max_participation`` is a single-order
      execution limit. A strategy whose AUM pushes per-name participation
      past that limit is already beyond capacity — which is exactly what the
      break-even point measures. Capping participation here would flatten the
      curve and hide the limit.
    """
    adv_now = float(adv.rolling(5, min_periods=1).mean().iloc[-1])
    aums = np.geomspace(1e6, max(adv_now * 1e6, 1e10), n_points)
    rows = []
    for aum in aums:
        part = aum * turnover_x / 21.0 / n_names / max(adv_now, 1.0)
        cost_per_dollar = model.fixed_cost + model.impact_k * model.sigma_daily * np.sqrt(part)
        net = gross_edge - cost_per_dollar
        rows.append({"aum": float(aum), "net_edge_per_dollar": float(net), "participation": float(part)})
    df = pd.DataFrame(rows)
    df["break_even"] = df["net_edge_per_dollar"] <= 0
    return df


__all__ = ["CostModel", "calibrated_cost", "capacity_curve"]
