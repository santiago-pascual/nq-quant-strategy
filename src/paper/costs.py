"""Explicit, versioned execution-cost schedules for realtime Paper."""
from __future__ import annotations

from dataclasses import dataclass
import json
from hashlib import sha256
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True)
class PaperCostPolicy:
    profile_id: str
    platform: str
    applicable_accounts: tuple[str, ...]
    symbol: str
    currency: str
    commission_per_contract_side: float
    exchange_fee_per_contract_side: float
    regulatory_fee_per_contract_side: float
    artificial_slippage_ticks: float
    source_url: str
    verified_date: str

    def __post_init__(self) -> None:
        if not self.profile_id or not self.platform or not self.symbol:
            raise ValueError("cost profile identity, platform, and symbol are required")
        if not self.source_url.startswith("https://") or not self.verified_date:
            raise ValueError("cost profile requires an HTTPS source and verification date")
        for name in ("commission_per_contract_side", "exchange_fee_per_contract_side",
                     "regulatory_fee_per_contract_side", "artificial_slippage_ticks"):
            value = float(getattr(self, name))
            if value < 0:
                raise ValueError(f"{name} must be non-negative")

    @property
    def total_cost_per_contract_side(self) -> float:
        return (self.commission_per_contract_side + self.exchange_fee_per_contract_side
                + self.regulatory_fee_per_contract_side)

    @property
    def identity(self) -> str:
        body=json.dumps(self.__dict__,sort_keys=True,separators=(",",":"))
        return sha256(body.encode()).hexdigest()

    @property
    def round_turn_per_contract(self) -> float:
        return 2.0 * self.total_cost_per_contract_side

    @classmethod
    def from_json(cls, path: str | Path) -> "PaperCostPolicy":
        return cls.from_mapping(json.loads(Path(path).read_text(encoding="utf-8")))

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "PaperCostPolicy":
        data = dict(value)
        data["applicable_accounts"] = tuple(data["applicable_accounts"])
        return cls(**data)


def calculate_trade_economics(
    *, direction: str, entry_price: float, exit_price: float, quantity: int,
    point_value: float, initial_risk_usd: float | None,
    policy: PaperCostPolicy,
) -> dict[str, float | None]:
    """Calculate full-position economics from executed prices and explicit costs."""
    if direction.lower() not in {"long", "short"}:
        raise ValueError("direction must be long or short")
    if quantity <= 0 or point_value <= 0:
        raise ValueError("quantity and point_value must be positive")
    signed_points = exit_price - entry_price
    if direction.lower() == "short":
        signed_points = -signed_points
    gross = signed_points * quantity * point_value
    commission = policy.commission_per_contract_side * quantity * 2
    exchange = policy.exchange_fee_per_contract_side * quantity * 2
    regulatory = policy.regulatory_fee_per_contract_side * quantity * 2
    total_costs = commission + exchange + regulatory
    net = gross - total_costs
    risk = float(initial_risk_usd) if initial_risk_usd is not None else None
    return {
        "gross_pnl": gross, "commission": commission, "exchange_fees": exchange,
        "regulatory_fees": regulatory, "total_costs": total_costs, "net_pnl": net,
        "realized_r": net / risk if risk is not None and risk > 0 else None,
    }
