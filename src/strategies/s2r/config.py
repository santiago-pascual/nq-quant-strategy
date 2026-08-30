from dataclasses import dataclass


@dataclass(frozen=True)
class S2RConfig:
    stop_points: float = 25.0
    rr: float = 1.75
    horizon_bars: int = 20

    target_state: int = 2
    tail_percent: float = 17.5
    quality_threshold: float = 0.75

    volatility_low: float = 0.40
    volatility_high: float = 0.60

    mae_threshold_r: float = 0.70
    recovery_level_r: float = 0.20
    recovery_deadline_bars: int = 6
