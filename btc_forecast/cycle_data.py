# cycle_data.py
# ------------------------------------------------------------
# Data structures for Bitcoin cycle forecasting.
# ------------------------------------------------------------

from dataclasses import dataclass
from datetime import date
from typing import Optional

@dataclass
class CycleData:
    i: int
    halving: date
    H: float                       # halving-day price (USD)
    peak_date: Optional[date] = None
    P: Optional[float] = None      # peak price (USD)

    @property
    def M(self) -> Optional[float]:
        # Peak multiple vs halving-day price
        return (self.P / self.H) if (self.P is not None and self.H) else None

    @property
    def L(self) -> Optional[int]:
        # Lag (days) from halving to peak
        return (self.peak_date - self.halving).days if (self.peak_date is not None) else None