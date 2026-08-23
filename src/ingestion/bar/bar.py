from dataclasses import dataclass
from datetime import datetime


@dataclass(slots=True, frozen=True)
class Bar:
    symbol: str
    open: float
    high: float
    low: float
    close: float
    volume: float
    dollar_volume: float
    n_ticks: int
    start_ts: datetime
    end_ts: datetime