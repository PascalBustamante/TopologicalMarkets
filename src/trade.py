from dataclasses import dataclass, field
from datetime import datetime


@dataclass(slots=True, frozen=True)
class Trade:
    """
    Canonical representation of a market trade.

    Every data source (Alpaca, Databento, Polygon, etc.)
    should be converted into this object.
    """

    symbol: str
    timestamp: datetime

    price: float
    volume: float

    exchange: str | None = None
    trade_id: int | None = None
    conditions: tuple[str, ...] = field(default_factory=tuple)
    tape: str | None = None

    @property
    def dollar_value(self) -> float:
        """Dollar value exchanged in this trade."""
        return self.price * self.volume