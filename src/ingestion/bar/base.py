from abc import ABC, abstractmethod
from src.trade import Trade
from src.ingestion.bar.bar import Bar


class BarBuilder(ABC):
    """
    Stateful, incremental bar builder. Feed trades one at a time via
    update(); emits a completed Bar once the threshold is crossed.
    Works identically for historical replay and live streaming.
    """

    def __init__(self, threshold: float):
        self.threshold = threshold
        self._reset()

    def _reset(self) -> None:
        self._open: float | None = None
        self._high = float("-inf")
        self._low = float("inf")
        self._close: float | None = None
        self._volume = 0.0
        self._dollar_volume = 0.0
        self._n_ticks = 0
        self._start_ts = None
        self._end_ts = None
        self._symbol: str | None = None

    def update(self, trade: Trade) -> Bar | None:
        if self._open is None:
            self._open = trade.price
            self._start_ts = trade.timestamp
            self._symbol = trade.symbol

        self._high = max(self._high, trade.price)
        self._low = min(self._low, trade.price)
        self._close = trade.price
        self._volume += trade.volume
        self._dollar_volume += trade.dollar_value
        self._n_ticks += 1
        self._end_ts = trade.timestamp

        if self._is_complete(trade):
            bar = self._emit()
            self._reset()
            return bar

        return None

    def _emit(self) -> Bar:
        return Bar(
            symbol=self._symbol,
            open=self._open,
            high=self._high,
            low=self._low,
            close=self._close,
            volume=self._volume,
            dollar_volume=self._dollar_volume,
            n_ticks=self._n_ticks,
            start_ts=self._start_ts,
            end_ts=self._end_ts,
        )

    @abstractmethod
    def _is_complete(self, trade: Trade) -> bool:
        """Return True if the current bar should close on this trade."""