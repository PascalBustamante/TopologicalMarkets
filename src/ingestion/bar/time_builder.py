from src.trade import Trade
from src.ingestion.bar.base import BarBuilder


class TimeBarBuilder(BarBuilder):
    """Closes on elapsed wall-clock time (threshold in seconds) — the
    classical baseline sampling scheme, as opposed to the information-driven
    tick/volume/dollar builders."""

    def _is_complete(self, trade: Trade) -> bool:
        return (trade.timestamp - self._start_ts).total_seconds() >= self.threshold
