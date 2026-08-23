from src.trade import Trade
from src.ingestion.bar.base import BarBuilder


class TickBarBuilder(BarBuilder):
    def _is_complete(self, trade: Trade) -> bool:
        return self._n_ticks >= self.threshold