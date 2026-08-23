from src.trade import Trade
from src.ingestion.bar.base import BarBuilder


class DollarBarBuilder(BarBuilder):
    def _is_complete(self, trade: Trade) -> bool:
        return self._dollar_volume >= self.threshold
