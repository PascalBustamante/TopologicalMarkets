from src.trade import Trade
from src.ingestion.bar.base import BarBuilder


class VolumeBarBuilder(BarBuilder):
    def _is_complete(self, trade: Trade) -> bool:
        return self._volume >= self.threshold