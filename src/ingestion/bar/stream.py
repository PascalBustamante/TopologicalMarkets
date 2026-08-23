from collections.abc import Iterable, Iterator

from src.trade import Trade
from src.ingestion.bar.bar import Bar
from src.ingestion.bar.base import BarBuilder


def sample_bars(trades: Iterable[Trade], builder: BarBuilder) -> Iterator[Bar]:
    """
    Consume a trade stream and yield each Bar as `builder` closes it.

    `trades` can be any iterable of Trade — a historical replay generator
    (e.g. AlpacaTradeSource.iter_trades()) or a live feed — since BarBuilder
    is fed one trade at a time either way.
    """
    for trade in trades:
        bar = builder.update(trade)
        if bar is not None:
            yield bar
