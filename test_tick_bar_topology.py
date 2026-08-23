from datetime import date

import numpy as np
from alpaca.data.enums import DataFeed
from dotenv import load_dotenv

from src.ingestion.bar.factory import SamplingMethod, make_bar_builder
from src.ingestion.bar.stream import sample_bars
from src.ingestion.date_ranges import DateRange
from src.sources.alpacha_ingestion import AlpacaTradeSource
from src.topology.embedding import takens_embedding
from src.topology.persistence import rips_persistence


def main() -> None:
    load_dotenv()

    symbol = "SPY"
    date_range = DateRange.single_day(date(2024, 1, 2))
    ticks_per_bar = 5

    # IEX is what most (non-SIP-subscribed) accounts have access to.
    source = AlpacaTradeSource(feed=DataFeed.IEX)
    trades = source.iter_trades(symbol, date_range.start, date_range.end)

    builder = make_bar_builder(SamplingMethod.TICK, threshold=ticks_per_bar)
    bars = list(sample_bars(trades, builder))

    print(f"{symbol} {date_range.start.date()}: {len(bars)} tick bars (ticks/bar={ticks_per_bar})")
    if len(bars) < 10:
        raise SystemExit("not enough bars for a meaningful embedding - widen the date range")

    closes = np.array([bar.close for bar in bars])

    dimension, delay = 2, 3
    cloud = takens_embedding(closes, dimension=dimension, delay=delay)
    print(f"embedding: dimension={dimension}, delay={delay} -> cloud shape {cloud.shape}")

    diagram = rips_persistence(cloud, max_edge_length=cloud.std() * 4)
    h1 = sorted((death - birth, birth, death) for dim, (birth, death) in diagram if dim == 1)

    if not h1:
        print("no H1 features found")
        return

    lifetime, birth, death = h1[-1]
    print(f"most persistent H1 loop: birth={birth:.3f} death={death:.3f} lifetime={lifetime:.3f}")
    if len(h1) > 1:
        print(f"next runner-up lifetime: {h1[-2][0]:.3f}")


if __name__ == "__main__":
    main()
