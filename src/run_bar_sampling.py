import logging
from datetime import date

from dotenv import load_dotenv

from src.ingestion.bar.factory import SamplingMethod, make_bar_builder
from src.ingestion.bar.stream import sample_bars
from src.ingestion.date_ranges import DateRange
from src.sources.alpacha_ingestion import AlpacaTradeSource

logger = logging.getLogger(__name__)

THRESHOLDS = {
    SamplingMethod.TICK: 100,
    SamplingMethod.VOLUME: 10_000,
    SamplingMethod.DOLLAR: 1_000_000,
}


def run(symbol: str, date_range: DateRange) -> None:
    source = AlpacaTradeSource()

    for method, threshold in THRESHOLDS.items():
        builder = make_bar_builder(method, threshold)
        trades = source.iter_trades(symbol, date_range.start, date_range.end)
        bars = list(sample_bars(trades, builder))

        print(f"{method.value}: {len(bars)} bars (threshold={threshold})")
        for bar in bars[:3]:
            print(" ", bar)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    load_dotenv()
    run("SPY", DateRange.single_day(date(2024, 1, 2)))
