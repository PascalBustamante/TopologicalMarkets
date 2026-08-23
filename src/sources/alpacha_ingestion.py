import argparse
import logging
import os
from datetime import datetime
from zipfile import Path

from alpaca.data.enums import DataFeed
from alpaca.data.historical.stock import StockHistoricalDataClient
from alpaca.data.requests import StockTradesRequest

from src.trade import Trade

logger = logging.getLogger(__name__)


class AlpacaTradeSource:
    """
    Pulls raw trades from Alpaca and converts them into canonical
    Trade objects. Handles pagination transparently — callers never
    see page tokens.
    """

    def __init__(
        self,
        api_key: str | None = None,
        secret_key: str | None = None,
        feed: DataFeed = DataFeed.SIP,
    ):
        api_key = api_key or os.environ.get("APCA_API_KEY_ID")
        secret_key = secret_key or os.environ.get("APCA_API_SECRET_KEY")

        if not api_key or not secret_key:
            raise ValueError(
                "Alpaca API credentials missing. Pass them explicitly or "
                "set APCA_API_KEY_ID / APCA_SECRET_KEY."
            )

        self.client = StockHistoricalDataClient(api_key, secret_key)
        self.feed = feed

    def get_trades(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        limit: int | None = None,
    ) -> list[Trade]:
        """
        Fetch all trades for `symbol` between start and end, handling
        pagination. Returns trades sorted chronologically.
        """
        all_trades: list[Trade] = []
        page_token: str | None = None

        while True:
            request = StockTradesRequest(
                symbol_or_symbols=symbol,
                start=start,
                end=end,
                feed=self.feed,
                limit=limit,
                page_token=page_token,
            )

            response = self.client.get_stock_trades(request)
            raw_trades = response[symbol] if hasattr(response, "__getitem__") else response.data.get(symbol, [])

            if not raw_trades:
                break

            all_trades.extend(self._to_trade(symbol, t) for t in raw_trades)

            page_token = getattr(response, "next_page_token", None)
            if not page_token:
                break

        logger.info("Fetched %d trades for %s (%s to %s)", len(all_trades), symbol, start, end)
        return all_trades

    @staticmethod
    def _to_trade(symbol: str, raw) -> Trade:
        return Trade(
            symbol=symbol,
            timestamp=raw.timestamp,
            price=raw.price,
            volume=raw.size,
            exchange=getattr(raw, "exchange", None),
            trade_id=getattr(raw, "id", None),
            conditions=tuple(getattr(raw, "conditions", None) or ()),
            tape=getattr(raw, "tape", None),
        )


if __name__ == "__main__":

    from dotenv import load_dotenv
    import os


    load_dotenv()
    import os
    from pathlib import Path

    print("cwd:", os.getcwd())
    print("this file:", Path(__file__).resolve())
    print(".env exists:", Path(".env").exists())
    parser = argparse.ArgumentParser(description="Fetch a small sample of Alpaca trades")
    parser.add_argument("symbol", nargs="?", default="SPY", help="Ticker symbol to test")
    parser.add_argument("--start", default="2024-01-02T00:00:00", help="Start time in ISO format")
    parser.add_argument("--end", default="2024-01-03T00:00:00", help="End time in ISO format")
    parser.add_argument("--limit", type=int, default=5, help="Number of trades to fetch per page")
    args = parser.parse_args()

    try:
        source = AlpacaTradeSource()
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    start = datetime.fromisoformat(args.start)
    end = datetime.fromisoformat(args.end)
    trades = source.get_trades(args.symbol, start, end, limit=args.limit)

    print(f"Fetched {len(trades)} trades for {args.symbol}")
    for trade in trades[:5]:
        print(trade)
