
APCA_API_ENDPOINT="https://paper-api.alpaca.markets/v2"
API_KEY = "PK43BORIQOXSF3X3UAB74RB76H"
SECRET_KEY = "66qe9NvsjxY1iTjLA4snHoo7EXRvvLLVLAjWkNRj4cSp"


from datetime import datetime
from alpaca.data.historical.stock import StockHistoricalDataClient
from alpaca.data.requests import StockTradesRequest


class AlpacaTradeSource:

    def __init__(self, api_key: str, secret_key: str):
        self.client = StockHistoricalDataClient(api_key, secret_key)

    def get_trades(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
    ):
        request = StockTradesRequest(
            symbol_or_symbols=symbol,
            start=start,
            end=end,
        )

        return self.client.get_stock_trades(request)

source = AlpacaTradeSource(API_KEY, SECRET_KEY)

trades = source.get_trades(
    "SPY",
    datetime(2024, 1, 2),
    datetime(2024, 1, 3),
)

df = trades.df.rename(
    columns={
        "size": "volume",
    }
)

df["dollar_value"] = df["price"] * df["volume"]

print(type(trades))

print(trades.df.head())
print(trades.df.columns)
print(trades.df.dtypes)

print(dir(trades))
print(trades.df.shape())