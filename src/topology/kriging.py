from collections.abc import Sequence

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel

from src.ingestion.bar.bar import Bar
from src.topology.landscape import landscape_vector
from src.topology.rolling import windowed_diagrams


def topological_features(
    bars: Sequence[Bar],
    window: int,
    step: int,
    horizon: int = 1,
    embedding_dimension: int = 2,
    embedding_delay: int = 3,
    homology_dimension: int = 1,
    resolution: int = 50,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Persistence-landscape feature matrix (one row per rolling window) paired
    with the forward log-return `horizon` bars past each window's end bar —
    the (X, y) training pair for a topologically-kerneled GP forecast.
    """
    vectors: list[np.ndarray] = []
    targets: list[float] = []

    for end_index, as_of_bar, diagram in windowed_diagrams(
        bars, window, step, embedding_dimension, embedding_delay
    ):
        target_index = end_index + horizon
        if target_index >= len(bars):
            continue

        vectors.append(
            landscape_vector(diagram, homology_dimension=homology_dimension, resolution=resolution)
        )
        targets.append(np.log(bars[target_index].close / as_of_bar.close))

    return np.array(vectors), np.array(targets)


def fit_topological_gp(X: np.ndarray, y: np.ndarray) -> GaussianProcessRegressor:
    """
    Kriging (GP regression) whose covariance is purely a function of
    persistence-landscape distance between windows: a standard RBF kernel
    applied to landscape vectors, so two windows are treated as correlated
    when their topology is similar, regardless of when they occurred.
    """
    kernel = ConstantKernel(1.0) * RBF(length_scale=1.0) + WhiteKernel(noise_level=1e-3)
    gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True, n_restarts_optimizer=3)
    gp.fit(X, y)
    return gp


if __name__ == "__main__":
    from datetime import datetime

    from alpaca.data.enums import DataFeed
    from dotenv import load_dotenv

    from src.ingestion.bar.factory import SamplingMethod, make_bar_builder
    from src.ingestion.bar.stream import sample_bars
    from src.ingestion.date_ranges import DateRange
    from src.sources.alpacha_ingestion import AlpacaTradeSource
    from src.topology.whitening import mahalanobis_whiten

    load_dotenv()

    symbol = "SPY"
    # Half a year instead of two weeks. Full (non-sparse) GP fitting is
    # O(n^3) in training points, so `step` is widened from 5 -> 20 to keep
    # the resulting window count (and thus fit time) tractable rather than
    # naively multiplying the 2-week window count by ~13x.
    date_range = DateRange.explicit(datetime(2024, 1, 2), datetime(2024, 7, 2))
    dollar_threshold = 1_000_000
    window, step, horizon = 30, 20, 3

    source = AlpacaTradeSource(feed=DataFeed.IEX)
    trades = source.iter_trades(symbol, date_range.start, date_range.end)

    builder = make_bar_builder(SamplingMethod.DOLLAR, threshold=dollar_threshold)
    bars = list(sample_bars(trades, builder))
    print(
        f"{symbol} {date_range.start.date()} to {date_range.end.date()}: "
        f"{len(bars)} dollar bars (threshold=${dollar_threshold:,})"
    )

    X, y = topological_features(bars, window=window, step=step, horizon=horizon)
    print(f"windows: {len(y)} (window={window}, step={step}, horizon={horizon}) -> X {X.shape}")

    # Chronological split — windows overlap (step < window), so the split
    # is approximate near the boundary, not a leak-free walk-forward backtest.
    n_train = int(len(y) * 0.8)
    X_train, X_test = X[:n_train], X[n_train:]
    y_train, y_test = y[:n_train], y[n_train:]
    print(f"train: {len(y_train)}  test: {len(y_test)}")

    n_components = 5
    X_train, X_test, variance_retained = mahalanobis_whiten(X_train, X_test, n_components)
    print(f"whitened to {n_components} components ({variance_retained:.2%} variance retained)")

    gp = fit_topological_gp(X_train, y_train)
    pred, std = gp.predict(X_test, return_std=True)

    residuals = pred - y_test
    rmse = float(np.sqrt(np.mean(residuals**2)))
    mae = float(np.mean(np.abs(residuals)))
    directional_accuracy = float(np.mean(np.sign(pred) == np.sign(y_test)))

    baseline_pred = np.full_like(y_test, y_train.mean())
    baseline_rmse = float(np.sqrt(np.mean((baseline_pred - y_test) ** 2)))

    print()
    print(f"RMSE:                 {rmse:.6f}")
    print(f"MAE:                  {mae:.6f}")
    print(f"directional accuracy: {directional_accuracy:.2%}")
    print(f"mean predictive std:  {std.mean():.6f}")
    print(f"baseline RMSE (predict train mean): {baseline_rmse:.6f}")
