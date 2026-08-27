from collections.abc import Sequence
from typing import Callable

import numpy as np

from src.ingestion.bar.bar import Bar

TargetFn = Callable[[Sequence[Bar], int], float | None]


def forward_log_return(horizon: int) -> TargetFn:
    """log(close[end_index + horizon] / close[end_index]) -- None past the series end."""

    def _target(bars: Sequence[Bar], end_index: int) -> float | None:
        target_index = end_index + horizon
        if target_index >= len(bars):
            return None
        return float(np.log(bars[target_index].close / bars[end_index].close))

    return _target


def forward_realized_volatility(vol_window: int, log: bool = True) -> TargetFn:
    """
    Std of consecutive log-returns over the `vol_window` bars following
    end_index -- realized volatility, not a single-bar return (volatility
    isn't defined over one observation). `log=True` returns log(std): unlike
    returns, realized vol is strictly positive and right-skewed, so a GP
    should regress its log rather than the raw value to keep residuals
    roughly symmetric and predictions positive after exponentiating back.
    """

    def _target(bars: Sequence[Bar], end_index: int) -> float | None:
        if end_index + vol_window >= len(bars):
            return None
        closes = np.array([bars[i].close for i in range(end_index, end_index + vol_window + 1)])
        log_returns = np.diff(np.log(closes))
        vol = float(np.std(log_returns))
        return float(np.log(vol)) if log else vol

    return _target
