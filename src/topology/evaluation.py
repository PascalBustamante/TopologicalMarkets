import numpy as np
from scipy.stats import spearmanr


def information_coefficient(pred: np.ndarray, actual: np.ndarray) -> float:
    """
    Spearman rank correlation between predicted and realized values.

    Alphas in finance are typically tiny relative to per-observation noise
    (here: mean returns ~1e-5 against a std of ~5e-4), so RMSE/directional
    accuracy on individual predictions are dominated by that noise floor
    and can look like "no signal" even when a real, small mean-shift
    exists. Rank correlation is the standard way to detect that shift
    without requiring individual predictions to be accurate.
    """
    return float(spearmanr(pred, actual).statistic)


def quantile_spread(
    pred: np.ndarray, actual: np.ndarray, n_quantiles: int = 5
) -> list[tuple[int, float, float, int]]:
    """
    Sort test windows into `n_quantiles` buckets by predicted value, and
    report each bucket's mean (and std-of-the-mean) realized return.

    Averaging over many windows per bucket cancels idiosyncratic noise in a
    way a single-point RMSE can't, so a real but small alpha shows up as a
    monotonic trend in bucket means even when it's invisible point-by-point.
    Returns (bucket_index, mean_return, standard_error, n_in_bucket).
    """
    order = np.argsort(pred)
    buckets = np.array_split(order, n_quantiles)
    return [
        (i, float(actual[b].mean()), float(actual[b].std() / np.sqrt(len(b))), len(b))
        for i, b in enumerate(buckets)
    ]
