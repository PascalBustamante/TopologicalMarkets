import numpy as np


def takens_embedding(series: np.ndarray, dimension: int, delay: int) -> np.ndarray:
    """
    Time-delay (Takens) embedding of a 1D series into R^dimension.

    Point i is [series[i], series[i + delay], ..., series[i + (dimension-1)*delay]],
    turning a scalar time series into a point cloud whose shape reflects the
    underlying dynamics (e.g. a periodic series embeds as a loop).
    """
    n_points = len(series) - (dimension - 1) * delay
    if n_points <= 0:
        raise ValueError(
            f"series of length {len(series)} too short for dimension={dimension}, delay={delay}"
        )

    return np.array(
        [series[i : i + dimension * delay : delay] for i in range(n_points)]
    )
