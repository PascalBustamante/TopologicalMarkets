from collections.abc import Sequence

import numpy as np

from src.ingestion.bar.bar import Bar
from src.topology.embedding import takens_embedding
from src.topology.landscape import landscape_norm
from src.topology.persistence import rips_persistence


def rolling_landscape_norm(
    bars: Sequence[Bar],
    window: int,
    step: int,
    embedding_dimension: int = 2,
    embedding_delay: int = 3,
    homology_dimension: int = 1,
) -> list[tuple[Bar, float]]:
    """
    Slide a window of `window` bars (stepping by `step`) over a bar series,
    embedding each window's closes and computing a landscape-norm statistic.

    Returns one (window's last Bar, statistic) pair per window, giving a
    time series of topological signal to compare across sampling schemes
    or plot against known market events.
    """
    closes = np.array([bar.close for bar in bars])
    results: list[tuple[Bar, float]] = []

    for start in range(0, len(bars) - window + 1, step):
        end = start + window
        window_closes = closes[start:end]

        cloud = takens_embedding(window_closes, embedding_dimension, embedding_delay)
        diagram = rips_persistence(cloud, max_edge_length=cloud.std() * 4)
        stat = landscape_norm(diagram, homology_dimension=homology_dimension)

        results.append((bars[end - 1], stat))

    return results
