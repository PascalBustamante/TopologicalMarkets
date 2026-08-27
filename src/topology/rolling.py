from collections.abc import Iterator, Sequence

import numpy as np

from src.ingestion.bar.bar import Bar
from src.topology.embedding import takens_embedding
from src.topology.landscape import landscape_norm
from src.topology.persistence import PersistencePair, rips_persistence


def windowed_diagrams(
    bars: Sequence[Bar],
    window: int,
    step: int,
    embedding_dimension: int = 2,
    embedding_delay: int = 3,
    rips_max_dimension: int = 2,
) -> Iterator[tuple[int, Bar, list[PersistencePair]]]:
    """
    Slide a window of `window` bars (stepping by `step`) over a bar series,
    yielding (window's end index, window's last Bar, persistence diagram)
    per window. Shared by any statistic computed per rolling window.

    `rips_max_dimension` must be >= 2 for H1 (loops) and >= 3 for H2
    (voids) to be present in the diagram — the Rips complex needs
    (k+1)-simplices to detect a k-dimensional hole. H2 also requires
    `embedding_dimension` >= 3: a 2D point cloud cannot contain a 3D void.
    """
    closes = np.array([bar.close for bar in bars])

    for start in range(0, len(bars) - window + 1, step):
        end = start + window
        cloud = takens_embedding(closes[start:end], embedding_dimension, embedding_delay)
        diagram = rips_persistence(cloud, max_edge_length=cloud.std() * 4, max_dimension=rips_max_dimension)
        yield end - 1, bars[end - 1], diagram


def rolling_landscape_norm(
    bars: Sequence[Bar],
    window: int,
    step: int,
    embedding_dimension: int = 2,
    embedding_delay: int = 3,
    homology_dimension: int = 1,
    rips_max_dimension: int = 2,
) -> list[tuple[Bar, float]]:
    """
    Landscape-norm statistic per rolling window, giving a time series of
    topological signal to compare across sampling schemes or plot against
    known market events.
    """
    return [
        (bar, landscape_norm(diagram, homology_dimension=homology_dimension))
        for _, bar, diagram in windowed_diagrams(
            bars, window, step, embedding_dimension, embedding_delay, rips_max_dimension
        )
    ]
