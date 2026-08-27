import numpy as np
from gudhi.representations import Landscape

from src.topology.persistence import PersistencePair


def landscape_vector(
    diagram: list[PersistencePair],
    homology_dimension: int = 1,
    num_landscapes: int = 1,
    resolution: int = 100,
) -> np.ndarray:
    """
    Vectorized persistence landscape for one homology dimension — a
    finite-dimensional Hilbert-space representation of a diagram. Unlike a
    raw diagram (a multiset of points, no natural vector-space structure),
    this vector supports ordinary Euclidean distance/inner products, so
    standard kernels (RBF, etc.) applied to it are automatically valid.
    """
    pairs = np.array(
        [(birth, death) for dim, (birth, death) in diagram if dim == homology_dimension and death < float("inf")]
    )
    if len(pairs) == 0:
        return np.zeros(num_landscapes * resolution)

    landscape = Landscape(num_landscapes=num_landscapes, resolution=resolution)
    return landscape.fit_transform([pairs])[0]


def landscape_norm(
    diagram: list[PersistencePair],
    homology_dimension: int = 1,
    num_landscapes: int = 1,
    resolution: int = 100,
) -> float:
    """
    L2 norm of the persistence landscape — a single scalar summarizing "how
    much topology" (e.g. loop structure, for dimension=1) a diagram
    contains. The statistic Gidea & Katz track over rolling windows as a
    regime-change indicator.
    """
    vector = landscape_vector(diagram, homology_dimension, num_landscapes, resolution)
    return float(np.linalg.norm(vector))
