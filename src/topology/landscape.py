import numpy as np
from gudhi.representations import Landscape

from src.topology.persistence import PersistencePair


def landscape_norm(
    diagram: list[PersistencePair],
    homology_dimension: int = 1,
    num_landscapes: int = 1,
    resolution: int = 100,
) -> float:
    """
    L2 norm of the persistence landscape for one homology dimension —
    a single scalar summarizing "how much topology" (e.g. loop structure,
    for dimension=1) a diagram contains. This is the statistic Gidea & Katz
    track over rolling windows as a regime-change indicator.
    """
    pairs = np.array(
        [(birth, death) for dim, (birth, death) in diagram if dim == homology_dimension and death < float("inf")]
    )
    if len(pairs) == 0:
        return 0.0

    landscape = Landscape(num_landscapes=num_landscapes, resolution=resolution)
    values = landscape.fit_transform([pairs])[0]
    return float(np.linalg.norm(values))
