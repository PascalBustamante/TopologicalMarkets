import gudhi
import numpy as np

PersistencePair = tuple[int, tuple[float, float]]


def rips_persistence(
    point_cloud: np.ndarray,
    max_edge_length: float | None = None,
    max_dimension: int = 2,
) -> list[PersistencePair]:
    """
    Persistent homology of a point cloud via a Vietoris-Rips filtration.

    Returns (homology_dimension, (birth, death)) pairs — e.g. dimension-0
    pairs describe connected components merging, dimension-1 pairs describe
    loops opening and closing as the filtration radius grows.
    """
    rips_complex = gudhi.RipsComplex(points=point_cloud, max_edge_length=max_edge_length)
    simplex_tree = rips_complex.create_simplex_tree(max_dimension=max_dimension)
    return simplex_tree.persistence()
