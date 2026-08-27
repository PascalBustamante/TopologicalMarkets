import numpy as np
from gudhi.representations import SlicedWassersteinKernel

from src.topology.persistence import PersistencePair


def diagrams_for_dimension(
    diagrams: list[list[PersistencePair]], homology_dimension: int
) -> list[np.ndarray]:
    """
    Extract the finite (birth, death) pairs for one homology dimension from
    each window's mixed-dimension diagram — the input format GUDHI's
    diagram-native kernels (e.g. SlicedWassersteinKernel) expect, as
    opposed to landscape_vector's fixed-size numeric summary.
    """
    out = []
    for diagram in diagrams:
        pairs = [(b, d) for dim, (b, d) in diagram if dim == homology_dimension and d < float("inf")]
        out.append(np.array(pairs) if pairs else np.empty((0, 2)))
    return out


def sliced_wasserstein_gram(
    train_diagrams: list[np.ndarray],
    test_diagrams: list[np.ndarray],
    num_directions: int = 10,
    bandwidth: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Gram matrices for the sliced Wasserstein kernel (Carriere, Cuturi &
    Oudot, 2017) — a proven positive-semidefinite kernel operating directly
    on persistence diagrams, rather than a fixed-size vectorization like
    landscape_vector. Captures diagram structure an L2 landscape distance
    can discard (e.g. differences among many small, non-dominant features).
    """
    kernel = SlicedWassersteinKernel(num_directions=num_directions, bandwidth=bandwidth)
    K_train = kernel.fit_transform(train_diagrams)
    K_test_train = kernel.transform(test_diagrams)
    return K_train, K_test_train
