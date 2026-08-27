from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Kernel, WhiteKernel

from src.topology.evaluation import information_coefficient, quantile_spread
from src.topology.whitening import mahalanobis_whiten


@dataclass
class ExperimentResult:
    name: str
    n_train: int
    n_test: int
    rmse: float
    baseline_rmse: float
    mae: float
    directional_accuracy: float
    information_coefficient: float
    quantile_means: list[float]
    learned_kernel: str

    def __str__(self) -> str:
        spread = "  ".join(f"{m:+.6f}" for m in self.quantile_means)
        return (
            f"{self.name}\n"
            f"  n_train={self.n_train}  n_test={self.n_test}\n"
            f"  RMSE={self.rmse:.6f}  baseline_RMSE={self.baseline_rmse:.6f}  MAE={self.mae:.6f}\n"
            f"  directional_accuracy={self.directional_accuracy:.2%}  IC={self.information_coefficient:+.4f}\n"
            f"  quantile means: {spread}\n"
            f"  learned kernel: {self.learned_kernel}"
        )


def _evaluate(
    name: str, pred: np.ndarray, y_train: np.ndarray, y_test: np.ndarray, learned_kernel_desc: str
) -> ExperimentResult:
    residuals = pred - y_test
    rmse = float(np.sqrt(np.mean(residuals**2)))
    mae = float(np.mean(np.abs(residuals)))
    directional_accuracy = float(np.mean(np.sign(pred) == np.sign(y_test)))
    baseline_rmse = float(np.sqrt(np.mean((y_train.mean() - y_test) ** 2)))
    ic = information_coefficient(pred, y_test)
    quantiles = quantile_spread(pred, y_test, n_quantiles=5)

    return ExperimentResult(
        name=name,
        n_train=len(y_train),
        n_test=len(y_test),
        rmse=rmse,
        baseline_rmse=baseline_rmse,
        mae=mae,
        directional_accuracy=directional_accuracy,
        information_coefficient=ic,
        quantile_means=[mean for _, mean, _, _ in quantiles],
        learned_kernel=learned_kernel_desc,
    )


def run_experiment(
    name: str,
    X: np.ndarray,
    y: np.ndarray,
    kernel_factory: Callable[[], Kernel],
    train_frac: float = 0.8,
    whiten_components: int | None = None,
) -> ExperimentResult:
    """
    One consistent train/fit/evaluate path for comparing metrics (via
    `whiten_components`), kriging kernels (via `kernel_factory`), and
    feature representations (via what's passed as X) on equal footing.
    `kernel_factory` returns the structural (signal) kernel; a WhiteKernel
    nugget is always added on top, same as every prior kernel in this
    project's history.
    """
    n_train = int(len(y) * train_frac)
    X_train, X_test = X[:n_train], X[n_train:]
    y_train, y_test = y[:n_train], y[n_train:]

    if whiten_components is not None:
        X_train, X_test, _ = mahalanobis_whiten(X_train, X_test, whiten_components)

    kernel = kernel_factory() + WhiteKernel(noise_level=1e-3)
    gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True, n_restarts_optimizer=3)
    gp.fit(X_train, y_train)
    pred = gp.predict(X_test)

    return _evaluate(name, pred, y_train, y_test, str(gp.kernel_))


def run_precomputed_kernel_experiment(
    name: str,
    K_train: np.ndarray,
    K_test_train: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    noise_level: float = 1e-3,
) -> ExperimentResult:
    """
    Kriging posterior mean from a precomputed Gram matrix — for kernels
    (e.g. sliced Wasserstein on raw diagrams) that aren't naturally a
    function of numeric feature vectors, so sklearn's
    GaussianProcessRegressor (and its gradient-based hyperparameter
    optimizer) doesn't directly apply. Implements the same BLUP formula
    sklearn uses internally:
        y_hat(x*) = k(x*, X)^T (K + noise*I)^-1 (y - mean(y)) + mean(y)
    """
    n = K_train.shape[0]
    y_mean = y_train.mean()
    alpha = np.linalg.solve(K_train + noise_level * np.eye(n), y_train - y_mean)
    pred = K_test_train @ alpha + y_mean

    return _evaluate(name, pred, y_train, y_test, f"precomputed Gram matrix (noise_level={noise_level})")
