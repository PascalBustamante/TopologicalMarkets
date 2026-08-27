from dataclasses import dataclass

import numpy as np

from src.topology.evaluation import information_coefficient
from src.topology.kriging import fit_topological_gp
from src.topology.whitening import mahalanobis_whiten


@dataclass
class WalkForwardFold:
    fold: int
    n_train: int
    n_test: int
    gp_rmse: float
    gp_ic: float
    ar1_rmse: float
    ar1_ic: float
    incremental_ic: float  # IC(GP prediction, out-of-sample AR(1) residuals)


def _ar1_out_of_sample(y_train: np.ndarray, y_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Fit AR(1) (y_t ~ a*y_{t-1} + b) on train only, predict out-of-sample on
    test. Returns (predictions, residuals) — the residuals are what a
    simple persistence model leaves unexplained, the bar any topological
    signal has to clear to be "incremental" rather than redundant.
    """
    prev_train, target_train = y_train[:-1], y_train[1:]
    A_train = np.vstack([prev_train, np.ones_like(prev_train)]).T
    coef, *_ = np.linalg.lstsq(A_train, target_train, rcond=None)

    prev_test = np.concatenate([[y_train[-1]], y_test[:-1]])
    A_test = np.vstack([prev_test, np.ones_like(prev_test)]).T
    pred = A_test @ coef
    return pred, y_test - pred


def walk_forward_validation(
    X: np.ndarray,
    y: np.ndarray,
    n_folds: int = 5,
    whiten_components: int = 5,
) -> list[WalkForwardFold]:
    """
    Expanding-window walk-forward validation: split the series
    chronologically into n_folds+1 chunks, and for fold i train on
    everything before chunk i, test on chunk i. Respects time order (no
    shuffling, no future leakage) and checks whether a finding from a
    single train/test split — e.g. a positive incremental IC over an AR(1)
    baseline — holds up across multiple, independent out-of-sample periods,
    rather than being a one-split artifact.
    """
    n = len(y)
    chunk_edges = np.linspace(0, n, n_folds + 2, dtype=int)

    results: list[WalkForwardFold] = []
    for i in range(1, n_folds + 1):
        train_end, test_end = chunk_edges[i], chunk_edges[i + 1]
        X_train, y_train = X[:train_end], y[:train_end]
        X_test, y_test = X[train_end:test_end], y[train_end:test_end]
        if len(y_test) < 10 or len(y_train) < 50:
            continue

        Xw_train, Xw_test, _ = mahalanobis_whiten(X_train, X_test, whiten_components)
        gp = fit_topological_gp(Xw_train, y_train)
        gp_pred = gp.predict(Xw_test)

        gp_rmse = float(np.sqrt(np.mean((gp_pred - y_test) ** 2)))
        gp_ic = information_coefficient(gp_pred, y_test)

        ar1_pred, ar1_resid = _ar1_out_of_sample(y_train, y_test)
        ar1_rmse = float(np.sqrt(np.mean((ar1_pred - y_test) ** 2)))
        ar1_ic = information_coefficient(ar1_pred, y_test)

        incremental_ic = information_coefficient(gp_pred, ar1_resid)

        results.append(
            WalkForwardFold(
                fold=i,
                n_train=len(y_train),
                n_test=len(y_test),
                gp_rmse=gp_rmse,
                gp_ic=gp_ic,
                ar1_rmse=ar1_rmse,
                ar1_ic=ar1_ic,
                incremental_ic=incremental_ic,
            )
        )

    return results


def summarize(results: list[WalkForwardFold]) -> str:
    ics = np.array([r.incremental_ic for r in results])
    lines = [
        f"fold {r.fold}: n_train={r.n_train} n_test={r.n_test}  "
        f"GP(RMSE={r.gp_rmse:.4f} IC={r.gp_ic:+.4f})  "
        f"AR1(RMSE={r.ar1_rmse:.4f} IC={r.ar1_ic:+.4f})  "
        f"incremental_IC={r.incremental_ic:+.4f}"
        for r in results
    ]
    lines.append("")
    lines.append(
        f"incremental IC across {len(results)} folds: "
        f"mean={ics.mean():+.4f}  std={ics.std():.4f}  "
        f"positive in {int((ics > 0).sum())}/{len(ics)} folds"
    )
    return "\n".join(lines)
