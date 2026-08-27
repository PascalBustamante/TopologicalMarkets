import numpy as np
from sklearn.decomposition import PCA


def mahalanobis_whiten(
    X_train: np.ndarray, X_test: np.ndarray, n_components: int
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    PCA-whitening fit on X_train only (never X_test, to avoid leaking test
    covariance structure): project onto the top `n_components` principal
    components and divide each by its standard deviation. Euclidean distance
    in the resulting space equals Mahalanobis distance (w.r.t. the training
    covariance, truncated to `n_components`) in the original space -- this
    is the d_M an isotropic RBF should be using instead of raw Euclidean d.

    Returns (X_train whitened, X_test whitened, cumulative explained variance
    ratio retained by the truncation).
    """
    pca = PCA(n_components=n_components, whiten=True)
    X_train_w = pca.fit_transform(X_train)
    X_test_w = pca.transform(X_test)
    return X_train_w, X_test_w, float(np.sum(pca.explained_variance_ratio_))
