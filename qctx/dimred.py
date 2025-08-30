from __future__ import annotations
from typing import Optional
import numpy as np

try:
    import cupy as cp  # type: ignore
except Exception:  # pragma: no cover - optional
    cp = None

from sklearn.decomposition import PCA


def _to_numpy(x):
    if cp is not None and isinstance(x, cp.ndarray):
        return cp.asnumpy(x)
    return np.asarray(x)


def pca_project(vectors, n_components: int = 2):
    """Project vectors to n_components using scikit-learn's PCA.


    Accepts numpy or cupy arrays; will copy GPU -> CPU if needed for now.
    """
    X = _to_numpy(vectors)
    pca = PCA(n_components=n_components)
    return pca.fit_transform(X)
