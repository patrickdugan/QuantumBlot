from __future__ import annotations
import numpy as np
from sklearn.decomposition import PCA
try:
    import cupy as cp
except Exception:
    cp=None
def _to_numpy(x):
    if cp is not None and isinstance(x,cp.ndarray): return cp.asnumpy(x)
    return np.asarray(x)
def pca_project(vectors, n_components:int=2):
    X=_to_numpy(vectors); pca=PCA(n_components=n_components)
    return pca.fit_transform(X)
