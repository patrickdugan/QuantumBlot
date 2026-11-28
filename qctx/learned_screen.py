"""Opt-in cosine retrieval with trained orthogonal subspaces.

Retains the original normalized rows. Training uses only supplied training data;
no centering, truncation of the returned scores, or learned similarity metric.
The floating-point safeguards are engineering margins, not interval certificates.
"""
from __future__ import annotations
import numpy as np
from .shell_spectrum import shell_probabilities


def normalize(x):
    a = np.asarray(x)
    if np.iscomplexobj(a):
        raise ValueError('real vectors only')
    a = np.asarray(a, dtype=np.float64)
    if a.ndim not in (1, 2) or not a.shape[-1] or not np.all(np.isfinite(a)):
        raise ValueError('expected finite real vector or matrix')
    scale = np.max(abs(a), axis=-1, keepdims=True)
    if np.any(scale == 0):
        raise ValueError('zero vector')
    a = a/scale
    return a/np.linalg.norm(a, axis=-1, keepdims=True)


def fit_basis(train, rank, method='pca', labels=None, seed=0):
    """Fit a fixed uncentered orthonormal frame; never fit on search queries.

    class_energy: equal quotas of top class-vs-rest second-moment contrast
    eigenvectors, followed by a global polar orthogonalization. It is a
    heuristic supervised objective, with no guarantee of better screening.
    """
    x = normalize(train)
    if x.ndim != 2 or len(x) < rank or not 1 <= rank <= x.shape[1]:
        raise ValueError('invalid training dimensions/rank')
    d = x.shape[1]
    if method == 'random':
        return np.linalg.qr(np.random.default_rng(seed).normal(size=(d,rank)), mode='reduced')[0]
    if method == 'pca':
        return np.linalg.svd(x, full_matrices=False)[2][:rank].T.copy()
    if method != 'class_energy':
        raise ValueError('unknown basis method')
    if labels is None or np.asarray(labels).shape != (len(x),):
        raise ValueError('class_energy needs one training label per row')
    labels = np.asarray(labels); classes = np.unique(labels)
    if len(classes) < 2 or len(classes) > rank:
        raise ValueError('need 2..rank classes')
    all_sum = x.T @ x; directions = []
    for i, c in enumerate(classes):
        z = x[labels == c]; a = z.T @ z
        contrast = a/len(z) - (all_sum-a)/(len(x)-len(z))
        quota = rank//len(classes) + (i < rank%len(classes))
        _, v = np.linalg.eigh(contrast)
        directions.append(v[:, -quota:][:, ::-1])
    candidates = np.concatenate(directions, axis=1)
    u, _, vt = np.linalg.svd(candidates, full_matrices=False)
    return u @ vt


def rank_top(scores, ids, k):
    """Deterministic top-k including ties (descending score, ascending row)."""
    scores = np.asarray(scores); ids = np.asarray(ids)
    if len(scores) > k:
        kth = np.partition(scores, len(scores)-k)[len(scores)-k]
        mask = scores >= kth
        scores, ids = scores[mask], ids[mask]
    ix = np.lexsort((ids, -scores))[:k]
    return ids[ix], scores[ix]


class LearnedScreen:
    """Energy and signed bounds on a common fixed orthogonal decomposition.

    A rank-r frame extends to an orthogonal basis. For power-of-two rank,
    the first r coordinates have dyadic block sizes 1,1,2,4,...,r/2; the
    complement is one merged block. Mapping this basis onto the real
    Fourier-shell subspaces implements an implicit learned rotation.
    """
    def __init__(self, vectors, basis=None, *, mode='signed', batch=512):
        self.x = normalize(vectors)
        if self.x.ndim != 2 or not len(self.x) or batch < 1:
            raise ValueError('invalid corpus')
        self.m, self.d = self.x.shape; self.mode = mode
        self.slack = 256*np.finfo(np.float64).eps*max(self.d, 1)
        self.ids = np.arange(self.m)
        if mode == 'shell':
            self.basis = None
            self.profile = np.empty((self.m, self.d.bit_length()))
            for start in range(0,self.m,batch):
                self.profile[start:start+batch] = np.sqrt(shell_probabilities(self.x[start:start+batch]))
            return
        if mode not in ('energy','signed'):
            raise ValueError('unknown screen mode')
        u = np.asarray(basis, dtype=np.float64)
        if u.ndim != 2 or u.shape[0] != self.d or not 1 <= u.shape[1] <= self.d or not np.all(np.isfinite(u)):
            raise ValueError('invalid basis')
        r = u.shape[1]
        if mode == 'energy' and (r & (r-1) or self.d & (self.d-1)):
            raise ValueError('dyadic energy mode needs power-of-two rank and dimension')
        defect = float(np.linalg.norm(u.T@u-np.eye(r), ord=2))
        if defect > 1e-8:
            raise ValueError('basis is not orthonormal')
        # Scale to a contraction, so I-UU^T is PSD in exact arithmetic.
        # The extra slack protects against computed norm/inner-product error.
        self.basis = u/np.sqrt(1+defect+self.slack)
        self.rank = r; self.defect = defect
        self.c = self.x @ self.basis
        self.residual = self._residual(self.c)
        if mode == 'energy':
            self.profile = self._profile(self.c, self.residual)

    def _residual(self, c):
        return np.sqrt(np.maximum(0., 1-np.sum(c*c,axis=-1)+self.slack))

    def _profile(self, c, residual):
        blocks = [abs(c[...,0])]
        for start in (1<<j for j in range(self.rank.bit_length()-1)):
            blocks.append(np.linalg.norm(c[...,start:2*start],axis=-1))
        return np.stack([*blocks,residual], axis=-1)

    def bounds(self, q):
        """Return upper bounds and cheap seed scores for a normalized query."""
        if self.mode == 'shell':
            upper = self.profile @ np.sqrt(shell_probabilities(q))
            return upper+self.slack, upper
        p = q @ self.basis
        residual = self._residual(p)
        signed = self.c @ p
        if self.mode == 'signed':
            upper = signed+self.residual*residual
        else:
            upper = self.profile @ self._profile(p,residual)
        return upper+self.slack, signed

    def search(self, query, k=20, *, pilot=64, fallback_fraction=.5, screen=True):
        """Seed an exact kth score, then evaluate every possible survivor.

        When the survivor set is large, use a contiguous full BLAS scan.
        Work counts include repeated pilot scores in that fallback.
        """
        q=normalize(query)
        if q.shape != (self.d,) or not 1 <= k <= self.m or pilot < 1 or not 0 < fallback_fraction <= 1:
            raise ValueError('invalid query parameters')
        if not screen:
            ids,scores=rank_top(self.x@q,self.ids,k)
            return dict(indices=ids.tolist(),scores=scores.tolist(),evaluated=self.m,unique_evaluated=self.m,fallback=True,survivors=self.m)
        upper, seed_score = self.bounds(q)
        n=min(self.m,max(k,pilot)); seeds,_=rank_top(seed_score,self.ids,n)
        seed_exact=self.x[seeds]@q
        _,top=rank_top(seed_exact,seeds,k); threshold=top[-1]
        mask=upper>=threshold-self.slack
        mask[seeds]=False; remaining=self.ids[mask]; survivors=n+len(remaining)
        fallback=survivors >= fallback_fraction*self.m
        if fallback:
            ids,scores=rank_top(self.x@q,self.ids,k); evaluated=self.m+n; unique=self.m
        else:
            ids,scores=rank_top(np.concatenate([seed_exact,self.x[remaining]@q]),np.concatenate([seeds,remaining]),k)
            evaluated=unique=survivors
        return dict(indices=ids.tolist(),scores=scores.tolist(),evaluated=int(evaluated),unique_evaluated=int(unique),fallback=bool(fallback),survivors=int(survivors))
