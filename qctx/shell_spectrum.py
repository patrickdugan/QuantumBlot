"""Dyadic Fourier-shell energy and coarse-to-exact cosine retrieval (NumPy only).

Shell j comprises frequencies of additive order 2**j, with j=0 for DC.
These are new valuation bands, not the contiguous FFT bands of prerank_wedge.
"""
from __future__ import annotations
import numpy as np


def _array(x):
    x = np.asarray(x, dtype=np.complex128 if np.iscomplexobj(x) else np.float64)
    if x.ndim == 0 or x.shape[-1] == 0 or not np.all(np.isfinite(x)):
        raise ValueError('expected finite nonempty vectors')
    return x


def _pow2(x):
    n = x.shape[-1]
    if n & (n-1):
        raise ValueError('last dimension must be a power of two; pad explicitly')
    return n


def _energy(x):
    e = np.einsum("...i,...i->...", x.real, x.real)
    if np.iscomplexobj(x):
        e = e + np.einsum("...i,...i->...", x.imag, x.imag)
    return e


def shell_probabilities(x):
    """Exact shell masses of the normalized QFT in exact arithmetic, O(N) work.

    Accept batches on leading axes; no FFT or symmetry assumption on x.
    Output last axis is ordered DC, additive order 2, 4, ..., N.
    """
    z = _array(x); n = _pow2(z); b = n.bit_length()-1
    scale = np.max(np.maximum(abs(z.real), abs(z.imag)), axis=-1, keepdims=True)
    if np.any(scale == 0):
        raise ValueError('zero vector has no normalized Fourier distribution')
    z = z/scale
    total = _energy(z)
    out = np.empty(z.shape[:-1]+(b+1,), dtype=np.float64)
    for j in range(b, 0, -1):
        half = z.shape[-1]//2
        u, v = z[..., :half], z[..., half:]
        out[..., j] = _energy(u-v) * (2.**(j-1-b)) / total
        z = u+v
    out[..., 0] = _energy(z) / (n*total)
    return out


def counts_shell_probabilities(counts, n_qubits, *, reverse_bits=False):
    """Aggregate measured frequency counts, O(nonzero bins) space/time plus output.

    Bit ordering must be provided by the caller; never inferred by best retrieval.
    """
    if not isinstance(n_qubits, int) or n_qubits < 0:
        raise ValueError('invalid qubit count')
    out = np.zeros(n_qubits+1, dtype=np.float64); total = 0
    for key, value in counts.items():
        if not isinstance(key, str) or len(key) != n_qubits or any(c not in '01' for c in key):
            raise ValueError('invalid bitstring')
        if not isinstance(value, (int, np.integer)) or isinstance(value, bool) or value < 0:
            raise ValueError('invalid count')
        k = int((key[::-1] if reverse_bits else key) or '0', 2)
        j = n_qubits-(k & -k).bit_length()+1 if k else 0
        out[j] += value; total += value
    if total <= 0:
        raise ValueError('empty counts')
    return out/total


def shell_coordinates(x):
    """Project to the unit-invariant shell basis; return coordinates and loss.

    Residual is squared relative norm outside the invariant subspace. This
    projection loses phase/amplitude detail for generic inputs; no silent use.
    """
    z = _array(x); n = _pow2(z)
    scale = np.max(np.maximum(abs(z.real), abs(z.imag)), axis=-1, keepdims=True)
    if np.any(scale == 0):
        raise ValueError('zero vector')
    z = z/scale
    z = z/np.sqrt(_energy(z))[..., None]
    c = [z[..., 0]]; residual = np.zeros(z.shape[:-1])
    for j in range(1, n.bit_length()):
        w = 1 << (j-1); step = n//(2*w); a = z[..., step::2*step]
        mean = np.mean(a, axis=-1)
        c.append(mean*np.sqrt(w)); residual += _energy(a-mean[..., None])
    return np.stack(c, axis=-1), residual


def shell_transform(c):
    """Normalized dyadic invariant QFT, O(number of shell coordinates)."""
    c = _array(c); n = c.shape[-1]-1; out = np.empty_like(c)
    # Prefix normalized by sqrt(2**j) avoids exponential weights.
    p = c[..., 0].copy(); t = 1/np.sqrt(2.)
    for j in range(1, n+1):
        out[..., n+1-j] = (p-c[..., j])*t
        p = (p+c[..., j])*t
    out[..., 0] = p
    return out


def _normalized(x):
    z = _array(x)
    scale = np.max(np.maximum(abs(z.real), abs(z.imag)), axis=-1, keepdims=True)
    if np.any(scale == 0):
        raise ValueError('zero vector')
    z = z/scale
    return z/np.sqrt(_energy(z))[..., None]


class ShellIndex:
    """Coarse shell Cauchy-Schwarz upper bounds; exact cosine reranking.

    Real part of complex inner product, stable row-index tie break. Floating
    point screening uses a conservative engineering margin, not interval math.
    Set screen=False for full exact-score scan. O(M log N) extra descriptors.
    """
    def __init__(self, vectors, batch=256):
        x = _array(vectors)
        if x.ndim != 2 or len(x) == 0 or batch < 1:
            raise ValueError('expected nonempty matrix and positive batch')
        n = _pow2(x)
        self.vectors = np.empty_like(x)
        self.profile = np.empty((len(x), n.bit_length()))
        for a in range(0, len(x), batch):
            z = _normalized(x[a:a+batch]); self.vectors[a:a+batch] = z
            self.profile[a:a+batch] = np.sqrt(shell_probabilities(z))

    def search(self, query, k=20, *, batch=64, screen=True):
        q = _normalized(query)
        if q.ndim != 1 or q.shape[0] != self.vectors.shape[1] or not 1 <= k <= len(self.vectors) or batch < 1:
            raise ValueError('invalid query, k, or batch')
        upper = self.profile @ np.sqrt(shell_probabilities(q))
        order = np.lexsort((np.arange(len(upper)), -upper)) if screen else np.arange(len(upper))
        margin = 128*np.finfo(float).eps*q.size
        keep = np.empty(0, dtype=int); scores = np.empty(0); evaluated = 0
        for a in range(0, len(order), batch):
            ix = order[a:a+batch]
            if screen and len(keep) == k and upper[ix[0]]+margin < scores[-1]:
                break
            ss = np.real(self.vectors[ix] @ q.conj()); evaluated += len(ix)
            keep = np.concatenate((keep, ix)); scores = np.concatenate((scores, ss))
            take = np.lexsort((keep, -scores))[:k]; keep, scores = keep[take], scores[take]
        return dict(indices=keep.tolist(), scores=scores.tolist(), evaluated=evaluated,
                    corpus_size=len(upper), screened=screen, numerical_margin=margin)


def main():
    import argparse, json
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('vectors', help='local .npy array with power-of-two final dimension')
    p.add_argument('--query', help='optional local query .npy; vectors must be a matrix')
    p.add_argument('--k', type=int, default=20)
    p.add_argument('--no-screen', action='store_true')
    args = p.parse_args(); x = np.load(args.vectors, allow_pickle=False)
    if args.query:
        result = ShellIndex(x).search(np.load(args.query, allow_pickle=False), args.k, screen=not args.no_screen)
    else:
        result = dict(schema='dyadic-additive-order-energy-v1', masses=shell_probabilities(x).tolist())
    print(json.dumps(result))


if __name__ == '__main__':
    main()
